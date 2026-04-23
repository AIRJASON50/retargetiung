"""
Interaction-mesh-based hand retargeter.

Adapted from OmniRetarget for dexterous hands. Supports:
- robot_only: fixed base, 20 DOF, hand keypoints only
- object mode: 6DOF wrist + 20 finger = 26 DOF, hand + object surface points

Core algorithm: minimize Laplacian deformation energy subject to joint limits
and trust region, solved via iterative QP with daqp solver.
"""

from __future__ import annotations

import numpy as np
import qpsolvers
from scipy import sparse as sp
from scipy.spatial.transform import Rotation as RotLib
from tqdm import tqdm
from wuji_retargeting.mediapipe import estimate_frame_from_hand_points

from .config import (
    JOINTS_MAPPING_LEFT,
    JOINTS_MAPPING_RIGHT,
    MIDPOINT_SEGMENTS,
    HandRetargetConfig,
)
from .mediapipe_io import load_pkl_sequence
from .mesh_utils import (
    calculate_laplacian_coordinates,
    calculate_laplacian_matrix,
    create_interaction_mesh,
    filter_adjacency_by_distance,
    get_adjacency_list,
    get_midpoint_skeleton_adjacency,
    get_skeleton_adjacency,
)
from .mujoco_hand import PinocchioHandModel

# ==============================================================================
# Constants
# ==============================================================================

DEFAULT_LAPLACIAN_WEIGHT: float = 5.0
"""Laplacian deformation energy weight. Default 5.0 for 5:5:1 ratio (anchor:lap:smooth)."""

PINCH_DISTANCE_MIN: float = 0.010
"""Full pinch distance threshold in meters (10mm)."""

PINCH_DISTANCE_MAX: float = 0.030
"""Approaching-pinch distance threshold in meters (30mm)."""

PINCH_MAX_BOOST: float = 5.0
"""Maximum semantic weight boost for pinch-proximate keypoints."""

CAPSULE_RADIUS: float = 0.0075
"""Fingertip capsule radius in meters (7.5mm) for non-penetration constraint."""

PENETRATION_QUERY_THRESHOLD: float = 0.05
"""Max signed distance (meters) for fingertip-object proximity queries."""

HOCAP_N_ITER_FIRST: int = 20
"""SQP iterations for the first frame in HO-Cap object-interaction mode.

Measured first-frame usage is ~4 iter (see experiments/archive/warmup_diagnosis/
probe_s2_iters.py); the cap is kept intentionally loose but no longer a misleading
200. Early-stop on ||Δq|| < config.s2_convergence_delta always triggers first."""

FINGER_CHAINS_MP: tuple[tuple[int, ...], ...] = (
    (0, 1, 2, 3, 4),  # thumb:  WRIST, CMC,  MCP,  IP,  TIP
    (0, 5, 6, 7, 8),  # index:  WRIST, MCP,  PIP,  DIP, TIP
    (0, 9, 10, 11, 12),  # middle
    (0, 13, 14, 15, 16),  # ring
    (0, 17, 18, 19, 20),  # pinky
)
"""MediaPipe chain indices (WRIST → finger TIP) for each of the 5 fingers.

Shared by ``solve_angle_warmup`` and ``_compute_bone_dir_residuals_and_jac``
so the bone topology lives in exactly one place."""

_NON_THUMB_MCP_MP: frozenset[int] = frozenset({5, 9, 13, 17})
"""MediaPipe indices for non-thumb MCP landmarks. These are the only endpoints
whose robot-body surrogate is governed by ``config.mcp_surrogate``."""


# ==============================================================================
# Classes
# ==============================================================================


class InteractionMeshHandRetargeter:
    """Retargets MediaPipe hand landmarks to robot joint angles using
    interaction mesh Laplacian deformation minimization.

    Attributes:
        config: Retargeting configuration (weights, flags, joint mapping).
        hand: MuJoCo/Pinocchio hand model used for FK and Jacobians.
        mp_indices: Sorted list of MediaPipe keypoint indices used in mapping.
        body_names: Robot body names corresponding to ``mp_indices``.
        n_keypoints: Number of mapped keypoints (typically 21).
        q_lb: Lower joint limits, shape ``(nq,)``.
        q_ub: Upper joint limits, shape ``(nq,)``.
        nq: Number of joint DOFs (typically 20).
        laplacian_weight: Scalar weight for Laplacian deformation cost.
        global_scale: Uniform scale applied to source landmarks.
    """

    # ==========================================================================
    # Dunder Methods
    # ==========================================================================

    def __init__(self, config: HandRetargetConfig) -> None:
        """Initialize the retargeter from a configuration object.

        Args:
            config: Complete retargeting configuration including joint mapping,
                solver parameters, and optional feature flags.
        """
        self.config = config
        if config.floating_base:
            from .mujoco_hand import MuJoCoFloatingHandModel

            self.hand = MuJoCoFloatingHandModel(config.mjcf_path, config.hand_side)
        else:
            self.hand = PinocchioHandModel(config.mjcf_path)

        # Keypoint mapping: sorted by MediaPipe index. ``body_names`` is a
        # @property so it always reflects the current MCP/thumb-CMC surrogate
        # — downstream callers (visualization, tests) see the same body the
        # solver optimized against.
        self.mp_indices = sorted(config.joints_mapping.keys())
        self.n_keypoints = len(self.mp_indices)  # 21

        # Joint limits
        self.q_lb = self.hand.q_lb
        self.q_ub = self.hand.q_ub
        self.nq = self.hand.nq  # 20

        self.laplacian_weight = DEFAULT_LAPLACIAN_WEIGHT

        # Global scale: WujiHand is human-scale, no scaling needed (unlike OmniRetarget's humanoid)
        # Config allows override if ever needed for a differently-sized hand
        self.global_scale = self.config.global_scale

        self._pinch_d_min = PINCH_DISTANCE_MIN
        self._pinch_d_max = PINCH_DISTANCE_MAX
        self._pinch_max_boost = PINCH_MAX_BOOST

        # Thumb tip and other fingertip indices in mapped array
        self._thumb_tip_mapped = self.mp_indices.index(4)
        self._other_tips_mapped = [self.mp_indices.index(i) for i in [8, 12, 16, 20]]

        # Cached topology per frame (rebuilt each frame from source points)
        self._adj_list: list[list[int]] | None = None

        # OPERATOR2MANO: fixed rotation from MediaPipe wrist convention -> robot palm convention
        if config.hand_side == "left":
            from wuji_retargeting.mediapipe import OPERATOR2MANO_LEFT as OP2MANO
        else:
            from wuji_retargeting.mediapipe import OPERATOR2MANO_RIGHT as OP2MANO
        self._R_mano = np.array(OP2MANO, dtype=np.float64)

        # Link-midpoint mode: override keypoint count. We keep the mp-index
        # pair specification (``MIDPOINT_SEGMENTS``) and resolve bodies on
        # demand through ``_mp_body_pos_jacp`` so the surrogate flags reach
        # this mode too — avoiding a stale pre-computed body-name list.
        if config.use_link_midpoints:
            self._midpoint_spec = MIDPOINT_SEGMENTS
            self.n_keypoints = len(self._midpoint_spec)  # 20
            # TIP is at index 4f+3 for finger f
            self._thumb_tip_mapped = 3
            self._other_tips_mapped = [7, 11, 15, 19]

        # Excluded fingers from Laplacian gradient (angle-only control)
        self._excluded_finger_kp_indices: list[int] = []
        if config.exclude_fingers_from_laplacian:
            for finger_idx in config.exclude_fingers_from_laplacian:
                if config.use_link_midpoints:
                    # 20-midpoint mode: finger f has keypoints [4f, 4f+1, 4f+2, 4f+3]
                    self._excluded_finger_kp_indices.extend(range(4 * finger_idx, 4 * finger_idx + 4))
                else:
                    # 21-point mode: finger f has mp indices at positions in mp_indices
                    starts = [1, 5, 9, 13, 17]
                    for k in range(4):
                        mp_idx = starts[finger_idx] + k
                        if mp_idx in self.mp_indices:
                            self._excluded_finger_kp_indices.append(self.mp_indices.index(mp_idx))

    # ==========================================================================
    # Public Methods
    # ==========================================================================

    def solve_single_iteration(
        self,
        q_current: np.ndarray,
        q_prev_frame: np.ndarray,
        target_laplacian: np.ndarray,
        adj_list: list[list[int]],
        semantic_weights: np.ndarray | None = None,
        object_pts: np.ndarray | None = None,
        obj_frame: tuple[np.ndarray, np.ndarray] | None = None,
        angle_targets: tuple[np.ndarray, np.ndarray] | None = None,
        cosik_live_landmarks: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float]:
        """Single iteration: linearize Laplacian and solve QP.

        Args:
            q_current: (nq,) current joint angles (linearization point).
            q_prev_frame: (nq,) joint angles from previous frame (for smoothness).
            target_laplacian: (V, 3) target Laplacian coordinates from source.
            adj_list: Adjacency list from Delaunay.
            semantic_weights: (V,) per-keypoint weights, or None for uniform.
            object_pts: (M, 3) object points -- in object-local frame if
                obj_frame set, else world frame.
            obj_frame: (R_obj_inv, t_obj) for object-frame Laplacian, or None
                for world frame.
            angle_targets: (target_q, confidence) L2 anchor from Stage 1 cosine IK.
                Used when anchor_mode="l2". Mutually exclusive with cosik_live_landmarks.
            cosik_live_landmarks: (21, 3) MediaPipe landmarks for live cos-IK cost
                term (anchor_mode="cosik_live"). Mutually exclusive with angle_targets.
                Contributes ``w_rot · Σ_bones ‖d_rob(q) − d_src‖²`` at current q —
                anisotropic anchor that penalizes bone-direction changes but leaves
                low-sensitivity directions (e.g. MCP abduction, cross-finger
                geometry) for IM to explore.

        Returns:
            Tuple of (q_new, cost): updated joint angles and optimization cost.
        """
        n_hand = self.n_keypoints

        # FK at current q
        self.hand.forward(q_current)
        hand_pts = self._get_robot_keypoints()  # (n_hand, 3) in world/wrist frame

        # Jacobians in world/wrist frame
        J_hand = self._get_robot_jacobians()  # (3*n_hand, nq)

        # Transform to object-local frame if requested
        if obj_frame is not None:
            R_inv, t_obj = obj_frame
            hand_pts = (hand_pts - t_obj) @ R_inv.T  # (n_hand, 3) in object frame
            # Rotate each 3-row block of the Jacobian: J_obj = R_inv @ J_world
            J_blocks = J_hand.reshape(n_hand, 3, self.nq)
            J_hand = np.einsum("ij,njk->nik", R_inv, J_blocks).reshape(n_hand * 3, self.nq)

        # Build combined vertices: hand + object (both in same frame)
        if object_pts is not None:
            robot_pts = np.vstack([hand_pts, object_pts])
        else:
            robot_pts = hand_pts

        V = len(robot_pts)

        # Jacobians: hand points have J, object points have J=0
        J_V = np.zeros((3 * V, self.nq))
        J_V[: 3 * n_hand, :] = J_hand
        # J_V[3*n_hand:, :] = 0  (object points are fixed in object frame)

        # Zero out Jacobian for excluded fingers (angle-only control).
        # Their positions still participate in Delaunay/Laplacian as passive anchors.
        if self._excluded_finger_kp_indices:
            for kp_idx in self._excluded_finger_kp_indices:
                if kp_idx < n_hand:
                    J_V[3 * kp_idx : 3 * kp_idx + 3, :] = 0.0

        # Recompute Laplacian matrix from current positions
        L = calculate_laplacian_matrix(
            robot_pts,
            adj_list,
            distance_decay_k=self.config.laplacian_distance_weight_k,
        )
        L_sp = sp.csr_matrix(L)
        Kron = sp.kron(L_sp, sp.eye(3, format="csr"), format="csr")
        J_L = Kron @ J_V  # (3V, nq)

        # Current Laplacian of robot configuration
        lap0 = L_sp @ robot_pts  # (V, 3)
        lap0_vec = lap0.reshape(-1)  # (3V,)
        target_lap_vec = target_laplacian.reshape(-1)  # (3V,)

        # Laplacian weight: base weight * semantic weight per keypoint
        if semantic_weights is not None:
            w_v = self.laplacian_weight * semantic_weights
        else:
            w_v = self.laplacian_weight * np.ones(V)
        sqrt_w3 = np.sqrt(np.repeat(w_v, 3))

        # Optional: normalize Laplacian residual + Jacobian by a characteristic
        # length, turning the IM cost into a dimensionless quantity so that
        # laplacian_weight / anchor_weight / smooth_weight are pure relative
        # importance ratios. Default None = legacy behavior (raw meters).
        # Typical values: ~0.03 m (avg bone length) or ~0.08 m (palm-tip dist).
        L_char = self.config.interaction_mesh_length_scale
        if L_char is not None and L_char > 0:
            sqrt_w3 = sqrt_w3 / float(L_char)

        # Eliminate lap_var via equality constraint:
        # lap_var = J_L @ dq + lap0_vec
        # Cost becomes: ||sqrt_w * (J_L @ dq + lap0 - target_lap)||^2
        lap_residual = lap0_vec - target_lap_vec  # (3V,)
        # Row-scale J_L by sqrt_w3 without materializing the 3V x 3V diag matrix.
        # Old: np.diag(sqrt_w3) @ dense is an O(V^2 * nq) GEMM with a mostly-zero
        # operand. New: broadcast scalar-multiply each row, O(V * nq).
        J_L_dense = J_L.toarray() if sp.issparse(J_L) else J_L
        J_w = sqrt_w3[:, None] * J_L_dense  # (3V, nq)
        r_w = sqrt_w3 * lap_residual

        # Build QP: min (1/2) dq^T H dq + c^T dq
        nq = self.nq
        dq_smooth = q_prev_frame - q_current

        # H = J_w^T J_w + smooth * I + damping
        H = J_w.T @ J_w + self.config.smooth_weight * np.eye(nq) + 1e-12 * np.eye(nq)
        c = J_w.T @ r_w - self.config.smooth_weight * dq_smooth

        # Angle anchor contribution
        if angle_targets is not None:
            target_q, confidence = angle_targets
            angle_diff = q_current - target_q  # residual at current q
            for j in range(nq):
                if confidence[j] > 0:
                    w_a = confidence[j] * self.config.angle_anchor_weight
                    H[j, j] += w_a
                    c[j] += w_a * angle_diff[j]

        # Live cos-IK anchor: penalise ||d_rob(q) - d_src||² per bone at CURRENT q.
        # Hessian J^T J is anisotropic (has 1/bone_length factors), so directions that
        # don't affect bone direction (e.g. MCP yaw, cross-finger) remain free for IM.
        if cosik_live_landmarks is not None:
            w_rot = self.config.anchor_cosik_weight
            r_bones, J_bones = self._compute_bone_dir_residuals_and_jac(cosik_live_landmarks)
            if r_bones.size:
                H += w_rot * (J_bones.T @ J_bones)
                c += w_rot * (J_bones.T @ r_bones)

        # Box constraints: joint limits + trust region
        lb = -self.config.step_size * np.ones(nq)
        ub = self.config.step_size * np.ones(nq)
        if self.config.activate_joint_limits:
            lb = np.maximum(lb, self.q_lb - q_current)
            ub = np.minimum(ub, self.q_ub - q_current)

        # Non-penetration: linear inequality constraints G @ dq <= h (S2 scope)
        G, h = self._build_penetration_constraints(
            q_current, active=self.config.activate_non_penetration_s2
        )

        dq_val, n_shrinks, found = self._solve_qp_trust_shrink(H, c, G, h, lb, ub)
        self._s2_shrinks_this_frame += n_shrinks
        if not found:
            self._s2_stall_iters += 1
            return q_current.copy(), float("inf")

        q_new = q_current + dq_val
        q_new = np.clip(q_new, self.q_lb, self.q_ub)

        # Compute cost for convergence check
        cost = 0.5 * dq_val @ H @ dq_val + c @ dq_val

        return q_new, cost

    def solve_angle_warmup(
        self,
        q_current: np.ndarray,
        q_prev: np.ndarray,
        landmarks_21: np.ndarray,
        n_iters: int | None = None,
    ) -> np.ndarray:
        """Stage 1: bone-direction cosine IK (GMR-inspired, but unit-vector residuals
        instead of mink SE3 FrameTask).

        For each finger chain, match the unit bone direction between source and robot.
        Uses FK + Jacobian per iteration. Scale-invariant by construction -- does not
        depend on bone length, only direction.

        Cost per bone: w_rot * ||robot_bone_dir - source_bone_dir||^2

        Previously included a tip-position anchor (w_tip=100). Validated as near-no-op
        on Manus (max Δq ~1.9°, RMS 0.34°) and HO-Cap (RMS 0.24° across 710 frames / 3
        clips); removed for cleaner cost-constraint semantics (see
        doc/exp_warmup_tip_anchor_removal.md).

        Args:
            q_current: (nq,) current joint angles.
            q_prev: (nq,) joint angles from previous frame (for smoothness).
            landmarks_21: (21, 3) MediaPipe hand landmarks.

        Returns:
            (nq,) updated joint angles after bone direction alignment.
        """
        w_rot = self.config.angle_warmup_weight
        eps = 1e-8

        iters = n_iters if n_iters is not None else self.config.angle_warmup_iters
        for _ in range(iters):
            self.hand.forward(q_current)

            residuals = []
            J_rows = []

            for f, chain in enumerate(FINGER_CHAINS_MP):
                for k in range(4):  # 4 bones per finger
                    parent_mp = chain[k]
                    child_mp = chain[k + 1]

                    # Robot bone endpoints (honors config.mcp_surrogate for non-thumb MCP)
                    rp, Jp = self._mp_body_pos_jacp(parent_mp)
                    rc, Jc = self._mp_body_pos_jacp(child_mp)
                    e_rob = rc - rp
                    e_len = np.linalg.norm(e_rob)
                    if e_len < eps:
                        continue
                    d_rob = e_rob / e_len

                    # Source bone direction
                    e_src = landmarks_21[child_mp] - landmarks_21[parent_mp]
                    s_len = np.linalg.norm(e_src)
                    if s_len < eps:
                        continue
                    d_src = e_src / s_len

                    # Direction residual
                    res = d_rob - d_src

                    # Jacobian of unit direction: (I - d*d^T)/||e|| @ J_e
                    Je = Jc - Jp
                    P = (np.eye(3) - np.outer(d_rob, d_rob)) / e_len
                    J_dir = P @ Je

                    residuals.append(np.sqrt(w_rot) * res)
                    J_rows.append(np.sqrt(w_rot) * J_dir)

            if not residuals:
                break

            # Build QP directly: min (1/2) dq^T H dq + c^T dq
            # from ||r + J @ dq||^2 + smooth * ||dq - dq_smooth||^2
            nq = self.nq
            dq_smooth = q_prev - q_current

            # Accumulate H = J^T J + smooth * I, c = J^T r - smooth * dq_smooth
            H = self.config.smooth_weight * np.eye(nq) + 1e-12 * np.eye(nq)
            c = -self.config.smooth_weight * dq_smooth

            for res, J_row in zip(residuals, J_rows):
                # res: (3,), J_row: (3, nq)
                H += J_row.T @ J_row
                c += J_row.T @ res

            # Box constraints: joint limits + trust region
            lb = np.maximum(self.q_lb - q_current, -self.config.step_size)
            ub = np.minimum(self.q_ub - q_current, self.config.step_size)

            # Non-penetration hard constraint (warmup scope, pipeline invariant
            # when enabled). Linearized at current q, re-queried every iter.
            G, h = self._build_penetration_constraints(
                q_current, active=self.config.activate_non_penetration_warmup
            )

            dq, n_shrinks, found = self._solve_qp_trust_shrink(H, c, G, h, lb, ub)
            self._warmup_shrinks_this_frame += n_shrinks
            if not found:
                # Trust-region shrink exhausted: this iter is structurally
                # infeasible. Stall (q unchanged), exit warmup to let S2 try
                # with a clean cost set.
                self._warmup_stall_iters += 1
                break

            q_current = np.clip(q_current + dq, self.q_lb, self.q_ub)

        return q_current

    def _compute_bone_dir_residuals_and_jac(self, landmarks_21: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute stacked unit-bone-direction residuals and Jacobians at current q.

        Assumes ``self.hand.forward(q)`` has already been called at the desired
        linearization point. Used by the cosik_live anchor mode in
        ``solve_single_iteration`` and mirrored (but re-computed inline) in
        ``solve_angle_warmup``.

        For each of 20 bones (5 fingers × 4 bones):
            r_bone = (p_child - p_parent).normalized - (lm_child - lm_parent).normalized
            J_bone = (I - d_rob d_robᵀ) / ‖e_rob‖ · (J_child - J_parent)

        Returns:
            residuals: (n_valid_bones * 3,) stacked unit-vector differences.
            J_stacked: (n_valid_bones * 3, nq) stacked direction Jacobians.
        """
        eps = 1e-8

        res_list: list[np.ndarray] = []
        J_list: list[np.ndarray] = []
        for chain in FINGER_CHAINS_MP:
            for k in range(4):
                rp, Jp = self._mp_body_pos_jacp(chain[k])
                rc, Jc = self._mp_body_pos_jacp(chain[k + 1])
                e_rob = rc - rp
                e_len = float(np.linalg.norm(e_rob))
                if e_len < eps:
                    continue
                d_rob = e_rob / e_len

                e_src = landmarks_21[chain[k + 1]] - landmarks_21[chain[k]]
                s_len = float(np.linalg.norm(e_src))
                if s_len < eps:
                    continue
                d_src = e_src / s_len

                P = (np.eye(3) - np.outer(d_rob, d_rob)) / e_len

                res_list.append(d_rob - d_src)
                J_list.append(P @ (Jc - Jp))

        if not res_list:
            return np.zeros(0), np.zeros((0, self.nq))
        return np.concatenate(res_list), np.vstack(J_list)

    def _mp_body_name(self, mp_idx: int) -> str:
        """Return the robot body name that the current surrogate config resolves
        ``mp_idx`` to. For ``mcp_surrogate="midpoint"`` there is no single body —
        we return the link1 name (the body the visualization closest matches)
        and callers that need exact positions should use ``_mp_body_pos_jacp``.
        """
        _jm = JOINTS_MAPPING_LEFT if self.config.hand_side == "left" else JOINTS_MAPPING_RIGHT
        body = _jm[mp_idx]
        if mp_idx == 1 and self.config.thumb_cmc_surrogate == "link2":
            return body[: -len("link1")] + "link2"
        if mp_idx in _NON_THUMB_MCP_MP and self.config.mcp_surrogate == "link2":
            return body[: -len("link1")] + "link2"
        return body

    @property
    def body_names(self) -> list[str]:
        """Robot body names for the 21 (or 20, in midpoint mode) mapped keypoints.

        Surrogate-aware: non-thumb MCP and thumb CMC reflect the current
        ``config.mcp_surrogate`` / ``config.thumb_cmc_surrogate``. Kept as a
        property (not a field) so runtime config changes take effect without
        reconstructing the retargeter.

        Note: when ``config.mcp_surrogate == "midpoint"`` there is no single
        body; this property returns the link1 name as the "display-closest"
        answer. Consumers needing exact positions should call
        ``_mp_body_pos_jacp`` (or iterate ``get_body_positions(body_names)`` for
        an approximation).
        """
        return [self._mp_body_name(i) for i in self.mp_indices]

    def _mp_body_pos_jacp(self, mp_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (world position, positional Jacobian 3×nq) for the robot body
        corresponding to a MediaPipe landmark, honoring surrogate flags.

        Non-thumb MCP (mp_idx ∈ {5, 9, 13, 17}) — ``config.mcp_surrogate``
        (default ``"link2"``):

            "link1"     joints_mapping direct (flex-pivot body, legacy)
            "link2"     finger{f}_link1 → finger{f}_link2 (abd-pivot)
            "midpoint"  0.5·(link1 + link2) position + Jacobian

        Plus an optional constant offset ``config.mcp_surface_offset_m`` along the
        palm back-normal to approximate the MediaPipe skin-surface landmark
        (default 0; C5 ablation showed positive values degraded quality).

        Thumb CMC (mp_idx=1) — ``config.thumb_cmc_surrogate`` (default ``"link2"``):

            "link1"     finger1_link1 (first CMC hinge, legacy)
            "link2"     finger1_link2 (after 2nd CMC hinge)

        Other landmarks (wrist, thumb MCP/IP/TIP, other finger PIP/DIP/TIP)
        always use ``joints_mapping`` directly. Callers must have already called
        ``self.hand.forward(q)``. This is the single entry point for landmark →
        body mapping across cos-IK bones, warmup tip anchor, and IM keypoints.
        """
        _jm = JOINTS_MAPPING_LEFT if self.config.hand_side == "left" else JOINTS_MAPPING_RIGHT
        body = _jm[mp_idx]

        # Thumb CMC override
        if mp_idx == 1 and self.config.thumb_cmc_surrogate == "link2":
            body_l2 = body[: -len("link1")] + "link2"
            return self.hand.get_body_pos(body_l2), self.hand.get_body_jacp(body_l2)

        # Non-thumb MCP surrogate
        if mp_idx in _NON_THUMB_MCP_MP:
            mode = self.config.mcp_surrogate
            body_l2 = body[: -len("link1")] + "link2"
            if mode == "link1":
                p, J = self.hand.get_body_pos(body), self.hand.get_body_jacp(body)
            elif mode == "link2":
                p, J = self.hand.get_body_pos(body_l2), self.hand.get_body_jacp(body_l2)
            else:  # midpoint
                p = 0.5 * (self.hand.get_body_pos(body) + self.hand.get_body_pos(body_l2))
                J = 0.5 * (self.hand.get_body_jacp(body) + self.hand.get_body_jacp(body_l2))
            # Back-of-hand surface offset
            if self.config.mcp_surface_offset_m != 0.0:
                n = self._palm_back_normal()
                p = p + self.config.mcp_surface_offset_m * n
                # Jacobian unchanged: when wrist is locked, n is constant in world;
                # when wrist floats, this is a first-order approximation that
                # ignores the Jacobian of n w.r.t. wrist DOFs (small, tolerable).
            return p, J

        # Default: direct mapping
        return self.hand.get_body_pos(body), self.hand.get_body_jacp(body)

    def _palm_back_normal(self) -> np.ndarray:
        """Unit vector normal to the palm plane, pointing toward the back of the hand.

        Derived from three anchors: WRIST (palm_link), INDEX_MCP (finger2_link1),
        PINKY_MCP (finger5_link1). Cross product order is hand-side dependent so
        the resulting normal consistently points toward the back of the hand
        (opposite side of the palm surface where MediaPipe knuckle landmarks sit).
        """
        _jm = JOINTS_MAPPING_LEFT if self.config.hand_side == "left" else JOINTS_MAPPING_RIGHT
        p_wrist = self.hand.get_body_pos(_jm[0])
        p_idx = self.hand.get_body_pos(_jm[5])
        p_pinky = self.hand.get_body_pos(_jm[17])
        # Left hand: cross in this order points to the back of hand by OPERATOR2MANO
        # convention; right hand is mirrored so we flip the sign.
        n = np.cross(p_idx - p_wrist, p_pinky - p_wrist)
        if self.config.hand_side == "right":
            n = -n
        norm = np.linalg.norm(n)
        return n / norm if norm > 1e-8 else np.array([0.0, 0.0, 1.0])

    def _build_anchor_confidence(self) -> np.ndarray:
        """Per-joint confidence mask for the S2 angle-anchor cost term.

        Wrist translation (0-2): 0.0  (position, not an angle — never anchored)
        Wrist rotation    (3-5): 0.5  (weak anchor; S2 IM decides finer orientation)
        Finger MCP abduction    : 0.5 (cosine IK extracts abduction poorly from a
                                       single palm plane; let S2 correct it)
        Other finger joints     : 1.0 (full anchor strength)

        Used to be co-located with a redundant cosine-IK pass in
        ``_extract_cosik_targets``; that pass was removed as a zombie
        (see experiments/archive/warmup_diagnosis/probe_extract_targets.py —
        99.4% of frames showed q_target − q_S1 ~ 3e-5 rad after the extra iters).
        """
        confidence = np.ones(self.nq)
        if self.config.floating_base and self.nq > 20:
            confidence[:3] = 0.0
            confidence[3:6] = 0.5
            for f in range(5):
                confidence[6 + 4 * f + 1] = 0.5
        else:
            for f in range(5):
                confidence[4 * f + 1] = 0.5
        return confidence

    def retarget_frame(
        self,
        landmarks: np.ndarray,
        q_prev: np.ndarray,
        is_first_frame: bool = False,
        use_semantic_weights: bool = False,
        object_pts_world: np.ndarray | None = None,
        obj_frame: tuple[np.ndarray, np.ndarray] | None = None,
        object_pts_local: np.ndarray | None = None,
    ) -> np.ndarray:
        """Retarget a single frame of MediaPipe landmarks to joint angles.

        Args:
            landmarks: (21, 3) preprocessed landmarks (in aligned wrist frame).
            q_prev: (nq,) joint angles from previous frame (warm start).
            is_first_frame: If True, use more iterations.
            use_semantic_weights: Enable pinch-aware loss weighting.
            object_pts_world: (M, 3) object surface points in world/wrist
                frame, or None.
            obj_frame: (R_obj_inv, t_obj) for object-frame Laplacian, or None.
            object_pts_local: (M, 3) object points in object-local frame (for
                obj_frame mode).

        Returns:
            (nq,) optimized joint angles.
        """
        source_pts = self._extract_source_keypoints(landmarks)
        source_pts_full, adj_list, target_lap = self._build_topology(
            source_pts,
            object_pts_world,
            obj_frame,
            object_pts_local,
        )
        sem_w = self._compute_weights(source_pts, object_pts_world, use_semantic_weights)
        return self._run_optimization(
            q_prev,
            landmarks,
            source_pts_full,
            adj_list,
            target_lap,
            sem_w,
            object_pts_world,
            obj_frame,
            object_pts_local,
            is_first_frame,
        )

    def _build_topology(
        self,
        source_pts: np.ndarray,
        object_pts_world: np.ndarray | None = None,
        obj_frame: tuple[np.ndarray, np.ndarray] | None = None,
        object_pts_local: np.ndarray | None = None,
    ) -> tuple[np.ndarray, list[list[int]], np.ndarray]:
        """Build interaction mesh topology and compute target Laplacian.

        Returns:
            source_pts_full: Combined hand + object points.
            adj_list: Adjacency list from Delaunay/skeleton.
            target_laplacian: Target Laplacian coordinates.
        """
        if object_pts_world is not None and len(object_pts_world) == 0:
            object_pts_world = None

        if obj_frame is not None and object_pts_local is not None:
            R_inv, t_obj = obj_frame
            source_pts_obj = (source_pts - t_obj) @ R_inv.T
            source_pts_full = np.vstack([source_pts_obj, object_pts_local])
        elif object_pts_world is not None:
            source_pts_full = np.vstack([source_pts, object_pts_world])
        else:
            source_pts_full = source_pts

        V = len(source_pts_full)

        if self.config.use_link_midpoints and self.config.use_skeleton_topology:
            adj_list = get_midpoint_skeleton_adjacency(self.n_keypoints)
            if V > self.n_keypoints:
                adj_list = adj_list + [[] for _ in range(V - self.n_keypoints)]
        elif self.config.use_skeleton_topology:
            adj_list = get_skeleton_adjacency(self.n_keypoints)
            if V > self.n_keypoints:
                adj_list = adj_list + [[] for _ in range(V - self.n_keypoints)]
        else:
            _, simplices = create_interaction_mesh(source_pts_full)
            adj_list = get_adjacency_list(simplices, V)
            if self.config.delaunay_edge_threshold is not None:
                adj_list = filter_adjacency_by_distance(
                    adj_list,
                    source_pts_full,
                    self.config.delaunay_edge_threshold,
                )
        self._adj_list = adj_list

        target_lap = calculate_laplacian_coordinates(
            source_pts_full,
            adj_list,
            distance_decay_k=self.config.laplacian_distance_weight_k,
        )
        return source_pts_full, adj_list, target_lap

    def _compute_weights(
        self,
        source_pts: np.ndarray,
        object_pts_world: np.ndarray | None,
        use_semantic: bool,
    ) -> np.ndarray | None:
        """Compute per-vertex semantic weights for Laplacian cost."""
        if not use_semantic:
            return None
        sem_w = self._compute_semantic_weights(source_pts)
        if object_pts_world is not None:
            sem_w = np.concatenate([sem_w, np.ones(len(object_pts_world))])
        return sem_w

    def _build_penetration_constraints(
        self, q: np.ndarray, active: bool,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Linearize non-penetration at ``q`` into QP inequality rows.

        Each hand-geom × object pair within ``PENETRATION_QUERY_THRESHOLD``
        becomes one row of ``G @ dq <= h`` where:

            h_row =  phi + penetration_tolerance          (meters)
            G_row = -J_contact                             (1, nq)

        Equivalent to ``J_contact · dq >= -phi - tol`` — "the separation
        velocity along the contact normal must be at least large enough to
        clear (or stay clear of) penetration by ``tol`` meters within one
        linearized step". Matches OmniRetarget's
        ``_update_jacobians_and_phis_from_q`` + constraint assembly.

        Returns ``(None, None)`` when:
          * feature disabled (``active=False``),
          * no object is injected in the hand model,
          * no hand-object pairs are within the query threshold.

        Args:
            q: Current joint configuration. ``self.hand.forward(q)`` is
                invoked inside to refresh kinematics.
            active: Whether the caller (warmup / S2) wants the constraint
                this iteration. When ``False`` this function returns early.

        Returns:
            (G, h) arrays or (None, None) if there are no active rows.
        """
        if not active:
            return None, None
        if not hasattr(self.hand, "query_hand_penetration"):
            return None, None
        if not getattr(self.hand, "_has_object", False):
            return None, None

        self.hand.forward(q)
        rows = self.hand.query_hand_penetration(threshold=PENETRATION_QUERY_THRESHOLD)
        if not rows:
            return None, None

        tol = self.config.penetration_tolerance
        G = np.vstack([(-r[0]).reshape(1, -1) for r in rows])
        h = np.array([r[1] + tol for r in rows])
        return G, h

    def _solve_qp_trust_shrink(
        self,
        H: np.ndarray,
        c: np.ndarray,
        G: np.ndarray | None,
        h: np.ndarray | None,
        lb: np.ndarray,
        ub: np.ndarray,
    ) -> tuple[np.ndarray, int, bool]:
        """Solve a QP with progressive trust-region halving on infeasibility.

        Standard SQP practice: when the linearized QP is infeasible at the
        current q, it often means the trust-region box is asking for a bigger
        step than the active constraint set allows. Halving the box may
        recover a feasible small step; repeat up to
        ``config.penetration_max_trust_shrinks`` times.

        If all shrinks fail, the iteration is structurally infeasible
        (cost gradient and constraint set have no compatible direction in
        any sub-trust-region). Caller should stall and log.

        This is strictly preferable to either "stall at infeasibility"
        (loses progress) or "drop the hard constraint and redo"
        (breaks the invariant), both of which were considered earlier.

        Returns:
            (dq, n_shrinks, found) — dq has shape (nq,); zero-filled when
            not found. ``n_shrinks`` counts halvings actually used (0 on
            first-try success).
        """
        lb_s, ub_s = lb.copy(), ub.copy()
        max_shrinks = self.config.penetration_max_trust_shrinks
        for k in range(max_shrinks + 1):
            problem = qpsolvers.Problem(H, c, G=G, h=h, lb=lb_s, ub=ub_s)
            sol = qpsolvers.solve_problem(problem, solver="daqp")
            if sol.found:
                return sol.x, k, True
            lb_s *= 0.5
            ub_s *= 0.5
        return np.zeros_like(lb), max_shrinks + 1, False

    def _run_optimization(
        self,
        q_prev: np.ndarray,
        landmarks: np.ndarray,
        source_pts_full: np.ndarray,
        adj_list: list[list[int]],
        target_lap: np.ndarray,
        sem_w: np.ndarray | None,
        object_pts_world: np.ndarray | None = None,
        obj_frame: tuple[np.ndarray, np.ndarray] | None = None,
        object_pts_local: np.ndarray | None = None,
        is_first_frame: bool = False,
    ) -> np.ndarray:
        """Run S1 angle warmup + S2 Laplacian convergence loop.

        Pipeline (both stages stop on q-space delta < configured threshold):
          1. Warmup: run ``solve_angle_warmup`` in an outer loop to convergence.
             First frame uses ``angle_warmup_iters_first`` (larger budget, since
             q starts at default pose); subsequent frames use ``angle_warmup_iters``.
          2. Anchor target = warmup's converged ``q_current`` directly (no extra
             redundant cosine-IK pass; see ``_build_anchor_confidence`` docstring).
          3. S2: iterate ``solve_single_iteration`` (IM + anchor + smooth) until
             ``||Δq|| < s2_convergence_delta`` or ``n_iter`` cap.
        """
        n_iter = self.config.n_iter_first if is_first_frame else self.config.n_iter
        q_current = q_prev.copy()
        warmup_conv = self.config.warmup_convergence_delta
        s2_conv = self.config.s2_convergence_delta

        # Per-frame non-penetration telemetry counters (reset each frame).
        # Populated by solve_angle_warmup / solve_single_iteration via their
        # trust-shrink helper. Consumer: driver scripts pull these into
        # per-frame metrics dict for the ablation report.
        self._warmup_shrinks_this_frame = 0
        self._warmup_stall_iters = 0
        self._s2_shrinks_this_frame = 0
        self._s2_stall_iters = 0

        # Stage 1: cosine IK bone direction alignment
        if self.config.use_angle_warmup:
            lm_21 = landmarks[:21] if landmarks.shape[0] > 21 else landmarks
            max_warmup = self.config.angle_warmup_iters_first if is_first_frame else self.config.angle_warmup_iters
            for _ in range(max_warmup):
                q_before = q_current.copy()
                q_current = self.solve_angle_warmup(q_current, q_prev, lm_21, n_iters=1)
                if np.linalg.norm(q_current - q_before) < warmup_conv:
                    break

        # Anchor setup for S2. Two modes (see config.anchor_mode):
        #   "cosik_live" (current default) : bone-direction cos-IK cost re-evaluated
        #                                    at every S2 iter — single-level joint form
        #   "l2"         (legacy)          : q-space L2 pull toward warmup's q_S1
        # getattr fallback matches the dataclass default so out-of-date pickled
        # configs still pick the principled mode.
        anchor_mode = getattr(self.config, "anchor_mode", "cosik_live")
        use_anchor = self.config.use_angle_warmup and (
            self.config.angle_anchor_weight > 0 or self.config.anchor_cosik_weight > 0
        )
        angle_targets_pair = None
        cosik_landmarks = None
        if use_anchor:
            if anchor_mode == "cosik_live":
                cosik_landmarks = landmarks[:21] if landmarks.shape[0] > 21 else landmarks
            else:  # "l2"
                angle_targets_pair = (q_current.copy(), self._build_anchor_confidence())

        # Stage 2: Laplacian position refinement (stop on q-norm, not cost-delta)
        solve_obj_pts = object_pts_local if obj_frame is not None else object_pts_world
        for _ in range(n_iter):
            q_before = q_current.copy()
            q_current, _cost = self.solve_single_iteration(
                q_current,
                q_prev,
                target_lap,
                adj_list,
                sem_w,
                object_pts=solve_obj_pts,
                obj_frame=obj_frame,
                angle_targets=angle_targets_pair,
                cosik_live_landmarks=cosik_landmarks,
            )
            if np.linalg.norm(q_current - q_before) < s2_conv:
                break

        # Final residual penetration (post-SQP). Computed regardless of whether
        # constraints were enabled so ablation can compare D (no constraint)
        # against A/B/C. Populated onto self._frame_np_metrics for driver.
        final_pen_max = 0.0
        if hasattr(self.hand, "query_hand_penetration") and getattr(self.hand, "_has_object", False):
            self.hand.forward(q_current)
            pen = [-phi for _, phi, _, _ in self.hand.query_hand_penetration(
                threshold=PENETRATION_QUERY_THRESHOLD
            ) if phi < 0]
            final_pen_max = max(pen) if pen else 0.0

        self._frame_np_metrics = {
            "warmup_shrinks": self._warmup_shrinks_this_frame,
            "warmup_stalls":  self._warmup_stall_iters,
            "s2_shrinks":     self._s2_shrinks_this_frame,
            "s2_stalls":      self._s2_stall_iters,
            "final_pen_max_mm": float(final_pen_max * 1000.0),
            "struct_infeas":  (self._warmup_stall_iters + self._s2_stall_iters) > 5,
        }

        return q_current

    def retarget_sequence(
        self,
        pkl_path: str,
        hand_side: str = "left",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Retarget a full .pkl sequence.

        Args:
            pkl_path: Path to MediaPipe .pkl file.
            hand_side: "left" or "right".

        Returns:
            Tuple of (qpos_seq, timestamps):
                qpos_seq: (T, nq) joint angle sequence.
                timestamps: (T,) timestamps.
        """
        landmarks_seq, timestamps = load_pkl_sequence(pkl_path, hand_side)

        from .mediapipe_io import preprocess_sequence

        landmarks_seq = preprocess_sequence(
            landmarks_seq,
            self.config.mediapipe_rotation,
            hand_side=hand_side,
            global_scale=self.global_scale,
        )

        T = len(landmarks_seq)
        qpos_seq = np.zeros((T, self.nq))
        q_prev = self.hand.get_default_qpos()

        for t in tqdm(range(T), desc="Retargeting"):
            q_opt = self.retarget_frame(
                landmarks_seq[t],
                q_prev,
                is_first_frame=(t == 0),
            )
            qpos_seq[t] = q_opt
            q_prev = q_opt

        return qpos_seq, timestamps

    def retarget_hocap_sequence(
        self,
        clip: dict,
        use_semantic_weights: bool = False,
    ) -> np.ndarray:
        """Retarget a HO-Cap clip with object interaction.

        Preprocessing: wrist-center landmarks + rotate to robot frame.
        Primary path: wrist_q (6D pose from data) + OPERATOR2MANO (convention fix).
        Fallback: SVD estimation + OPERATOR2MANO (when wrist_q unavailable).

        Args:
            clip: dict from load_hocap_clip() with keys:
                landmarks, object_pts_local, object_t, object_q,
                wrist_q (optional), fps.
            use_semantic_weights: Enable pinch-aware weighting.

        Returns:
            qpos_seq: (T, nq) joint angle sequence.
        """
        from .mediapipe_io import transform_object_points

        landmarks_raw = clip["landmarks"]  # (T, 21, 3) world frame
        obj_pts_local = clip["object_pts_local"]  # (M, 3) mesh local
        obj_t = clip["object_t"]  # (T, 3)
        obj_q = clip["object_q"]  # (T, 4)
        T = len(landmarks_raw)

        qpos_seq = np.zeros((T, self.nq))
        q_prev = self.hand.get_default_qpos()

        # Save state -- restore on exit to avoid side effects across calls
        saved_q_lb = self.q_lb.copy()
        saved_q_ub = self.q_ub.copy()
        saved_n_iter_first = self.config.n_iter_first

        if self.config.floating_base and self.nq > 20:
            # Lock all 6 wrist DOF: position (0:3) and rotation (3:6)
            # SVD+MANO alignment puts landmarks in robot wrist frame already
            self.q_lb[:6] = 0.0
            self.q_ub[:6] = 0.0
        # Override n_iter_first with HOCAP_N_ITER_FIRST (currently 20 — loose upper
        # bound well above measured usage of ~4 iter; early-stop on ||Δq|| always
        # fires first). Uses save/restore because retarget_frame reads
        # self.config.n_iter_first directly.
        self.config.n_iter_first = HOCAP_N_ITER_FIRST

        # Inject object mesh only when at least one stage actually needs the
        # hard-constraint path. Injecting unconditionally rebuilds the MuJoCo
        # model (spec.compile()) and subtly changes retargeting behavior even
        # in D baseline -- observed as visual wrist misalignment on HO-Cap
        # clips that work correctly on main. Keep the gate here to preserve
        # D-mode parity with main; D-mode pen_max telemetry will therefore
        # report 0 (no object injected to measure against).
        np_any = (self.config.activate_non_penetration_warmup
                  or self.config.activate_non_penetration_s2)
        if np_any and hasattr(self.hand, "inject_object_mesh"):
            mesh_path = clip.get("mesh_path")
            if mesh_path:
                self.hand.inject_object_mesh(mesh_path, self.config.hand_side)
                # Update limits after model rebuild. Re-apply the SAME 6-DOF
                # wrist lock used in the non-inject path above (translation
                # 0:3 AND rotation 3:6). The previous [:3] slice was a typo —
                # leaving wrist rotation unlocked made C mode diverge from D
                # by up to 32° even on frames with zero penetration.
                self.q_lb = self.hand.q_lb
                self.q_ub = self.hand.q_ub
                saved_q_lb = self.q_lb.copy()
                saved_q_ub = self.q_ub.copy()
                if self.config.floating_base and self.nq > 20:
                    self.q_lb[:6] = 0.0
                    self.q_ub[:6] = 0.0

        try:
            for t in tqdm(range(T), desc="Retargeting (obj mode)"):
                obj_world = transform_object_points(obj_pts_local, obj_q[t], obj_t[t])
                # Force SVD alignment (more reliable than wrist_q for HO-Cap data)
                lm, obj_transformed = self._align_frame(landmarks_raw[t], None, obj_world)

                # Update object pose in retargeter's MuJoCo model — only when
                # the hand's object mesh was actually injected (gated above by
                # np_any). This places the object in the SAME hand-local frame
                # that play_hocap's viz model reconstructs via R_inv, so the
                # mj_geomDistance queries here correspond 1:1 to the
                # penetration the user sees on screen.
                #
                # R_align matches play_hocap.retarget_hand's formula:
                #   R_svd = estimate_frame_from_hand_points(lm_centered)
                #   R_align = R_svd @ OPERATOR2MANO   (and R_inv = R_align.T)
                #
                # Object position: (obj_t - wrist_world) @ R_align  (row-vector form)
                # Object rotation: R_align.T @ R_obj_world
                # This is the inverse of qpos_to_world: p_local = (p_world-wrist) @ R_align
                if np_any and self.hand._has_object:
                    wrist_world = landmarks_raw[t, 0]
                    lm_centered = landmarks_raw[t] - wrist_world
                    R_svd = estimate_frame_from_hand_points(lm_centered)
                    R_align = R_svd @ self._R_mano
                    obj_center_aligned = (obj_t[t] - wrist_world) @ R_align
                    R_obj_world = RotLib.from_quat(obj_q[t]).as_matrix()
                    R_obj_aligned = R_align.T @ R_obj_world
                    obj_quat_aligned = RotLib.from_matrix(R_obj_aligned).as_quat()
                    self.hand.set_object_pose(obj_center_aligned, obj_quat_aligned)

                # Build object-frame transform if enabled
                # obj_transformed and lm are in wrist-aligned frame
                # SVD alignment is forced (wrist_q=None), so use identity rotation
                if self.config.use_object_frame:
                    wrist_world = landmarks_raw[t, 0]
                    obj_c = obj_t[t] - wrist_world
                    R_obj_a = np.eye(3)
                    R_obj_inv = R_obj_a.T
                    frame_arg = (R_obj_inv, obj_c)
                    # Object points in their own local frame (pre-sampled mesh coords)
                    local_arg = obj_pts_local
                else:
                    frame_arg = None
                    local_arg = None

                q_opt = self.retarget_frame(
                    lm,
                    q_prev,
                    is_first_frame=(t == 0),
                    use_semantic_weights=use_semantic_weights,
                    object_pts_world=obj_transformed,
                    obj_frame=frame_arg,
                    object_pts_local=local_arg,
                )
                qpos_seq[t] = q_opt
                q_prev = q_opt
        finally:
            self.q_lb[:] = saved_q_lb
            self.q_ub[:] = saved_q_ub
            self.config.n_iter_first = saved_n_iter_first

        return qpos_seq

    # ==========================================================================
    # Private Methods
    # ==========================================================================

    def _align_frame(
        self,
        landmarks_t: np.ndarray,
        wrist_q_t: np.ndarray | None,
        obj_world_t: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Align one frame of hand landmarks + object points to robot convention.

        Primary path (wrist_q available): R_wrist.T @ OPERATOR2MANO
        Fallback (no wrist_q): SVD estimation via apply_mediapipe_transformations

        Args:
            landmarks_t: (21, 3) raw landmarks in world frame.
            wrist_q_t: (4,) wrist quaternion (xyzw) or None for SVD fallback.
            obj_world_t: (M, 3) object surface points in world frame.

        Returns:
            lm: (21, 3) aligned landmarks (wrist at origin, robot convention).
            obj: (M, 3) aligned object points (same frame as landmarks).
        """
        wrist = landmarks_t[0]
        lm_centered = landmarks_t - wrist
        obj_wrist = obj_world_t - wrist

        if wrist_q_t is not None:
            R_wrist = RotLib.from_quat(wrist_q_t).as_matrix()
            R_align = R_wrist.T @ self._R_mano
            lm = lm_centered @ R_align
            obj = obj_wrist @ R_align
        else:
            from .mediapipe_io import preprocess_landmarks

            lm = preprocess_landmarks(
                landmarks_t,
                self.config.mediapipe_rotation,
                hand_side=self.config.hand_side,
                global_scale=1.0,  # scale applied separately below
                use_mano_rotation=self.config.use_mano_rotation,
            )
            # SVD fallback for object rotation
            if self.config.use_mano_rotation:
                from wuji_retargeting.mediapipe import apply_mediapipe_transformations

                trans = apply_mediapipe_transformations(landmarks_t.copy(), self.config.hand_side)
                R_t, _, _, _ = np.linalg.lstsq(lm_centered[1:6], trans[1:6], rcond=None)
                obj = obj_wrist @ R_t
                angles = [self.config.mediapipe_rotation.get(k, 0) for k in "xyz"]
                if any(a != 0 for a in angles):
                    R_extra = RotLib.from_euler("xyz", angles, degrees=True).as_matrix()
                    obj = obj @ R_extra.T
            else:
                obj = obj_wrist

        if self.global_scale != 1.0:
            lm *= self.global_scale
            obj *= self.global_scale

        return lm, obj

    def _extract_source_keypoints(self, landmarks: np.ndarray) -> np.ndarray:
        """Extract the mapped keypoints from landmarks array.

        If link midpoints enabled: returns 20 midpoints from MediaPipe 21 pts.
        Default: returns landmarks[mp_indices] (21 pts).
        """
        if self.config.use_link_midpoints:
            pts = np.empty((self.n_keypoints, 3))
            for k, (parent, child) in enumerate(self._midpoint_spec):
                if child is None:
                    pts[k] = landmarks[parent]
                else:
                    pts[k] = (landmarks[parent] + landmarks[child]) / 2
            return pts
        return landmarks[self.mp_indices]

    def _compute_semantic_weights(self, source_pts: np.ndarray) -> np.ndarray:
        """Compute per-keypoint weights based on pinch proximity.

        Only monitors thumb-to-other-fingertip pairs (4 pairs).

        Args:
            source_pts: (n_keypoints, 3) hand-only keypoints (NOT including
                object points).

        Returns:
            (n_keypoints,) weight array, 1.0 for non-pinch, up to max_boost
            for pinch.
        """
        if len(source_pts) != self.n_keypoints:
            raise ValueError(f"Expected {self.n_keypoints} keypoints, got {len(source_pts)}")
        w = np.ones(self.n_keypoints)

        for tip_mapped in self._other_tips_mapped:
            dist = np.linalg.norm(source_pts[self._thumb_tip_mapped] - source_pts[tip_mapped])
            if dist < self._pinch_d_max:
                t = (self._pinch_d_max - dist) / (self._pinch_d_max - self._pinch_d_min)
                boost = 1.0 + (self._pinch_max_boost - 1.0) * np.clip(t, 0.0, 1.0)
                w[self._thumb_tip_mapped] = max(w[self._thumb_tip_mapped], boost)
                w[tip_mapped] = max(w[tip_mapped], boost)

        return w

    def _get_robot_keypoints(self) -> np.ndarray:
        """Get robot body positions for mapped keypoints. Returns (N, 3).

        All paths (standard 21-point and link-midpoint 20-point) resolve
        bodies through ``_mp_body_pos_jacp``, so ``config.mcp_surrogate`` and
        ``config.thumb_cmc_surrogate`` reach every keypoint consistently.
        """
        if self.config.use_link_midpoints:
            pts = np.empty((self.n_keypoints, 3))
            for k, (parent_mp, child_mp) in enumerate(self._midpoint_spec):
                p_parent, _ = self._mp_body_pos_jacp(parent_mp)
                if child_mp is None:
                    pts[k] = p_parent
                else:
                    p_child, _ = self._mp_body_pos_jacp(child_mp)
                    pts[k] = 0.5 * (p_parent + p_child)
            return pts
        pts = np.empty((self.n_keypoints, 3))
        for k, mp_idx in enumerate(self.mp_indices):
            pts[k], _ = self._mp_body_pos_jacp(mp_idx)
        return pts

    def _get_robot_jacobians(self) -> np.ndarray:
        """Get stacked positional Jacobians for mapped keypoints. Returns (3N, nq).

        Mirrors ``_get_robot_keypoints``: uses ``_mp_body_pos_jacp`` so MCP
        surrogate choice is consistent across cos-IK and IM Jacobians in both
        the standard and link-midpoint keypoint layouts.
        """
        if self.config.use_link_midpoints:
            J = np.zeros((3 * self.n_keypoints, self.nq))
            for k, (parent_mp, child_mp) in enumerate(self._midpoint_spec):
                _, J_parent = self._mp_body_pos_jacp(parent_mp)
                if child_mp is None:
                    J[3 * k : 3 * k + 3] = J_parent
                else:
                    _, J_child = self._mp_body_pos_jacp(child_mp)
                    J[3 * k : 3 * k + 3] = 0.5 * (J_parent + J_child)
            return J
        J = np.zeros((3 * self.n_keypoints, self.nq))
        for k, mp_idx in enumerate(self.mp_indices):
            _, J[3 * k : 3 * k + 3] = self._mp_body_pos_jacp(mp_idx)
        return J
