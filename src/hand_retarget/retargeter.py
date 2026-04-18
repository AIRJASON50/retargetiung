"""
Interaction-mesh-based hand retargeter.

Adapted from OmniRetarget for dexterous hands. Supports:
- robot_only: fixed base, 20 DOF, hand keypoints only
- object mode: 6DOF wrist + 20 finger = 26 DOF, hand + object surface points

Core algorithm: minimize Laplacian deformation energy subject to joint limits
and trust region, solved via SQP with CVXPY + Clarabel SOCP solver.
"""

from __future__ import annotations

import numpy as np
import qpsolvers
from scipy import sparse as sp
from scipy.spatial.transform import Rotation as RotLib
from tqdm import tqdm

from .config import (
    JOINTS_MAPPING_LEFT,
    JOINTS_MAPPING_RIGHT,
    MIDPOINT_SEGMENTS,
    HandRetargetConfig,
    _build_midpoint_body_pairs,
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
from .mujoco_hand import MuJoCoHandModel

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

HOCAP_N_ITER_FIRST: int = 200
"""SQP iterations for the first frame in HO-Cap object-interaction mode."""


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
            self.hand = MuJoCoHandModel(config.mjcf_path)

        # Keypoint mapping: sorted by MediaPipe index
        self.mp_indices = sorted(config.joints_mapping.keys())
        self.body_names = [config.joints_mapping[i] for i in self.mp_indices]
        self.n_keypoints = len(self.mp_indices)  # 21

        # Joint limits
        self.q_lb = self.hand.q_lb
        self.q_ub = self.hand.q_ub
        self.nq = self.hand.nq  # 20

        self.laplacian_weight = DEFAULT_LAPLACIAN_WEIGHT

        # Global scale: WujiHand is human-scale, no scaling needed (unlike OmniRetarget's humanoid)
        # Config allows override if ever needed for a differently-sized hand
        self.global_scale = self.config.global_scale if self.config.global_scale is not None else 1.0

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

        # Link-midpoint mode: override keypoint count and build midpoint specs
        if config.use_link_midpoints:
            self._midpoint_spec = MIDPOINT_SEGMENTS
            self._midpoint_body_pairs = _build_midpoint_body_pairs(
                JOINTS_MAPPING_LEFT if config.hand_side == "left" else JOINTS_MAPPING_RIGHT
            )
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

        # Edge ratio: T-pose refs (computed on first frame)

        # Angle warmup: cache T-pose body points at default pose
        if config.use_angle_warmup:
            q0 = self.hand.get_default_qpos()
            self.hand.forward(q0)
            _jm = JOINTS_MAPPING_LEFT if config.hand_side == "left" else JOINTS_MAPPING_RIGHT
            self._tpose_body_pts = self.hand.get_body_positions([_jm[i] for i in range(21)])
            self._angle_q0 = q0.copy()

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
    ) -> tuple[np.ndarray, float]:
        """Single SQP iteration: linearize and solve SOCP sub-problem.

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
            angle_targets: (target_q, confidence) from Stage 1 cosine IK, or None.

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
            uniform_weight=True,
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

        # Eliminate lap_var via equality constraint:
        # lap_var = J_L @ dq + lap0_vec
        # Cost becomes: ||sqrt_w * (J_L @ dq + lap0 - target_lap)||^2
        lap_residual = lap0_vec - target_lap_vec  # (3V,)
        J_w = np.diag(sqrt_w3) @ J_L.toarray() if sp.issparse(J_L) else np.diag(sqrt_w3) @ J_L
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

        # Box constraints: joint limits + trust region
        lb = -self.config.step_size * np.ones(nq)
        ub = self.config.step_size * np.ones(nq)
        if self.config.activate_joint_limits:
            lb = np.maximum(lb, self.q_lb - q_current)
            ub = np.minimum(ub, self.q_ub - q_current)

        # Non-penetration: linear inequality constraints G @ dq <= h
        G_rows = []
        h_vals = []
        if self.config.activate_non_penetration and hasattr(self.hand, "query_tip_penetration"):
            for J_contact, phi, _ in self.hand.query_tip_penetration(threshold=PENETRATION_QUERY_THRESHOLD):
                # J_contact @ dq >= -phi - capsule_radius  →  -J_contact @ dq <= phi + capsule_radius
                G_rows.append(-J_contact.reshape(1, -1))
                h_vals.append(phi + CAPSULE_RADIUS)

        G = np.vstack(G_rows) if G_rows else None
        h = np.array(h_vals) if h_vals else None

        problem = qpsolvers.Problem(H, c, G=G, h=h, lb=lb, ub=ub)
        solution = qpsolvers.solve_problem(problem, solver="daqp")

        if not solution.found:
            return q_current.copy(), float("inf")

        dq_val = solution.x
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
    ) -> np.ndarray:
        """Stage 1: GMR-style bone direction alignment using IK.

        For each finger chain, match the bone direction (cosine) between
        source and robot. Uses FK + Jacobian per iteration.

        Cost per bone: w_rot * ||robot_bone_dir - source_bone_dir||^2
        End-effector (TIP): w_tip * ||robot_tip_pos - source_tip_pos||^2

        Args:
            q_current: (nq,) current joint angles.
            q_prev: (nq,) joint angles from previous frame (for smoothness).
            landmarks_21: (21, 3) MediaPipe hand landmarks.

        Returns:
            (nq,) updated joint angles after bone direction alignment.
        """
        chains_mp = [
            [0, 1, 2, 3, 4],
            [0, 5, 6, 7, 8],
            [0, 9, 10, 11, 12],
            [0, 13, 14, 15, 16],
            [0, 17, 18, 19, 20],
        ]
        _jm = JOINTS_MAPPING_LEFT if self.config.hand_side == "left" else JOINTS_MAPPING_RIGHT
        w_rot = self.config.angle_warmup_weight
        w_tip = 100.0  # strong anchor on fingertips
        eps = 1e-8

        for _ in range(self.config.angle_warmup_iters):
            self.hand.forward(q_current)

            residuals = []
            J_rows = []

            for f, chain in enumerate(chains_mp):
                for k in range(4):  # 4 bones per finger
                    parent_mp = chain[k]
                    child_mp = chain[k + 1]
                    parent_body = _jm[parent_mp]
                    child_body = _jm[child_mp]

                    # Robot bone direction
                    rp = self.hand.get_body_pos(parent_body)
                    rc = self.hand.get_body_pos(child_body)
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
                    Jp = self.hand.get_body_jacp(parent_body)
                    Jc = self.hand.get_body_jacp(child_body)
                    Je = Jc - Jp
                    P = (np.eye(3) - np.outer(d_rob, d_rob)) / e_len
                    J_dir = P @ Je

                    residuals.append(np.sqrt(w_rot) * res)
                    J_rows.append(np.sqrt(w_rot) * J_dir)

                # TIP position anchor
                tip_mp = chain[4]
                tip_body = _jm[tip_mp]
                rob_tip = self.hand.get_body_pos(tip_body)
                src_tip = landmarks_21[tip_mp]
                res_tip = rob_tip - src_tip
                J_tip = self.hand.get_body_jacp(tip_body)

                residuals.append(np.sqrt(w_tip) * res_tip)
                J_rows.append(np.sqrt(w_tip) * J_tip)

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

            problem = qpsolvers.Problem(H, c, lb=lb, ub=ub)
            solution = qpsolvers.solve_problem(problem, solver="daqp")

            if solution.found:
                q_current = q_current + solution.x
                q_current = np.clip(q_current, self.q_lb, self.q_ub)

        return q_current

    def _extract_cosik_targets(
        self,
        q_current: np.ndarray,
        q_prev: np.ndarray,
        landmarks_21: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract cosine IK angle targets for use as anchor in joint cost.

        Runs a few S1 iterations to convergence, then returns (target_q, confidence).

        Args:
            q_current: (nq,) current joint angles.
            q_prev: (nq,) joint angles from previous frame.
            landmarks_21: (21, 3) MediaPipe landmarks.

        Returns:
            Tuple of (target_q, confidence): target joint angles and per-joint confidence.
        """
        old_iters = self.config.angle_warmup_iters
        self.config.angle_warmup_iters = 1
        q_target = q_current.copy()
        for _ in range(3):
            q_before = q_target.copy()
            q_target = self.solve_angle_warmup(q_target, q_prev, landmarks_21)
            if np.linalg.norm(q_target - q_before) < 1e-3:
                break
        self.config.angle_warmup_iters = old_iters

        confidence = np.ones(self.nq)
        if self.config.floating_base and self.nq > 20:
            # Wrist translation (first 3 DOF): no angle anchor (position, not angle)
            confidence[:3] = 0.0
            # Wrist rotation (DOF 3-5): anchor to S1's alignment
            confidence[3:6] = 0.5
            # Finger MCP abduction: lower confidence
            for f in range(5):
                confidence[6 + 4 * f + 1] = 0.5
        else:
            for f in range(5):
                confidence[4 * f + 1] = 0.5
        return q_target, confidence

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
        # Extract mapped keypoints
        source_pts = self._extract_source_keypoints(landmarks)  # (n_hand, 3)

        # Guard against empty object points
        if object_pts_world is not None and len(object_pts_world) == 0:
            object_pts_world = None

        # Combine hand + object points for mesh construction
        # When obj_frame is set, transform source points to object-local frame
        if obj_frame is not None and object_pts_local is not None:
            R_inv, t_obj = obj_frame
            source_pts_obj = (source_pts - t_obj) @ R_inv.T
            source_pts_full = np.vstack([source_pts_obj, object_pts_local])
        elif object_pts_world is not None:
            source_pts_full = np.vstack([source_pts, object_pts_world])
        else:
            source_pts_full = source_pts

        V = len(source_pts_full)

        # Build topology
        if self.config.use_link_midpoints and self.config.use_skeleton_topology:
            adj_list = get_midpoint_skeleton_adjacency(self.n_keypoints)
            if V > self.n_keypoints:
                adj_list = adj_list + [[] for _ in range(V - self.n_keypoints)]
        elif self.config.use_skeleton_topology:
            # Hand skeleton: per-finger chains, no cross-finger connections
            adj_list = get_skeleton_adjacency(self.n_keypoints)
            if V > self.n_keypoints:
                adj_list = adj_list + [[] for _ in range(V - self.n_keypoints)]
        else:
            # Delaunay tetrahedralization (default)
            _, simplices = create_interaction_mesh(source_pts_full)
            adj_list = get_adjacency_list(simplices, V)
            if self.config.delaunay_edge_threshold is not None:
                adj_list = filter_adjacency_by_distance(adj_list, source_pts_full, self.config.delaunay_edge_threshold)
        self._adj_list = adj_list

        # Compute target Laplacian from combined source
        target_laplacian = calculate_laplacian_coordinates(
            source_pts_full,
            adj_list,
            distance_decay_k=self.config.laplacian_distance_weight_k,
        )

        # Semantic weights (on full vertex set)
        if use_semantic_weights:
            sem_w_hand = self._compute_semantic_weights(source_pts)
            if object_pts_world is not None:
                # Object points get base weight (no semantic boost)
                sem_w = np.concatenate([sem_w_hand, np.ones(len(object_pts_world))])
            else:
                sem_w = sem_w_hand
        else:
            sem_w = None

        # SQP iterations with convergence-based dispatch
        n_iter = self.config.n_iter_first if is_first_frame else self.config.n_iter
        q_current = q_prev.copy()
        last_cost = float("inf")
        convergence_delta = 1e-3

        # Stage 1: cosine IK bone direction alignment (convergence-based, max 5 iter)
        if self.config.use_angle_warmup:
            lm_21 = landmarks[:21] if landmarks.shape[0] > 21 else landmarks
            old_iters = self.config.angle_warmup_iters
            self.config.angle_warmup_iters = 1
            for _ in range(5):
                q_before = q_current.copy()
                q_current = self.solve_angle_warmup(q_current, q_prev, lm_21)
                delta = np.linalg.norm(q_current - q_before)
                if delta < convergence_delta:
                    break
            self.config.angle_warmup_iters = old_iters

        # Compute angle targets for joint cost (from S1 or fresh extraction)
        angle_targets_pair = None
        if self.config.use_angle_warmup and self.config.angle_anchor_weight > 0:
            lm_21 = landmarks[:21] if landmarks.shape[0] > 21 else landmarks
            angle_targets_pair = self._extract_cosik_targets(q_current, q_prev, lm_21)

        # Stage 2: position refinement (convergence-based)
        solve_obj_pts = object_pts_local if obj_frame is not None else object_pts_world
        last_cost = float("inf")

        for _ in range(n_iter):
            q_current, cost = self.solve_single_iteration(
                q_current,
                q_prev,
                target_laplacian,
                adj_list,
                sem_w,
                object_pts=solve_obj_pts,
                obj_frame=obj_frame,
                angle_targets=angle_targets_pair,
            )
            if abs(last_cost - cost) < convergence_delta:
                break
            last_cost = cost

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
        # Override n_iter_first for HO-Cap (200 iters for complex object scenes).
        # Uses save/restore because retarget_frame reads self.config.n_iter_first directly.
        self.config.n_iter_first = HOCAP_N_ITER_FIRST

        # Inject object mesh for non-penetration constraint
        if self.config.activate_non_penetration and hasattr(self.hand, "inject_object_mesh"):
            mesh_path = clip.get("mesh_path")
            if mesh_path:
                self.hand.inject_object_mesh(mesh_path, self.config.hand_side)
                # Update limits after model rebuild
                self.q_lb = self.hand.q_lb
                self.q_ub = self.hand.q_ub
                saved_q_lb = self.q_lb.copy()
                saved_q_ub = self.q_ub.copy()
                if self.config.floating_base and self.nq > 20:
                    self.q_lb[:3] = 0.0
                    self.q_ub[:3] = 0.0

        try:
            for t in tqdm(range(T), desc="Retargeting (obj mode)"):
                obj_world = transform_object_points(obj_pts_local, obj_q[t], obj_t[t])
                # Force SVD alignment (more reliable than wrist_q for HO-Cap data)
                lm, obj_transformed = self._align_frame(landmarks_raw[t], None, obj_world)

                # Update object pose in retargeter's MuJoCo model (for collision queries)
                # Object pose must be in the same aligned frame as hand landmarks
                # SVD alignment is forced (wrist_q=None), so use raw object pose
                if self.config.activate_non_penetration and self.hand._has_object:
                    wrist_world = landmarks_raw[t, 0]
                    obj_center_aligned = obj_t[t] - wrist_world
                    obj_quat_aligned = obj_q[t]
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
        If orientation probes enabled: landmarks should be (26, 3).
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
        """Get robot body positions for mapped keypoints. Returns (N, 3)."""
        if self.config.use_link_midpoints:
            pts = np.empty((self.n_keypoints, 3))
            for k, (parent_body, child_body) in enumerate(self._midpoint_body_pairs):
                if child_body is None:
                    pts[k] = self.hand.get_body_pos(parent_body)
                else:
                    pts[k] = (self.hand.get_body_pos(parent_body) + self.hand.get_body_pos(child_body)) / 2
            return pts
        return self.hand.get_body_positions(self.body_names)

    def _get_robot_jacobians(self) -> np.ndarray:
        """Get stacked Jacobians for mapped keypoints. Returns (3N, nq)."""
        if self.config.use_link_midpoints:
            J = np.zeros((3 * self.n_keypoints, self.nq))
            for k, (parent_body, child_body) in enumerate(self._midpoint_body_pairs):
                if child_body is None:
                    J[3 * k : 3 * k + 3] = self.hand.get_body_jacp(parent_body)
                else:
                    J[3 * k : 3 * k + 3] = (
                        self.hand.get_body_jacp(parent_body) + self.hand.get_body_jacp(child_body)
                    ) / 2
            return J
        return self.hand.get_body_jacobians(self.body_names)
