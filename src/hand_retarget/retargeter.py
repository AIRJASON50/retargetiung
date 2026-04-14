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
import cvxpy as cp
from scipy import sparse as sp
from tqdm import tqdm

from .mujoco_hand import MuJoCoHandModel
from .mesh_utils import (
    create_interaction_mesh,
    get_adjacency_list,
    get_skeleton_adjacency,
    calculate_laplacian_coordinates,
    calculate_laplacian_matrix,
    estimate_per_vertex_rotations,
    compute_arap_edge_data,
)
from .config import HandRetargetConfig
from .mediapipe_io import load_pkl_sequence


class InteractionMeshHandRetargeter:
    """
    Retargets MediaPipe hand landmarks to robot joint angles using
    interaction mesh Laplacian deformation minimization.
    """

    def __init__(self, config: HandRetargetConfig):
        self.config = config
        if config.floating_base:
            from .mujoco_hand import MuJoCoFloatingHandModel
            self.hand = MuJoCoFloatingHandModel(config.mjcf_path, config.hand_side)
        else:
            probe_offset = config.probe_offset if config.use_orientation_probes else 0.0
            self.hand = MuJoCoHandModel(config.mjcf_path, probe_offset=probe_offset)

        # Keypoint mapping: sorted by MediaPipe index
        self.mp_indices = sorted(config.joints_mapping.keys())
        self.body_names = [config.joints_mapping[i] for i in self.mp_indices]
        self.n_keypoints = len(self.mp_indices)  # 21

        # Joint limits
        self.q_lb = self.hand.q_lb
        self.q_ub = self.hand.q_ub
        self.nq = self.hand.nq  # 20

        # Laplacian weight (matches OmniRetarget: self.laplacian_weights = 10)
        self.laplacian_weight = 10.0

        # Global scale: WujiHand is human-scale, no scaling needed (unlike OmniRetarget's humanoid)
        # Config allows override if ever needed for a differently-sized hand
        self.global_scale = self.config.global_scale if self.config.global_scale is not None else 1.0

        # Semantic weight config
        self._pinch_d_min = 0.010   # 10mm: full pinch
        self._pinch_d_max = 0.030   # 30mm: approaching
        self._pinch_max_boost = 5.0

        # Thumb tip and other fingertip indices in mapped array
        self._thumb_tip_mapped = self.mp_indices.index(4)
        self._other_tips_mapped = [self.mp_indices.index(i) for i in [8, 12, 16, 20]]

        # Cached topology per frame (rebuilt each frame from source points)
        self._adj_list = None

        # Per-segment bone-ratio auto-scaling
        # Each finger has 4 segments: Wrist-MCP, MCP-PIP, PIP-DIP, DIP-TIP
        # Segment chains: list of consecutive MediaPipe index pairs per finger
        self._bone_ratios = None  # (5, 4) array, computed after warmup
        self._warmup_seg_lengths = []  # list of (5, 4) per-frame segment lengths
        self._finger_segments = [
            [(0, 1), (1, 2), (2, 3), (3, 4)],       # Thumb: Wrist-CMC, CMC-MCP, MCP-IP, IP-TIP
            [(0, 5), (5, 6), (6, 7), (7, 8)],        # Index: Wrist-MCP, MCP-PIP, PIP-DIP, DIP-TIP
            [(0, 9), (9, 10), (10, 11), (11, 12)],   # Middle
            [(0, 13), (13, 14), (14, 15), (15, 16)],  # Ring
            [(0, 17), (17, 18), (18, 19), (19, 20)],  # Pinky
        ]
        if config.use_bone_scaling:
            self._robot_seg_lengths = self._compute_robot_segment_lengths()

        # OPERATOR2MANO: fixed rotation from MediaPipe wrist convention → robot palm convention
        if config.hand_side == "left":
            from wuji_retargeting.mediapipe import OPERATOR2MANO_LEFT as OP2MANO
        else:
            from wuji_retargeting.mediapipe import OPERATOR2MANO_RIGHT as OP2MANO
        self._R_mano = np.array(OP2MANO, dtype=np.float64)

    def _compute_robot_segment_lengths(self) -> np.ndarray:
        """Compute robot per-segment bone lengths at default pose. Returns (5, 4)."""
        q0 = self.hand.get_default_qpos()
        self.hand.forward(q0)
        n_segs = len(self._finger_segments[0])
        lengths = np.zeros((5, n_segs))
        mapping = self.config.joints_mapping
        for f, segs in enumerate(self._finger_segments):
            for s, (prox_id, dist_id) in enumerate(segs):
                prox_pos = self.hand.get_body_pos(mapping[prox_id])
                dist_pos = self.hand.get_body_pos(mapping[dist_id])
                lengths[f, s] = np.linalg.norm(dist_pos - prox_pos)
        return lengths

    def _compute_source_segment_lengths(self, landmarks: np.ndarray) -> np.ndarray:
        """Compute source per-segment bone lengths. Returns (5, 4)."""
        n_segs = len(self._finger_segments[0])
        lengths = np.zeros((5, n_segs))
        for f, segs in enumerate(self._finger_segments):
            for s, (prox_id, dist_id) in enumerate(segs):
                lengths[f, s] = np.linalg.norm(landmarks[dist_id] - landmarks[prox_id])
        return lengths

    def _apply_bone_scaling(self, landmarks: np.ndarray, source_pts: np.ndarray) -> np.ndarray:
        """Apply per-segment bone-ratio scaling to source keypoints.

        Each bone segment (MCP-PIP, PIP-DIP, DIP-TIP) gets its own ratio.
        Scaling is applied cumulatively along the kinematic chain: each keypoint
        is repositioned by scaling its segment relative to the proximal joint.

        During warmup (first N frames): collect segment lengths, return unscaled.
        After warmup: compute ratios, freeze and apply.
        """
        # Warmup: collect source segment lengths
        if self._bone_ratios is None:
            src_lengths = self._compute_source_segment_lengths(landmarks)
            self._warmup_seg_lengths.append(src_lengths)

            if len(self._warmup_seg_lengths) < self.config.bone_scaling_warmup:
                return source_pts  # not enough data yet

            # Compute per-segment ratios from median
            all_lengths = np.array(self._warmup_seg_lengths)  # (N, 5, 4)
            median_src = np.median(all_lengths, axis=0)  # (5, 4)
            self._bone_ratios = self._robot_seg_lengths / (median_src + 1e-8)  # (5, 4)

            finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
            n_segs = self._bone_ratios.shape[1]
            seg_names = [f"s{i}" for i in range(n_segs)]
            for f in range(5):
                ratios_str = ", ".join(
                    f"{seg_names[s]}={self._bone_ratios[f, s]:.3f}"
                    f"({self._robot_seg_lengths[f, s]*1000:.1f}/{median_src[f, s]*1000:.1f}mm)"
                    for s in range(n_segs)
                )
                print(f"  Bone ratio {finger_names[f]}: {ratios_str}")

        # Apply per-segment scaling cumulatively along each finger chain
        # Use original segment vectors (from source_pts), scale each, accumulate from wrist
        scaled = source_pts.copy()
        for f, segs in enumerate(self._finger_segments):
            for s, (prox_mp, dist_mp) in enumerate(segs):
                prox_mapped = self.mp_indices.index(prox_mp)
                dist_mapped = self.mp_indices.index(dist_mp)
                # Original segment vector (unscaled)
                seg_vec = source_pts[dist_mapped] - source_pts[prox_mapped]
                # Accumulate: scaled proximal (already positioned) + ratio * original segment
                scaled[dist_mapped] = scaled[prox_mapped] + self._bone_ratios[f, s] * seg_vec
        return scaled

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
            landmarks_t: (21, 3) raw landmarks in world frame
            wrist_q_t: (4,) wrist quaternion (xyzw) or None for SVD fallback
            obj_world_t: (M, 3) object surface points in world frame

        Returns:
            lm: (21, 3) aligned landmarks (wrist at origin, robot convention)
            obj: (M, 3) aligned object points (same frame as landmarks)
        """
        from scipy.spatial.transform import Rotation as RotLib

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
                landmarks_t, self.config.mediapipe_rotation,
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

    def _augment_with_probes(self, landmarks: np.ndarray) -> np.ndarray:
        """Augment (21, 3) MediaPipe landmarks with 5 fingertip direction probe points.

        Each probe point is offset from the fingertip along the DIP->TIP direction,
        encoding orientation information as a position difference.
        Returns (26, 3) array: original 21 landmarks + 5 probe points.
        """
        tip_dip_pairs = [(4, 3), (8, 7), (12, 11), (16, 15), (20, 19)]
        probes = np.zeros((5, 3))
        for i, (tip_id, dip_id) in enumerate(tip_dip_pairs):
            direction = landmarks[tip_id] - landmarks[dip_id]
            norm = np.linalg.norm(direction)
            if norm > 1e-8:
                direction /= norm
            probes[i] = landmarks[tip_id] + self.config.probe_offset * direction
        return np.vstack([landmarks, probes])

    def _extract_source_keypoints(self, landmarks: np.ndarray) -> np.ndarray:
        """Extract the mapped keypoints from landmarks array.

        If orientation probes are enabled, landmarks should already be augmented
        to (26, 3) via _augment_with_probes before calling this method.
        """
        return landmarks[self.mp_indices]

    def _compute_semantic_weights(self, source_pts: np.ndarray) -> np.ndarray:
        """
        Compute per-keypoint weights based on pinch proximity.
        Only monitors thumb-to-other-fingertip pairs (4 pairs).

        Args:
            source_pts: (n_keypoints, 3) hand-only keypoints (NOT including object points)

        Returns:
            (n_keypoints,) weight array, 1.0 for non-pinch, up to max_boost for pinch
        """
        if len(source_pts) != self.n_keypoints:
            raise ValueError(
                f"Expected {self.n_keypoints} keypoints, got {len(source_pts)}"
            )
        w = np.ones(self.n_keypoints)

        for tip_mapped in self._other_tips_mapped:
            dist = np.linalg.norm(
                source_pts[self._thumb_tip_mapped] - source_pts[tip_mapped]
            )
            if dist < self._pinch_d_max:
                t = (self._pinch_d_max - dist) / (self._pinch_d_max - self._pinch_d_min)
                boost = 1.0 + (self._pinch_max_boost - 1.0) * np.clip(t, 0.0, 1.0)
                w[self._thumb_tip_mapped] = max(w[self._thumb_tip_mapped], boost)
                w[tip_mapped] = max(w[tip_mapped], boost)

        return w

    def _get_robot_keypoints(self) -> np.ndarray:
        """Get robot body positions for mapped keypoints. Returns (N, 3)."""
        return self.hand.get_body_positions(self.body_names)

    def _get_robot_jacobians(self) -> np.ndarray:
        """Get stacked Jacobians for mapped keypoints. Returns (3N, nq)."""
        return self.hand.get_body_jacobians(self.body_names)

    def solve_single_iteration(
        self,
        q_current: np.ndarray,
        q_prev_frame: np.ndarray,
        target_laplacian: np.ndarray,
        adj_list: list[list[int]],
        semantic_weights: np.ndarray | None = None,
        object_pts: np.ndarray | None = None,
        obj_frame: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, float]:
        """
        Single SQP iteration: linearize and solve SOCP sub-problem.

        Args:
            q_current: (nq,) current joint angles (linearization point)
            q_prev_frame: (nq,) joint angles from previous frame (for smoothness)
            target_laplacian: (V, 3) target Laplacian coordinates from source
            adj_list: adjacency list from Delaunay
            semantic_weights: (V,) per-keypoint weights, or None for uniform
            object_pts: (M, 3) object points -- in object-local frame if obj_frame set, else world frame
            obj_frame: (R_obj_inv, t_obj) for object-frame Laplacian, or None for world frame

        Returns:
            (q_new, cost): updated joint angles and optimization cost
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
            J_hand = np.einsum('ij,njk->nik', R_inv, J_blocks).reshape(n_hand * 3, self.nq)

        # Build combined vertices: hand + object (both in same frame)
        if object_pts is not None:
            robot_pts = np.vstack([hand_pts, object_pts])
        else:
            robot_pts = hand_pts

        V = len(robot_pts)

        # Jacobians: hand points have J, object points have J=0
        J_V = np.zeros((3 * V, self.nq))
        J_V[:3 * n_hand, :] = J_hand
        # J_V[3*n_hand:, :] = 0  (object points are fixed in object frame)

        # Recompute Laplacian matrix from current positions
        L = calculate_laplacian_matrix(robot_pts, adj_list, uniform_weight=True)
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

        # Decision variables
        dq = cp.Variable(self.nq, name="dq")
        lap_var = cp.Variable(3 * V, name="lap")

        # Constraints
        constraints = []

        # Laplacian linearization equality (OmniRetarget L582)
        constraints.append(cp.Constant(J_L) @ dq - lap_var == -lap0_vec)

        # Joint limits (OmniRetarget L640-644)
        if self.config.activate_joint_limits:
            constraints.append(dq >= self.q_lb - q_current)
            constraints.append(dq <= self.q_ub - q_current)

        # Trust region SOC (OmniRetarget L647)
        constraints.append(cp.SOC(self.config.step_size, dq))

        # Objective
        obj_terms = []

        # Weighted Laplacian deformation energy (OmniRetarget L652)
        obj_terms.append(cp.sum_squares(cp.multiply(sqrt_w3, lap_var - target_lap_vec)))

        # Temporal smoothness (OmniRetarget L666-668)
        dq_smooth = q_prev_frame - q_current
        obj_terms.append(self.config.smooth_weight * cp.sum_squares(dq - dq_smooth))

        # Non-penetration hard constraint (OmniRetarget formula 3b)
        # Allow penetration up to capsule radius (normal contact = capsule surface on object)
        if self.config.activate_non_penetration and hasattr(self.hand, "query_tip_penetration"):
            capsule_radius = 0.0075  # 7.5mm
            for J_contact, phi, _ in self.hand.query_tip_penetration(threshold=0.05):
                constraints.append(J_contact @ dq >= -phi - capsule_radius)

        # Solve
        problem = cp.Problem(cp.Minimize(cp.sum(obj_terms)), constraints)
        problem.solve(solver=cp.CLARABEL, verbose=False)

        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            # Fallback: remove SOC and retry (OmniRetarget L682-685)
            constraints_no_soc = [c for c in constraints
                                  if not isinstance(c, cp.constraints.second_order.SOC)]
            problem = cp.Problem(cp.Minimize(cp.sum(obj_terms)), constraints_no_soc)
            problem.solve(solver=cp.CLARABEL, verbose=False)

        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return q_current.copy(), float("inf")

        q_new = q_current + dq.value
        q_new = np.clip(q_new, self.q_lb, self.q_ub)

        return q_new, problem.value

    def solve_single_iteration_arap(
        self,
        q_current: np.ndarray,
        q_prev_frame: np.ndarray,
        source_pts: np.ndarray,
        adj_list: list[list[int]],
        semantic_weights: np.ndarray | None = None,
        object_pts_world: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float]:
        """Single SQP iteration with ARAP per-edge energy.

        Replaces Laplacian deformation cost with:
            sum_{(i,j)} w_ij * ||(robot_i - robot_j) - R_i @ (source_i - source_j)||^2

        No auxiliary lap_var variable needed -- directly quadratic in dq.
        """
        n_hand = self.n_keypoints

        # FK at current q
        self.hand.forward(q_current)
        hand_pts = self._get_robot_keypoints()

        # Combined vertices
        if object_pts_world is not None:
            robot_pts = np.vstack([hand_pts, object_pts_world])
            source_full = source_pts
        else:
            robot_pts = hand_pts
            source_full = source_pts

        V = len(robot_pts)

        # Jacobians
        J_hand = self._get_robot_jacobians()
        J_V = np.zeros((3 * V, self.nq))
        J_V[:3 * n_hand, :] = J_hand

        # Per-vertex rotations (ARAP local step)
        rotations = estimate_per_vertex_rotations(source_full, robot_pts, adj_list)

        # Edge residuals and Jacobian
        edges, residuals, J_edge = compute_arap_edge_data(
            source_full, robot_pts, adj_list, rotations, J_V, self.nq,
        )
        E = len(edges)

        # Per-edge weights from semantic weights
        if semantic_weights is not None:
            # Edge weight = mean of endpoint weights
            w_e = np.array([
                0.5 * (semantic_weights[i] + semantic_weights[j])
                for i, j in edges
            ])
            w_e *= self.laplacian_weight
        else:
            w_e = self.laplacian_weight * np.ones(E)
        sqrt_w3 = np.sqrt(np.repeat(w_e, 3))

        # Decision variable: just dq (no auxiliary lap_var)
        dq = cp.Variable(self.nq, name="dq")

        # Constraints
        constraints = []
        if self.config.activate_joint_limits:
            constraints.append(dq >= self.q_lb - q_current)
            constraints.append(dq <= self.q_ub - q_current)
        constraints.append(cp.SOC(self.config.step_size, dq))

        # ARAP edge cost: ||sqrt_w * (residual + J_edge @ dq)||^2
        obj_terms = []
        obj_terms.append(
            cp.sum_squares(cp.multiply(sqrt_w3, residuals + cp.Constant(J_edge) @ dq))
        )

        # Temporal smoothness
        dq_smooth = q_prev_frame - q_current
        obj_terms.append(self.config.smooth_weight * cp.sum_squares(dq - dq_smooth))

        # Non-penetration
        if self.config.activate_non_penetration and hasattr(self.hand, "query_tip_penetration"):
            capsule_radius = 0.0075
            for J_contact, phi, _ in self.hand.query_tip_penetration(threshold=0.05):
                constraints.append(J_contact @ dq >= -phi - capsule_radius)

        # Solve
        problem = cp.Problem(cp.Minimize(cp.sum(obj_terms)), constraints)
        problem.solve(solver=cp.CLARABEL, verbose=False)

        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            constraints_no_soc = [c for c in constraints
                                  if not isinstance(c, cp.constraints.second_order.SOC)]
            problem = cp.Problem(cp.Minimize(cp.sum(obj_terms)), constraints_no_soc)
            problem.solve(solver=cp.CLARABEL, verbose=False)

        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return q_current.copy(), float("inf")

        q_new = q_current + dq.value
        q_new = np.clip(q_new, self.q_lb, self.q_ub)
        return q_new, problem.value

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
        """
        Retarget a single frame of MediaPipe landmarks to joint angles.

        Args:
            landmarks: (21, 3) preprocessed landmarks (in aligned wrist frame)
            q_prev: (nq,) joint angles from previous frame (warm start)
            is_first_frame: if True, use more iterations
            use_semantic_weights: enable pinch-aware loss weighting
            object_pts_world: (M, 3) object surface points in world/wrist frame, or None
            obj_frame: (R_obj_inv, t_obj) for object-frame Laplacian, or None
            object_pts_local: (M, 3) object points in object-local frame (for obj_frame mode)

        Returns:
            (nq,) optimized joint angles
        """
        # Augment with probe points if enabled (21 -> 26 landmarks)
        if self.config.use_orientation_probes and landmarks.shape[0] == 21:
            landmarks = self._augment_with_probes(landmarks)

        # Extract mapped keypoints
        source_pts = self._extract_source_keypoints(landmarks)  # (n_hand, 3)

        # Per-finger bone-ratio scaling (warmup-based)
        if self.config.use_bone_scaling:
            # Pass original 21-point landmarks for finger length computation
            lm_21 = landmarks[:21] if landmarks.shape[0] > 21 else landmarks
            source_pts = self._apply_bone_scaling(lm_21, source_pts)

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
        if self.config.use_skeleton_topology:
            # Hand skeleton: per-finger chains, no cross-finger connections
            adj_list = get_skeleton_adjacency(self.n_keypoints)
            # If object points present, they are disconnected (no edges to hand)
            if V > self.n_keypoints:
                adj_list = adj_list + [[] for _ in range(V - self.n_keypoints)]
        else:
            # Delaunay tetrahedralization (default)
            _, simplices = create_interaction_mesh(source_pts_full)
            adj_list = get_adjacency_list(simplices, V)
        self._adj_list = adj_list

        # Compute target Laplacian from combined source
        target_laplacian = calculate_laplacian_coordinates(source_pts_full, adj_list)

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

        # SQP iterations
        n_iter = self.config.n_iter_first if is_first_frame else self.config.n_iter
        q_current = q_prev.copy()
        last_cost = float("inf")

        if self.config.use_arap_edge:
            # Two-phase: Laplacian warmup then ARAP edge refinement
            if is_first_frame:
                # Phase 1: Laplacian warmup to get a reasonable starting pose
                warmup_n = min(20, n_iter // 2)
                for _ in range(warmup_n):
                    q_current, cost = self.solve_single_iteration(
                        q_current, q_prev, target_laplacian, adj_list, sem_w,
                        object_pts=object_pts_world,
                    )
                    if abs(last_cost - cost) < 1e-8:
                        break
                    last_cost = cost
                last_cost = float("inf")
                remaining = n_iter - warmup_n
            else:
                remaining = n_iter

            # Phase 2: ARAP per-edge energy
            for _ in range(remaining):
                q_current, cost = self.solve_single_iteration_arap(
                    q_current, q_prev, source_pts_full, adj_list, sem_w, object_pts_world,
                )
                if abs(last_cost - cost) < 1e-8:
                    break
                last_cost = cost
        else:
            # Original Laplacian energy (with optional rotation compensation)
            use_rot_comp = self.config.rotation_compensation
            target_laplacian_raw = target_laplacian.copy() if use_rot_comp else None
            solve_obj_pts = object_pts_local if obj_frame is not None else object_pts_world

            for _ in range(n_iter):
                if use_rot_comp:
                    self.hand.forward(q_current)
                    robot_pts = self._get_robot_keypoints()
                    if object_pts_world is not None:
                        robot_pts_full = np.vstack([robot_pts, object_pts_world])
                    else:
                        robot_pts_full = robot_pts
                    R = estimate_per_vertex_rotations(source_pts_full, robot_pts_full, adj_list)
                    target_laplacian = np.einsum('vij,vj->vi', R, target_laplacian_raw)

                q_current, cost = self.solve_single_iteration(
                    q_current, q_prev, target_laplacian, adj_list, sem_w,
                    object_pts=solve_obj_pts,
                    obj_frame=obj_frame,
                )
                if abs(last_cost - cost) < 1e-8:
                    break
                last_cost = cost

        return q_current

    def retarget_sequence(
        self,
        pkl_path: str,
        hand_side: str = "left",
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Retarget a full .pkl sequence.

        Args:
            pkl_path: path to MediaPipe .pkl file
            hand_side: "left" or "right"

        Returns:
            qpos_seq: (T, nq) joint angle sequence
            timestamps: (T,) timestamps
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
        """
        Retarget a HO-Cap clip with object interaction.

        Preprocessing: wrist-center landmarks + rotate to robot frame.
        Primary path: wrist_q (6D pose from data) + OPERATOR2MANO (convention fix).
        Fallback: SVD estimation + OPERATOR2MANO (when wrist_q unavailable).

        Args:
            clip: dict from load_hocap_clip() with keys:
                landmarks, object_pts_local, object_t, object_q,
                wrist_q (optional), fps
            use_semantic_weights: enable pinch-aware weighting

        Returns:
            qpos_seq: (T, nq) joint angle sequence
        """
        from .mediapipe_io import transform_object_points

        landmarks_raw = clip["landmarks"]       # (T, 21, 3) world frame
        obj_pts_local = clip["object_pts_local"]  # (M, 3) mesh local
        obj_t = clip["object_t"]                # (T, 3)
        obj_q = clip["object_q"]                # (T, 4)
        wrist_q_seq = clip.get("wrist_q")       # (T, 4) xyzw or None
        T = len(landmarks_raw)

        qpos_seq = np.zeros((T, self.nq))
        q_prev = self.hand.get_default_qpos()

        # Save state — restore on exit to avoid side effects across calls
        saved_q_lb = self.q_lb.copy()
        saved_q_ub = self.q_ub.copy()
        saved_n_iter_first = self.config.n_iter_first

        if self.config.floating_base and self.nq > 20:
            self.q_lb[:3] = 0.0
            self.q_ub[:3] = 0.0
        self.config.n_iter_first = 200

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
                wrist_q_t = wrist_q_seq[t] if wrist_q_seq is not None else None
                lm, obj_transformed = self._align_frame(landmarks_raw[t], wrist_q_t, obj_world)

                # Update object pose in retargeter's MuJoCo model (for collision queries)
                # Object pose must be in the same aligned frame as hand landmarks
                if self.config.activate_non_penetration and self.hand._has_object:
                    # Object center in aligned frame = transform(obj_center_world) via same R_align
                    from scipy.spatial.transform import Rotation as RotLib
                    wrist_world = landmarks_raw[t, 0]
                    obj_center_aligned = obj_t[t] - wrist_world
                    if wrist_q_t is not None:
                        R_wrist = RotLib.from_quat(wrist_q_t).as_matrix()
                        R_align = R_wrist.T @ self._R_mano
                        obj_center_aligned = obj_center_aligned @ R_align
                        # Object rotation in aligned frame
                        R_obj_world = RotLib.from_quat(obj_q[t]).as_matrix()
                        R_obj_aligned = R_align.T @ R_obj_world
                        obj_quat_aligned = RotLib.from_matrix(R_obj_aligned).as_quat()  # xyzw
                    else:
                        obj_quat_aligned = obj_q[t]
                    self.hand.set_object_pose(obj_center_aligned, obj_quat_aligned)

                # Build object-frame transform if enabled
                # obj_transformed and lm are in wrist-aligned frame
                # Object pose in aligned frame: center + rotation
                if self.config.use_object_frame:
                    from scipy.spatial.transform import Rotation as RotLib2
                    wrist_world = landmarks_raw[t, 0]
                    obj_c = obj_t[t] - wrist_world
                    if wrist_q_t is not None:
                        R_wrist = RotLib2.from_quat(wrist_q_t).as_matrix()
                        R_align = R_wrist.T @ self._R_mano
                        obj_c = obj_c @ R_align
                        R_obj_w = RotLib2.from_quat(obj_q[t]).as_matrix()
                        R_obj_a = R_align.T @ R_obj_w
                    else:
                        R_obj_a = np.eye(3)
                    R_obj_inv = R_obj_a.T
                    frame_arg = (R_obj_inv, obj_c)
                    # Object points in their own local frame (pre-sampled mesh coords)
                    local_arg = obj_pts_local
                else:
                    frame_arg = None
                    local_arg = None

                q_opt = self.retarget_frame(
                    lm, q_prev,
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
