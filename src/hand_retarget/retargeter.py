"""
Interaction-mesh-based hand retargeter.

Adapted from OmniRetarget for dexterous hands. Supports:
- robot_only: fixed base, 20 DOF, hand keypoints only
- object mode: 6DOF wrist + 20 finger = 26 DOF, hand + object surface points

Core algorithm: minimize Laplacian deformation energy subject to joint limits
and trust region, solved via SQP with CVXPY + Clarabel SOCP solver.
"""

import numpy as np
import cvxpy as cp
from scipy import sparse as sp
from tqdm import tqdm

from .mujoco_hand import MuJoCoHandModel
from .mesh_utils import (
    create_interaction_mesh,
    get_adjacency_list,
    calculate_laplacian_coordinates,
    calculate_laplacian_matrix,
)
from .config import HandRetargetConfig
from .mediapipe_io import preprocess_landmarks, load_pkl_sequence


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
            self.hand = MuJoCoHandModel(config.mjcf_path)

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

    def _extract_source_keypoints(self, landmarks: np.ndarray) -> np.ndarray:
        """Extract the mapped keypoints from full 21-point MediaPipe landmarks."""
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
        assert len(source_pts) == self.n_keypoints
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
        object_pts_world: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float]:
        """
        Single SQP iteration: linearize and solve SOCP sub-problem.

        Args:
            q_current: (nq,) current joint angles (linearization point)
            q_prev_frame: (nq,) joint angles from previous frame (for smoothness)
            target_laplacian: (V, 3) target Laplacian coordinates from source
            adj_list: adjacency list from Delaunay
            semantic_weights: (V,) per-keypoint weights, or None for uniform
            object_pts_world: (M, 3) object surface points in world frame, or None

        Returns:
            (q_new, cost): updated joint angles and optimization cost
        """
        n_hand = self.n_keypoints

        # FK at current q
        self.hand.forward(q_current)
        hand_pts = self._get_robot_keypoints()  # (n_hand, 3)

        # Build combined vertices: hand + object
        if object_pts_world is not None:
            robot_pts = np.vstack([hand_pts, object_pts_world])  # (n_hand+M, 3)
        else:
            robot_pts = hand_pts

        V = len(robot_pts)

        # Jacobians: hand points have J, object points have J=0
        J_hand = self._get_robot_jacobians()  # (3*n_hand, nq)
        J_V = np.zeros((3 * V, self.nq))
        J_V[:3 * n_hand, :] = J_hand
        # J_V[3*n_hand:, :] = 0  (object points, already zero)

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

    def retarget_frame(
        self,
        landmarks: np.ndarray,
        q_prev: np.ndarray,
        is_first_frame: bool = False,
        use_semantic_weights: bool = False,
        object_pts_world: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Retarget a single frame of MediaPipe landmarks to joint angles.

        Args:
            landmarks: (21, 3) preprocessed landmarks
            q_prev: (nq,) joint angles from previous frame (warm start)
            is_first_frame: if True, use more iterations
            use_semantic_weights: enable pinch-aware loss weighting
            object_pts_world: (M, 3) object surface points in world frame, or None

        Returns:
            (nq,) optimized joint angles
        """
        # Extract mapped keypoints
        source_pts = self._extract_source_keypoints(landmarks)  # (n_hand, 3)

        # Guard against empty object points
        if object_pts_world is not None and len(object_pts_world) == 0:
            object_pts_world = None

        # Combine hand + object points for mesh construction
        if object_pts_world is not None:
            source_pts_full = np.vstack([source_pts, object_pts_world])
        else:
            source_pts_full = source_pts

        V = len(source_pts_full)

        # Rebuild Delaunay every frame on combined point set
        _, simplices = create_interaction_mesh(source_pts_full)
        adj_list = get_adjacency_list(simplices, V)
        self._adj_list = adj_list  # cache for visualization

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

        for _ in range(n_iter):
            q_current, cost = self.solve_single_iteration(
                q_current, q_prev, target_laplacian, adj_list, sem_w, object_pts_world
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
        from scipy.spatial.transform import Rotation as RotLib

        landmarks_raw = clip["landmarks"]       # (T, 21, 3) world frame
        obj_pts_local = clip["object_pts_local"]  # (M, 3) mesh local
        obj_t = clip["object_t"]                # (T, 3)
        obj_q = clip["object_q"]                # (T, 4)
        T = len(landmarks_raw)

        qpos_seq = np.zeros((T, self.nq))
        q_prev = self.hand.get_default_qpos()
        self._source_wrist_world = landmarks_raw[:, 0, :].copy()

        # Lock wrist translation at origin (source wrist = 0 after centering)
        if self.config.floating_base and self.nq > 20:
            self.q_lb[:3] = 0.0
            self.q_ub[:3] = 0.0

        # More iterations for first frame convergence
        saved_n_iter = self.config.n_iter_first
        self.config.n_iter_first = 200

        # --- Frame alignment setup ---
        # OPERATOR2MANO: fixed rotation from MediaPipe wrist convention → robot palm convention
        if self.config.hand_side == "left":
            from wuji_retargeting.mediapipe import OPERATOR2MANO_LEFT as OP2MANO
        else:
            from wuji_retargeting.mediapipe import OPERATOR2MANO_RIGHT as OP2MANO
        R_mano = np.array(OP2MANO, dtype=np.float64)

        # Primary: wrist_q from data (direct 6D pose measurement)
        wrist_q_seq = clip.get("wrist_q")  # (T, 4) xyzw quaternion, or None
        use_wrist_q = wrist_q_seq is not None

        # Fallback: SVD estimation (only when no wrist_q)
        if not use_wrist_q:
            from .mediapipe_io import preprocess_landmarks

        for t in tqdm(range(T), desc="Retargeting (obj mode)"):
            wrist_world = landmarks_raw[t, 0]
            lm_centered = landmarks_raw[t] - wrist_world

            if use_wrist_q:
                # wrist_q derotates world→local, OPERATOR2MANO fixes convention
                R_wrist = RotLib.from_quat(wrist_q_seq[t]).as_matrix()
                R_align = R_wrist.T @ R_mano
                lm = lm_centered @ R_align
            else:
                # SVD estimates orientation from landmarks, OPERATOR2MANO fixes convention
                lm = preprocess_landmarks(
                    landmarks_raw[t],
                    self.config.mediapipe_rotation,
                    hand_side=self.config.hand_side,
                    global_scale=self.global_scale,
                    use_mano_rotation=self.config.use_mano_rotation,
                )

            # Object points: local → world → wrist-relative → same rotation as hand
            obj_world = transform_object_points(obj_pts_local, obj_q[t], obj_t[t])
            obj_transformed = (obj_world - wrist_world) @ R_align if use_wrist_q else obj_world - wrist_world

            if not use_wrist_q and self.config.use_mano_rotation:
                # SVD fallback: apply same rotation to object points
                from wuji_retargeting.mediapipe import apply_mediapipe_transformations
                raw_c = landmarks_raw[t] - wrist_world
                trans = apply_mediapipe_transformations(landmarks_raw[t].copy(), self.config.hand_side)
                R_t, _, _, _ = np.linalg.lstsq(raw_c[1:6], trans[1:6], rcond=None)
                obj_transformed = (obj_world - wrist_world) @ R_t
                angles = [self.config.mediapipe_rotation.get(k, 0) for k in "xyz"]
                if any(a != 0 for a in angles):
                    R_extra = RotLib.from_euler("xyz", angles, degrees=True).as_matrix()
                    obj_transformed = obj_transformed @ R_extra.T

            if self.global_scale != 1.0:
                lm *= self.global_scale
                obj_transformed *= self.global_scale

            q_opt = self.retarget_frame(
                lm, q_prev,
                is_first_frame=(t == 0),
                use_semantic_weights=use_semantic_weights,
                object_pts_world=obj_transformed,
            )
            qpos_seq[t] = q_opt
            q_prev = q_opt

            # Restore normal iteration count after first frame
            if t == 0:
                self.config.n_iter_first = saved_n_iter

        return qpos_seq
