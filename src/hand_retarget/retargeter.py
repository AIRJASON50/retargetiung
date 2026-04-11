"""
Interaction-mesh-based hand retargeter.

Adapted from OmniRetarget's InteractionMeshRetargeter for fixed-base dexterous hands.
Core algorithm: minimize Laplacian deformation energy subject to joint limits and trust region,
solved via SQP with CVXPY + Clarabel SOCP solver.

Matches OmniRetarget robot_only mode behavior:
- Delaunay topology rebuilt every frame from source keypoints
- Laplacian matrix recomputed every SQP iteration from current robot positions
- Laplacian weight = 10 (uniform across all keypoints)
- No floating base (T(q) = I for all-hinge)
- No foot sticking, no object interaction
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

    def _build_topology(self, source_positions: np.ndarray):
        """
        Build Delaunay interaction mesh from source keypoints.
        Called every frame (matching OmniRetarget behavior).

        Args:
            source_positions: (N, 3) keypoint positions
        Returns:
            adj_list: adjacency list
        """
        _, simplices = create_interaction_mesh(source_positions)
        self._adj_list = get_adjacency_list(simplices, self.n_keypoints)
        return self._adj_list

    def _extract_source_keypoints(self, landmarks: np.ndarray) -> np.ndarray:
        """Extract the mapped keypoints from full 21-point MediaPipe landmarks."""
        return landmarks[self.mp_indices]

    def _compute_semantic_weights(self, source_pts: np.ndarray) -> np.ndarray:
        """
        Compute per-keypoint weights based on pinch proximity.
        Only monitors thumb-to-other-fingertip pairs (4 pairs).
        Non-pinch points get weight 1.0, pinch points get up to max_boost.
        """
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
    ) -> tuple[np.ndarray, float]:
        """
        Single SQP iteration: linearize and solve SOCP sub-problem.

        Matches OmniRetarget's solve_single_iteration:
        - Recompute Laplacian matrix from current robot positions each iteration
        - Apply laplacian_weight to deformation cost
        - Smoothness cost on dq

        Args:
            q_current: (nq,) current joint angles (linearization point)
            q_prev_frame: (nq,) joint angles from previous frame (for smoothness)
            target_laplacian: (N, 3) target Laplacian coordinates from source
            adj_list: adjacency list from Delaunay

        Returns:
            (q_new, cost): updated joint angles and optimization cost
        """
        V = self.n_keypoints

        # FK at current q
        self.hand.forward(q_current)
        robot_pts = self._get_robot_keypoints()  # (V, 3)

        # Jacobians for robot keypoints
        J_V = self._get_robot_jacobians()  # (3V, nq)

        # Recompute Laplacian matrix from current robot positions (matches OmniRetarget L560)
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
    ) -> np.ndarray:
        """
        Retarget a single frame of MediaPipe landmarks to joint angles.

        Matches OmniRetarget retarget_motion loop:
        - Rebuild Delaunay every frame from source keypoints
        - Compute target Laplacian from source vertices + adjacency
        - Run SQP iterations with per-iteration L matrix update

        Args:
            landmarks: (21, 3) preprocessed landmarks
            q_prev: (nq,) joint angles from previous frame (warm start)
            is_first_frame: if True, use more iterations

        Returns:
            (nq,) optimized joint angles
        """
        # Extract mapped keypoints
        source_pts = self._extract_source_keypoints(landmarks)  # (16, 3)

        # Rebuild Delaunay every frame (OmniRetarget L385-388)
        _, simplices = create_interaction_mesh(source_pts)
        adj_list = get_adjacency_list(simplices, self.n_keypoints)
        self._adj_list = adj_list  # cache for visualization

        # Compute target Laplacian from source (OmniRetarget L411)
        target_laplacian = calculate_laplacian_coordinates(source_pts, adj_list)

        # Semantic weights (only if enabled)
        sem_w = self._compute_semantic_weights(source_pts) if use_semantic_weights else None

        # SQP iterations (OmniRetarget L727-747)
        n_iter = self.config.n_iter_first if is_first_frame else self.config.n_iter
        q_current = q_prev.copy()
        last_cost = float("inf")

        for _ in range(n_iter):
            q_current, cost = self.solve_single_iteration(
                q_current, q_prev, target_laplacian, adj_list, sem_w
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
