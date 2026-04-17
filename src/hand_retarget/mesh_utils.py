"""
Interaction mesh utilities: Delaunay triangulation and Laplacian deformation.
Extracted from OmniRetarget (holosoma_retargeting/src/utils.py).
"""

from __future__ import annotations

# ==============================================================================
# Imports
# ==============================================================================
import numpy as np
from scipy.spatial import Delaunay

# ==============================================================================
# Constants
# ==============================================================================

_FINGER_ROOTS = [1, 5, 9, 13, 17]
"""MediaPipe indices of each finger's root (MCP/CMC) joint."""

# ==============================================================================
# Functions
# ==============================================================================


def create_interaction_mesh(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Creates a tetrahedral mesh from keypoints using Delaunay triangulation.

    Args:
        vertices: (N, 3) array of vertex positions.

    Returns:
        tuple: (vertices, simplices) - input points and generated tetrahedra indices.
    """
    tri = Delaunay(vertices)
    return vertices, tri.simplices


def get_adjacency_list(tetrahedra: np.ndarray, num_vertices: int) -> list[list[int]]:
    """
    Creates an adjacency list from tetrahedra.
    Two vertices are neighbors if they share a tetrahedron.

    Args:
        tetrahedra: (M, 4) array of tetrahedron vertex indices.
        num_vertices: total number of vertices in the mesh.

    Returns:
        Adjacency list where ``adj[i]`` is the list of vertex indices
        adjacent to vertex *i*.
    """
    adj = [set() for _ in range(num_vertices)]
    for tet in tetrahedra:
        for i in range(4):
            for j in range(i + 1, 4):
                u, v = tet[i], tet[j]
                adj[u].add(v)
                adj[v].add(u)
    return [list(s) for s in adj]


def filter_adjacency_by_distance(
    adj_list: list[list[int]],
    vertices: np.ndarray,
    threshold: float,
) -> list[list[int]]:
    """Remove edges longer than threshold from adjacency list.

    Filters out long-range Delaunay edges (e.g. cross-finger connections)
    that pollute Laplacian neighborhood averages. Vertices that lose all
    neighbors contribute zero Laplacian error (neutral, not harmful).

    Args:
        adj_list: input adjacency list.
        vertices: (N, 3) vertex positions used to measure edge lengths.
        threshold: max edge length to keep (meters).

    Returns:
        Filtered adjacency list with the same length as adj_list.
    """
    filtered = [[] for _ in range(len(adj_list))]
    for i, neighbors in enumerate(adj_list):
        for j in neighbors:
            if np.linalg.norm(vertices[j] - vertices[i]) < threshold:
                filtered[i].append(j)
    return filtered


def calculate_laplacian_coordinates(
    vertices: np.ndarray,
    adj_list: list[list[int]],
    uniform_weight: bool = True,
    epsilon: float = 1e-6,
    distance_decay_k: float | None = None,
) -> np.ndarray:
    """
    Calculates the Laplacian coordinates for each vertex: ``L @ vertices``.

    Args:
        vertices: (N, 3) array of vertex positions.
        adj_list: adjacency list for the mesh.
        uniform_weight: if True, use uniform weights; else distance-based (1/d).
        distance_decay_k: if set, use exp(-k*d) weights instead of uniform/1/d.

    Returns:
        (N, 3) array of Laplacian coordinates.
    """
    L = calculate_laplacian_matrix(vertices, adj_list, uniform_weight, epsilon, distance_decay_k)
    return L @ vertices


def calculate_laplacian_matrix(
    vertices: np.ndarray,
    adj_list: list[list[int]],
    uniform_weight: bool = True,
    epsilon: float = 1e-6,
    distance_decay_k: float | None = None,
) -> np.ndarray:
    """
    Calculates the (N, N) Laplacian matrix.

    L[i, i] = 1.0
    L[i, j] = -w_ij / sum(w)  for neighbor j

    Weight options (in priority order):
      distance_decay_k set: w_ij = exp(-k * ||e_ij||)  (shorter edges → higher weight)
      uniform_weight=False: w_ij = 1 / ||e_ij||
      default (uniform_weight=True): w_ij = 1 / degree(i)

    Args:
        vertices: (N, 3) array of vertex positions.
        adj_list: adjacency list for the mesh.
        uniform_weight: if True, use uniform weights; else distance-based (1/d).
        epsilon: small value to avoid division by zero.
        distance_decay_k: if set, use exp(-k*d) weights instead of uniform/1/d.

    Returns:
        (N, N) Laplacian matrix.
    """
    n_verts = len(vertices)
    L = np.zeros((n_verts, n_verts))

    for i in range(n_verts):
        neighbors = adj_list[i]
        if len(neighbors) > 0:
            if distance_decay_k is not None:
                vi = vertices[i]
                neighbor_pos = vertices[neighbors]
                distances = np.linalg.norm(vi - neighbor_pos, axis=1)
                weights = np.exp(-distance_decay_k * distances)
                w_sum = weights.sum()
                weights = weights / (w_sum if w_sum > epsilon else 1.0)
            elif uniform_weight:
                weights = np.ones(len(neighbors)) / len(neighbors)
            else:
                vi = vertices[i]
                neighbor_pos = vertices[neighbors]
                distances = np.linalg.norm(vi - neighbor_pos, axis=1)
                weights = 1.0 / (distances + epsilon)
                weights = weights / np.sum(weights)

            L[i, i] = 1.0
            for j, idx in enumerate(neighbors):
                L[i, idx] = -weights[j]

    return L


def get_skeleton_adjacency(n_keypoints: int = 21) -> list[list[int]]:
    """Hand skeleton topology: each finger is an independent chain.

    MediaPipe 21-point layout:
        0=Wrist
        1-4=Thumb (CMC, MCP, IP, TIP)
        5-8=Index (MCP, PIP, DIP, TIP)
        9-12=Middle, 13-16=Ring, 17-20=Pinky

    Edges: Wrist-to-each-MCP + intra-finger chain (4 per finger).
    Total: 5 (wrist-MCP) + 5×3 (intra-finger) = 20 edges.
    No cross-finger connections.
    """
    adj = [[] for _ in range(n_keypoints)]

    def _add(i: int, j: int) -> None:
        adj[i].append(j)
        adj[j].append(i)

    # Wrist to each finger root
    for root in _FINGER_ROOTS:
        _add(0, root)

    # Intra-finger chains
    for start in _FINGER_ROOTS:
        for k in range(3):
            _add(start + k, start + k + 1)

    return adj


def get_midpoint_skeleton_adjacency(n_midpoints: int = 20) -> list[list[int]]:
    """Skeleton topology for 20 link-midpoint vertices (4 per finger, no wrist).

    Per-finger chain: indices 4f, 4f+1, 4f+2, 4f+3 for finger f=0..4.
    Total edges: 5 fingers * 3 = 15. No cross-finger connections.

    Args:
        n_midpoints: Number of midpoint keypoints (default 20).

    Returns:
        Adjacency list of length ``n_midpoints``.
    """
    adj: list[list[int]] = [[] for _ in range(n_midpoints)]

    def _add(i: int, j: int) -> None:
        adj[i].append(j)
        adj[j].append(i)

    for f in range(5):
        base = f * 4
        for k in range(3):
            _add(base + k, base + k + 1)
    return adj


def compute_edge_ratio_data(
    source_pts: np.ndarray,
    robot_pts: np.ndarray,
    adj_list: list[list[int]],
    J_V: np.ndarray,
    nq: int,
    robot_T_positions: np.ndarray | None = None,
    human_T_positions: np.ndarray | None = None,
    distance_threshold: float = 0.06,
    distance_decay_k: float = 20.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build per-edge ratio residuals and Jacobian for SQP (Zhang 2023 style).

    T-pose mode (robot_T_positions and human_T_positions provided):
        r_ij = e_robot / ||e_T,robot|| - e_source / ||e_T,source||
        Each embodiment normalized by its own rest-pose bone length.

    Fallback mode:
        r_ij = (e_robot - e_source) / ||e_source||

    Edges filtered by source distance < threshold. Weights: exp(-k * ||e_source||).

    Args:
        source_pts: (V, 3) source keypoints.
        robot_pts: (V, 3) current robot keypoints.
        adj_list: adjacency list.
        J_V: (3V, nq) stacked translational Jacobians.
        nq: number of joint DOFs.
        robot_T_positions: (V, 3) robot T-pose positions, or None.
        human_T_positions: (V, 3) human T-pose positions, or None.
        distance_threshold: max source edge length to keep (meters).
        distance_decay_k: exponential weight decay parameter.

    Returns:
        edges: (E, 2) filtered edge pairs.
        residuals: (3E,) flattened residual vector.
        J_edge: (3E, nq) Jacobian matrix.
        weights: (E,) per-edge weights.
    """
    all_edges = get_edge_list(adj_list)
    eps = 1e-8

    # Filter edges by source distance
    filtered = []
    for i, j in all_edges:
        d = np.linalg.norm(source_pts[j] - source_pts[i])
        if d < distance_threshold:
            filtered.append((i, j))

    if len(filtered) == 0:
        return (
            np.empty((0, 2), dtype=np.intp),
            np.empty(0),
            np.empty((0, nq)),
            np.empty(0),
        )

    edges = np.array(filtered, dtype=np.intp)
    E = len(edges)

    use_tpose = robot_T_positions is not None and human_T_positions is not None
    residuals = np.empty(3 * E)
    J_edge = np.empty((3 * E, nq))
    weights = np.empty(E)

    for k, (i, j) in enumerate(edges):
        e_robot = robot_pts[j] - robot_pts[i]
        e_source = source_pts[j] - source_pts[i]
        J_e = J_V[3 * j : 3 * j + 3] - J_V[3 * i : 3 * i + 3]

        if use_tpose:
            e_T_robot = np.linalg.norm(robot_T_positions[j] - robot_T_positions[i]) + eps
            e_T_human = np.linalg.norm(human_T_positions[j] - human_T_positions[i]) + eps
            residuals[3 * k : 3 * k + 3] = e_robot / e_T_robot - e_source / e_T_human
            J_edge[3 * k : 3 * k + 3] = J_e / e_T_robot
            weights[k] = np.exp(-distance_decay_k * np.linalg.norm(e_source))
        else:
            ref_len = np.linalg.norm(e_source) + eps
            residuals[3 * k : 3 * k + 3] = (e_robot - e_source) / ref_len
            J_edge[3 * k : 3 * k + 3] = J_e / ref_len
            weights[k] = np.exp(-distance_decay_k * ref_len)

    return edges, residuals, J_edge, weights


def extract_inter_bone_angles(landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract inter-bone angles from MediaPipe 21-point landmarks.

    Uses palm plane to decompose MCP into flexion (in-plane bending)
    and abduction (out-of-plane lateral spread). PIP/DIP use simple arccos.

    Args:
        landmarks: (21, 3) MediaPipe hand landmarks.

    Returns:
        target_angles: (20,) target joint angles in radians.
        confidence: (20,) per-joint confidence (0-1).
    """
    chains = [
        [0, 1, 2, 3, 4],  # Thumb
        [0, 5, 6, 7, 8],  # Index
        [0, 9, 10, 11, 12],  # Middle
        [0, 13, 14, 15, 16],  # Ring
        [0, 17, 18, 19, 20],  # Pinky
    ]
    target_angles = np.zeros(20)
    confidence = np.ones(20)
    eps = 1e-8

    # Palm plane from wrist, index_MCP, pinky_MCP
    v_idx = landmarks[5] - landmarks[0]  # wrist -> index MCP
    v_pnk = landmarks[17] - landmarks[0]  # wrist -> pinky MCP
    palm_normal = np.cross(v_idx, v_pnk)
    pn_norm = np.linalg.norm(palm_normal)
    if pn_norm > eps:
        palm_normal /= pn_norm
    else:
        palm_normal = np.array([0.0, 0.0, 1.0])

    for f, chain in enumerate(chains):
        base = 4 * f
        bones = [landmarks[chain[k + 1]] - landmarks[chain[k]] for k in range(4)]
        norms = [np.linalg.norm(b) for b in bones]

        # --- MCP flexion + abduction (2 DOF) ---
        if norms[0] > eps and norms[1] > eps:
            b0_hat = bones[0] / norms[0]

            flex_axis = np.cross(palm_normal, b0_hat)
            fa_norm = np.linalg.norm(flex_axis)
            if fa_norm > eps:
                flex_axis /= fa_norm
            else:
                flex_axis = np.array([1.0, 0.0, 0.0])

            abd_axis = palm_normal
            b1_rel = bones[1] / norms[1]

            cos_flex = np.dot(b1_rel, b0_hat)
            sin_flex = np.dot(b1_rel, flex_axis)
            flex_angle = np.arctan2(sin_flex, cos_flex)
            target_angles[base] = max(0.0, flex_angle)
            confidence[base] = 1.0

            abd_component = np.dot(b1_rel, abd_axis)
            in_plane_mag = np.sqrt(max(0, 1.0 - abd_component**2))
            abd_angle = np.arctan2(abd_component, in_plane_mag)
            target_angles[base + 1] = abd_angle
            confidence[base + 1] = 0.8
        else:
            confidence[base] = 0.0
            confidence[base + 1] = 0.0

        # --- PIP (1 DOF): angle between bone[1] and bone[2] ---
        if norms[1] > eps and norms[2] > eps:
            cos_a = np.clip(np.dot(bones[1], bones[2]) / (norms[1] * norms[2]), -1.0, 1.0)
            target_angles[base + 2] = np.arccos(cos_a)
        else:
            confidence[base + 2] = 0.0

        # --- DIP (1 DOF): angle between bone[2] and bone[3] ---
        if norms[2] > eps and norms[3] > eps:
            cos_a = np.clip(np.dot(bones[2], bones[3]) / (norms[2] * norms[3]), -1.0, 1.0)
            target_angles[base + 3] = np.arccos(cos_a)
        else:
            confidence[base + 3] = 0.0

    return target_angles, confidence


def get_edge_list(adj_list: list[list[int]]) -> np.ndarray:
    """Extract unique undirected edges from adjacency list.

    Returns (E, 2) array of (i, j) pairs where i < j.
    """
    edges = set()
    for i, nbrs in enumerate(adj_list):
        for j in nbrs:
            edges.add((min(i, j), max(i, j)))
    return np.array(sorted(edges), dtype=np.intp)


def compute_arap_edge_data(
    source_pts: np.ndarray,
    robot_pts: np.ndarray,
    adj_list: list[list[int]],
    rotations: np.ndarray,
    J_V: np.ndarray,
    nq: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build ARAP per-edge residuals and Jacobian for SQP.

    ARAP energy per edge (i,j):
        ||(robot_i - robot_j) - R_i @ (source_i - source_j)||^2

    Linearized around current robot_pts:
        residual_ij + (J_i - J_j) @ dq

    Args:
        source_pts: (V, 3) source keypoints.
        robot_pts: (V, 3) current robot keypoints.
        adj_list: adjacency list.
        rotations: (V, 3, 3) per-vertex rotations from SVD.
        J_V: (3V, nq) stacked translational Jacobians (hand rows nonzero, object rows zero).
        nq: number of joint DOFs.

    Returns:
        edges: (E, 2) edge index pairs.
        residuals: (3E,) flattened residual vector.
        J_edge: (3E, nq) edge Jacobian matrix.
    """
    edges = get_edge_list(adj_list)
    E = len(edges)

    residuals = np.empty(3 * E)
    J_edge = np.empty((3 * E, nq))

    for k, (i, j) in enumerate(edges):
        e_robot = robot_pts[i] - robot_pts[j]  # (3,)
        e_source = source_pts[i] - source_pts[j]  # (3,)
        e_target = rotations[i] @ e_source  # R_i rotated source edge
        residuals[3 * k : 3 * k + 3] = e_robot - e_target
        J_edge[3 * k : 3 * k + 3] = J_V[3 * i : 3 * i + 3] - J_V[3 * j : 3 * j + 3]

    return edges, residuals, J_edge


def estimate_per_vertex_rotations(
    source_pts: np.ndarray,
    current_pts: np.ndarray,
    adj_list: list[list[int]],
) -> np.ndarray:
    """
    ARAP-style per-vertex rotation estimation via SVD.

    For each vertex i, finds R_i minimizing:
        sum_{j in N(i)} ||R_i @ (source_j - source_i) - (current_j - current_i)||^2

    Args:
        source_pts: (N, 3) source keypoints.
        current_pts: (N, 3) current robot keypoints.
        adj_list: adjacency list from Delaunay.

    Returns:
        (N, 3, 3) per-vertex rotation matrices.
    """
    n_verts = len(source_pts)
    rotations = np.empty((n_verts, 3, 3))
    for i in range(n_verts):
        nbrs = adj_list[i]
        if len(nbrs) < 2:
            rotations[i] = np.eye(3)
            continue
        e_src = source_pts[nbrs] - source_pts[i]
        e_cur = current_pts[nbrs] - current_pts[i]
        H = e_src.T @ e_cur
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        rotations[i] = R
    return rotations
