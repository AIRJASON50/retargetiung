"""
Interaction mesh utilities: Delaunay triangulation and Laplacian deformation.
Extracted from OmniRetarget (holosoma_retargeting/src/utils.py).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay


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
    """
    adj = [set() for _ in range(num_vertices)]
    for tet in tetrahedra:
        for i in range(4):
            for j in range(i + 1, 4):
                u, v = tet[i], tet[j]
                adj[u].add(v)
                adj[v].add(u)
    return [list(s) for s in adj]


def calculate_laplacian_coordinates(
    vertices: np.ndarray,
    adj_list: list[list[int]],
    uniform_weight: bool = True,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """
    Calculates the Laplacian coordinates for each vertex: ``L @ vertices``.

    Args:
        vertices: (N, 3) array of vertex positions.
        adj_list: adjacency list for the mesh.
        uniform_weight: if True, use uniform weights; else distance-based.

    Returns:
        (N, 3) array of Laplacian coordinates.

    Note:
        The old standalone implementation used ``1/(1.5*d + eps)`` for non-uniform
        weights, while ``calculate_laplacian_matrix`` uses ``1/(d + eps)``.  All
        production code uses ``uniform_weight=True``, so the difference was moot.
        This function now delegates to ``calculate_laplacian_matrix`` to keep one
        canonical weight computation (the ``1/(d + eps)`` variant).
    """
    L = calculate_laplacian_matrix(vertices, adj_list, uniform_weight, epsilon)
    return L @ vertices


def calculate_laplacian_matrix(
    vertices: np.ndarray,
    adj_list: list[list[int]],
    uniform_weight: bool = True,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """
    Calculates the (N, N) Laplacian matrix.

    L[i, i] = 1.0
    L[i, j] = -w_ij / sum(w)  for neighbor j

    With uniform weights: L[i, j] = -1 / degree(i)

    Returns:
        (N, N) Laplacian matrix.
    """
    N = len(vertices)
    L = np.zeros((N, N))

    for i in range(N):
        neighbors = adj_list[i]
        if len(neighbors) > 0:
            if uniform_weight:
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

    def _add(i, j):
        adj[i].append(j)
        adj[j].append(i)

    # Wrist to each finger root
    for root in [1, 5, 9, 13, 17]:
        _add(0, root)

    # Intra-finger chains
    for start in [1, 5, 9, 13, 17]:
        for k in range(3):
            _add(start + k, start + k + 1)

    return adj


def get_edge_list(adj_list: list[list[int]]) -> np.ndarray:
    """Extract unique directed edges from adjacency list.

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
        e_robot = robot_pts[i] - robot_pts[j]            # (3,)
        e_source = source_pts[i] - source_pts[j]         # (3,)
        e_target = rotations[i] @ e_source                # R_i rotated source edge
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
    N = len(source_pts)
    rotations = np.empty((N, 3, 3))
    for i in range(N):
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
