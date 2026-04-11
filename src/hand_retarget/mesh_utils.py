"""
Interaction mesh utilities: Delaunay triangulation and Laplacian deformation.
Extracted from OmniRetarget (holosoma_retargeting/src/utils.py).
"""

import numpy as np
from scipy.spatial import Delaunay


def create_interaction_mesh(vertices: np.ndarray):
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
    Calculates the Laplacian coordinates for each vertex.

    L(p_i) = p_i - weighted_mean(neighbors)

    Args:
        vertices: (N, 3) array of vertex positions.
        adj_list: adjacency list for the mesh.
        uniform_weight: if True, use uniform weights; else distance-based.

    Returns:
        (N, 3) array of Laplacian coordinates.
    """
    laplacian = np.zeros_like(vertices)

    for i in range(len(vertices)):
        neighbors = adj_list[i]
        if len(neighbors) > 0:
            vi = vertices[i]
            neighbor_pos = vertices[neighbors]
            distances = np.linalg.norm(vi - neighbor_pos, axis=1)

            if uniform_weight:
                weights = np.ones_like(distances)
            else:
                weights = 1.0 / (1.5 * distances + epsilon)

            w_sum = np.sum(weights)
            center = np.sum(weights[:, np.newaxis] * neighbor_pos, axis=0) / w_sum
            laplacian[i] = vi - center

    return laplacian


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
