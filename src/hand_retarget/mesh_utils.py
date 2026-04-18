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
    epsilon: float = 1e-6,
    distance_decay_k: float | None = None,
) -> np.ndarray:
    """
    Calculates the Laplacian coordinates for each vertex: ``L @ vertices``.

    Args:
        vertices: (N, 3) array of vertex positions.
        adj_list: adjacency list for the mesh.
        distance_decay_k: if set, use exp(-k*d) weights instead of uniform.

    Returns:
        (N, 3) array of Laplacian coordinates.
    """
    L = calculate_laplacian_matrix(vertices, adj_list, epsilon, distance_decay_k=distance_decay_k)
    return L @ vertices


def calculate_laplacian_matrix(
    vertices: np.ndarray,
    adj_list: list[list[int]],
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
            else:
                weights = np.ones(len(neighbors)) / len(neighbors)

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


def get_edge_list(adj_list: list[list[int]]) -> np.ndarray:
    """Extract unique undirected edges from adjacency list.

    Returns (E, 2) array of (i, j) pairs where i < j.
    """
    edges = set()
    for i, nbrs in enumerate(adj_list):
        for j in nbrs:
            edges.add((min(i, j), max(i, j)))
    return np.array(sorted(edges), dtype=np.intp)


