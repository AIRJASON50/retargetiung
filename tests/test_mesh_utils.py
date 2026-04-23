"""Unit tests for ``hand_retarget.mesh_utils`` numeric correctness.

Covers small, hand-verifiable inputs that exercise every public function in
``src/hand_retarget/mesh_utils.py`` without needing a MuJoCo/Pinocchio model:

  * Delaunay mesh construction on a single tetrahedron.
  * Adjacency <-> edge-list conversions (uniqueness, (i<j) canonicalization).
  * MediaPipe 21-point hand skeleton topology (20 edges, per-finger chain).
  * Midpoint skeleton topology (15 intra-finger edges, no cross-finger).
  * Laplacian matrix algebraic properties:
      - each row sums to 0  (shift-invariant operator);
      - ``L @ (points + c)`` == ``L @ points``;
      - exponential distance weights give higher magnitude to closer neighbors.
  * Distance-based edge filtering removes over-long edges.

Fast by design (no model loading). Target runtime < 500 ms for the whole file,
each individual test well under 50 ms.

Run: ``PYTHONPATH=src pytest tests/test_mesh_utils.py -v``
"""

from __future__ import annotations

import numpy as np
import pytest

from hand_retarget.mesh_utils import (
    calculate_laplacian_coordinates,
    calculate_laplacian_matrix,
    create_interaction_mesh,
    filter_adjacency_by_distance,
    get_adjacency_list,
    get_edge_list,
    get_midpoint_skeleton_adjacency,
    get_skeleton_adjacency,
)

# Numerical tolerance for shift-invariance and row-sum identities.
NUM_TOL = 1e-10


# ==========================================================================
# Delaunay + adjacency
# ==========================================================================


def test_delaunay_small_tetrahedron():
    """4 non-coplanar points yield a single 4-simplex; every vertex is adjacent
    to the remaining 3 via ``get_adjacency_list``.
    """
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    verts_out, simplices = create_interaction_mesh(vertices)

    # create_interaction_mesh returns the input points verbatim.
    assert verts_out is vertices
    # Exactly one tetrahedron covering all 4 vertices (order may vary).
    assert simplices.shape == (1, 4)
    assert set(simplices[0].tolist()) == {0, 1, 2, 3}

    adj = get_adjacency_list(simplices, num_vertices=4)
    assert len(adj) == 4
    for i in range(4):
        assert sorted(adj[i]) == [j for j in range(4) if j != i]


# ==========================================================================
# Edge list
# ==========================================================================


def test_edge_list_symmetric():
    """``get_edge_list`` emits each undirected edge exactly once with ``i < j``.

    The adjacency list records both directions (u->v and v->u); the edge list
    must canonicalize to (min, max) and deduplicate.
    """
    # Tetrahedron adjacency: every pair (0,1,2,3) connected.
    adj = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]
    edges = get_edge_list(adj)

    # 4 vertices fully connected => C(4,2)=6 unique edges.
    assert edges.shape == (6, 2)
    # Each row i<j.
    assert np.all(edges[:, 0] < edges[:, 1])
    # Rows are sorted lexicographically.
    lex_key = edges[:, 0] * 1000 + edges[:, 1]
    assert np.all(np.diff(lex_key) > 0)
    # Content matches the fully-connected edge set.
    expected = {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)}
    assert {tuple(e) for e in edges.tolist()} == expected


# ==========================================================================
# Skeleton topology
# ==========================================================================


def test_skeleton_adjacency_hand():
    """MediaPipe 21-point hand skeleton has 20 edges: 5 wrist->MCP + 5x3 chain.

    Topology invariants:
      * wrist (index 0) has degree 5 (one edge per finger root).
      * each fingertip (4, 8, 12, 16, 20) has degree 1 (chain end).
      * finger roots (1, 5, 9, 13, 17) have degree 2 (wrist + next joint).
    """
    adj = get_skeleton_adjacency(21)
    assert len(adj) == 21

    # Total undirected edges: sum(deg) / 2.
    total_edges = sum(len(n) for n in adj) // 2
    assert total_edges == 20

    # Wrist degree = number of fingers.
    assert len(adj[0]) == 5
    assert sorted(adj[0]) == [1, 5, 9, 13, 17]

    # Each fingertip has one neighbor (the DIP/IP below it).
    for tip in (4, 8, 12, 16, 20):
        assert sorted(adj[tip]) == [tip - 1]

    # Finger roots (MCP/CMC) touch wrist + next joint in the chain.
    for root in (1, 5, 9, 13, 17):
        assert sorted(adj[root]) == [0, root + 1]


def test_midpoint_skeleton_adjacency():
    """20-midpoint skeleton: 5 fingers x 3 intra-finger edges = 15, no wrist."""
    adj = get_midpoint_skeleton_adjacency(20)
    assert len(adj) == 20

    total_edges = sum(len(n) for n in adj) // 2
    assert total_edges == 15

    # Per-finger chain: 4f, 4f+1, 4f+2, 4f+3 connected linearly.
    for f in range(5):
        base = f * 4
        # Interior midpoints (base+1, base+2) have degree 2.
        assert sorted(adj[base + 1]) == [base, base + 2]
        assert sorted(adj[base + 2]) == [base + 1, base + 3]
        # Chain endpoints (base, base+3) have degree 1.
        assert sorted(adj[base]) == [base + 1]
        assert sorted(adj[base + 3]) == [base + 2]


# ==========================================================================
# Laplacian matrix
# ==========================================================================


def test_laplacian_matrix_row_sum_zero():
    """Each row of ``L`` must sum to 0: diagonal = 1, off-diagonals = -w_ij/sum(w).

    Guarantees ``L`` is a valid graph Laplacian up to sign normalization.
    """
    # Tetrahedron vertices + fully connected adjacency.
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    adj = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]

    L = calculate_laplacian_matrix(vertices, adj)
    assert L.shape == (4, 4)

    row_sums = L.sum(axis=1)
    assert row_sums == pytest.approx(np.zeros(4), abs=NUM_TOL)

    # Uniform weights: each off-diagonal should be -1/3.
    for i in range(4):
        assert L[i, i] == pytest.approx(1.0, abs=NUM_TOL)
        for j in range(4):
            if i != j:
                assert L[i, j] == pytest.approx(-1.0 / 3.0, abs=NUM_TOL)


def test_laplacian_coordinates_invariant_under_translation():
    """Laplacian coordinates ``L @ P`` are shift-invariant: ``L @ (P + c) == L @ P``.

    This follows from row-sum = 0 and is a defining property we rely on for
    position-based retargeting. Uses a non-trivial (non-regular) point cloud
    so the result is not coincidentally zero.
    """
    # Non-degenerate tetrahedron with asymmetric coordinates.
    points = np.array(
        [[0.1, 0.2, 0.3], [1.5, 0.4, -0.2], [0.6, 1.1, 0.9], [-0.3, 0.7, 1.4]],
        dtype=float,
    )
    adj = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]

    lap_coords = calculate_laplacian_coordinates(points, adj)
    translated = points + np.array([1.0, 2.0, 3.0])
    lap_coords_translated = calculate_laplacian_coordinates(translated, adj)

    assert lap_coords.shape == points.shape
    # Shift-invariance: identical Laplacian coords under translation.
    assert lap_coords_translated == pytest.approx(lap_coords, abs=NUM_TOL)


# ==========================================================================
# Edge filtering
# ==========================================================================


def test_filter_adjacency_by_distance_removes_long_edges():
    """Long edges (>= threshold) are dropped; short edges are kept.

    Setup: 3-vertex path graph with adjacency ``[[1], [0, 2], [1]]`` and
    coordinates placing v0 and v1 close (~0.03) but v2 far (~100).
    With threshold=0.05, only edge (0,1) should survive; vertex 2 ends up
    isolated (empty list) since its only neighbor v1 is >threshold away.
    """
    adj = [[1], [0, 2], [1]]
    vertices = np.array([[0.0, 0.0, 0.0], [0.03, 0.0, 0.0], [100.0, 0.0, 0.0]])

    filtered = filter_adjacency_by_distance(adj, vertices, threshold=0.05)

    # Structural invariant: output length matches input.
    assert len(filtered) == 3
    # Edge (0,1) distance 0.03 < 0.05, kept.
    assert sorted(filtered[0]) == [1]
    # Vertex 1 loses neighbor 2 (distance 99.97), retains 0.
    assert sorted(filtered[1]) == [0]
    # Vertex 2's only neighbor (v1) is far, so it becomes isolated.
    assert filtered[2] == []


# ==========================================================================
# Distance-decay weighting
# ==========================================================================


def test_laplacian_distance_weight_decay():
    """With ``distance_decay_k > 0``, closer neighbors receive larger weights.

    Build a star with vertex 0 connected to v1 (close) and v2 (far). The
    exponential weight ``exp(-k*d)`` must produce ``|L[0, 1]| > |L[0, 2]|``.
    """
    vertices = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.05, 0.0, 0.0]])
    adj = [[1, 2], [0], [0]]

    L = calculate_laplacian_matrix(vertices, adj, distance_decay_k=100.0)

    assert L.shape == (3, 3)
    # Closer neighbor v1 (d=0.01) must outweigh far neighbor v2 (d=0.05).
    assert abs(L[0, 1]) > abs(L[0, 2])
    # Row sum still zero (normalization invariant).
    assert L[0].sum() == pytest.approx(0.0, abs=NUM_TOL)
