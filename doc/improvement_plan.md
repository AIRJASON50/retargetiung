# Hand Retargeting — Improvement Plan

> Last updated: 2026-04-18
> Status: Production refactoring complete. S1+S2 pipeline + daqp solver deployed.

## Current Architecture

```
S1 (cosine IK, daqp)  →  S2 (Laplacian IM, daqp)  →  output qpos
    bone direction        position refinement           330+ fps (manus)
    scale-invariant       contact-aware (IM)            160+ fps (HO-Cap)
```

Default: 5:5:1 ratio (anchor=5, laplacian=5, smooth=1).

## Completed Experiments

| # | Name | Result | Status |
|---|------|--------|--------|
| 1 | Fixed topology | Jerk -43%, direction -11% | **Adopted** |
| 2 | Distance weight | Position worsened | Rejected |
| 3 | Semantic weight | Pinch -32% | **Adopted** |
| 4 | ARAP rotation comp | Hyper -25pp, speed -2.5x | Rejected (code deleted) |
| 5 | Orientation probes | Marginal improvement | Rejected (code deleted) |
| 6 | Bone scaling | Laplacian resists non-uniform scaling | Rejected (code deleted) |
| 7 | ARAP edge + skeleton | DIP hyper 0%, overall 63.6% | Not merged (code deleted) |
| 8 | Link midpoint + joint opt | Index tip -24%, pinky protected | **Adopted** |
| 9 | daqp solver | 10x speedup, identical output | **Adopted** |

## Completed Refactoring (2026-04-18)

- Deleted 6 experimental methods, 9 config fields, 4 mesh_utils functions
- Replaced CVXPY+Clarabel with daqp QP solver (10x speedup)
- Split retarget_frame() into _build_topology, _compute_weights, _run_optimization
- Renamed MuJoCoHandModel → PinocchioHandModel
- Replaced hardcoded paths with WUJI_SDK_PATH env var
- Added pytest gate tests (4 tests, all passing)
- Total: 2927 → ~2100 lines (-28%)
- Gate: 0.000006° max diff from baseline
- Details: doc/refactoring_progress.md

## Known Issues

1. **HO-Cap S1 DIP hyperextension**: Robot default DIP ≈ 31°, source hand fingers nearly straight (5-16°). S1 cosine IK drives DIP negative to match direction. Best HO-Cap config: no S1 (33.7% hyper vs 97% with S1).

2. **MediaPipe fixed bone template**: All 9 HO-Cap subjects have identical bone lengths (MediaPipe 3D model template, not real anatomy). MCP-PIP is 21-33mm vs robot 47.7mm (0.44-0.69x ratio).

3. **Wrist alignment**: SVD+MANO alignment quality 0.97-0.99 cos. Procrustes to robot MCPs gives 0.999 but absorbs wrist rotation. SVD is correct for fixed-base retargeting.

## Future Directions

1. **DIP offset calibration**: Compute per-joint q_offset from robot T-pose to fix S1 DIP hyperextension
2. **Per-bone Laplacian scaling**: Scale Laplacian target per-edge (edge ratio direction), not per-vertex
3. **mink orientation IK**: Use mink's SO3 log-map error for more principled direction matching
4. **Contact graph energy**: Explicit hand-object contact preservation as primary objective
5. **SAME-style GCN**: Skeleton-agnostic learned retargeting for arbitrary hand topologies
