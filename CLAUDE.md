# CLAUDE.md -- Hand Retargeting

## Overview

Hand retargeting system: MediaPipe 21-point landmarks → WujiHand 20 DOF robot.
Pipeline: SVD+MANO alignment → S1 cosine IK (bone direction) → S2 Laplacian IM (position refinement).
Solver: daqp QP (330+ fps manus, 160+ fps HO-Cap).

## Environment

```bash
conda activate mjplgd  # Python 3.11, pinocchio, mujoco, qpsolvers[daqp]
export WUJI_SDK_PATH="/home/l/ws/doc/WujiRepo/wuji_retargeting_private/public"  # optional, has default fallback
```

## Directory Structure

```
retargeting/
├── src/hand_retarget/           # Core library
│   ├── retargeter.py            # InteractionMeshHandRetargeter (S1+S2 pipeline)
│   ├── mesh_utils.py            # Delaunay, Laplacian, skeleton adjacency
│   ├── mujoco_hand.py           # PinocchioHandModel (fixed base) + MuJoCoFloatingHandModel
│   ├── mediapipe_io.py          # SVD+MANO preprocessing, HO-Cap clip loading
│   └── config.py                # HandRetargetConfig (17 fields) + JOINTS_MAPPING
├── src/scene_builder/           # MuJoCo scene injection (tip sites, wrist6dof, collision)
├── assets/                      # Robot models (MuJoCo XML, STL meshes)
├── config/                      # YAML configs
├── demos/
│   ├── legacy/                  # Manus visualization (play_interaction_mesh.py, play_manus.py)
│   ├── hocap/                   # HO-Cap visualization (play_hocap.py)
│   └── shared/                  # Overlay, playback, cache utils
├── scripts/                     # Batch retargeting (retarget_hocap.py)
├── tests/                       # pytest gate tests (test_gate.py)
├── experiments/archive/         # Completed experiment results (EXP-1~8)
├── doc/                         # Algorithm notes, experiment docs
└── lib/                         # Reference libraries (gitignored)
```

## Quick Start

```bash
# Manus visualization
PYTHONPATH=src python demos/legacy/play_interaction_mesh.py --semantic-weight

# HO-Cap visualization
PYTHONPATH=src python demos/hocap/play_hocap.py --clip hocap__subject_3__20231024_161306__seg00

# Run tests
PYTHONPATH=src pytest tests/test_gate.py -v

# Lint
ruff check src/ && ruff format src/
```

## Core Pipeline

```
MediaPipe 21pts → mediapipe_io (SVD+MANO align) → retarget_frame()
                                                    ├── _build_topology (Delaunay → adj_list → Laplacian)
                                                    ├── _compute_weights (pinch-aware semantic)
                                                    └── _run_optimization
                                                          ├── S1: solve_angle_warmup (cosine IK, daqp)
                                                          └── S2: solve_single_iteration (Laplacian, daqp)
```

Default config: `anchor=5, laplacian=5, smooth=1` (5:5:1 ratio). S1+S2 enabled.

## Key Config Fields

| Field | Default | Description |
|-------|---------|-------------|
| `use_angle_warmup` | True | S1 cosine IK bone direction alignment |
| `angle_anchor_weight` | 5.0 | S1 angle anchor in S2 joint cost |
| `smooth_weight` | 1.0 | Temporal smoothness |
| `use_link_midpoints` | False | 20 link midpoints instead of 21 joint origins |
| `exclude_fingers_from_laplacian` | None | Finger indices (0-4) excluded from S2 gradient |
| `delaunay_edge_threshold` | 0.06 | Filter Delaunay edges > 60mm |
| `laplacian_distance_weight_k` | 20.0 | Exponential decay weight for Laplacian |

## HO-Cap Notes

- SVD alignment forced (wrist_q unreliable, SVD 胜率 92%)
- All 6 wrist DOF locked (SVD puts landmarks in wrist frame)
- MediaPipe bone lengths are fixed template (identical across all 9 subjects)
- S1 causes DIP hyperextension on HO-Cap (robot default DIP ≈ 31° vs source ≈ 5-16°)
- Best HO-Cap config: baseline without S1 (33.7% hyper vs 97% with S1)

## Joint Index

```
20 DOF, 5 fingers × 4 joints: q[4f], q[4f+1], q[4f+2], q[4f+3] = MCP_flex, MCP_abd, PIP, DIP
Thumb: q[0-3]  Index: q[4-7]  Middle: q[8-11]  Ring: q[12-15]  Pinky: q[16-19]
```

## Experiment History

| # | Name | Result | Status |
|---|------|--------|--------|
| 1 | Fixed topology | Jerk -43%, direction -11% | Adopted |
| 2 | Distance weight | Position worsened | Rejected |
| 3 | Semantic weight | Pinch -32% | Adopted |
| 4 | ARAP rotation comp | Hyper -25pp, speed -2.5x | Rejected |
| 5 | Orientation probes | Marginal improvement | Rejected |
| 6 | Bone scaling | Laplacian resists non-uniform scaling | Rejected |
| 7 | ARAP edge + skeleton | DIP hyper 0%, overall 63.6% | Not merged |
| 8 | Link midpoint + joint opt | Index tip -24%, pinky protected | Adopted |
| 9 | daqp solver | 10x speedup, identical output | Adopted |
