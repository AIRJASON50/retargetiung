# Dexterous Hand Retargeting

Interaction mesh retargeting for WujiHand, ported from OmniRetarget robot_only mode.

## Algorithm

```
MediaPipe 21 landmarks (human hand)
  -> coordinate transform (SVD frame + MANO rotation)
  -> Delaunay tetrahedralization -> adjacency list
  -> Laplacian coordinates (source target)
  -> SQP optimization: find robot joint angles that minimize
     || Laplacian_source - Laplacian_robot ||^2
     subject to joint limits + trust region
  -> joint angles (20 DOF)
```

## Structure

```
retargeting/
├── src/
│   ├── hand_retarget/              # Core algorithm
│   │   ├── retargeter.py           #   InteractionMeshHandRetargeter (SQP solver)
│   │   ├── mujoco_hand.py          #   Pinocchio FK/Jacobian (uses baseline URDF with tip_link)
│   │   ├── mesh_utils.py           #   Delaunay + Laplacian functions
│   │   ├── mediapipe_io.py         #   PKL loading + preprocessing (global scale only)
│   │   └── config.py               #   Config + 21-point keypoint mapping
│   └── input_devices/              #   PKL replay module (from wuji_retargeting)
├── config/
│   ├── interaction_mesh_left.yaml  #   Interaction mesh config
│   └── baseline_left.yaml          #   Baseline (wuji_retargeting) config
├── demos/
│   ├── hand/
│   │   ├── run_interaction_mesh.py #   Batch retarget -> results/interaction_mesh/*.npz
│   │   ├── run_baseline_batch.py   #   Batch baseline -> results/baseline/*.npz
│   │   ├── play_interaction_mesh.py#   MuJoCo viewer (precompute/live, mesh overlay)
│   │   ├── play_manus.py          #   Baseline replay viewer
│   │   ├── play_mesh_only.py      #   Delaunay topology visualization (no robot)
│   │   ├── play_compare.py        #   Side-by-side baseline vs interaction mesh
│   │   └── compare.py             #   Numerical comparison (smoothness, limits, tips)
│   └── humanoid/
│       └── play_omniretarget_demo.py  # OmniRetarget reference visualization
├── data/
│   ├── manus1.pkl                  #   Manus glove trajectory (13341 frames, left hand)
│   └── cache/                      #   Precomputed retargeting cache
├── results/
│   └── interaction_mesh/           #   Retargeting output (.npz)
├── lib/
│   └── 25_OmniRetarget/           #   OmniRetarget source (reference only)
├── doc/
│   ├── improvement_plan.md         #   Phase 0 + Phase 1 plan
│   └── omni.md                     #   OmniRetarget codebase notes
└── wuji_manus_demo/                #   [legacy] standalone baseline demo, data migrated to data/
```

## Usage

```bash
# Batch retarget
python demos/hand/run_interaction_mesh.py

# Visualize (precompute mode, with cache)
python demos/hand/play_interaction_mesh.py --speed 0.3

# Live retargeting
python demos/hand/play_interaction_mesh.py --live --speed 0.3

# Baseline comparison
python demos/hand/play_manus.py

# Side-by-side comparison
python demos/hand/play_compare.py
```

## Dependencies

- `wuji-retargeting` (pip install -e, preprocessing + baseline)
- `pinocchio` (conda install -c conda-forge, FK/Jacobian)
- `cvxpy` + `clarabel` (SOCP solver)
- `mujoco` (visualization)
- `scipy` (Delaunay)

## Description Assets

- Retargeting: `wuji_hand_description/urdf/left.urdf` (Pinocchio, has tip_link)
- Visualization: `urdf_cali/reference/result/xml/left.xml` (MuJoCo, visual only)

## Status

See `doc/improvement_plan.md`.
