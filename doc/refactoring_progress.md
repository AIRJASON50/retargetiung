# Code Refactoring Progress

> Started: 2026-04-18
> Gate baseline: tests/refactor_gate_baseline.npz (100 frames, max diff threshold: 0.01°)
> Backup tag: `pre-refactor`

## Completed

### Phase 1: Infrastructure (commit 26ef509)
- [x] Replace hardcoded `/home/l/ws/...` paths with `WUJI_SDK_PATH` env var (5 files)
- [x] Remove unused imports (cvxpy, experimental mesh_utils functions)
- Gate: 0.000000° ✓

### Phase 2a: Delete experimental methods from retargeter.py (commit 0adf952)
- [x] Delete `solve_single_iteration_arap` (111 lines)
- [x] Delete `solve_single_iteration_edge_ratio` (94 lines)
- [x] Delete `_augment_with_probes` (17 lines)
- [x] Delete `_apply_bone_scaling` (102 lines)
- [x] Delete `_compute_robot_segment_lengths` (22 lines)
- [x] Delete `_compute_source_segment_lengths` (16 lines)
- [x] Remove edge ratio / ARAP / rotation compensation dispatch in retarget_frame
- [x] Remove bone scaling / probe init from __init__
- [x] Remove `ARAP_LAPLACIAN_WARMUP` constant
- retargeter.py: 1366 → 895 lines (-35%)
- Gate: 0.000006° ✓

### Phase 2b: Delete experimental config + mesh_utils dead code (commit df0c407)
- [x] Delete 9 config fields (probes, bone scaling, ARAP, edge ratio, self_collision)
- [x] Delete `_PROBE_MAPPING_LEFT/RIGHT` constants
- [x] Delete `fingertip_links` property
- [x] Simplify `joints_mapping` (no probe merging)
- [x] Fix `global_scale` default: None → 1.0
- [x] Clean `_YAML_FIELD_MAP`, `_YAML_ENABLED_MAP`, `make_stamp()`
- [x] Delete 4 mesh_utils functions (extract_inter_bone_angles, compute_edge_ratio_data, compute_arap_edge_data, estimate_per_vertex_rotations)
- [x] Remove `uniform_weight` parameter from Laplacian functions
- config.py: 357 → 298 (-17%), mesh_utils.py: 488 → 228 (-53%)
- Gate: 0.000006° ✓

## Remaining

### Phase 2c: Demo script cleanup
- [x] Remove 8 CLI flags from play_interaction_mesh.py (--fixed-topology, --distance-weight, --probes, --bone-scale, --no-palm-spread, --rotation-comp, --arap-edge, --edge-ratio)
- [x] Remove inline SQP solver from play_interaction_mesh.py
- [x] Fix play_manus.py DEFAULT_PKL path
- [x] Remove --edge-ratio from play_hocap.py
- [ ] Remove batch_retarget_hocap.py (deferred) (superseded by retarget_hocap.py)

### Phase 3: Structure refactoring
- [x] Split retarget_frame() into _build_topology, _compute_weights, _run_optimization into _preprocess_landmarks, _build_topology, _run_optimization
- [x] Eliminate config mutability (n_iters parameter) in retarget_hocap_sequence (pass overrides)
- [x] Rename MuJoCoHandModel → PinocchioHandModel → PinocchioHandModel (with compat alias)
- [x] Remove _last_q dead store dead store in mujoco_hand.py
- [x] Clean retargeter.py __init__ (remove empty "Edge ratio" comment)
- [x] Fix global_scale default 1.0 (remove `is not None` check since default is now 1.0)

### Phase 4: Testing
- [x] Create tests/test_gate.py (4 tests, all passing) (pytest-compatible)
- [x] Add test_config_defaults
- [x] Add test_synthetic_roundtrip (FK → landmarks → retarget → verify)
- [x] Update YAML config (smooth=1.0, angle_warmup enabled) (smooth_weight 0.2 → 1.0, add angle_warmup section)

### Phase 5: Documentation
- [ ] Update CLAUDE.md
- [ ] Update doc/improvement_plan.md
- [x] Delete doc/trajmd
- [x] ruff format all files all files

## Line Count Tracking

| File | Original | Current | Target |
|------|----------|---------|--------|
| retargeter.py | 1366 | 894 | ~750 |
| config.py | 357 | 298 | ~250 |
| mesh_utils.py | 488 | 228 | 228 |
| mujoco_hand.py | 486 | 486 | ~480 |
| mediapipe_io.py | 230 | 230 | 230 |
| **Total** | **2927** | **2136** | **~1938** |
