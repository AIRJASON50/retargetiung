"""
EXP-1: Fixed topology vs per-frame Delaunay rebuild.

Compares three conditions:
  A. Baseline (wuji_retargeting IK)
  B. Interaction Mesh — per-frame Delaunay (current, matches OmniRetarget)
  C. Interaction Mesh — fixed topology (first frame only, matches Ho 2010)

Uses benchmark metrics on manus1_5k.pkl.

Usage:
    cd /home/l/ws/RL/retargeting
    PYTHONPATH=src python experiments/exp1_fixed_topology.py
    PYTHONPATH=src python experiments/exp1_fixed_topology.py --frames 1000
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "experiments"))

from hand_retarget import InteractionMeshHandRetargeter, HandRetargetConfig
from hand_retarget.mediapipe_io import load_pkl_sequence, preprocess_sequence
from hand_retarget.mesh_utils import create_interaction_mesh, get_adjacency_list, calculate_laplacian_matrix
from benchmark import RetargetBenchmark

DEFAULT_PKL = PROJECT_DIR / "data" / "manus1_5k.pkl"
DEFAULT_CONFIG = PROJECT_DIR / "config" / "interaction_mesh_left.yaml"
DEFAULT_URDF = Path("/home/l/ws/doc/WujiRepo/wuji_retargeting_private/public/wuji_retargeting/wuji_hand_description/urdf/left.urdf")
BASELINE_CONFIG = PROJECT_DIR / "demos" / "hand" / "baseline" / "wuji_manus_demo" / "config" / "retarget_manus_left.yaml"
HAND_SIDE = "left"


def run_baseline(frames, N):
    """Run baseline retargeting."""
    from wuji_retargeting import Retargeter
    retargeter = Retargeter.from_yaml(str(BASELINE_CONFIG), HAND_SIDE)

    qpos = np.zeros((N, 20))
    t0 = time.time()
    for t in range(N):
        qpos[t] = retargeter.retarget(frames[t][f"{HAND_SIDE}_fingers"])
    elapsed = time.time() - t0
    print(f"  Baseline: {N} frames in {elapsed:.1f}s ({N/elapsed:.0f} fps)")
    return qpos


def run_im_perframe(proc_seq, config, N):
    """Run interaction mesh with per-frame Delaunay (current behavior)."""
    retargeter = InteractionMeshHandRetargeter(config)
    qpos = np.zeros((N, retargeter.nq))
    q_prev = retargeter.hand.get_default_qpos()

    t0 = time.time()
    for t in range(N):
        q = retargeter.retarget_frame(proc_seq[t], q_prev, is_first_frame=(t == 0))
        qpos[t] = q
        q_prev = q
    elapsed = time.time() - t0
    print(f"  IM per-frame: {N} frames in {elapsed:.1f}s ({N/elapsed:.0f} fps)")
    return qpos


def run_im_fixed(proc_seq, config, N):
    """Run interaction mesh with fixed topology (first frame only)."""
    retargeter = InteractionMeshHandRetargeter(config)
    qpos = np.zeros((N, retargeter.nq))
    q_prev = retargeter.hand.get_default_qpos()

    # Build topology from first frame and fix it
    source_pts_0 = retargeter._extract_source_keypoints(proc_seq[0])
    _, simplices = create_interaction_mesh(source_pts_0)
    fixed_adj = get_adjacency_list(simplices, retargeter.n_keypoints)

    t0 = time.time()
    for t in range(N):
        source_pts = retargeter._extract_source_keypoints(proc_seq[t])

        # Compute target Laplacian using fixed topology
        from hand_retarget.mesh_utils import calculate_laplacian_coordinates
        target_lap = calculate_laplacian_coordinates(source_pts, fixed_adj)

        # SQP iterations with fixed adj_list
        n_iter = 50 if t == 0 else 10
        q_current = q_prev.copy()
        last_cost = float("inf")
        for _ in range(n_iter):
            q_current, cost = retargeter.solve_single_iteration(
                q_current, q_prev, target_lap, fixed_adj
            )
            if abs(last_cost - cost) < 1e-8:
                break
            last_cost = cost

        qpos[t] = q_current
        q_prev = q_current
        retargeter._adj_list = fixed_adj  # keep for visualization compatibility

    elapsed = time.time() - t0
    print(f"  IM fixed: {N} frames in {elapsed:.1f}s ({N/elapsed:.0f} fps)")
    return qpos


def main():
    parser = argparse.ArgumentParser(description="EXP-1: Fixed vs per-frame topology")
    parser.add_argument("--pkl", type=str, default=str(DEFAULT_PKL))
    parser.add_argument("--frames", type=int, default=None, help="Limit number of frames")
    args = parser.parse_args()

    # Load data
    import pickle
    with open(args.pkl, "rb") as f:
        recording = pickle.load(f)
    frames = [f for f in recording if not np.allclose(f[f"{HAND_SIDE}_fingers"], 0)]

    landmarks_seq, _ = load_pkl_sequence(args.pkl, HAND_SIDE)
    config = HandRetargetConfig.from_yaml(str(DEFAULT_CONFIG), mjcf_path=str(DEFAULT_URDF))

    # Preprocess
    retargeter_tmp = InteractionMeshHandRetargeter(config)
    proc_seq = preprocess_sequence(
        landmarks_seq, config.mediapipe_rotation,
        hand_side=HAND_SIDE, global_scale=retargeter_tmp.global_scale
    )

    N = min(len(frames), len(proc_seq))
    if args.frames:
        N = min(N, args.frames)
    frames = frames[:N]
    proc_seq = proc_seq[:N]

    print(f"Running EXP-1 on {N} frames from {args.pkl}")
    print("=" * 60)

    # Run all three conditions
    print("\nCondition A: Baseline (NLopt IK)")
    qpos_bl = run_baseline(frames, N)

    print("\nCondition B: IM per-frame Delaunay (current)")
    qpos_im_pf = run_im_perframe(proc_seq, config, N)

    print("\nCondition C: IM fixed topology (Ho 2010)")
    qpos_im_fix = run_im_fixed(proc_seq, config, N)

    # Benchmark
    bench = RetargetBenchmark(str(DEFAULT_URDF), HAND_SIDE)

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    results_bl = bench.evaluate(proc_seq, qpos_bl)
    RetargetBenchmark.print_results(results_bl, "A: Baseline (NLopt IK)")

    results_pf = bench.evaluate(proc_seq, qpos_im_pf)
    RetargetBenchmark.print_results(results_pf, "B: IM Per-Frame Delaunay")

    results_fix = bench.evaluate(proc_seq, qpos_im_fix)
    RetargetBenchmark.print_results(results_fix, "C: IM Fixed Topology (Ho 2010)")

    # Summary comparison
    print(f"\n{'='*60}")
    print(f"  SUMMARY COMPARISON")
    print(f"{'='*60}")
    print(f"{'Metric':<35} {'Baseline':>10} {'IM PerFrame':>12} {'IM Fixed':>10}")
    print(f"{'-'*67}")
    print(f"{'Tip Pos Error (mm)':<35} {results_bl['tip_pos_error_mm']['mean']:>10.2f} {results_pf['tip_pos_error_mm']['mean']:>12.2f} {results_fix['tip_pos_error_mm']['mean']:>10.2f}")
    print(f"{'Tip Dir Error (deg)':<35} {results_bl['tip_dir_error_deg']['mean']:>10.2f} {results_pf['tip_dir_error_deg']['mean']:>12.2f} {results_fix['tip_dir_error_deg']['mean']:>10.2f}")
    print(f"{'Inter-Tip Dist Error (mm)':<35} {results_bl['inter_tip_distance_error_mm']['mean']:>10.2f} {results_pf['inter_tip_distance_error_mm']['mean']:>12.2f} {results_fix['inter_tip_distance_error_mm']['mean']:>10.2f}")
    print(f"{'Jerk (rad/s^3)':<35} {results_bl['smoothness']['jerk_rad_s3']:>10.1f} {results_pf['smoothness']['jerk_rad_s3']:>12.1f} {results_fix['smoothness']['jerk_rad_s3']:>10.1f}")
    print(f"{'Temporal Consistency':<35} {results_bl['smoothness']['temporal_consistency']:>10.5f} {results_pf['smoothness']['temporal_consistency']:>12.5f} {results_fix['smoothness']['temporal_consistency']:>10.5f}")
    print(f"{'C-Space Coverage':<35} {results_bl['cspace_coverage']['mean']:>10.3f} {results_pf['cspace_coverage']['mean']:>12.3f} {results_fix['cspace_coverage']['mean']:>10.3f}")
    print(f"{'Joint Limit Violation %':<35} {results_bl['joint_limit_violation_rate']*100:>10.1f} {results_pf['joint_limit_violation_rate']*100:>12.1f} {results_fix['joint_limit_violation_rate']*100:>10.1f}")


if __name__ == "__main__":
    main()
