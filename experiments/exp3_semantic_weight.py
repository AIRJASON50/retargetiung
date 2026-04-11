"""
EXP-3: Semantic weight on Laplacian loss.

Compares three conditions:
  A. Baseline (wuji_retargeting IK)
  B. IM fixed topology + uniform weight
  C. IM fixed topology + semantic weight (pinch-aware)

Uses manus1_pinch.pkl (frames 700-1200, contains pinch region).

Usage:
    cd /home/l/ws/RL/retargeting
    PYTHONPATH=src python experiments/exp3_semantic_weight.py
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
from hand_retarget.mesh_utils import create_interaction_mesh, get_adjacency_list, calculate_laplacian_coordinates
from benchmark import RetargetBenchmark

DEFAULT_PKL = PROJECT_DIR / "data" / "manus1_pinch.pkl"
DEFAULT_CONFIG = PROJECT_DIR / "config" / "interaction_mesh_left.yaml"
DEFAULT_URDF = Path("/home/l/ws/doc/WujiRepo/wuji_retargeting_private/public/wuji_retargeting/wuji_hand_description/urdf/left.urdf")
BASELINE_CONFIG = PROJECT_DIR / "demos" / "hand" / "baseline" / "wuji_manus_demo" / "config" / "retarget_manus_left.yaml"
HAND_SIDE = "left"


def run_baseline(frames, N):
    from wuji_retargeting import Retargeter
    retargeter = Retargeter.from_yaml(str(BASELINE_CONFIG), HAND_SIDE)
    qpos = np.zeros((N, 20))
    t0 = time.time()
    for t in range(N):
        qpos[t] = retargeter.retarget(frames[t][f"{HAND_SIDE}_fingers"])
    elapsed = time.time() - t0
    print(f"  Baseline: {N} frames in {elapsed:.1f}s ({N/elapsed:.0f} fps)")
    return qpos


def run_im_fixed(proc_seq, config, N, use_semantic_weights=False, label=""):
    retargeter = InteractionMeshHandRetargeter(config)
    qpos = np.zeros((N, retargeter.nq))
    q_prev = retargeter.hand.get_default_qpos()

    # Build fixed topology from first frame
    source_pts_0 = retargeter._extract_source_keypoints(proc_seq[0])
    _, simplices = create_interaction_mesh(source_pts_0)
    fixed_adj = get_adjacency_list(simplices, retargeter.n_keypoints)

    t0 = time.time()
    for t in range(N):
        source_pts = retargeter._extract_source_keypoints(proc_seq[t])
        target_lap = calculate_laplacian_coordinates(source_pts, fixed_adj)

        # Semantic weights from source keypoints
        sem_w = retargeter._compute_semantic_weights(source_pts) if use_semantic_weights else None

        n_iter = 50 if t == 0 else 10
        q_current = q_prev.copy()
        last_cost = float("inf")
        for _ in range(n_iter):
            q_current, cost = retargeter.solve_single_iteration(
                q_current, q_prev, target_lap, fixed_adj, sem_w
            )
            if abs(last_cost - cost) < 1e-8:
                break
            last_cost = cost

        qpos[t] = q_current
        q_prev = q_current

        if (t + 1) % 100 == 0:
            fps = (t + 1) / (time.time() - t0)
            # Show semantic weight status
            if sem_w is not None:
                max_w = sem_w.max()
                active = np.sum(sem_w > 1.01)
                print(f"  {t+1}/{N} ({fps:.0f} fps) | max_w={max_w:.1f}, active={active} pts")
            else:
                print(f"  {t+1}/{N} ({fps:.0f} fps)")

    elapsed = time.time() - t0
    print(f"  {label}: {N} frames in {elapsed:.1f}s ({N/elapsed:.0f} fps)")
    return qpos


def main():
    parser = argparse.ArgumentParser(description="EXP-3: Semantic weight")
    parser.add_argument("--pkl", type=str, default=str(DEFAULT_PKL))
    parser.add_argument("--frames", type=int, default=None)
    args = parser.parse_args()

    import pickle
    with open(args.pkl, "rb") as f:
        recording = pickle.load(f)
    frames = [f for f in recording if not np.allclose(f[f"{HAND_SIDE}_fingers"], 0)]

    landmarks_seq, _ = load_pkl_sequence(args.pkl, HAND_SIDE)
    config = HandRetargetConfig.from_yaml(str(DEFAULT_CONFIG), mjcf_path=str(DEFAULT_URDF))

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

    # Show pinch statistics
    thumb_idx_dists = np.linalg.norm(proc_seq[:, 4, :] - proc_seq[:, 8, :], axis=1) * 1000
    print(f"Data: {N} frames from {args.pkl}")
    print(f"Thumb-Index dist: min={thumb_idx_dists.min():.1f}mm, mean={thumb_idx_dists.mean():.1f}mm")
    print(f"  <30mm: {np.sum(thumb_idx_dists<30)} frames, <20mm: {np.sum(thumb_idx_dists<20)} frames")
    print("=" * 60)

    print("\nCondition A: Baseline")
    qpos_bl = run_baseline(frames, N)

    print("\nCondition B: IM fixed + uniform")
    qpos_uni = run_im_fixed(proc_seq, config, N, use_semantic_weights=False, label="IM uniform")

    print("\nCondition C: IM fixed + semantic weight")
    qpos_sem = run_im_fixed(proc_seq, config, N, use_semantic_weights=True, label="IM semantic")

    # Benchmark
    bench = RetargetBenchmark(str(DEFAULT_URDF), HAND_SIDE)

    results_bl = bench.evaluate(proc_seq, qpos_bl)
    results_uni = bench.evaluate(proc_seq, qpos_uni)
    results_sem = bench.evaluate(proc_seq, qpos_sem)

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"{'Metric':<35} {'Baseline':>10} {'IM Uniform':>12} {'IM Semantic':>12}")
    print(f"{'-'*69}")
    print(f"{'Tip Pos Error (mm)':<35} {results_bl['tip_pos_error_mm']['mean']:>10.2f} {results_uni['tip_pos_error_mm']['mean']:>12.2f} {results_sem['tip_pos_error_mm']['mean']:>12.2f}")
    print(f"{'Tip Dir Error (deg)':<35} {results_bl['tip_dir_error_deg']['mean']:>10.2f} {results_uni['tip_dir_error_deg']['mean']:>12.2f} {results_sem['tip_dir_error_deg']['mean']:>12.2f}")
    print(f"{'Inter-Tip Dist Error (mm)':<35} {results_bl['inter_tip_distance_error_mm']['mean']:>10.2f} {results_uni['inter_tip_distance_error_mm']['mean']:>12.2f} {results_sem['inter_tip_distance_error_mm']['mean']:>12.2f}")
    print(f"{'Jerk (rad/s^3)':<35} {results_bl['smoothness']['jerk_rad_s3']:>10.1f} {results_uni['smoothness']['jerk_rad_s3']:>12.1f} {results_sem['smoothness']['jerk_rad_s3']:>12.1f}")

    # Pinch-specific
    print(f"\n{'='*70}")
    print(f"  PINCH ANALYSIS (Thumb-Index distance preservation)")
    print(f"{'='*70}")
    for label, r in [("Baseline", results_bl), ("IM Uniform", results_uni), ("IM Semantic", results_sem)]:
        p = r["pinch"]
        print(f"\n  {label}:")
        print(f"    Pinch frames (<30mm): {p['n_pinch_frames']}")
        print(f"    Thumb-Index dist error: all={p['thumb_index_dist_error_mm']['all']:.2f}mm")
        if p['n_pinch_frames'] > 0:
            print(f"      pinch frames: {p['thumb_index_dist_error_mm']['pinch']:.2f}mm")
            print(f"      normal frames: {p['thumb_index_dist_error_mm']['normal']:.2f}mm")
        print(f"    All thumb pairs: {p['per_pair']}")


if __name__ == "__main__":
    main()
