"""
EXP-2: Distance-based Laplacian weights vs uniform weights.

Compares three conditions:
  A. Baseline (wuji_retargeting IK)
  B. IM fixed topology + uniform weight (EXP-1 winner)
  C. IM fixed topology + distance weight (Ho 2010 original)

Uses benchmark metrics on manus1_5k.pkl.

Usage:
    cd /home/l/ws/RL/retargeting
    PYTHONPATH=src python experiments/exp2_distance_weight.py
    PYTHONPATH=src python experiments/exp2_distance_weight.py --frames 500
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
from hand_retarget.mesh_utils import (
    create_interaction_mesh, get_adjacency_list,
    calculate_laplacian_coordinates, calculate_laplacian_matrix,
)
from benchmark import RetargetBenchmark

DEFAULT_PKL = PROJECT_DIR / "data" / "manus1_5k.pkl"
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


def run_im_fixed_topology(proc_seq, config, N, uniform_weight=True, label=""):
    """Run IM with fixed topology and configurable weight scheme."""
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

        # Compute target Laplacian from source with fixed topology
        # Use the SAME weight scheme for both target Laplacian and robot Laplacian
        target_lap = calculate_laplacian_coordinates(
            source_pts, fixed_adj, uniform_weight=uniform_weight
        )

        # SQP iterations — need to override the weight scheme in solve_single_iteration
        n_iter = 50 if t == 0 else 10
        q_current = q_prev.copy()
        last_cost = float("inf")

        for _ in range(n_iter):
            # FK
            retargeter.hand.forward(q_current)
            robot_pts = retargeter._get_robot_keypoints()
            J_V = retargeter._get_robot_jacobians()

            # Recompute L matrix from robot positions with chosen weight scheme
            from scipy import sparse as sp
            import cvxpy as cp

            L = calculate_laplacian_matrix(
                robot_pts, fixed_adj, uniform_weight=uniform_weight
            )
            L_sp = sp.csr_matrix(L)
            Kron = sp.kron(L_sp, sp.eye(3, format="csr"), format="csr")
            J_L = Kron @ J_V

            lap0 = L_sp @ robot_pts
            lap0_vec = lap0.reshape(-1)
            target_lap_vec = target_lap.reshape(-1)

            V = retargeter.n_keypoints
            w_v = retargeter.laplacian_weight * np.ones(V)
            sqrt_w3 = np.sqrt(np.repeat(w_v, 3))

            dq = cp.Variable(retargeter.nq, name="dq")
            lap_var = cp.Variable(3 * V, name="lap")

            constraints = [
                cp.Constant(J_L) @ dq - lap_var == -lap0_vec,
                dq >= retargeter.q_lb - q_current,
                dq <= retargeter.q_ub - q_current,
                cp.SOC(config.step_size, dq),
            ]

            obj_terms = [
                cp.sum_squares(cp.multiply(sqrt_w3, lap_var - target_lap_vec)),
                config.smooth_weight * cp.sum_squares(dq - (q_prev - q_current)),
            ]

            problem = cp.Problem(cp.Minimize(cp.sum(obj_terms)), constraints)
            problem.solve(solver=cp.CLARABEL, verbose=False)

            if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                break

            q_current = q_current + dq.value
            q_current = np.clip(q_current, retargeter.q_lb, retargeter.q_ub)

            if abs(last_cost - problem.value) < 1e-8:
                break
            last_cost = problem.value

        qpos[t] = q_current
        q_prev = q_current

    elapsed = time.time() - t0
    print(f"  {label}: {N} frames in {elapsed:.1f}s ({N/elapsed:.0f} fps)")
    return qpos


def main():
    parser = argparse.ArgumentParser(description="EXP-2: Distance weight vs uniform")
    parser.add_argument("--pkl", type=str, default=str(DEFAULT_PKL))
    parser.add_argument("--frames", type=int, default=500)
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

    N = min(len(frames), len(proc_seq), args.frames)
    frames = frames[:N]
    proc_seq = proc_seq[:N]

    print(f"Running EXP-2 on {N} frames")
    print("=" * 60)

    print("\nCondition A: Baseline (NLopt IK)")
    qpos_bl = run_baseline(frames, N)

    print("\nCondition B: IM fixed topology + uniform weight")
    qpos_uniform = run_im_fixed_topology(proc_seq, config, N,
                                          uniform_weight=True,
                                          label="IM uniform")

    print("\nCondition C: IM fixed topology + distance weight (Ho 2010)")
    qpos_distance = run_im_fixed_topology(proc_seq, config, N,
                                           uniform_weight=False,
                                           label="IM distance")

    # Benchmark
    bench = RetargetBenchmark(str(DEFAULT_URDF), HAND_SIDE)

    results_bl = bench.evaluate(proc_seq, qpos_bl)
    results_uni = bench.evaluate(proc_seq, qpos_uniform)
    results_dist = bench.evaluate(proc_seq, qpos_distance)

    print(f"\n{'='*60}")
    print(f"  SUMMARY COMPARISON")
    print(f"{'='*60}")
    print(f"{'Metric':<35} {'Baseline':>10} {'IM Uniform':>12} {'IM Distance':>12}")
    print(f"{'-'*69}")
    print(f"{'Tip Pos Error (mm)':<35} {results_bl['tip_pos_error_mm']['mean']:>10.2f} {results_uni['tip_pos_error_mm']['mean']:>12.2f} {results_dist['tip_pos_error_mm']['mean']:>12.2f}")
    print(f"{'Tip Dir Error (deg)':<35} {results_bl['tip_dir_error_deg']['mean']:>10.2f} {results_uni['tip_dir_error_deg']['mean']:>12.2f} {results_dist['tip_dir_error_deg']['mean']:>12.2f}")
    print(f"{'Inter-Tip Dist Error (mm)':<35} {results_bl['inter_tip_distance_error_mm']['mean']:>10.2f} {results_uni['inter_tip_distance_error_mm']['mean']:>12.2f} {results_dist['inter_tip_distance_error_mm']['mean']:>12.2f}")
    print(f"{'Jerk (rad/s^3)':<35} {results_bl['smoothness']['jerk_rad_s3']:>10.1f} {results_uni['smoothness']['jerk_rad_s3']:>12.1f} {results_dist['smoothness']['jerk_rad_s3']:>12.1f}")
    print(f"{'Temporal Consistency':<35} {results_bl['smoothness']['temporal_consistency']:>10.5f} {results_uni['smoothness']['temporal_consistency']:>12.5f} {results_dist['smoothness']['temporal_consistency']:>12.5f}")
    print(f"{'C-Space Coverage':<35} {results_bl['cspace_coverage']['mean']:>10.3f} {results_uni['cspace_coverage']['mean']:>12.3f} {results_dist['cspace_coverage']['mean']:>12.3f}")

    # Per-finger breakdown
    print(f"\n{'='*60}")
    print(f"  PER-FINGER TIP POSITION ERROR (mm)")
    print(f"{'='*60}")
    print(f"{'Finger':<12} {'Baseline':>10} {'IM Uniform':>12} {'IM Distance':>12}")
    print(f"{'-'*46}")
    for name in ["Thumb", "Index", "Middle", "Ring", "Pinky"]:
        bl = results_bl['tip_pos_error_mm']['per_finger'][name]
        uni = results_uni['tip_pos_error_mm']['per_finger'][name]
        dist = results_dist['tip_pos_error_mm']['per_finger'][name]
        print(f"{name:<12} {bl:>10.2f} {uni:>12.2f} {dist:>12.2f}")

    print(f"\n{'='*60}")
    print(f"  PER-FINGER TIP DIRECTION ERROR (deg)")
    print(f"{'='*60}")
    print(f"{'Finger':<12} {'Baseline':>10} {'IM Uniform':>12} {'IM Distance':>12}")
    print(f"{'-'*46}")
    for name in ["Thumb", "Index", "Middle", "Ring", "Pinky"]:
        bl = results_bl['tip_dir_error_deg']['per_finger'][name]
        uni = results_uni['tip_dir_error_deg']['per_finger'][name]
        dist = results_dist['tip_dir_error_deg']['per_finger'][name]
        print(f"{name:<12} {bl:>10.2f} {uni:>12.2f} {dist:>12.2f}")


if __name__ == "__main__":
    main()
