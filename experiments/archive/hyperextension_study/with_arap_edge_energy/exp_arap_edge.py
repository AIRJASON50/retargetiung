"""
EXP: ARAP per-edge energy vs Laplacian vs ARAP rotation compensation.

Three-way comparison:
  A. IM Laplacian (current default)
  B. IM + ARAP rotation compensation (rotates Laplacian target)
  C. IM + ARAP edge energy (replaces Laplacian with per-edge cost)

Metrics: hyperextension, tip position/direction error, smoothness, speed.

Usage:
    cd /home/l/ws/RL/retargeting
    python experiments/exp_arap_edge.py --frames 500
    python experiments/exp_arap_edge.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "experiments"))
sys.path.insert(0, "/home/l/ws/doc/WujiRepo/wuji_retargeting_private/public")

from hand_retarget import InteractionMeshHandRetargeter, HandRetargetConfig
from hand_retarget.mediapipe_io import load_pkl_sequence, preprocess_sequence

DEFAULT_PKL = PROJECT_DIR / "data" / "manus_for_pinch" / "manus1_5k.pkl"
DEFAULT_CONFIG = PROJECT_DIR / "config" / "interaction_mesh_left.yaml"
DEFAULT_URDF = Path(
    "/home/l/ws/doc/WujiRepo/wuji_retargeting_private/public/"
    "wuji_retargeting/wuji_hand_description/urdf/left.urdf"
)
HAND_SIDE = "left"

PIP_INDICES = [2, 6, 10, 14, 18]
DIP_INDICES = [3, 7, 11, 15, 19]
FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]


def run_im(proc_seq, config, N, label="IM"):
    retargeter = InteractionMeshHandRetargeter(config)
    qpos = np.zeros((N, retargeter.nq))
    q_prev = retargeter.hand.get_default_qpos()
    t0 = time.time()
    for t in range(N):
        q = retargeter.retarget_frame(proc_seq[t], q_prev, is_first_frame=(t == 0))
        qpos[t] = q
        q_prev = q
    elapsed = time.time() - t0
    print(f"  {label}: {N} frames in {elapsed:.1f}s ({N / elapsed:.0f} fps)")
    return qpos


def analyze(qpos):
    T = len(qpos)
    results = {}

    # Hyperextension
    any_hyper = np.zeros(T, dtype=bool)
    per_finger = {}
    for f, name in enumerate(FINGER_NAMES):
        pip_neg = qpos[:, PIP_INDICES[f]] < 0
        dip_neg = qpos[:, DIP_INDICES[f]] < 0
        finger_hyper = pip_neg | dip_neg
        any_hyper |= finger_hyper
        per_finger[name] = {
            "pip_pct": pip_neg.mean() * 100,
            "dip_pct": dip_neg.mean() * 100,
            "any_pct": finger_hyper.mean() * 100,
        }
    results["hyper_pct"] = any_hyper.mean() * 100
    results["per_finger"] = per_finger

    # Smoothness (jerk)
    if T > 3:
        jerk = np.diff(qpos, n=3, axis=0)
        results["jerk"] = np.mean(np.linalg.norm(jerk, axis=1))
    else:
        results["jerk"] = 0.0

    # C-space coverage
    q_range = qpos.max(axis=0) - qpos.min(axis=0)
    results["cspace_mean"] = q_range.mean()

    return results


def print_comparison(all_results):
    labels = list(all_results.keys())
    w = 14

    print()
    print("=" * 65)
    print("  ARAP EDGE ENERGY COMPARISON")
    print("=" * 65)

    header = f"{'':>22s}" + "".join(f"{l:>{w}s}" for l in labels)
    print(header)
    print("-" * (22 + w * len(labels)))

    # Overall hyperextension
    row = f"{'Hyperext. %':>22s}"
    for l in labels:
        row += f"{all_results[l]['hyper_pct']:>{w}.1f}"
    print(row)

    # Per-finger
    print()
    for name in FINGER_NAMES:
        row = f"{'  ' + name + ' PIP<0 %':>22s}"
        for l in labels:
            row += f"{all_results[l]['per_finger'][name]['pip_pct']:>{w}.1f}"
        print(row)
        row = f"{'  ' + name + ' DIP<0 %':>22s}"
        for l in labels:
            row += f"{all_results[l]['per_finger'][name]['dip_pct']:>{w}.1f}"
        print(row)

    # Smoothness
    print()
    row = f"{'Jerk':>22s}"
    for l in labels:
        row += f"{all_results[l]['jerk']:>{w}.1f}"
    print(row)

    row = f"{'C-space range':>22s}"
    for l in labels:
        row += f"{all_results[l]['cspace_mean']:>{w}.3f}"
    print(row)

    print("=" * 65)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", type=str, default=str(DEFAULT_PKL))
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--urdf", type=str, default=str(DEFAULT_URDF))
    parser.add_argument("--frames", type=int, default=None)
    args = parser.parse_args()

    landmarks_seq, _ = load_pkl_sequence(args.pkl, HAND_SIDE)
    N = len(landmarks_seq)
    if args.frames:
        N = min(N, args.frames)
    landmarks_seq = landmarks_seq[:N]

    config_base = HandRetargetConfig.from_yaml(args.config, mjcf_path=args.urdf)
    proc_seq = preprocess_sequence(
        landmarks_seq, config_base.mediapipe_rotation,
        hand_side=HAND_SIDE,
        global_scale=InteractionMeshHandRetargeter(config_base).global_scale,
    )

    print(f"Data: {args.pkl} ({N} frames)")
    print()

    # A: Laplacian (default)
    config_a = HandRetargetConfig.from_yaml(args.config, mjcf_path=args.urdf)
    qpos_a = run_im(proc_seq, config_a, N, label="Laplacian")

    # B: ARAP rotation compensation
    config_b = HandRetargetConfig.from_yaml(args.config, mjcf_path=args.urdf)
    config_b.rotation_compensation = True
    qpos_b = run_im(proc_seq, config_b, N, label="ARAP rot-comp")

    # C: ARAP edge energy (Delaunay topology)
    config_c = HandRetargetConfig.from_yaml(args.config, mjcf_path=args.urdf)
    config_c.use_arap_edge = True
    qpos_c = run_im(proc_seq, config_c, N, label="ARAP edge (Delaunay)")

    # D: ARAP edge energy + skeleton topology
    config_d = HandRetargetConfig.from_yaml(args.config, mjcf_path=args.urdf)
    config_d.use_arap_edge = True
    config_d.use_skeleton_topology = True
    qpos_d = run_im(proc_seq, config_d, N, label="ARAP edge (skeleton)")

    # Analyze
    results = {
        "Laplacian": analyze(qpos_a),
        "ARAP rot": analyze(qpos_b),
        "Edge+Dela": analyze(qpos_c),
        "Edge+Skel": analyze(qpos_d),
    }
    print_comparison(results)


if __name__ == "__main__":
    main()
