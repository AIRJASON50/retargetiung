"""Baseline capture & regression test for refactoring.

Captures retarget qpos from manus1_5k.pkl (first 100 frames) and saves as
baseline .npz.  On subsequent runs, compares against saved baseline.

Usage:
    cd /home/l/ws/RL/retargeting
    PYTHONPATH=src python tests/test_refactor_baseline.py --capture   # save baseline
    PYTHONPATH=src python tests/test_refactor_baseline.py --verify    # compare against baseline
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
import os; _WUJI_SDK = os.environ.get("WUJI_SDK_PATH", "/home/l/ws/doc/WujiRepo/wuji_retargeting_private/public"); sys.path.insert(0, _WUJI_SDK)

from hand_retarget import HandRetargetConfig, InteractionMeshHandRetargeter
from hand_retarget.mediapipe_io import load_pkl_sequence, preprocess_sequence

URDF = Path(
    _WUJI_SDK
    "wuji_retargeting/wuji_hand_description/urdf/left.urdf"
)
PKL = PROJECT_DIR / "data" / "manus_for_pinch" / "manus1_5k.pkl"
CONFIG_YAML = PROJECT_DIR / "config" / "interaction_mesh_left.yaml"
BASELINE_NPZ = PROJECT_DIR / "tests" / "refactor_baseline.npz"
N_FRAMES = 100
HAND_SIDE = "left"


def run_retarget(n_frames: int = N_FRAMES) -> np.ndarray:
    """Run retarget on first n_frames and return qpos_seq."""
    cfg = HandRetargetConfig.from_yaml(str(CONFIG_YAML), mjcf_path=str(URDF))
    retargeter = InteractionMeshHandRetargeter(cfg)

    landmarks_seq, _ = load_pkl_sequence(str(PKL), HAND_SIDE)
    landmarks_seq = preprocess_sequence(
        landmarks_seq[:n_frames],
        cfg.mediapipe_rotation,
        hand_side=HAND_SIDE,
        global_scale=retargeter.global_scale,
    )

    qpos_seq = np.zeros((n_frames, retargeter.nq))
    q_prev = retargeter.hand.get_default_qpos()

    for t in range(n_frames):
        q_opt = retargeter.retarget_frame(
            landmarks_seq[t], q_prev, is_first_frame=(t == 0)
        )
        qpos_seq[t] = q_opt
        q_prev = q_opt
        if (t + 1) % 20 == 0:
            print(f"  frame {t + 1}/{n_frames}")

    return qpos_seq


def capture():
    """Capture baseline results."""
    print(f"Capturing baseline ({N_FRAMES} frames)...")
    qpos = run_retarget()
    np.savez(str(BASELINE_NPZ), qpos=qpos)
    print(f"Saved to {BASELINE_NPZ}")
    print(f"  qpos shape: {qpos.shape}")
    print(f"  qpos range: [{qpos.min():.4f}, {qpos.max():.4f}]")
    print(f"  qpos mean:  {qpos.mean():.6f}")


def verify():
    """Verify current code matches baseline."""
    if not BASELINE_NPZ.exists():
        print(f"ERROR: baseline not found at {BASELINE_NPZ}")
        print("Run with --capture first.")
        sys.exit(1)

    baseline = np.load(str(BASELINE_NPZ))["qpos"]
    print(f"Running retarget ({N_FRAMES} frames)...")
    current = run_retarget()

    max_diff = np.abs(current - baseline).max()
    mean_diff = np.abs(current - baseline).mean()
    print(f"\nComparison vs baseline:")
    print(f"  max  abs diff: {max_diff:.2e}")
    print(f"  mean abs diff: {mean_diff:.2e}")

    # Tolerance: SOCP solver is deterministic with same input,
    # but float rounding can cause ~1e-10 drift
    tol = 1e-6
    if max_diff < tol:
        print(f"  PASS (max diff < {tol:.0e})")
    else:
        print(f"  FAIL (max diff >= {tol:.0e})")
        # Show which frames/joints differ most
        frame_idx, joint_idx = np.unravel_index(
            np.abs(current - baseline).argmax(), current.shape
        )
        print(f"  Worst: frame={frame_idx}, joint={joint_idx}")
        print(f"    baseline: {baseline[frame_idx, joint_idx]:.8f}")
        print(f"    current:  {current[frame_idx, joint_idx]:.8f}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--capture", action="store_true", help="Capture baseline")
    group.add_argument("--verify", action="store_true", help="Verify against baseline")
    args = parser.parse_args()

    if args.capture:
        capture()
    else:
        verify()
