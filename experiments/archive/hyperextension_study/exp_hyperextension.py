"""
EXP: Hyperextension (反弓) comparison with ARAP rotation compensation.

Compares three conditions:
  A. Baseline (wuji_retargeting IK)
  B. Interaction Mesh -- original Laplacian
  C. Interaction Mesh -- with ARAP per-vertex rotation compensation

Metric: percentage of frames where PIP/DIP joints have negative angles.

Usage:
    cd /home/l/ws/RL/retargeting
    PYTHONPATH=src python experiments/exp_hyperextension.py
    PYTHONPATH=src python experiments/exp_hyperextension.py --frames 1000
    PYTHONPATH=src python experiments/exp_hyperextension.py --no-cache
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "experiments"))
# editable install fallback: wuji_retargeting source
sys.path.insert(0, "/home/l/ws/doc/WujiRepo/wuji_retargeting_private/public")

from hand_retarget import InteractionMeshHandRetargeter, HandRetargetConfig
from hand_retarget.mediapipe_io import load_pkl_sequence, preprocess_sequence

DEFAULT_PKL = PROJECT_DIR / "data" / "manus_for_pinch" / "manus1_5k.pkl"
DEFAULT_CONFIG = PROJECT_DIR / "config" / "interaction_mesh_left.yaml"
DEFAULT_URDF = Path(
    "/home/l/ws/doc/WujiRepo/wuji_retargeting_private/public/"
    "wuji_retargeting/wuji_hand_description/urdf/left.urdf"
)
BASELINE_CONFIG = (
    PROJECT_DIR / "demos" / "hand" / "baseline" / "wuji_manus_demo"
    / "config" / "retarget_manus_left.yaml"
)
HAND_SIDE = "left"
CACHE_DIR = PROJECT_DIR / "data" / "cache"

# Joint indices: 5 fingers x 4 joints (MCP-flex, MCP-abd, PIP, DIP)
PIP_INDICES = [2, 6, 10, 14, 18]
DIP_INDICES = [3, 7, 11, 15, 19]
FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]


def analyze_hyperextension(qpos_seq: np.ndarray) -> dict:
    """Analyze hyperextension in retargeted joint angles.

    Args:
        qpos_seq: (T, 20) joint angle sequence

    Returns:
        dict with overall and per-finger hyperextension stats
    """
    T = len(qpos_seq)
    any_hyper = np.zeros(T, dtype=bool)
    per_finger = {}

    for f, name in enumerate(FINGER_NAMES):
        pip_idx = PIP_INDICES[f]
        dip_idx = DIP_INDICES[f]
        pip_neg = qpos_seq[:, pip_idx] < 0
        dip_neg = qpos_seq[:, dip_idx] < 0
        finger_hyper = pip_neg | dip_neg
        any_hyper |= finger_hyper

        per_finger[name] = {
            "pip_neg_pct": pip_neg.mean() * 100,
            "dip_neg_pct": dip_neg.mean() * 100,
            "any_pct": finger_hyper.mean() * 100,
            "pip_min_deg": np.degrees(qpos_seq[:, pip_idx].min()),
            "dip_min_deg": np.degrees(qpos_seq[:, dip_idx].min()),
        }

    return {
        "overall_pct": any_hyper.mean() * 100,
        "per_finger": per_finger,
        "total_frames": T,
    }


def run_baseline(frames, N):
    """Run baseline IK retargeting."""
    from wuji_retargeting import Retargeter

    retargeter = Retargeter.from_yaml(str(BASELINE_CONFIG), HAND_SIDE)
    qpos = np.zeros((N, 20))
    t0 = time.time()
    for t in range(N):
        qpos[t] = retargeter.retarget(frames[t][f"{HAND_SIDE}_fingers"])
    elapsed = time.time() - t0
    print(f"  Baseline IK: {N} frames in {elapsed:.1f}s ({N / elapsed:.0f} fps)")
    return qpos


def run_im(proc_seq, config, N, label="IM"):
    """Run interaction mesh retargeting."""
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


def print_comparison(results: dict[str, dict]):
    """Print formatted comparison table."""
    labels = list(results.keys())
    w = max(12, max(len(l) for l in labels) + 2)

    header = f"{'':>20s}" + "".join(f"{l:>{w}s}" for l in labels)
    sep = "-" * (20 + w * len(labels))

    print()
    print("=" * len(sep))
    print("  HYPEREXTENSION COMPARISON")
    print("=" * len(sep))
    print(header)
    print(sep)

    # Overall
    row = f"{'Overall %':>20s}"
    for l in labels:
        row += f"{results[l]['overall_pct']:>{w}.1f}"
    print(row)
    print()

    # Per finger
    print("Per-finger (any PIP/DIP < 0):")
    for name in FINGER_NAMES:
        row = f"  {name:>18s}"
        for l in labels:
            row += f"{results[l]['per_finger'][name]['any_pct']:>{w}.1f}"
        print(row)
    print()

    # Per joint
    print("Per-joint (% frames < 0):")
    for name in FINGER_NAMES:
        row_pip = f"  {name + ' PIP':>18s}"
        row_dip = f"  {name + ' DIP':>18s}"
        for l in labels:
            pf = results[l]["per_finger"][name]
            row_pip += f"{pf['pip_neg_pct']:>{w}.1f}"
            row_dip += f"{pf['dip_neg_pct']:>{w}.1f}"
        print(row_pip)
        print(row_dip)
    print()

    # Worst angles
    print("Worst hyperextension (deg):")
    for name in FINGER_NAMES:
        row_pip = f"  {name + ' PIP':>18s}"
        row_dip = f"  {name + ' DIP':>18s}"
        for l in labels:
            pf = results[l]["per_finger"][name]
            row_pip += f"{pf['pip_min_deg']:>{w}.1f}"
            row_dip += f"{pf['dip_min_deg']:>{w}.1f}"
        print(row_pip)
        print(row_dip)

    print("=" * len(sep))


def main():
    parser = argparse.ArgumentParser(description="Hyperextension comparison experiment")
    parser.add_argument("--pkl", type=str, default=str(DEFAULT_PKL))
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--urdf", type=str, default=str(DEFAULT_URDF))
    parser.add_argument("--frames", type=int, default=None, help="Limit frame count")
    parser.add_argument("--no-cache", action="store_true", help="Force recompute all")
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pkl_stem = Path(args.pkl).stem

    # Load raw data (baseline needs raw frames)
    import pickle

    with open(args.pkl, "rb") as f:
        raw_frames = pickle.load(f)
    # Filter non-zero frames
    raw_frames = [fr for fr in raw_frames if np.any(fr.get(f"{HAND_SIDE}_fingers", np.zeros(1)))]

    # Load and preprocess for IM
    landmarks_seq, timestamps = load_pkl_sequence(args.pkl, HAND_SIDE)

    N = min(len(landmarks_seq), len(raw_frames))
    if args.frames:
        N = min(N, args.frames)
    raw_frames = raw_frames[:N]
    landmarks_seq = landmarks_seq[:N]

    config = HandRetargetConfig.from_yaml(args.config, mjcf_path=args.urdf)
    retargeter_tmp = InteractionMeshHandRetargeter(config)
    proc_seq = preprocess_sequence(
        landmarks_seq, config.mediapipe_rotation,
        hand_side=HAND_SIDE, global_scale=retargeter_tmp.global_scale,
    )
    del retargeter_tmp

    print(f"Data: {args.pkl} ({N} frames)")
    print(f"Config: {args.config}")
    print()

    # --- Condition A: Baseline IK ---
    cache_bl = CACHE_DIR / f"{pkl_stem}_bl_cache.npz"
    if cache_bl.exists() and not args.no_cache:
        qpos_bl = np.load(cache_bl)["qpos"][:N]
        print(f"  Baseline IK: loaded from cache ({len(qpos_bl)} frames)")
    else:
        qpos_bl = run_baseline(raw_frames, N)
        np.savez(cache_bl, qpos=qpos_bl)

    # --- Condition B: IM original ---
    cache_im = CACHE_DIR / f"{pkl_stem}_im_cache.npz"
    if cache_im.exists() and not args.no_cache:
        qpos_im = np.load(cache_im)["qpos"][:N]
        print(f"  IM original: loaded from cache ({len(qpos_im)} frames)")
    else:
        config_im = HandRetargetConfig.from_yaml(args.config, mjcf_path=args.urdf)
        qpos_im = run_im(proc_seq, config_im, N, label="IM original")
        np.savez(cache_im, qpos=qpos_im)

    # --- Condition C: IM + ARAP rotation compensation ---
    cache_arap = CACHE_DIR / f"{pkl_stem}_im_rotcomp_cache.npz"
    if cache_arap.exists() and not args.no_cache:
        qpos_arap = np.load(cache_arap)["qpos"][:N]
        print(f"  IM+ARAP: loaded from cache ({len(qpos_arap)} frames)")
    else:
        config_arap = HandRetargetConfig.from_yaml(args.config, mjcf_path=args.urdf)
        config_arap.rotation_compensation = True
        qpos_arap = run_im(proc_seq, config_arap, N, label="IM+ARAP")
        np.savez(cache_arap, qpos=qpos_arap)

    # --- Analysis ---
    results = {
        "Baseline": analyze_hyperextension(qpos_bl),
        "IM Orig": analyze_hyperextension(qpos_im),
        "IM+ARAP": analyze_hyperextension(qpos_arap),
    }

    print_comparison(results)


if __name__ == "__main__":
    main()
