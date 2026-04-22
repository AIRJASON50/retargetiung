"""
EXP: Palm spread correction in bone scaling.

Compares three conditions:
  A. IM without bone scaling (reference)
  B. IM + radial-only bone scaling (per-finger chain wrist→tip)
  C. IM + radial + palm spread bone scaling (adds cross-finger MCP correction)

Hypothesis: palm spread correction reduces topology mismatch between human and robot,
leading to better finger tracking and fewer unnatural postures.

Metric: hyperextension rate (PIP/DIP < 0), plus FPS.

Usage:
    cd /home/l/ws/RL/retargeting
    conda activate mjplgd
    PYTHONPATH=src python experiments/exp_palm_spread.py
    PYTHONPATH=src python experiments/exp_palm_spread.py --frames 500
    PYTHONPATH=src python experiments/exp_palm_spread.py --no-cache
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, "/home/l/ws/doc/WujiRepo/wuji_retargeting_private/public")

from hand_retarget import InteractionMeshHandRetargeter, HandRetargetConfig
from hand_retarget.mediapipe_io import load_pkl_sequence, preprocess_sequence

DEFAULT_PKL = PROJECT_DIR / "data" / "manus_for_pinch" / "manus1_5k.pkl"
DEFAULT_CONFIG = PROJECT_DIR / "config" / "manus.yaml"
DEFAULT_URDF = Path(
    "/home/l/ws/doc/WujiRepo/wuji_retargeting_private/public/"
    "wuji_retargeting/wuji_hand_description/urdf/left.urdf"
)
HAND_SIDE = "left"
CACHE_DIR = PROJECT_DIR / "data" / "cache"

PIP_INDICES = [2, 6, 10, 14, 18]
DIP_INDICES = [3, 7, 11, 15, 19]
FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]


def analyze_hyperextension(qpos_seq: np.ndarray) -> dict:
    """Analyze per-finger hyperextension rate.

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


def run_im(proc_seq: np.ndarray, config: HandRetargetConfig, N: int, label: str = "IM") -> tuple[np.ndarray, float]:
    """Run interaction mesh retargeting.

    Returns:
        (qpos_seq, fps)
    """
    retargeter = InteractionMeshHandRetargeter(config)
    qpos = np.zeros((N, retargeter.nq))
    q_prev = retargeter.hand.get_default_qpos()

    t0 = time.time()
    for t in range(N):
        q = retargeter.retarget_frame(proc_seq[t], q_prev, is_first_frame=(t == 0))
        qpos[t] = q
        q_prev = q
    elapsed = time.time() - t0
    fps = N / elapsed
    print(f"  {label}: {N} frames in {elapsed:.1f}s ({fps:.0f} fps)")
    return qpos, fps


def print_comparison(results: dict[str, dict], fps: dict[str, float]):
    """Print formatted comparison table."""
    labels = list(results.keys())
    w = max(14, max(len(l) for l in labels) + 2)

    header = f"{'':>22s}" + "".join(f"{l:>{w}s}" for l in labels)
    sep = "-" * (22 + w * len(labels))

    print()
    print("=" * len(sep))
    print("  PALM SPREAD SCALING -- HYPEREXTENSION COMPARISON")
    print("=" * len(sep))
    print(header)
    print(sep)

    # FPS
    row = f"{'FPS':>22s}"
    for l in labels:
        row += f"{fps[l]:>{w}.0f}"
    print(row)
    print()

    # Overall hyperextension
    row = f"{'Overall hyper %':>22s}"
    for l in labels:
        row += f"{results[l]['overall_pct']:>{w}.1f}"
    print(row)
    print()

    # Per-finger any PIP/DIP < 0
    print("Per-finger (any PIP/DIP < 0):")
    for name in FINGER_NAMES:
        row = f"  {name:>20s}"
        for l in labels:
            row += f"{results[l]['per_finger'][name]['any_pct']:>{w}.1f}"
        print(row)
    print()

    # PIP and DIP separately
    print("Per-joint (% frames < 0):")
    for name in FINGER_NAMES:
        row_pip = f"  {name + ' PIP':>20s}"
        row_dip = f"  {name + ' DIP':>20s}"
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
        row_pip = f"  {name + ' PIP':>20s}"
        row_dip = f"  {name + ' DIP':>20s}"
        for l in labels:
            pf = results[l]["per_finger"][name]
            row_pip += f"{pf['pip_min_deg']:>{w}.1f}"
            row_dip += f"{pf['dip_min_deg']:>{w}.1f}"
        print(row_pip)
        print(row_dip)

    print("=" * len(sep))

    # Summary deltas vs condition A
    baseline_label = labels[0]
    baseline_pct = results[baseline_label]["overall_pct"]
    print()
    print("  Deltas vs A (overall hyperextension %):")
    for l in labels[1:]:
        delta = results[l]["overall_pct"] - baseline_pct
        sign = "+" if delta >= 0 else ""
        print(f"    {l}: {sign}{delta:.1f} pp")


def main():
    parser = argparse.ArgumentParser(description="Palm spread bone scaling experiment")
    parser.add_argument("--pkl", type=str, default=str(DEFAULT_PKL))
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--urdf", type=str, default=str(DEFAULT_URDF))
    parser.add_argument("--frames", type=int, default=None, help="Limit frame count")
    parser.add_argument("--no-cache", action="store_true", help="Force recompute all")
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pkl_stem = Path(args.pkl).stem

    # Load and preprocess
    landmarks_seq, _ = load_pkl_sequence(args.pkl, HAND_SIDE)

    # Use a temporary config to get the global scale for preprocessing
    cfg_tmp = HandRetargetConfig.from_yaml(args.config, mjcf_path=args.urdf)
    retargeter_tmp = InteractionMeshHandRetargeter(cfg_tmp)
    global_scale = retargeter_tmp.global_scale
    del retargeter_tmp

    N = len(landmarks_seq)
    if args.frames:
        N = min(N, args.frames)
    landmarks_seq = landmarks_seq[:N]

    proc_seq = preprocess_sequence(
        landmarks_seq, cfg_tmp.mediapipe_rotation,
        hand_side=HAND_SIDE, global_scale=global_scale,
    )

    print(f"Data: {args.pkl} ({N} frames)")
    print(f"Config: {args.config}")
    print()

    fps_results: dict[str, float] = {}

    # --- Condition A: IM, no bone scaling ---
    cache_a = CACHE_DIR / f"{pkl_stem}_im_cache.npz"
    if cache_a.exists() and not args.no_cache:
        qpos_a = np.load(cache_a)["qpos"][:N]
        fps_results["A: No scaling"] = float(np.load(cache_a).get("fps", np.array(0.0)))
        print(f"  A (no scaling): loaded from cache ({len(qpos_a)} frames)")
    else:
        cfg_a = HandRetargetConfig.from_yaml(args.config, mjcf_path=args.urdf)
        cfg_a.use_bone_scaling = False
        qpos_a, fps_a = run_im(proc_seq, cfg_a, N, label="A (no scaling)")
        fps_results["A: No scaling"] = fps_a
        np.savez(cache_a, qpos=qpos_a, fps=np.array(fps_a))

    # --- Condition B: IM + radial-only bone scaling ---
    cache_b = CACHE_DIR / f"{pkl_stem}_im_bonescale_radial_cache.npz"
    if cache_b.exists() and not args.no_cache:
        qpos_b = np.load(cache_b)["qpos"][:N]
        fps_results["B: Radial only"] = float(np.load(cache_b).get("fps", np.array(0.0)))
        print(f"  B (radial only): loaded from cache ({len(qpos_b)} frames)")
    else:
        cfg_b = HandRetargetConfig.from_yaml(args.config, mjcf_path=args.urdf)
        cfg_b.use_bone_scaling = True
        cfg_b.use_palm_spread_scaling = False
        qpos_b, fps_b = run_im(proc_seq, cfg_b, N, label="B (radial only)")
        fps_results["B: Radial only"] = fps_b
        np.savez(cache_b, qpos=qpos_b, fps=np.array(fps_b))

    # --- Condition C: IM + radial + palm spread bone scaling ---
    cache_c = CACHE_DIR / f"{pkl_stem}_im_bonescale_full_cache.npz"
    if cache_c.exists() and not args.no_cache:
        qpos_c = np.load(cache_c)["qpos"][:N]
        fps_results["C: Radial+Palm"] = float(np.load(cache_c).get("fps", np.array(0.0)))
        print(f"  C (radial+palm): loaded from cache ({len(qpos_c)} frames)")
    else:
        cfg_c = HandRetargetConfig.from_yaml(args.config, mjcf_path=args.urdf)
        cfg_c.use_bone_scaling = True
        cfg_c.use_palm_spread_scaling = True
        qpos_c, fps_c = run_im(proc_seq, cfg_c, N, label="C (radial+palm)")
        fps_results["C: Radial+Palm"] = fps_c
        np.savez(cache_c, qpos=qpos_c, fps=np.array(fps_c))

    # --- Analysis ---
    hyper_results = {
        "A: No scaling": analyze_hyperextension(qpos_a),
        "B: Radial only": analyze_hyperextension(qpos_b),
        "C: Radial+Palm": analyze_hyperextension(qpos_c),
    }

    print_comparison(hyper_results, fps_results)


if __name__ == "__main__":
    main()
