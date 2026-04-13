"""
Compare SVD+MANO baseline vs simplified (no rotation) HO-Cap pipeline.

Usage:
    python experiments/compare_pipelines.py --mode baseline    # save current results
    python experiments/compare_pipelines.py --mode simplified  # save simplified results
    python experiments/compare_pipelines.py --mode compare     # compare both
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hand_retarget import InteractionMeshHandRetargeter, HandRetargetConfig
from hand_retarget.mediapipe_io import load_hocap_clip

PROJECT_DIR = Path(__file__).resolve().parents[1]
HOCAP_DIR = PROJECT_DIR / "data" / "hocap" / "hocap"
SCENE_LEFT = PROJECT_DIR / "assets" / "scenes" / "single_hand_obj_left.xml"
SCENE_RIGHT = PROJECT_DIR / "assets" / "scenes" / "single_hand_obj.xml"

# 5+ clips: mixed subjects, single + bimanual, different objects
CLIPS = [
    ("hocap__subject_1__20231025_165502__seg00", "left"),
    ("hocap__subject_1__20231025_165807__seg03", "left"),
    ("hocap__subject_2__20231022_200657__seg00", "left"),   # bimanual, test left
    ("hocap__subject_2__20231022_200657__seg00", "right"),  # bimanual, test right
    ("hocap__subject_3__20231024_161306__seg00", "left"),   # bimanual, test left
    ("hocap__subject_3__20231024_161306__seg00", "right"),  # bimanual, test right
]


def retarget_clip(clip_id: str, hand_side: str, obj_samples: int = 50,
                  use_mano_rotation: bool = True):
    """Retarget one clip, return qpos and per-frame metrics."""
    npz_path = str(HOCAP_DIR / "motions" / f"{clip_id}.npz")
    meta_path = str(HOCAP_DIR / "motions" / f"{clip_id}.meta.json")
    scene = SCENE_LEFT if hand_side == "left" else SCENE_RIGHT

    clip = load_hocap_clip(npz_path, meta_path, str(HOCAP_DIR / "assets"),
                           hand_side=hand_side, sample_count=obj_samples)

    config = HandRetargetConfig(
        mjcf_path=str(scene), hand_side=hand_side,
        floating_base=True, object_sample_count=obj_samples,
        use_mano_rotation=use_mano_rotation,
    )
    retargeter = InteractionMeshHandRetargeter(config)
    qpos = retargeter.retarget_hocap_sequence(clip)

    # Compute per-frame metrics — compare in the retargeter's optimization frame
    from hand_retarget.mediapipe_io import preprocess_landmarks
    from scipy.spatial.transform import Rotation as RotLib

    # Determine preprocessing used by retargeter
    wrist_q_seq = clip.get("wrist_q")
    has_wrist_q = wrist_q_seq is not None and not use_mano_rotation
    if has_wrist_q:
        if config.hand_side == "left":
            from wuji_retargeting.mediapipe import OPERATOR2MANO_LEFT as OP2MANO
        else:
            from wuji_retargeting.mediapipe import OPERATOR2MANO_RIGHT as OP2MANO
        R_mano = np.array(OP2MANO, dtype=np.float64)

    T = len(qpos)
    tip_errors = np.zeros((T, 5))    # per fingertip position error
    all_errors = np.zeros((T, 21))   # all 21 keypoints

    for t in range(T):
        if has_wrist_q:
            # Match wrist_q + MANO preprocessing
            R_wrist = RotLib.from_quat(wrist_q_seq[t]).as_matrix()
            centered = clip["landmarks"][t] - clip["landmarks"][t, 0:1, :]
            source = centered @ R_wrist.T @ R_mano
            if retargeter.global_scale != 1.0:
                source *= retargeter.global_scale
        else:
            # Match SVD+MANO preprocessing
            source = preprocess_landmarks(
                clip["landmarks"][t],
                config.mediapipe_rotation,
                hand_side=config.hand_side,
                global_scale=retargeter.global_scale,
                use_mano_rotation=config.use_mano_rotation,
            )

        # Robot FK (in same frame as retargeter)
        retargeter.hand.forward(qpos[t])
        robot = retargeter._get_robot_keypoints()

        # Per-keypoint error
        for k, mp_idx in enumerate(retargeter.mp_indices):
            all_errors[t, k] = np.linalg.norm(source[mp_idx] - robot[k])

        # Fingertip errors (mp indices 4,8,12,16,20 → mapped indices)
        tip_mp = [4, 8, 12, 16, 20]
        for i, mp_idx in enumerate(tip_mp):
            k = retargeter.mp_indices.index(mp_idx)
            tip_errors[t, i] = all_errors[t, k]

    # Jerk (joint angle stability)
    if T >= 3:
        jerk = np.sum(np.abs(qpos[2:] - 2 * qpos[1:-1] + qpos[:-2]), axis=0)
        mean_jerk = float(jerk.mean())
    else:
        mean_jerk = 0.0

    metrics = {
        "clip_id": clip_id,
        "hand_side": hand_side,
        "n_frames": T,
        "mean_tip_error_mm": float(tip_errors.mean() * 1000),
        "max_tip_error_mm": float(tip_errors.max() * 1000),
        "per_finger_mean_mm": [float(tip_errors[:, i].mean() * 1000) for i in range(5)],
        "mean_all_error_mm": float(all_errors.mean() * 1000),
        "mean_jerk": mean_jerk,
    }

    return qpos, metrics


def run_all(output_dir: Path, use_mano_rotation: bool = True):
    """Retarget all clips and save results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics = []

    for clip_id, hand_side in CLIPS:
        tag = f"{clip_id}__{hand_side}"
        print(f"\n{'='*60}")
        print(f"Clip: {tag}")
        print(f"{'='*60}")

        qpos, metrics = retarget_clip(clip_id, hand_side, use_mano_rotation=use_mano_rotation)

        # Save qpos
        np.savez_compressed(str(output_dir / f"qpos_{tag}.npz"), qpos=qpos)

        # Save metrics
        with open(str(output_dir / f"metrics_{tag}.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        all_metrics.append(metrics)
        print(f"  mean_tip_error: {metrics['mean_tip_error_mm']:.2f} mm")
        print(f"  max_tip_error:  {metrics['max_tip_error_mm']:.2f} mm")
        print(f"  mean_jerk:      {metrics['mean_jerk']:.4f}")

    # Summary
    with open(str(output_dir / "summary.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n{'='*60}")
    print(f"SUMMARY ({len(all_metrics)} clips)")
    print(f"{'='*60}")
    tips = [m["mean_tip_error_mm"] for m in all_metrics]
    jerks = [m["mean_jerk"] for m in all_metrics]
    print(f"  Avg mean_tip_error: {np.mean(tips):.2f} mm")
    print(f"  Avg mean_jerk:      {np.mean(jerks):.4f}")


def compare(baseline_dir: Path, simplified_dir: Path):
    """Compare baseline vs simplified results."""
    print(f"\n{'='*60}")
    print("COMPARISON: baseline (SVD+MANO) vs simplified (no rotation)")
    print(f"{'='*60}")

    with open(str(baseline_dir / "summary.json")) as f:
        baseline = json.load(f)
    with open(str(simplified_dir / "summary.json")) as f:
        simplified = json.load(f)

    passed = 0
    failed = 0

    print(f"\n{'Clip':<55} {'Baseline':>10} {'Simplified':>10} {'Delta':>8} {'Status':>8}")
    print("-" * 95)

    for b, s in zip(baseline, simplified):
        tag = f"{b['clip_id']}__{b['hand_side']}"
        b_err = b["mean_tip_error_mm"]
        s_err = s["mean_tip_error_mm"]
        delta = s_err - b_err
        status = "PASS" if delta <= 1.0 else "FAIL"
        if status == "PASS":
            passed += 1
        else:
            failed += 1
        print(f"  {tag:<53} {b_err:>8.2f}mm {s_err:>8.2f}mm {delta:>+7.2f} {status:>6}")

    print("-" * 95)

    b_avg = np.mean([m["mean_tip_error_mm"] for m in baseline])
    s_avg = np.mean([m["mean_tip_error_mm"] for m in simplified])
    b_jerk = np.mean([m["mean_jerk"] for m in baseline])
    s_jerk = np.mean([m["mean_jerk"] for m in simplified])

    print(f"  {'AVG tip error':<53} {b_avg:>8.2f}mm {s_avg:>8.2f}mm {s_avg-b_avg:>+7.2f}")
    print(f"  {'AVG jerk':<53} {b_jerk:>8.4f}  {s_jerk:>8.4f}  {s_jerk-b_jerk:>+7.4f}")
    print()
    print(f"  Result: {passed}/{passed+failed} clips passed (threshold: delta <= 1mm)")

    if failed == 0:
        print("  >>> ADOPT: simplified pipeline is equal or better <<<")
    else:
        print("  >>> REJECT: simplified pipeline is worse on some clips <<<")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "simplified", "compare"], required=True)
    args = parser.parse_args()

    baseline_dir = PROJECT_DIR / "experiments" / "baseline_svd_mano"
    simplified_dir = PROJECT_DIR / "experiments" / "simplified_no_svd"

    if args.mode == "baseline":
        run_all(baseline_dir, use_mano_rotation=True)
    elif args.mode == "simplified":
        run_all(simplified_dir, use_mano_rotation=False)
    elif args.mode == "compare":
        compare(baseline_dir, simplified_dir)


if __name__ == "__main__":
    main()
