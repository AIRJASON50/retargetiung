"""Compare SVD vs wrist_q alignment quality across all HO-Cap clips.

For each clip, measures:
- Frame 0 tip error (SVD vs wrist_q)
- Angular difference between SVD and wrist_q orientations
- Per-hand-side statistics

Usage:
    PYTHONPATH=src python experiments/analyze_alignment.py
"""

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as RotLib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hand_retarget import InteractionMeshHandRetargeter, HandRetargetConfig
from hand_retarget.mediapipe_io import load_hocap_clip, preprocess_landmarks, transform_object_points
from wuji_retargeting.mediapipe import apply_mediapipe_transformations

PROJECT_DIR = Path(__file__).resolve().parents[1]
HOCAP_DIR = PROJECT_DIR / "data" / "hocap" / "hocap"


def analyze_clip(clip_id: str, hand_side: str):
    """Compare SVD vs wrist_q alignment for one clip."""
    npz_path = str(HOCAP_DIR / "motions" / f"{clip_id}.npz")
    meta_path = str(HOCAP_DIR / "motions" / f"{clip_id}.meta.json")
    scene = "assets/scenes/single_hand_obj_left.xml" if hand_side == "left" else "assets/scenes/single_hand_obj.xml"

    clip = load_hocap_clip(npz_path, meta_path, str(HOCAP_DIR / "assets"),
                           hand_side=hand_side, sample_count=10)

    lm = clip["landmarks"]
    wrist_q_seq = clip.get("wrist_q")
    T = len(lm)

    # --- Angular difference at frame 0 ---
    centered_0 = lm[0] - lm[0, 0]
    svd_result = apply_mediapipe_transformations(lm[0].copy(), hand_side)
    svd_finger = svd_result[9] / np.linalg.norm(svd_result[9])

    if wrist_q_seq is not None:
        from wuji_retargeting.mediapipe import OPERATOR2MANO_LEFT, OPERATOR2MANO_RIGHT
        R_mano = np.array(OPERATOR2MANO_LEFT if hand_side == "left" else OPERATOR2MANO_RIGHT, dtype=float)
        R_wrist = RotLib.from_quat(wrist_q_seq[0]).as_matrix()
        wq_aligned = (centered_0 @ R_wrist.T) @ R_mano
        wq_finger = wq_aligned[9] / np.linalg.norm(wq_aligned[9])
        angle_diff = np.degrees(np.arccos(np.clip(np.dot(svd_finger, wq_finger), -1, 1)))
    else:
        angle_diff = -1

    # --- Retarget first 10 frames with each method ---
    N = min(10, T)
    clip_sub = {k: v[:N] if isinstance(v, np.ndarray) and v.ndim > 0 and v.shape[0] >= N else v
                for k, v in clip.items()}
    clip_sub["object_pts_local"] = clip["object_pts_local"]
    clip_sub["mesh_path"] = clip.get("mesh_path")

    config = HandRetargetConfig(
        mjcf_path=str(Path(scene).resolve()),
        hand_side=hand_side, floating_base=True, object_sample_count=10,
        activate_non_penetration=False)

    # SVD path (remove wrist_q)
    clip_svd = dict(clip_sub)
    clip_svd["wrist_q"] = None
    config.use_mano_rotation = True
    ret_svd = InteractionMeshHandRetargeter(config)
    qpos_svd = ret_svd.retarget_hocap_sequence(clip_svd)

    # wrist_q path
    config_wq = HandRetargetConfig(
        mjcf_path=str(Path(scene).resolve()),
        hand_side=hand_side, floating_base=True, object_sample_count=10,
        activate_non_penetration=False)
    ret_wq = InteractionMeshHandRetargeter(config_wq)
    qpos_wq = ret_wq.retarget_hocap_sequence(clip_sub)

    # Measure tip errors
    def tip_error(ret, qpos_t, landmarks_t, use_svd, wrist_q_t=None):
        if use_svd:
            source = preprocess_landmarks(landmarks_t, config.mediapipe_rotation,
                                          hand_side=hand_side, global_scale=1.0, use_mano_rotation=True)
        else:
            obj_w = transform_object_points(clip["object_pts_local"], clip["object_q"][0], clip["object_t"][0])
            source, _ = ret._align_frame(landmarks_t, wrist_q_t, obj_w)
        ret.hand.forward(qpos_t)
        robot = ret._get_robot_keypoints()
        return np.mean([np.linalg.norm(source[mp] - robot[ret.mp_indices.index(mp)])
                        for mp in [4, 8, 12, 16, 20]]) * 1000

    svd_f0 = tip_error(ret_svd, qpos_svd[0], clip_sub["landmarks"][0], use_svd=True)
    wq_f0 = tip_error(ret_wq, qpos_wq[0], clip_sub["landmarks"][0], use_svd=False,
                       wrist_q_t=wrist_q_seq[0] if wrist_q_seq is not None else None)

    svd_errs = [tip_error(ret_svd, qpos_svd[t], clip_sub["landmarks"][t], use_svd=True) for t in range(N)]
    wq_errs = [tip_error(ret_wq, qpos_wq[t], clip_sub["landmarks"][t], use_svd=False,
                          wrist_q_t=wrist_q_seq[t] if wrist_q_seq is not None else None) for t in range(N)]

    return {
        "clip_id": clip_id,
        "hand_side": hand_side,
        "n_frames": T,
        "angle_diff_deg": round(angle_diff, 1),
        "svd_f0_mm": round(svd_f0, 1),
        "wq_f0_mm": round(wq_f0, 1),
        "delta_f0_mm": round(wq_f0 - svd_f0, 1),
        "svd_avg10_mm": round(np.mean(svd_errs), 1),
        "wq_avg10_mm": round(np.mean(wq_errs), 1),
        "delta_avg10_mm": round(np.mean(wq_errs) - np.mean(svd_errs), 1),
    }


def main():
    meta_files = sorted(HOCAP_DIR.glob("motions/*.meta.json"))
    all_tasks = []
    for mf in meta_files:
        clip_id = mf.stem.replace(".meta", "")
        npz_path = str(mf).replace(".meta.json", ".npz")
        d = np.load(npz_path, allow_pickle=True)
        for side in ["left", "right"]:
            key = f"mediapipe_{side[0]}_world"
            if key in d and d[key].dtype != object:
                all_tasks.append((clip_id, side))

    print(f"Analyzing {len(all_tasks)} hand-clips...")

    output_dir = PROJECT_DIR / "experiments" / "alignment_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    t0 = time.perf_counter()

    for i, (clip_id, hand_side) in enumerate(all_tasks):
        tag = f"{clip_id}__{hand_side}"
        try:
            r = analyze_clip(clip_id, hand_side)
            results.append(r)
            winner = "SVD" if r["delta_f0_mm"] > 0 else "WQ"
            print(f"[{i+1}/{len(all_tasks)}] {tag}: angle={r['angle_diff_deg']}deg, "
                  f"f0: SVD={r['svd_f0_mm']} WQ={r['wq_f0_mm']} ({winner}), "
                  f"avg10: SVD={r['svd_avg10_mm']} WQ={r['wq_avg10_mm']}")
        except Exception as e:
            print(f"[{i+1}/{len(all_tasks)}] ERROR {tag}: {e}")

    elapsed = time.perf_counter() - t0

    # Save CSV
    csv_path = output_dir / "alignment_comparison.csv"
    fields = ["clip_id", "hand_side", "n_frames", "angle_diff_deg",
              "svd_f0_mm", "wq_f0_mm", "delta_f0_mm",
              "svd_avg10_mm", "wq_avg10_mm", "delta_avg10_mm"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow(r)

    # Summary
    print(f"\n{'='*70}")
    print(f"Complete: {elapsed:.0f}s ({elapsed/max(len(all_tasks),1):.1f}s per clip)")
    print(f"{'='*70}")

    left = [r for r in results if r["hand_side"] == "left"]
    right = [r for r in results if r["hand_side"] == "right"]

    for label, subset in [("ALL", results), ("LEFT", left), ("RIGHT", right)]:
        if not subset:
            continue
        angles = [r["angle_diff_deg"] for r in subset]
        svd_f0 = [r["svd_f0_mm"] for r in subset]
        wq_f0 = [r["wq_f0_mm"] for r in subset]
        svd_10 = [r["svd_avg10_mm"] for r in subset]
        wq_10 = [r["wq_avg10_mm"] for r in subset]
        svd_wins_f0 = sum(1 for r in subset if r["delta_f0_mm"] > 0)
        wq_wins_f0 = sum(1 for r in subset if r["delta_f0_mm"] < 0)

        print(f"\n  {label} ({len(subset)} clips):")
        print(f"    Angle SVD↔wrist_q: mean={np.mean(angles):.1f}deg, median={np.median(angles):.1f}deg, max={np.max(angles):.1f}deg")
        print(f"    Frame 0:  SVD={np.mean(svd_f0):.1f}mm, WQ={np.mean(wq_f0):.1f}mm, delta={np.mean(wq_f0)-np.mean(svd_f0):+.1f}mm")
        print(f"    Avg 10:   SVD={np.mean(svd_10):.1f}mm, WQ={np.mean(wq_10):.1f}mm, delta={np.mean(wq_10)-np.mean(svd_10):+.1f}mm")
        print(f"    Wins f0:  SVD={svd_wins_f0}, WQ={wq_wins_f0}, tie={len(subset)-svd_wins_f0-wq_wins_f0}")

    # Outliers: clips where wrist_q is >50mm worse at frame 0
    outliers = [r for r in results if r["delta_f0_mm"] > 50]
    if outliers:
        print(f"\n  OUTLIERS (wrist_q >50mm worse at f0): {len(outliers)} clips")
        for r in sorted(outliers, key=lambda x: -x["delta_f0_mm"])[:10]:
            print(f"    {r['clip_id']}__{r['hand_side']}: angle={r['angle_diff_deg']}deg, delta_f0={r['delta_f0_mm']:+.0f}mm")

    print(f"\n  CSV: {csv_path}")


if __name__ == "__main__":
    main()
