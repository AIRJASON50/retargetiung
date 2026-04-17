"""Screen all HO-Cap clips: retarget + measure tip error + penetration depth.

Outputs a CSV with per-clip metrics and copies "clean" clips to a separate folder.
Clean = mean_tip_error < threshold AND max_penetration_depth < threshold.

Usage:
    PYTHONPATH=src python experiments/screen_clips.py
    PYTHONPATH=src python experiments/screen_clips.py --tip-threshold 15 --pen-threshold 10
"""

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as RotLib
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hand_retarget import InteractionMeshHandRetargeter, HandRetargetConfig
from hand_retarget.mediapipe_io import load_hocap_clip, transform_object_points

PROJECT_DIR = Path(__file__).resolve().parents[1]
HOCAP_DIR = PROJECT_DIR / "data" / "hocap" / "hocap"
SCENE_LEFT = str((PROJECT_DIR / "assets" / "scenes" / "single_hand_obj_left.xml").resolve())
SCENE_RIGHT = str((PROJECT_DIR / "assets" / "scenes" / "single_hand_obj.xml").resolve())


def screen_one_clip(clip_id: str, hand_side: str, obj_samples: int = 50):
    """Retarget one clip and measure tip error + penetration depth."""
    npz_path = str(HOCAP_DIR / "motions" / f"{clip_id}.npz")
    meta_path = str(HOCAP_DIR / "motions" / f"{clip_id}.meta.json")
    scene = SCENE_LEFT if hand_side == "left" else SCENE_RIGHT

    clip = load_hocap_clip(npz_path, meta_path, str(HOCAP_DIR / "assets"),
                           hand_side=hand_side, sample_count=obj_samples)

    config = HandRetargetConfig(
        mjcf_path=scene, hand_side=hand_side,
        floating_base=True, object_sample_count=obj_samples,
        activate_non_penetration=False,  # measure natural quality, no constraint
    )
    retargeter = InteractionMeshHandRetargeter(config)
    qpos = retargeter.retarget_hocap_sequence(clip)

    # Inject object for penetration measurement only (after retarget)
    if clip.get("mesh_path"):
        retargeter.hand.inject_object_mesh(clip["mesh_path"], hand_side)

    T = len(qpos)
    wrist_q_seq = clip.get("wrist_q")
    obj_pts_local = clip["object_pts_local"]
    obj_t = clip["object_t"]
    obj_q = clip["object_q"]

    tip_errors = []
    pen_depths = []  # max penetration depth per frame (beyond capsule radius)

    capsule_radius = 0.0075

    for t in range(T):
        # Tip error in aligned frame
        wrist_q_t = wrist_q_seq[t] if wrist_q_seq is not None else None
        obj_world = transform_object_points(obj_pts_local, obj_q[t], obj_t[t])
        lm_aligned, _ = retargeter._align_frame(clip["landmarks"][t], wrist_q_t, obj_world)

        retargeter.hand.forward(qpos[t])
        robot = retargeter._get_robot_keypoints()

        tip_mp = [4, 8, 12, 16, 20]
        for mp_idx in tip_mp:
            k = retargeter.mp_indices.index(mp_idx)
            tip_errors.append(np.linalg.norm(lm_aligned[mp_idx] - robot[k]))

        # Penetration depth (if object injected)
        if retargeter.hand._has_object:
            wrist = clip["landmarks"][t, 0]
            R_wrist = RotLib.from_quat(wrist_q_t).as_matrix() if wrist_q_t is not None else np.eye(3)
            R_align = R_wrist.T @ retargeter._R_mano
            obj_center = (obj_t[t] - wrist) @ R_align
            R_obj = RotLib.from_quat(obj_q[t]).as_matrix()
            R_obj_al = R_align.T @ R_obj
            obj_q_xyzw = RotLib.from_matrix(R_obj_al).as_quat()
            retargeter.hand.set_object_pose(obj_center, obj_q_xyzw)
            retargeter.hand.forward(qpos[t])

            contacts = retargeter.hand.query_tip_penetration(threshold=0.05)
            frame_max_depth = 0.0
            for _, phi, _ in contacts:
                depth = max(0, -phi - capsule_radius)
                frame_max_depth = max(frame_max_depth, depth)
            pen_depths.append(frame_max_depth)

    mean_tip = float(np.mean(tip_errors) * 1000)
    max_pen = float(np.max(pen_depths) * 1000) if pen_depths else 0.0
    mean_pen = float(np.mean(pen_depths) * 1000) if pen_depths else 0.0
    jerk = 0.0
    if T >= 3:
        jerk = float(np.mean(np.sum(np.abs(qpos[2:] - 2 * qpos[1:-1] + qpos[:-2]), axis=0)))

    return {
        "clip_id": clip_id,
        "hand_side": hand_side,
        "n_frames": T,
        "mean_tip_mm": mean_tip,
        "max_pen_mm": max_pen,
        "mean_pen_mm": mean_pen,
        "jerk": jerk,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tip-threshold", type=float, default=15.0, help="Max mean tip error (mm)")
    parser.add_argument("--pen-threshold", type=float, default=10.0, help="Max penetration beyond capsule (mm)")
    parser.add_argument("--max-clips", type=int, default=None, help="Limit number of clips to process")
    args = parser.parse_args()

    # Discover all clips
    meta_files = sorted(HOCAP_DIR.glob("motions/*.meta.json"))
    all_tasks = []
    for mf in meta_files:
        clip_id = mf.stem.replace(".meta", "")
        npz_path = str(mf).replace(".meta.json", ".npz")
        data = np.load(npz_path, allow_pickle=True)
        for side in ["left", "right"]:
            key = f"mediapipe_{side[0]}_world"
            if key in data and data[key].dtype != object:
                all_tasks.append((clip_id, side))

    if args.max_clips:
        all_tasks = all_tasks[:args.max_clips]

    print(f"Screening {len(all_tasks)} hand-clips...")

    results = []
    clean = []
    t0 = time.perf_counter()

    for i, (clip_id, hand_side) in enumerate(all_tasks):
        tag = f"{clip_id}__{hand_side}"
        try:
            r = screen_one_clip(clip_id, hand_side)
            results.append(r)
            status = "CLEAN" if r["mean_tip_mm"] < args.tip_threshold and r["max_pen_mm"] < args.pen_threshold else "SKIP"
            if status == "CLEAN":
                clean.append(r)
            print(f"[{i+1}/{len(all_tasks)}] {status} {tag}: tip={r['mean_tip_mm']:.1f}mm pen={r['max_pen_mm']:.1f}mm")
        except Exception as e:
            print(f"[{i+1}/{len(all_tasks)}] ERROR {tag}: {e}")
            results.append({"clip_id": clip_id, "hand_side": hand_side, "error": str(e)})

    elapsed = time.perf_counter() - t0

    # Save CSV
    output_dir = PROJECT_DIR / "experiments" / "clip_screening"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "all_clips.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clip_id", "hand_side", "n_frames", "mean_tip_mm", "max_pen_mm", "mean_pen_mm", "jerk"])
        w.writeheader()
        for r in results:
            if "error" not in r:
                w.writerow(r)

    # Save clean list
    clean_path = output_dir / "clean_clips.json"
    with open(clean_path, "w") as f:
        json.dump([{"clip_id": r["clip_id"], "hand_side": r["hand_side"],
                     "mean_tip_mm": r["mean_tip_mm"], "max_pen_mm": r["max_pen_mm"]}
                    for r in clean], f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print(f"Screening complete: {elapsed:.0f}s ({elapsed/len(all_tasks):.1f}s per clip)")
    print(f"Total: {len(results)}, Clean: {len(clean)} ({len(clean)/len(results)*100:.0f}%)")
    print(f"Thresholds: tip<{args.tip_threshold}mm, pen<{args.pen_threshold}mm")
    print(f"Results: {csv_path}")
    print(f"Clean list: {clean_path}")


if __name__ == "__main__":
    main()
