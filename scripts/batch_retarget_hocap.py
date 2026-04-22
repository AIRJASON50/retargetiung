"""Batch retarget all HO-Cap clips and cache results.

Saves qpos + wrist transforms to data/cache/hocap/{clip_id}.npz
so play_hocap.py can load without recomputing.

Usage:
    python scripts/batch_retarget_hocap.py
    python scripts/batch_retarget_hocap.py --obj-samples 50
    python scripts/batch_retarget_hocap.py --skip-existing
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as RotLib

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR))
import os; _WUJI_SDK = os.environ.get("WUJI_SDK_PATH", "/home/l/ws/doc/WujiRepo/wuji_retargeting_private/public"); sys.path.insert(0, _WUJI_SDK)

from hand_retarget import InteractionMeshHandRetargeter, HandRetargetConfig  # noqa: E402
from hand_retarget.mediapipe_io import load_hocap_clip  # noqa: E402

HOCAP_DIR = PROJECT_DIR / "data" / "hocap" / "hocap"
CACHE_DIR = PROJECT_DIR / "data" / "cache" / "hocap"
HOCAP_CONFIG_YAML = PROJECT_DIR / "config" / "hocap.yaml"
SCENE_LEFT = PROJECT_DIR / "assets" / "scenes" / "single_hand_obj_left.xml"
SCENE_RIGHT = PROJECT_DIR / "assets" / "scenes" / "single_hand_obj.xml"


def detect_handedness(npz_path: str, meta_path: str) -> list[str]:
    """Return list of active hands for a clip."""
    data = np.load(npz_path, allow_pickle=True)
    with open(meta_path) as f:
        meta = json.load(f)
    handedness = meta.get("handedness", "unknown")

    hands = []
    if handedness == "left":
        hands = ["left"]
    elif handedness == "right":
        hands = ["right"]
    elif handedness == "both":
        has_l = data["mediapipe_l_world"].ndim > 0
        has_r = data["mediapipe_r_world"].ndim > 0
        if has_l:
            hands.append("left")
        if has_r:
            hands.append("right")
    else:
        # Fallback: check which hands have data
        if data["mediapipe_l_world"].ndim > 0:
            hands.append("left")
        if data["mediapipe_r_world"].ndim > 0:
            hands.append("right")
    return hands


def retarget_one_hand(clip: dict, hand_side: str, obj_samples: int) -> dict:
    """Retarget a single hand and return qpos + wrist transforms."""
    scene = SCENE_LEFT if hand_side == "left" else SCENE_RIGHT
    config = HandRetargetConfig.from_yaml(
        str(HOCAP_CONFIG_YAML),
        mjcf_path=str(scene),
        hand_side=hand_side,
        object_sample_count=obj_samples,
    )
    retargeter = InteractionMeshHandRetargeter(config)
    qpos = retargeter.retarget_hocap_sequence(clip, use_semantic_weights=False)

    T = len(clip["landmarks"])
    R_inv_list = np.zeros((T, 3, 3))
    wrist_list = np.zeros((T, 3))
    wrist_q_seq = clip.get("wrist_q")

    for t in range(T):
        wrist_list[t] = clip["landmarks"][t, 0]
        if wrist_q_seq is not None:
            R_wrist = RotLib.from_quat(wrist_q_seq[t]).as_matrix()
            R_align = R_wrist.T @ retargeter._R_mano
            R_inv_list[t] = R_align.T
        else:
            R_inv_list[t] = np.eye(3)

    return {"qpos": qpos, "R_inv": R_inv_list, "wrist": wrist_list}


def main():
    parser = argparse.ArgumentParser(description="Batch retarget HO-Cap clips")
    parser.add_argument("--obj-samples", type=int, default=50)
    parser.add_argument("--skip-existing", action="store_true", help="Skip clips with existing cache")
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    motion_dir = HOCAP_DIR / "motions"
    npz_files = sorted(motion_dir.glob("*.npz"))
    total_clips = len(npz_files)

    print(f"HO-Cap batch retarget: {total_clips} clips")
    print(f"Object samples: {args.obj_samples}")
    print(f"Cache dir: {CACHE_DIR}")
    print(f"Skip existing: {args.skip_existing}")
    print("=" * 60)

    t_global = time.time()
    done = 0
    skipped = 0
    failed = 0
    total_frames = 0

    for i, npz_path in enumerate(npz_files):
        clip_id = npz_path.stem
        meta_path = str(npz_path).replace(".npz", ".meta.json")
        cache_path = CACHE_DIR / f"{clip_id}.npz"

        if args.skip_existing and cache_path.exists():
            skipped += 1
            continue

        try:
            hands = detect_handedness(str(npz_path), meta_path)
            if not hands:
                print(f"  [{i+1}/{total_clips}] {clip_id}: no hand data, skipping")
                skipped += 1
                continue

            t0 = time.time()
            save_dict = {"hands": np.array(hands)}

            for hand_side in hands:
                clip = load_hocap_clip(
                    str(npz_path), meta_path, str(HOCAP_DIR / "assets"),
                    hand_side=hand_side, sample_count=args.obj_samples,
                )
                result = retarget_one_hand(clip, hand_side, args.obj_samples)
                n_frames = len(result["qpos"])
                total_frames += n_frames

                save_dict[f"qpos_{hand_side}"] = result["qpos"]
                save_dict[f"R_inv_{hand_side}"] = result["R_inv"]
                save_dict[f"wrist_{hand_side}"] = result["wrist"]

            np.savez_compressed(cache_path, **save_dict)
            elapsed = time.time() - t0
            fps = n_frames / elapsed if elapsed > 0 else 0
            done += 1
            print(f"  [{i+1}/{total_clips}] {clip_id}: {hands} {n_frames}f {elapsed:.1f}s ({fps:.0f}fps)")

        except Exception as e:
            failed += 1
            print(f"  [{i+1}/{total_clips}] {clip_id}: FAILED - {e}")

    elapsed_total = time.time() - t_global
    print("=" * 60)
    print(f"Done: {done}, Skipped: {skipped}, Failed: {failed}")
    print(f"Total frames: {total_frames}")
    print(f"Total time: {elapsed_total/60:.1f} min")


if __name__ == "__main__":
    main()
