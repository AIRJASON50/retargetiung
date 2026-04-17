"""
EXP: Object-frame Laplacian vs. wrist-frame Laplacian on HO-Cap data.

Compares retargeting quality when Laplacian is computed in:
  A. Wrist-aligned frame (current, default)
  B. Object-local frame (OmniRetarget approach)

Metrics: fingertip-to-object distance preservation, joint smoothness.

Usage:
    python experiments/exp_object_frame.py --clip 84
    python experiments/exp_object_frame.py --clip 101
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
sys.path.insert(0, "/home/l/ws/doc/WujiRepo/wuji_retargeting_private/public")

from hand_retarget import InteractionMeshHandRetargeter, HandRetargetConfig  # noqa: E402
from hand_retarget.mediapipe_io import load_hocap_clip, transform_object_points  # noqa: E402

HOCAP_DIR = PROJECT_DIR / "data" / "hocap" / "hocap"
SCENE_LEFT = PROJECT_DIR / "assets" / "scenes" / "single_hand_obj_left.xml"
SCENE_RIGHT = PROJECT_DIR / "assets" / "scenes" / "single_hand_obj.xml"
INDEX_PATH = PROJECT_DIR / "data" / "cache" / "hocap" / "clip_index.txt"


def resolve_clip(clip_arg: str) -> tuple[str, str, str]:
    """Resolve numeric ID or clip_id to (clip_id, npz_path, meta_path)."""
    if clip_arg.isdigit():
        num = int(clip_arg)
        with open(INDEX_PATH) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2 and parts[0] == f"{num:03d}":
                    clip_arg = parts[1]
                    break
    npz_path = str(HOCAP_DIR / "motions" / f"{clip_arg}.npz")
    meta_path = str(HOCAP_DIR / "motions" / f"{clip_arg}.meta.json")
    return clip_arg, npz_path, meta_path


def detect_hand(meta_path: str) -> str:
    """Return primary hand side from metadata."""
    with open(meta_path) as f:
        meta = json.load(f)
    h = meta.get("handedness", "right")
    if h == "both":
        return "right"  # default to right for comparison
    return h


def run_retarget(clip_id, npz_path, meta_path, hand_side, use_object_frame, obj_samples=50):
    """Run retargeting with or without object-frame Laplacian."""
    scene = SCENE_LEFT if hand_side == "left" else SCENE_RIGHT
    config = HandRetargetConfig(
        mjcf_path=str(scene), hand_side=hand_side,
        floating_base=True, object_sample_count=obj_samples,
    )
    config.use_object_frame = use_object_frame

    retargeter = InteractionMeshHandRetargeter(config)
    clip = load_hocap_clip(npz_path, meta_path, str(HOCAP_DIR / "assets"),
                           hand_side=hand_side, sample_count=obj_samples)

    t0 = time.time()
    qpos = retargeter.retarget_hocap_sequence(clip, use_semantic_weights=False)
    elapsed = time.time() - t0
    T = len(qpos)
    fps = T / elapsed

    label = "obj-frame" if use_object_frame else "wrist-frame"
    print(f"  {label}: {T} frames in {elapsed:.1f}s ({fps:.0f} fps)")

    return qpos, clip, retargeter


def compute_tip_object_distance(qpos, clip, retargeter):
    """Compute per-frame fingertip-to-object-center distance."""
    from hand_retarget.mediapipe_io import transform_object_points

    T = len(qpos)
    obj_pts_local = clip["object_pts_local"]
    obj_t = clip["object_t"]
    obj_q = clip["object_q"]
    wrist_q_seq = clip.get("wrist_q")
    landmarks_raw = clip["landmarks"]

    tip_names = retargeter.config.fingertip_links
    dists = np.zeros((T, 5))  # 5 fingertips

    for t in range(T):
        retargeter.hand.forward(qpos[t])
        tips = retargeter.hand.get_body_positions(tip_names)  # (5, 3) in robot frame

        # Object center in aligned frame
        wrist_q_t = wrist_q_seq[t] if wrist_q_seq is not None else None
        wrist_world = landmarks_raw[t, 0]
        obj_c = obj_t[t] - wrist_world
        if wrist_q_t is not None:
            R_wrist = RotLib.from_quat(wrist_q_t).as_matrix()
            R_align = R_wrist.T @ retargeter._R_mano
            obj_c = obj_c @ R_align

        for f in range(5):
            dists[t, f] = np.linalg.norm(tips[f] - obj_c)

    return dists


def compute_smoothness(qpos):
    """Compute jerk (3rd derivative) of joint angles."""
    if len(qpos) < 4:
        return 0.0
    vel = np.diff(qpos, axis=0)
    acc = np.diff(vel, axis=0)
    jerk = np.diff(acc, axis=0)
    return np.mean(np.linalg.norm(jerk, axis=1))


def main():
    parser = argparse.ArgumentParser(description="Object-frame Laplacian experiment")
    parser.add_argument("--clip", type=str, default="101", help="Clip ID (numeric or full name)")
    parser.add_argument("--obj-samples", type=int, default=50)
    args = parser.parse_args()

    clip_id, npz_path, meta_path = resolve_clip(args.clip)
    hand_side = detect_hand(meta_path)

    with open(meta_path) as f:
        meta = json.load(f)
    asset = meta["objects"][0]["asset_name"]

    print(f"Clip: {clip_id}")
    print(f"Hand: {hand_side}, Object: {asset}")
    print()

    # A: wrist-frame (current default)
    qpos_wrist, clip, retargeter_w = run_retarget(
        clip_id, npz_path, meta_path, hand_side, use_object_frame=False, obj_samples=args.obj_samples
    )

    # B: object-frame
    qpos_obj, _, retargeter_o = run_retarget(
        clip_id, npz_path, meta_path, hand_side, use_object_frame=True, obj_samples=args.obj_samples
    )

    # Metrics
    print()
    print("Computing metrics...")

    dist_w = compute_tip_object_distance(qpos_wrist, clip, retargeter_w)
    dist_o = compute_tip_object_distance(qpos_obj, clip, retargeter_o)

    jerk_w = compute_smoothness(qpos_wrist)
    jerk_o = compute_smoothness(qpos_obj)

    finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    w = 14

    print()
    print("=" * 50)
    print("  OBJECT-FRAME LAPLACIAN COMPARISON")
    print("=" * 50)

    print(f"\n{'Tip-Object dist (mm)':>25s}  {'Wrist':>{w}s}  {'ObjFrame':>{w}s}")
    print("-" * (25 + w * 2 + 4))
    for f, name in enumerate(finger_names):
        mw = dist_w[:, f].mean() * 1000
        mo = dist_o[:, f].mean() * 1000
        print(f"{name:>25s}  {mw:>{w}.1f}  {mo:>{w}.1f}")
    print(f"{'Mean':>25s}  {dist_w.mean()*1000:>{w}.1f}  {dist_o.mean()*1000:>{w}.1f}")

    print(f"\n{'Smoothness (jerk)':>25s}  {jerk_w:>{w}.1f}  {jerk_o:>{w}.1f}")

    # Joint angle difference
    diff = np.mean(np.abs(qpos_wrist - qpos_obj))
    print(f"{'Mean |q_diff| (rad)':>25s}  {diff:>{w}.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
