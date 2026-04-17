"""HO-Cap retargeting CLI — single clip or batch, with config stamp.

Produces stamped .npz files in a configurable output directory.
Same key format as the legacy batch_retarget_hocap.py, plus metadata:
    qpos_{side}       (T, 26)
    R_inv_{side}      (T, 3, 3)
    wrist_{side}      (T, 3)
    hands             (K,) str array
    stamp             () str scalar — config stamp string
    config_json       () str scalar — full config as JSON
    generated_at      () str scalar — ISO-8601 UTC timestamp

Output filename: {clip_id}__{stamp}.npz

Usage:
    # Single clip
    python scripts/retarget_hocap.py --clip hocap__subject_3__20231024_162409__seg00

    # Single clip, custom output dir and settings
    python scripts/retarget_hocap.py \\
        --clip hocap__subject_3__20231024_162409__seg00 \\
        --out-dir data/cache/hocap_exp \\
        --decay-k 0 \\
        --threshold 0.08

    # Batch (all clips), skip existing
    python scripts/retarget_hocap.py --all --skip-existing

    # Batch, limit frames for quick testing
    python scripts/retarget_hocap.py --all --frames 100
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as RotLib

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, "/home/l/ws/doc/WujiRepo/wuji_retargeting_private/public")

from hand_retarget import InteractionMeshHandRetargeter, HandRetargetConfig  # noqa: E402
from hand_retarget.mediapipe_io import load_hocap_clip  # noqa: E402

HOCAP_DIR = PROJECT_DIR / "data" / "hocap" / "hocap"
SCENE = {
    "left":  str(PROJECT_DIR / "assets" / "scenes" / "single_hand_obj_left.xml"),
    "right": str(PROJECT_DIR / "assets" / "scenes" / "single_hand_obj.xml"),
}


# ─────────────────────────────────────────────
# Clip discovery
# ─────────────────────────────────────────────

def discover_clips() -> list[str]:
    """Return sorted list of all clip IDs from motions/ directory."""
    return sorted(p.stem for p in (HOCAP_DIR / "motions").glob("*.npz"))


def detect_hands(clip_id: str) -> list[str]:
    """Detect which hands have data in a clip."""
    npz = np.load(str(HOCAP_DIR / "motions" / f"{clip_id}.npz"), allow_pickle=True)
    hands = []
    if "mediapipe_l_world" in npz and npz["mediapipe_l_world"].ndim > 0:
        hands.append("left")
    if "mediapipe_r_world" in npz and npz["mediapipe_r_world"].ndim > 0:
        hands.append("right")
    return hands


# ─────────────────────────────────────────────
# Retargeting
# ─────────────────────────────────────────────

def retarget_clip(clip_id: str, config_kw: dict, frames: int | None) -> dict:
    """Retarget all hands in one clip; return save_dict ready for np.savez_compressed."""
    npz_path  = str(HOCAP_DIR / "motions" / f"{clip_id}.npz")
    meta_path = str(HOCAP_DIR / "motions" / f"{clip_id}.meta.json")
    hands = detect_hands(clip_id)

    save_dict: dict[str, object] = {"hands": np.array(hands)}

    for hand_side in hands:
        clip = load_hocap_clip(
            npz_path, meta_path, str(HOCAP_DIR / "assets"),
            hand_side=hand_side, sample_count=config_kw["object_sample_count"],
        )

        if frames is not None:
            N = min(frames, len(clip["landmarks"]))
            pts_local = clip["object_pts_local"]
            clip = {k: (v[:N] if isinstance(v, np.ndarray) and v.ndim > 0 and len(v) >= N else v)
                    for k, v in clip.items()}
            clip["object_pts_local"] = pts_local

        cfg = HandRetargetConfig(
            mjcf_path=SCENE[hand_side],
            hand_side=hand_side,
            **config_kw,
        )
        ret = InteractionMeshHandRetargeter(cfg)

        t0 = time.time()
        qpos = ret.retarget_hocap_sequence(clip)
        elapsed = time.time() - t0
        T = len(qpos)
        print(f"    [{hand_side}] {T} frames in {elapsed:.1f}s ({T/elapsed:.0f} fps)")

        R_inv = np.zeros((T, 3, 3))
        wrist = np.zeros((T, 3))
        wq = clip.get("wrist_q")
        for t in range(T):
            wrist[t] = clip["landmarks"][t, 0]
            if wq is not None:
                R = RotLib.from_quat(wq[t]).as_matrix()
                R_inv[t] = (R.T @ ret._R_mano).T
            else:
                R_inv[t] = np.eye(3)

        save_dict[f"qpos_{hand_side}"]  = qpos
        save_dict[f"R_inv_{hand_side}"] = R_inv
        save_dict[f"wrist_{hand_side}"] = wrist

    return save_dict


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def build_config_kw(args: argparse.Namespace) -> dict:
    """Build HandRetargetConfig constructor kwargs from CLI args."""
    kw: dict[str, object] = {
        "floating_base":               True,
        "object_sample_count":         args.obj_samples,
        "delaunay_edge_threshold":     args.threshold,
        "laplacian_distance_weight_k": args.decay_k,
        "use_arap_edge":               args.arap,
        "use_skeleton_topology":       args.skeleton,
        "use_bone_scaling":            args.bone_scale,
        "use_orientation_probes":      args.probes,
        "rotation_compensation":       args.rotation_comp,
    }
    return kw


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retarget HO-Cap clips and save stamped .npz trajectory files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Target selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--clip", metavar="CLIP_ID",
                       help="Single clip ID or path to .npz")
    group.add_argument("--all", action="store_true",
                       help="Process all clips in the HO-Cap motions directory")

    # Output
    parser.add_argument("--out-dir", type=Path,
                        default=PROJECT_DIR / "data" / "cache" / "hocap",
                        help="Directory for output .npz files")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip clips whose stamped output already exists")
    parser.add_argument("--frames", type=int, default=None,
                        help="Limit frames per clip (for quick testing)")

    # Config overrides
    parser.add_argument("--obj-samples", type=int, default=50,
                        help="Object surface sample count")
    parser.add_argument("--threshold", type=float, default=0.06,
                        help="Delaunay edge threshold (m). Set to 0 for None (no filter)")
    parser.add_argument("--decay-k", type=float, default=20.0,
                        help="Laplacian distance-decay k. Set to 0 for None (uniform)")
    parser.add_argument("--arap", action="store_true",
                        help="Use ARAP per-edge energy")
    parser.add_argument("--skeleton", action="store_true",
                        help="Use skeleton topology")
    parser.add_argument("--bone-scale", action="store_true",
                        help="Use per-finger bone-ratio scaling")
    parser.add_argument("--probes", action="store_true",
                        help="Use fingertip orientation probes")
    parser.add_argument("--rotation-comp", action="store_true",
                        help="Use ARAP rotation compensation")

    args = parser.parse_args()

    # Convert sentinel 0 -> None for optional float params
    if args.threshold == 0:
        args.threshold = None
    if args.decay_k == 0:
        args.decay_k = None

    config_kw = build_config_kw(args)

    # Compute stamp from a representative config (hand_side doesn't affect stamp fields)
    probe_cfg = HandRetargetConfig(mjcf_path="", hand_side="right", **config_kw)
    stamp = probe_cfg.make_stamp()
    config_json = json.dumps(
        {k: v for k, v in dataclasses.asdict(probe_cfg).items() if k != "mjcf_path"},
        sort_keys=True, default=str,
    )
    generated_at = datetime.now(timezone.utc).isoformat()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Stamp:   {stamp}")
    print(f"Out dir: {args.out_dir}")
    print(f"Config:  threshold={args.threshold}  decay_k={args.decay_k}"
          f"  obj_samples={args.obj_samples}")
    print("=" * 60)

    clips = [args.clip] if args.clip else discover_clips()
    total = len(clips)
    done = skipped = failed = 0
    t_global = time.time()

    for i, clip_id in enumerate(clips):
        # Normalise: accept bare clip_id or full path
        if clip_id.endswith(".npz"):
            clip_id = Path(clip_id).stem

        out_path = args.out_dir / f"{clip_id}__{stamp}.npz"

        if args.skip_existing and out_path.exists():
            skipped += 1
            continue

        label = f"[{i+1}/{total}] {clip_id}"
        print(f"{label}")
        try:
            save_dict = retarget_clip(clip_id, config_kw, args.frames)
            save_dict["stamp"]        = np.bytes_(stamp)
            save_dict["config_json"]  = np.bytes_(config_json)
            save_dict["generated_at"] = np.bytes_(generated_at)

            np.savez_compressed(out_path, **save_dict)
            hands = list(save_dict["hands"])
            T = len(save_dict[f"qpos_{hands[0]}"])
            print(f"  -> {out_path.name}  ({hands}, {T} frames)")
            done += 1
        except Exception as exc:
            import traceback
            print(f"  FAILED: {exc}")
            traceback.print_exc()
            failed += 1

    elapsed = time.time() - t_global
    print("=" * 60)
    print(f"Done: {done}  Skipped: {skipped}  Failed: {failed}  "
          f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
