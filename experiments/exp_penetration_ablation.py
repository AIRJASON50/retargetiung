"""
EXP-13: Non-penetration hard constraint ablation (warmup × S2).

Four modes × N HO-Cap clips. Each mode toggles the pipeline-invariant
non-penetration linearized SQP constraint independently on warmup and S2:

    D  no constraint anywhere          (baseline, matches legacy)
    A  constraint in warmup only
    B  constraint in S2 only
    C  constraint in both              (pipeline invariant)

Per-frame telemetry collected via retargeter._frame_np_metrics:
    final_pen_max_mm   : residual penetration at final q (after retarget)
    warmup_shrinks     : trust-region halvings in the warmup stage
    s2_shrinks         : same for S2
    warmup_stalls      : warmup iters where all shrinks failed (structural)
    s2_stalls          : same for S2
    struct_infeas      : (warmup_stalls + s2_stalls) > 5

Per-clip summaries: pen stats, per-stage infeasibility, fps, qpos diff vs D.

Also writes stamped .npz caches (one per mode × clip) for use with
play_hocap.py --cache for 4-way visual comparison.

Usage:
    PYTHONPATH=src python experiments/exp_penetration_ablation.py            # full
    PYTHONPATH=src python experiments/exp_penetration_ablation.py --quick    # 100f × 1 clip
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
MAIN_REPO = Path("/home/l/ws/RL/retargeting")
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR))
_WUJI_SDK = os.environ.get(
    "WUJI_SDK_PATH", "/home/l/ws/doc/WujiRepo/wuji_retargeting_private/public"
)
sys.path.insert(0, _WUJI_SDK)

from hand_retarget import HandRetargetConfig, InteractionMeshHandRetargeter  # noqa: E402
from hand_retarget.mediapipe_io import load_hocap_clip  # noqa: E402

HOCAP_DIR = MAIN_REPO / "data" / "hocap" / "hocap"
HOCAP_YAML = PROJECT_DIR / "config" / "hocap.yaml"
SCENE = {
    "left":  str(PROJECT_DIR / "assets" / "scenes" / "single_hand_obj_left.xml"),
    "right": str(PROJECT_DIR / "assets" / "scenes" / "single_hand_obj.xml"),
}

OUT_DIR = PROJECT_DIR / "experiments" / "penetration_ablation_exp"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = PROJECT_DIR / "data" / "cache" / "hocap"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CLIPS = [
    "hocap__subject_1__20231025_165502__seg00",  # default clip
    "hocap__subject_3__20231024_161306__seg00",  # subject_3 grasp
    "hocap__subject_3__20231024_162409__seg00",  # subject_3 grasp
]

MODES = [
    ("D", False, False),
    ("A", True,  False),
    ("B", False, True),
    ("C", True,  True),
]


def detect_hands(clip_id: str) -> list[str]:
    """Return all active hands. Bimanual clips must retarget BOTH or
    play_hocap will fall back to on-the-fly retargeting for the missing
    hand, mixing the cached mode with current YAML defaults."""
    npz_path = HOCAP_DIR / "motions" / f"{clip_id}.npz"
    data = np.load(str(npz_path), allow_pickle=True)
    hands: list[str] = []
    if "mediapipe_l_world" in data and data["mediapipe_l_world"].ndim > 0:
        hands.append("left")
    if "mediapipe_r_world" in data and data["mediapipe_r_world"].ndim > 0:
        hands.append("right")
    return hands


def run_one(clip_id: str, mode_tag: str, warmup_flag: bool, s2_flag: bool,
            n_frames: int | None) -> dict:
    hands = detect_hands(clip_id)
    if not hands:
        raise RuntimeError(f"No hands detected for {clip_id}")

    npz = HOCAP_DIR / "motions" / f"{clip_id}.npz"
    meta = HOCAP_DIR / "motions" / f"{clip_id}.meta.json"

    all_per_frame: list[dict] = []
    save_kwargs: dict = {"hands": np.array(hands), "stamp": f"np-{mode_tag}".encode()}
    total_T = 0
    wall_total = 0.0

    for hand_side in hands:
        cfg = HandRetargetConfig.from_yaml(
            str(HOCAP_YAML),
            mjcf_path=SCENE[hand_side],
            hand_side=hand_side,
            activate_non_penetration_warmup=warmup_flag,
            activate_non_penetration_s2=s2_flag,
        )
        r = InteractionMeshHandRetargeter(cfg)

        clip = load_hocap_clip(
            str(npz), str(meta), str(HOCAP_DIR / "assets"),
            hand_side=hand_side, sample_count=50,
        )
        if n_frames is not None and n_frames < len(clip["landmarks"]):
            pts_local = clip["object_pts_local"]
            clip = {
                k: v[:n_frames] if isinstance(v, np.ndarray) and v.ndim > 0 and v.shape[0] >= n_frames else v
                for k, v in clip.items()
            }
            clip["object_pts_local"] = pts_local

        # Per-frame telemetry — aggregated across all hands for summary
        per_frame: list[dict] = []
        orig_retarget = r.retarget_frame

        def tap(*args, **kwargs):
            q = orig_retarget(*args, **kwargs)
            per_frame.append(dict(r._frame_np_metrics))
            return q
        r.retarget_frame = tap

        t0 = time.perf_counter()
        qpos = r.retarget_hocap_sequence(clip)
        wall_total += time.perf_counter() - t0
        T = len(qpos)
        total_T += T
        all_per_frame.extend(per_frame)

        # Per-hand cache entries. R_inv must be the SVD-derived rotation that
        # maps robot-local → world (same computation as play_hocap's retarget_hand
        # fallback path, otherwise the hand is rendered axis-aligned instead of
        # in the MediaPipe world pose).
        from wuji_retargeting.mediapipe import (
            OPERATOR2MANO_LEFT, OPERATOR2MANO_RIGHT,
            estimate_frame_from_hand_points,
        )
        op2mano = np.array(OPERATOR2MANO_LEFT if hand_side == "left" else OPERATOR2MANO_RIGHT)
        R_inv = np.zeros((T, 3, 3))
        wrist = np.zeros((T, 3))
        for t in range(T):
            wrist[t] = clip["landmarks"][t, 0]
            lm_centered = clip["landmarks"][t] - clip["landmarks"][t, 0]
            R_svd = estimate_frame_from_hand_points(lm_centered)
            R_align = R_svd @ op2mano
            R_inv[t] = R_align.T
        save_kwargs[f"qpos_{hand_side}"]  = qpos
        save_kwargs[f"R_inv_{hand_side}"] = R_inv
        save_kwargs[f"wrist_{hand_side}"] = wrist

    stamp = f"np-{mode_tag}"
    cache_path = CACHE_DIR / f"{clip_id}__{stamp}.npz"
    np.savez_compressed(str(cache_path), **save_kwargs)

    # Aggregate telemetry across all hands
    per_frame = all_per_frame
    T = total_T
    wall = wall_total

    pens = np.array([f["final_pen_max_mm"] for f in per_frame])
    w_sh = np.array([f["warmup_shrinks"] for f in per_frame])
    s_sh = np.array([f["s2_shrinks"] for f in per_frame])
    w_st = np.array([f["warmup_stalls"] for f in per_frame])
    s_st = np.array([f["s2_stalls"] for f in per_frame])
    struct = np.array([f["struct_infeas"] for f in per_frame])
    # qpos collected from last hand only is sufficient for Δqpos vs D comparison
    # (diffs are per-DOF and we just need a scalar summary)

    return {
        "clip": clip_id,
        "mode": mode_tag,
        "hands": list(hands),
        "T": T,
        "wall_s": wall,
        "fps": T / wall if wall > 0 else float("nan"),
        "qpos": qpos,  # last hand only; sufficient for Δqpos summary
        "pen_max_mm":  float(pens.max()),
        "pen_mean_mm": float(pens.mean()),
        "pen_p95_mm":  float(np.percentile(pens, 95)),
        "frames_with_pen": int((pens > 0.5).sum()),  # >0.5mm counts as meaningful
        "warmup_shrink_total": int(w_sh.sum()),
        "s2_shrink_total":     int(s_sh.sum()),
        "warmup_stall_total":  int(w_st.sum()),
        "s2_stall_total":      int(s_st.sum()),
        "struct_infeas_count": int(struct.sum()),
        "cache_path": str(cache_path),
    }


def compare_qpos(res_D: dict, res_X: dict) -> dict:
    qa, qb = res_D["qpos"], res_X["qpos"]
    diff_deg = np.degrees(qa - qb)
    return {
        "qpos_max_abs_deg":  float(np.abs(diff_deg).max()),
        "qpos_rms_deg":      float(np.sqrt((diff_deg ** 2).mean())),
        "qpos_mean_abs_deg": float(np.abs(diff_deg).mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="100f × 1 clip")
    parser.add_argument("--clips", type=str, default=None, help="Comma-separated clip IDs")
    parser.add_argument("--frames", type=int, default=None)
    args = parser.parse_args()

    if args.quick:
        args.frames = 100
        clips = CLIPS[:1]
    else:
        clips = args.clips.split(",") if args.clips else CLIPS

    print(f"Modes: {[m[0] for m in MODES]}")
    print(f"Clips: {clips}")
    print(f"Frames: {args.frames or 'all'}")
    print("=" * 60)

    all_results: list[dict] = []
    per_clip_summary: list[dict] = []

    for clip in clips:
        print(f"\n>>> CLIP {clip}")
        clip_results: dict[str, dict] = {}
        for tag, w, s in MODES:
            r = run_one(clip, tag, w, s, args.frames)
            clip_results[tag] = r
            all_results.append(r)
            print(f"  [{tag}] T={r['T']:3d} fps={r['fps']:6.1f}  "
                  f"pen max={r['pen_max_mm']:5.2f}mm mean={r['pen_mean_mm']:5.2f}mm "
                  f"np≥0.5mm={r['frames_with_pen']:3d}  "
                  f"w_sh={r['warmup_shrink_total']:3d} s_sh={r['s2_shrink_total']:3d} "
                  f"struct={r['struct_infeas_count']}")
        # qpos diffs vs D baseline
        for tag in ["A", "B", "C"]:
            clip_results[tag]["qpos_vs_D"] = compare_qpos(clip_results["D"], clip_results[tag])
        per_clip_summary.append({
            "clip": clip,
            **{tag: {k: v for k, v in clip_results[tag].items() if k != "qpos"}
               for tag in ["D", "A", "B", "C"]},
        })

    # Write report
    lines = ["# Penetration Hard-Constraint Ablation (EXP-13)\n\n"]
    lines.append(f"Run: clips={len(clips)}, frames={args.frames or 'all'}\n")
    lines.append(f"Modes: D (off), A (warmup only), B (S2 only), C (both)\n\n")
    lines.append(
        "## Per-clip penetration summary (mm)\n\n"
        "| Clip | Mode | T | fps | pen_max | pen_mean | pen_p95 | frames≥0.5mm | w_shrinks | s_shrinks | struct | Δqpos vs D (°) |\n"
        "|------|------|--:|----:|--------:|---------:|--------:|-------------:|----------:|----------:|-------:|---------------:|\n"
    )
    for s in per_clip_summary:
        short = s["clip"].split("__")[1] + "/" + s["clip"].split("__")[-1]
        for tag in ["D", "A", "B", "C"]:
            rr = s[tag]
            dq = rr.get("qpos_vs_D", {}).get("qpos_rms_deg", 0.0) if tag != "D" else 0.0
            lines.append(
                f"| {short} | {tag} | {rr['T']} | {rr['fps']:.1f} | "
                f"{rr['pen_max_mm']:.2f} | {rr['pen_mean_mm']:.2f} | {rr['pen_p95_mm']:.2f} | "
                f"{rr['frames_with_pen']} | {rr['warmup_shrink_total']} | "
                f"{rr['s2_shrink_total']} | {rr['struct_infeas_count']} | "
                f"{dq:.3f} |\n"
            )
    (OUT_DIR / "REPORT.md").write_text("".join(lines))
    (OUT_DIR / "summary.json").write_text(
        json.dumps(per_clip_summary, indent=2, default=str)
    )
    print(f"\nReport: {OUT_DIR / 'REPORT.md'}")
    print(f"JSON:   {OUT_DIR / 'summary.json'}")
    print(f"Cache stamps per mode: np-D / np-A / np-B / np-C in {CACHE_DIR}")


if __name__ == "__main__":
    main()
