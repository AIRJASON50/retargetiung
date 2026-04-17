"""Analyze raw HO-Cap data quality: MediaPipe landmark penetration into object mesh.

Uses MuJoCo mj_geomDistance with a probe sphere at each landmark position
to compute signed distance to the object convex hull.

Usage:
    PYTHONPATH=src python experiments/analyze_data_penetration.py
    PYTHONPATH=src python experiments/analyze_data_penetration.py --max-clips 20
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hand_retarget.mediapipe_io import load_hocap_clip

PROJECT_DIR = Path(__file__).resolve().parents[1]
HOCAP_DIR = PROJECT_DIR / "data" / "hocap" / "hocap"

# MediaPipe landmark names
TIP_INDICES = [4, 8, 12, 16, 20]
TIP_NAMES = ["thumb", "index", "middle", "ring", "pinky"]
ALL_NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]


def build_probe_model(mesh_path: str):
    """Build minimal MuJoCo model: object mesh + probe sphere."""
    xml = f"""<mujoco>
      <asset><mesh name="obj" file="{mesh_path}"/></asset>
      <worldbody>
        <body name="obj" mocap="true">
          <geom name="obj_geom" type="mesh" mesh="obj" contype="1" conaffinity="1"/>
        </body>
        <body name="probe" mocap="true">
          <geom name="probe_geom" type="sphere" size="0.001" contype="2" conaffinity="2"/>
        </body>
      </worldbody>
    </mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    obj_gid = model.geom("obj_geom").id
    probe_gid = model.geom("probe_geom").id
    return model, data, obj_gid, probe_gid


def analyze_clip(clip_id: str, hand_side: str):
    """Analyze penetration of raw MediaPipe landmarks into object mesh."""
    npz_path = str(HOCAP_DIR / "motions" / f"{clip_id}.npz")
    meta_path = str(HOCAP_DIR / "motions" / f"{clip_id}.meta.json")

    clip = load_hocap_clip(npz_path, meta_path, str(HOCAP_DIR / "assets"),
                           hand_side=hand_side, sample_count=10)

    mesh_path = str(Path(clip["mesh_path"]).resolve())
    model, data, obj_gid, probe_gid = build_probe_model(mesh_path)
    fromto = np.zeros(6)

    landmarks = clip["landmarks"]  # (T, 21, 3)
    obj_t = clip["object_t"]      # (T, 3)
    obj_q = clip["object_q"]      # (T, 4)
    T = len(landmarks)

    # Per-frame, per-landmark signed distance
    all_phi = np.zeros((T, 21))

    for t in range(T):
        data.mocap_pos[0] = obj_t[t]
        data.mocap_quat[0] = [obj_q[t, 3], obj_q[t, 0], obj_q[t, 1], obj_q[t, 2]]

        for j in range(21):
            data.mocap_pos[1] = landmarks[t, j]
            data.mocap_quat[1] = [1, 0, 0, 0]
            mujoco.mj_forward(model, data)
            all_phi[t, j] = mujoco.mj_geomDistance(model, data, probe_gid, obj_gid, 0.3, fromto)

    # --- Aggregate metrics ---
    tips_phi = all_phi[:, TIP_INDICES]  # (T, 5)
    all_21_phi = all_phi                # (T, 21)

    # Per-tip statistics
    per_tip = {}
    for i, name in enumerate(TIP_NAMES):
        phi_seq = tips_phi[:, i]
        inside_mask = phi_seq < 0
        per_tip[name] = {
            "frames_inside": int(inside_mask.sum()),
            "pct_inside": float(inside_mask.mean() * 100),
            "max_depth_mm": float(-phi_seq.min() * 1000) if inside_mask.any() else 0.0,
            "mean_depth_when_inside_mm": float(-phi_seq[inside_mask].mean() * 1000) if inside_mask.any() else 0.0,
        }

    # Consecutive penetration streaks (longest continuous run of phi < 0)
    def longest_streak(mask):
        max_streak = 0
        current = 0
        for v in mask:
            if v:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0
        return max_streak

    # Overall clip statistics
    any_tip_inside = (tips_phi < 0).any(axis=1)  # (T,) — any tip inside per frame
    n_tips_inside = (tips_phi < 0).sum(axis=1)   # (T,) — how many tips inside per frame
    any_lm_inside = (all_21_phi < 0).any(axis=1)
    n_lm_inside = (all_21_phi < 0).sum(axis=1)

    result = {
        "clip_id": clip_id,
        "hand_side": hand_side,
        "object": clip.get("asset_name", "?"),
        "n_frames": T,
        # Tip penetration summary
        "frames_any_tip_inside": int(any_tip_inside.sum()),
        "pct_any_tip_inside": float(any_tip_inside.mean() * 100),
        "frames_3plus_tips_inside": int((n_tips_inside >= 3).sum()),
        "pct_3plus_tips_inside": float((n_tips_inside >= 3).mean() * 100),
        "max_tip_depth_mm": float(-tips_phi.min() * 1000) if (tips_phi < 0).any() else 0.0,
        "mean_tip_depth_when_inside_mm": float(-tips_phi[tips_phi < 0].mean() * 1000) if (tips_phi < 0).any() else 0.0,
        "longest_tip_penetration_streak": longest_streak(any_tip_inside),
        # All 21 landmarks
        "frames_any_lm_inside": int(any_lm_inside.sum()),
        "pct_any_lm_inside": float(any_lm_inside.mean() * 100),
        "max_lm_depth_mm": float(-all_21_phi.min() * 1000) if (all_21_phi < 0).any() else 0.0,
        "mean_n_lm_inside": float(n_lm_inside[any_lm_inside].mean()) if any_lm_inside.any() else 0.0,
        # Per-tip breakdown
        "per_tip": per_tip,
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-clips", type=int, default=None)
    args = parser.parse_args()

    # Discover all hand-clips
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

    if args.max_clips:
        all_tasks = all_tasks[:args.max_clips]

    print(f"Analyzing {len(all_tasks)} hand-clips for raw data penetration...")

    output_dir = PROJECT_DIR / "experiments" / "data_penetration"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    t0 = time.perf_counter()

    for i, (clip_id, hand_side) in enumerate(all_tasks):
        tag = f"{clip_id}__{hand_side}"
        try:
            r = analyze_clip(clip_id, hand_side)
            results.append(r)
            print(f"[{i+1}/{len(all_tasks)}] {tag}: "
                  f"tips_inside={r['pct_any_tip_inside']:.0f}%, "
                  f"max_depth={r['max_tip_depth_mm']:.0f}mm, "
                  f"streak={r['longest_tip_penetration_streak']}f")
        except Exception as e:
            print(f"[{i+1}/{len(all_tasks)}] ERROR {tag}: {e}")

    elapsed = time.perf_counter() - t0

    # --- Save CSV ---
    csv_path = output_dir / "penetration_stats.csv"
    fields = ["clip_id", "hand_side", "object", "n_frames",
              "pct_any_tip_inside", "pct_3plus_tips_inside",
              "max_tip_depth_mm", "mean_tip_depth_when_inside_mm",
              "longest_tip_penetration_streak",
              "pct_any_lm_inside", "max_lm_depth_mm", "mean_n_lm_inside"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r)

    # --- Save full JSON ---
    json_path = output_dir / "penetration_full.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # --- Summary ---
    print(f"\n{'='*70}")
    print(f"Analysis complete: {elapsed:.0f}s ({elapsed/max(len(all_tasks),1):.1f}s per clip)")
    print(f"{'='*70}")

    if not results:
        return

    pcts = [r["pct_any_tip_inside"] for r in results]
    depths = [r["max_tip_depth_mm"] for r in results]
    streaks = [r["longest_tip_penetration_streak"] for r in results]

    print(f"\nDataset-wide statistics ({len(results)} clips):")
    print(f"  Tip penetration rate:  mean={np.mean(pcts):.0f}%, median={np.median(pcts):.0f}%, range=[{np.min(pcts):.0f}%, {np.max(pcts):.0f}%]")
    print(f"  Max penetration depth: mean={np.mean(depths):.0f}mm, median={np.median(depths):.0f}mm, range=[{np.min(depths):.0f}mm, {np.max(depths):.0f}mm]")
    print(f"  Longest streak:        mean={np.mean(streaks):.0f}f, median={np.median(streaks):.0f}f, max={np.max(streaks)}f")

    # Distribution buckets
    print(f"\n  Penetration rate distribution:")
    for lo, hi in [(0, 10), (10, 30), (30, 50), (50, 70), (70, 100)]:
        n = sum(1 for p in pcts if lo <= p < hi)
        print(f"    {lo:3d}-{hi:3d}%: {n:3d} clips ({n/len(results)*100:.0f}%)")

    print(f"\n  Max depth distribution:")
    for lo, hi in [(0, 5), (5, 15), (15, 30), (30, 50), (50, 999)]:
        n = sum(1 for d in depths if lo <= d < hi)
        label = f"{lo}-{hi}mm" if hi < 999 else f"{lo}+mm"
        print(f"    {label:>8s}: {n:3d} clips ({n/len(results)*100:.0f}%)")

    # Best clips (lowest penetration)
    by_depth = sorted(results, key=lambda r: r["max_tip_depth_mm"])
    print(f"\n  Top 10 cleanest clips (lowest max penetration):")
    for r in by_depth[:10]:
        print(f"    {r['clip_id']}__{r['hand_side']}: depth={r['max_tip_depth_mm']:.1f}mm, "
              f"inside={r['pct_any_tip_inside']:.0f}%, obj={r['object']}")

    print(f"\n  Results: {csv_path}")
    print(f"  Full:    {json_path}")


if __name__ == "__main__":
    main()
