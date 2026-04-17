"""Final report: separate penetration and alignment analyses with plots.

Usage:
    PYTHONPATH=src python experiments/report_final.py
"""

import csv
from pathlib import Path

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

# Register Chinese font via FontProperties (bypass rcParams cache issues)
_CN_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
_CN_FONT = fm.FontProperties(fname=_CN_FONT_PATH)
matplotlib.rcParams["axes.unicode_minus"] = False


def _set_cn(ax, title="", xlabel="", ylabel=""):
    """Set Chinese text on axes using explicit font."""
    if title:
        ax.set_title(title, fontproperties=_CN_FONT, fontsize=11)
    if xlabel:
        ax.set_xlabel(xlabel, fontproperties=_CN_FONT)
    if ylabel:
        ax.set_ylabel(ylabel, fontproperties=_CN_FONT)

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "experiments" / "reports"


def load_csv(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def report_penetration():
    """Report 1: Raw data penetration quality."""
    data = load_csv(PROJECT_DIR / "experiments" / "data_penetration" / "penetration_stats.csv")
    for r in data:
        for k in r:
            if k not in ("clip_id", "hand_side", "object"):
                r[k] = float(r[k])

    left = [r for r in data if r["hand_side"] == "left"]
    right = [r for r in data if r["hand_side"] == "right"]

    print("=" * 70)
    print("REPORT 1: HO-Cap Raw Data Penetration Analysis")
    print("=" * 70)
    print()
    print("Test: place 1mm probe sphere at each MediaPipe fingertip landmark,")
    print("measure signed distance to object mesh convex hull via mj_geomDistance.")
    print("phi < 0 = landmark inside object.")
    print()

    # Table 1: Overall stats
    print("--- Table 1: Overall Statistics ---")
    print(f"{'':>15s} | {'N':>4s} | {'pen_rate':>8s} {'med':>6s} | {'max_depth':>9s} {'med':>6s} | {'mean_depth':>10s} | {'streak':>6s}")
    print("-" * 80)
    for label, subset in [("ALL", data), ("LEFT", left), ("RIGHT", right)]:
        pcts = [r["pct_any_tip_inside"] for r in subset]
        depths = [r["max_tip_depth_mm"] for r in subset]
        means = [r["mean_tip_depth_when_inside_mm"] for r in subset]
        streaks = [r["longest_tip_penetration_streak"] for r in subset]
        print(f"{label:>15s} | {len(subset):4d} | {np.mean(pcts):6.1f}% {np.median(pcts):5.1f}% "
              f"| {np.mean(depths):7.1f}mm {np.median(depths):5.1f}mm "
              f"| {np.mean(means):8.1f}mm "
              f"| {np.mean(streaks):5.0f}f")
    print()

    # Table 2: Distribution
    print("--- Table 2: Penetration Rate Distribution ---")
    print(f"{'Bucket':>12s} | {'N':>4s} {'%':>5s} | {'avg_max_depth':>13s} | {'description':>20s}")
    print("-" * 65)
    descs = ["clean", "light", "moderate", "heavy", "severe"]
    for (lo, hi), desc in zip([(0, 10), (10, 30), (30, 50), (50, 70), (70, 100)], descs):
        b = [r for r in data if lo <= r["pct_any_tip_inside"] < hi]
        if not b:
            continue
        md = np.mean([r["max_tip_depth_mm"] for r in b])
        print(f"  {lo:3d}-{hi:3d}%   | {len(b):4d} {len(b)/len(data)*100:4.0f}% | {md:11.1f}mm | {desc}")
    print()

    # Table 3: Per-finger breakdown
    print("--- Table 3: Per-Finger Penetration (dataset-wide) ---")
    # Load full JSON for per-finger data
    import json
    full = json.load(open(PROJECT_DIR / "experiments" / "data_penetration" / "penetration_full.json"))
    fingers = ["thumb", "index", "middle", "ring", "pinky"]
    print(f"{'Finger':>10s} | {'clips_w_pen':>11s} | {'avg_%_inside':>12s} | {'avg_max_depth':>13s}")
    print("-" * 55)
    for fn in fingers:
        clips_pen = sum(1 for r in full if r["per_tip"][fn]["frames_inside"] > 0)
        avg_pct = np.mean([r["per_tip"][fn]["pct_inside"] for r in full])
        avg_depth = np.mean([r["per_tip"][fn]["max_depth_mm"] for r in full if r["per_tip"][fn]["max_depth_mm"] > 0]) if clips_pen > 0 else 0
        print(f"{fn:>10s} | {clips_pen:4d}/{len(full):<4d}  | {avg_pct:10.1f}% | {avg_depth:11.1f}mm")
    print()

    print("--- Conclusion ---")
    print("1. 37% of frames have at least one fingertip inside the object convex hull")
    print("2. Median max penetration depth = 17mm (far exceeds capsule radius 7.5mm)")
    print("3. Penetration is systematic, not noise — mean continuous streak = 100 frames")
    print("4. Left and right hands are equally affected (~34-39% penetration rate)")
    print("5. This is a MediaPipe 3D estimation limitation under hand-object occlusion")
    print()

    # --- Penetration plots ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Report 1: HO-Cap 原始数据穿透分析", fontproperties=_CN_FONT, fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.hist([r["pct_any_tip_inside"] for r in left], bins=20, alpha=0.6, color="blue", label=f"左手 ({len(left)})")
    ax.hist([r["pct_any_tip_inside"] for r in right], bins=20, alpha=0.6, color="red", label=f"右手 ({len(right)})")
    _set_cn(ax, "穿透率分布", "指尖在物体内部的帧占比 (%)", "轨迹数")
    ax.legend(fontsize=8, prop=_CN_FONT)

    ax = axes[1]
    ax.hist([r["max_tip_depth_mm"] for r in left], bins=20, alpha=0.6, color="blue", label="左手")
    ax.hist([r["max_tip_depth_mm"] for r in right], bins=20, alpha=0.6, color="red", label="右手")
    ax.axvline(7.5, color="green", linestyle="--", alpha=0.7, label="capsule 半径")
    _set_cn(ax, "最大穿透深度分布", "最大穿透深度 (mm)", "轨迹数")
    ax.legend(fontsize=8, prop=_CN_FONT)

    ax = axes[2]
    ax.scatter([r["pct_any_tip_inside"] for r in left], [r["max_tip_depth_mm"] for r in left],
               c="blue", alpha=0.4, s=15, label="左手")
    ax.scatter([r["pct_any_tip_inside"] for r in right], [r["max_tip_depth_mm"] for r in right],
               c="red", alpha=0.4, s=15, label="右手")
    ax.axhline(7.5, color="green", linestyle="--", alpha=0.5, label="capsule 半径")
    _set_cn(ax, "穿透率 vs 深度", "穿透率 (%)", "最大深度 (mm)")
    ax.legend(fontsize=8, prop=_CN_FONT)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "report1_penetration.png", dpi=150)
    print(f"Plot saved: {OUTPUT_DIR / 'report1_penetration.png'}")
    plt.close()


def report_alignment():
    """Report 2: SVD vs wrist_q alignment comparison."""
    data = load_csv(PROJECT_DIR / "experiments" / "alignment_comparison" / "alignment_comparison.csv")
    for r in data:
        for k in r:
            if k not in ("clip_id", "hand_side"):
                r[k] = float(r[k])

    left = [r for r in data if r["hand_side"] == "left"]
    right = [r for r in data if r["hand_side"] == "right"]

    print()
    print("=" * 70)
    print("REPORT 2: SVD vs wrist_q Alignment Comparison")
    print("=" * 70)
    print()
    print("Test: retarget first 10 frames with SVD+MANO vs wrist_q+MANO alignment,")
    print("measure fingertip position error in the aligned frame.")
    print()

    # Table 1: Summary
    print("--- Table 1: Summary ---")
    print(f"{'':>15s} | {'N':>4s} | {'SVD_f0':>7s} {'WQ_f0':>7s} {'delta':>7s} | {'SVD_avg10':>9s} {'WQ_avg10':>9s} {'delta':>7s} | {'SVD_win':>7s}")
    print("-" * 90)
    for label, subset in [("ALL", data), ("LEFT", left), ("RIGHT", right)]:
        sf = np.mean([r["svd_f0_mm"] for r in subset])
        wf = np.mean([r["wq_f0_mm"] for r in subset])
        sa = np.mean([r["svd_avg10_mm"] for r in subset])
        wa = np.mean([r["wq_avg10_mm"] for r in subset])
        sw = sum(1 for r in subset if r["svd_f0_mm"] < r["wq_f0_mm"])
        n = len(subset)
        print(f"{label:>15s} | {n:4d} | {sf:6.1f}mm {wf:6.1f}mm {wf-sf:+6.1f}mm "
              f"| {sa:8.1f}mm {wa:8.1f}mm {wa-sa:+6.1f}mm | {sw:4d}/{n}")
    print()

    # Table 2: Angle distribution
    print("--- Table 2: SVD-wrist_q Orientation Difference ---")
    print(f"{'Angle':>12s} | {'N':>4s} {'%':>5s} | {'avg_delta_f0':>12s} | {'interpretation':>25s}")
    print("-" * 70)
    interps = ["excellent match", "acceptable", "significant gap", "major mismatch", "near-orthogonal"]
    for (lo, hi), interp in zip([(0, 5), (5, 15), (15, 30), (30, 50), (50, 91)], interps):
        b = [r for r in data if lo <= r["angle_diff_deg"] < hi]
        if not b:
            continue
        avg_d = np.mean([r["delta_f0_mm"] for r in b])
        print(f"  {lo:3d}-{hi:3d} deg | {len(b):4d} {len(b)/len(data)*100:4.0f}% | {avg_d:+10.1f}mm | {interp}")
    print()

    # Table 3: Per-subject
    print("--- Table 3: Per-Subject Summary ---")
    subjects = sorted(set(r["clip_id"].split("__")[1] for r in data))
    print(f"{'Subject':>12s} | {'N':>4s} | {'SVD_f0':>7s} {'WQ_f0':>7s} | {'angle':>6s} | {'L/R':>5s}")
    print("-" * 55)
    for s in subjects:
        clips = [r for r in data if s in r["clip_id"]]
        if len(clips) < 3:
            continue
        sf = np.mean([r["svd_f0_mm"] for r in clips])
        wf = np.mean([r["wq_f0_mm"] for r in clips])
        ang = np.mean([r["angle_diff_deg"] for r in clips])
        ln = sum(1 for r in clips if r["hand_side"] == "left")
        rn = sum(1 for r in clips if r["hand_side"] == "right")
        print(f"{s:>12s} | {len(clips):4d} | {sf:6.1f}mm {wf:6.1f}mm | {ang:5.1f}d | {ln}L/{rn}R")
    print()

    print("--- Conclusion ---")
    print("1. SVD alignment is systematically better: wins 92% of clips (242/264)")
    print("2. RIGHT hand: SVD wins 100% (180/180), wrist_q is +32mm worse on average")
    print("3. LEFT hand: SVD wins 74%, gap is smaller (+7mm) — wrist_q occasionally competitive")
    print("4. Orientation gap (SVD vs wrist_q): mean=20deg, with outliers up to 91deg")
    print("5. Larger angle gap -> larger quality gap (linear correlation)")
    print("6. HO-Cap wrist_q data has systematic right-hand bias")
    print("7. Recommendation: SVD for retarget alignment, wrist_q for visualization only")
    print()

    # --- Alignment plots ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Report 2: SVD vs wrist_q 对齐对比", fontproperties=_CN_FONT, fontsize=13, fontweight="bold")

    # Plot 1: scatter SVD vs wrist_q
    ax = axes[0]
    ax.scatter([r["svd_f0_mm"] for r in left], [r["wq_f0_mm"] for r in left],
               c="blue", alpha=0.5, s=15, label=f"左手 ({len(left)})")
    ax.scatter([r["svd_f0_mm"] for r in right], [r["wq_f0_mm"] for r in right],
               c="red", alpha=0.5, s=15, label=f"右手 ({len(right)})")
    lim = max(max(r["svd_f0_mm"] for r in data), max(r["wq_f0_mm"] for r in data)) * 1.05
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3, label="相等线")
    _set_cn(ax, "首帧指尖误差: SVD vs wrist_q", "SVD 对齐 (mm)", "wrist_q 对齐 (mm)")
    ax.legend(fontsize=9, prop=_CN_FONT)
    ax.set_aspect("equal")

    # Plot 2: per-subject delta (wrist_q - SVD), split by hand
    ax = axes[1]
    subjects = sorted(set(r["clip_id"].split("__")[1] for r in data))
    x_pos = 0
    ticks, tick_labels = [], []
    for s in subjects:
        s_left = [r["delta_f0_mm"] for r in data if s in r["clip_id"] and r["hand_side"] == "left"]
        s_right = [r["delta_f0_mm"] for r in data if s in r["clip_id"] and r["hand_side"] == "right"]
        if s_left:
            bp = ax.boxplot([s_left], positions=[x_pos], widths=0.6, patch_artist=True,
                            boxprops=dict(facecolor="lightblue"), medianprops=dict(color="blue"))
            x_pos += 1
        if s_right:
            bp = ax.boxplot([s_right], positions=[x_pos], widths=0.6, patch_artist=True,
                            boxprops=dict(facecolor="lightsalmon"), medianprops=dict(color="red"))
            x_pos += 1
        ticks.append(x_pos - (2 if s_left and s_right else 1) + 0.5 * (1 if s_left and s_right else 0))
        tick_labels.append(s.replace("subject_", "S"))
        x_pos += 0.5
    ax.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, fontsize=8)
    _set_cn(ax, "各被试 wrist_q 相对 SVD 的误差增量", "", "wrist_q - SVD 误差 (mm)")
    # Manual legend
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="lightblue", label="左手"),
                       Patch(facecolor="lightsalmon", label="右手")],
              fontsize=9, prop=_CN_FONT)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "report2_alignment.png", dpi=150)
    print(f"Plot saved: {OUTPUT_DIR / 'report2_alignment.png'}")
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_penetration()
    report_alignment()


if __name__ == "__main__":
    main()
