"""Plot alignment comparison results: SVD vs wrist_q.

Reads alignment_comparison.csv and generates visualization plots.

Usage:
    PYTHONPATH=src python experiments/plot_alignment.py
"""

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_DIR / "experiments" / "alignment_comparison" / "alignment_comparison.csv"
OUTPUT_DIR = PROJECT_DIR / "experiments" / "alignment_comparison"


def load_data():
    results = []
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            for k in row:
                if k not in ("clip_id", "hand_side"):
                    row[k] = float(row[k])
            results.append(row)
    return results


def main():
    data = load_data()
    left = [r for r in data if r["hand_side"] == "left"]
    right = [r for r in data if r["hand_side"] == "right"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("SVD vs wrist_q Alignment Comparison (HO-Cap Full Dataset)", fontsize=14, fontweight="bold")

    # --- Plot 1: Frame 0 error scatter (SVD vs WQ) ---
    ax = axes[0, 0]
    ax.scatter([r["svd_f0_mm"] for r in left], [r["wq_f0_mm"] for r in left],
               c="blue", alpha=0.5, s=20, label=f"left ({len(left)})")
    ax.scatter([r["svd_f0_mm"] for r in right], [r["wq_f0_mm"] for r in right],
               c="red", alpha=0.5, s=20, label=f"right ({len(right)})")
    lim = max(max(r["svd_f0_mm"] for r in data), max(r["wq_f0_mm"] for r in data)) * 1.05
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3, label="equal")
    ax.set_xlabel("SVD frame 0 tip error (mm)")
    ax.set_ylabel("wrist_q frame 0 tip error (mm)")
    ax.set_title("Frame 0 Tip Error")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")

    # --- Plot 2: Avg 10 error scatter ---
    ax = axes[0, 1]
    ax.scatter([r["svd_avg10_mm"] for r in left], [r["wq_avg10_mm"] for r in left],
               c="blue", alpha=0.5, s=20, label="left")
    ax.scatter([r["svd_avg10_mm"] for r in right], [r["wq_avg10_mm"] for r in right],
               c="red", alpha=0.5, s=20, label="right")
    lim2 = max(max(r["svd_avg10_mm"] for r in data), max(r["wq_avg10_mm"] for r in data)) * 1.05
    ax.plot([0, lim2], [0, lim2], "k--", alpha=0.3)
    ax.set_xlabel("SVD avg 10 frames (mm)")
    ax.set_ylabel("wrist_q avg 10 frames (mm)")
    ax.set_title("Avg First 10 Frames Tip Error")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")

    # --- Plot 3: Angular difference histogram ---
    ax = axes[0, 2]
    angles_l = [r["angle_diff_deg"] for r in left if r["angle_diff_deg"] >= 0]
    angles_r = [r["angle_diff_deg"] for r in right if r["angle_diff_deg"] >= 0]
    bins = np.arange(0, max(max(angles_l, default=0), max(angles_r, default=0)) + 5, 3)
    ax.hist(angles_l, bins=bins, alpha=0.6, color="blue", label=f"left (mean={np.mean(angles_l):.1f}deg)")
    ax.hist(angles_r, bins=bins, alpha=0.6, color="red", label=f"right (mean={np.mean(angles_r):.1f}deg)")
    ax.set_xlabel("Angle between SVD and wrist_q (deg)")
    ax.set_ylabel("Count")
    ax.set_title("SVD vs wrist_q Orientation Difference")
    ax.legend(fontsize=8)

    # --- Plot 4: Delta f0 histogram (positive = SVD better) ---
    ax = axes[1, 0]
    delta_l = [r["delta_f0_mm"] for r in left]
    delta_r = [r["delta_f0_mm"] for r in right]
    bins_d = np.arange(min(min(delta_l, default=0), min(delta_r, default=0)) - 5,
                       max(max(delta_l, default=0), max(delta_r, default=0)) + 5, 5)
    ax.hist(delta_l, bins=bins_d, alpha=0.6, color="blue", label="left")
    ax.hist(delta_r, bins=bins_d, alpha=0.6, color="red", label="right")
    ax.axvline(0, color="k", linestyle="--", alpha=0.3)
    ax.set_xlabel("wrist_q - SVD frame 0 error (mm)")
    ax.set_ylabel("Count")
    ax.set_title("Frame 0 Delta (positive = SVD better)")
    ax.legend(fontsize=8)

    # --- Plot 5: Angle vs delta_f0 scatter ---
    ax = axes[1, 1]
    for subset, color, label in [(left, "blue", "left"), (right, "red", "right")]:
        angles = [r["angle_diff_deg"] for r in subset]
        deltas = [r["delta_f0_mm"] for r in subset]
        ax.scatter(angles, deltas, c=color, alpha=0.5, s=20, label=label)
    ax.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax.set_xlabel("SVD-wrist_q angle (deg)")
    ax.set_ylabel("wrist_q - SVD frame 0 error (mm)")
    ax.set_title("Orientation Gap vs Quality Gap")
    ax.legend(fontsize=8)

    # --- Plot 6: Per-subject box plot ---
    ax = axes[1, 2]
    subjects = sorted(set(r["clip_id"].split("__")[1] for r in data))
    box_data_svd = []
    box_data_wq = []
    labels = []
    for subj in subjects:
        s_clips = [r for r in data if subj in r["clip_id"]]
        if len(s_clips) < 3:
            continue
        box_data_svd.append([r["svd_f0_mm"] for r in s_clips])
        box_data_wq.append([r["wq_f0_mm"] for r in s_clips])
        labels.append(subj.replace("subject_", "S"))

    positions_svd = np.arange(len(labels)) * 3
    positions_wq = positions_svd + 1
    bp1 = ax.boxplot(box_data_svd, positions=positions_svd, widths=0.8,
                      patch_artist=True, boxprops=dict(facecolor="lightblue"))
    bp2 = ax.boxplot(box_data_wq, positions=positions_wq, widths=0.8,
                      patch_artist=True, boxprops=dict(facecolor="lightsalmon"))
    ax.set_xticks(positions_svd + 0.5)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Frame 0 tip error (mm)")
    ax.set_title("Per-Subject Frame 0 Error")
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ["SVD", "wrist_q"], fontsize=8)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "alignment_comparison.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
