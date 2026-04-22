"""Re-analyze probe_s2_ablation results with object-side metrics.

Reads the .npz output from probe_s2_ablation.py (qpos for each of 5 configs) and
computes object-aware metrics that don't need ground-truth contact labels:

  tip_obj_dist_err_mm  — fingertip-to-nearest-obj-point distance: robot vs source
                          (measures whether robot preserves hand-object proximity)
  corr_dist_timeseries — Pearson correlation of per-frame robot vs source
                          tip-to-object distance series (does robot reproduce
                          the approach/retreat motion?)
  penetration_rate     — fraction of (frame, tip) pairs where robot tip is
                          'inside' the object sample cloud (crude proxy, since
                          sample points aren't a closed surface)
  delaunay_ho_edges    — mean count of hand-object edges in the Delaunay
                          adjacency each frame; if near zero, IM never uses
                          object info
  delaunay_ho_share    — hand-object edges as a fraction of total edges
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from hand_retarget import HandRetargetConfig, InteractionMeshHandRetargeter  # noqa: E402
from hand_retarget.config import JOINTS_MAPPING_LEFT  # noqa: E402
from hand_retarget.mediapipe_io import load_hocap_clip, transform_object_points  # noqa: E402
from hand_retarget.mesh_utils import (  # noqa: E402
    create_interaction_mesh, filter_adjacency_by_distance, get_adjacency_list, get_edge_list,
)

HOCAP_DIR = PROJECT_DIR / "data" / "hocap" / "hocap"
CLIP_ID = "hocap__subject_1__20231025_165502__seg00"
HAND_SIDE = "left"
SCENE = PROJECT_DIR / "assets" / "scenes" / "single_hand_obj_left.xml"
FRAME_LIMIT = 200

CHAINS = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12],
          [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]]
TIP_MPS = [4, 8, 12, 16, 20]

CONFIGS = ["A_current", "B_s2_only", "C_warm_no_anc", "D_warmup_only", "E_strong_anchor"]


def min_dist_to_set(p, points):
    """Minimum Euclidean distance from point p to a point cloud."""
    d = np.linalg.norm(points - p, axis=1)
    return float(d.min())


def main():
    npz_path = PROJECT_DIR / "experiments" / "archive" / "warmup_diagnosis" / "probe_s2_ablation.npz"
    saved = np.load(npz_path)

    qpos_all = {name: saved[f"qpos_{name}"] for name in CONFIGS}

    # Load clip for source landmarks and object pose
    clip = load_hocap_clip(
        str(HOCAP_DIR / "motions" / f"{CLIP_ID}.npz"),
        str(HOCAP_DIR / "motions" / f"{CLIP_ID}.meta.json"),
        str(HOCAP_DIR / "assets"),
        hand_side=HAND_SIDE, sample_count=50,
    )
    if FRAME_LIMIT:
        N = min(FRAME_LIMIT, len(clip["landmarks"]))
        pts_local = clip["object_pts_local"]
        clip = {k: v[:N] if isinstance(v, np.ndarray) and v.ndim > 0 and v.shape[0] >= N else v
                for k, v in clip.items()}
        clip["object_pts_local"] = pts_local

    T = len(clip["landmarks"])
    landmarks_raw = clip["landmarks"]

    # Build a retargeter to get FK + aligned source landmarks
    cfg = HandRetargetConfig(mjcf_path=str(SCENE), hand_side=HAND_SIDE,
                             floating_base=True, object_sample_count=50)
    ret = InteractionMeshHandRetargeter(cfg)
    ret.q_lb[:6] = 0.0
    ret.q_ub[:6] = 0.0

    # Per-frame aligned source landmarks and object points (in same aligned frame)
    obj_pts_local = clip["object_pts_local"]
    obj_q = clip["object_q"]
    obj_t = clip["object_t"]

    lm_aligned = np.empty((T, 21, 3))
    obj_aligned = np.empty((T, len(obj_pts_local), 3))
    for t in range(T):
        obj_world = transform_object_points(obj_pts_local, obj_q[t], obj_t[t])
        lm, obj = ret._align_frame(landmarks_raw[t], None, obj_world)
        lm_aligned[t] = lm[:21]
        obj_aligned[t] = obj

    # ── Source tip-to-object distance time series ──
    src_tip_obj = np.empty((T, 5))
    for t in range(T):
        for f, tip_mp in enumerate(TIP_MPS):
            src_tip_obj[t, f] = min_dist_to_set(lm_aligned[t, tip_mp], obj_aligned[t])

    # ── Delaunay hand-object edge analysis (topology-level, config-independent
    # since source points drive topology — same for all configs) ──
    ho_edge_counts = []
    total_edge_counts = []
    for t in range(T):
        all_pts = np.vstack([lm_aligned[t], obj_aligned[t]])
        try:
            _, simplices = create_interaction_mesh(all_pts)
        except Exception:
            continue
        adj = get_adjacency_list(simplices, len(all_pts))
        adj = filter_adjacency_by_distance(adj, all_pts, 0.06)
        edges = get_edge_list(adj)
        n_hand = 21
        if len(edges) == 0:
            ho_edge_counts.append(0)
            total_edge_counts.append(0)
            continue
        # Hand-object edge: one endpoint < 21, other ≥ 21
        is_ho = ((edges[:, 0] < n_hand) & (edges[:, 1] >= n_hand)) | \
                ((edges[:, 0] >= n_hand) & (edges[:, 1] < n_hand))
        ho_edge_counts.append(int(is_ho.sum()))
        total_edge_counts.append(int(len(edges)))

    ho_edges_arr = np.array(ho_edge_counts)
    total_edges_arr = np.array(total_edge_counts)
    ho_share = ho_edges_arr / np.clip(total_edges_arr, 1, None)

    print(f"\n=== Object-side metrics on {CLIP_ID} [{T} frames] ===\n")
    print("[Delaunay topology — hand-object edge utilization (config-independent)]")
    print(f"  total edges / frame         : mean={total_edges_arr.mean():.1f}  "
          f"median={np.median(total_edges_arr):.0f}")
    print(f"  hand-object edges / frame   : mean={ho_edges_arr.mean():.1f}  "
          f"median={np.median(ho_edges_arr):.0f}  max={ho_edges_arr.max()}")
    print(f"  HO share of edges           : mean={100*ho_share.mean():.1f}%  "
          f"median={100*np.median(ho_share):.1f}%")
    print(f"  frames with any HO edge     : {int((ho_edges_arr>0).sum())}/{T} "
          f"({100*(ho_edges_arr>0).mean():.0f}%)")
    print("  → if HO share is small, object points barely participate in IM;")
    print("    S2's 'object-aware' story rests on these edges existing.")

    # ── Per-config tip-to-object distance metrics ──
    print("\n[Per-config tip-to-object distance preservation]")
    print(f"{'config':<18} {'|Δd| mean mm':>14} {'|Δd| p90 mm':>14} "
          f"{'corr(src,rob)':>15} {'rob<src frames':>16}")
    print("-" * 82)

    results_for_save = {}
    for name in CONFIGS:
        qpos = qpos_all[name]
        rob_tip_obj = np.empty((T, 5))
        for t in range(T):
            ret.hand.forward(qpos[t])
            for f, tip_mp in enumerate(TIP_MPS):
                # Use same joints mapping as ret; robot fingertip position
                tip_body = JOINTS_MAPPING_LEFT[tip_mp]
                p_rob = ret.hand.get_body_pos(tip_body)
                rob_tip_obj[t, f] = min_dist_to_set(p_rob, obj_aligned[t])

        # Error: absolute difference (in the same frame)
        err = np.abs(rob_tip_obj - src_tip_obj)  # (T, 5)
        err_mm = 1000 * err

        # Correlation per finger, then average
        corrs = []
        for f in range(5):
            s = src_tip_obj[:, f]
            r = rob_tip_obj[:, f]
            if s.std() < 1e-8 or r.std() < 1e-8:
                continue
            corrs.append(float(np.corrcoef(s, r)[0, 1]))
        mean_corr = float(np.mean(corrs)) if corrs else float("nan")

        # How often robot is CLOSER to object than source (potential penetration proxy)
        rob_closer = int((rob_tip_obj < src_tip_obj).sum())
        total_pairs = T * 5
        rob_closer_pct = 100 * rob_closer / total_pairs

        print(f"{name:<18} {err_mm.mean():>14.2f} {np.percentile(err_mm, 90):>14.2f} "
              f"{mean_corr:>+15.3f} {rob_closer_pct:>15.1f}%")

        results_for_save[f"rob_tip_obj_{name}"] = rob_tip_obj
        results_for_save[f"err_mm_{name}"] = err_mm

    print("\n[Interpretation]")
    print("  |Δd|  = per-(frame, finger) difference in tip-to-object distance")
    print("  corr  = temporal correlation: does robot reproduce approach/retreat timing?")
    print("  rob<src frames: robot closer than source (proxy for over-penetration)")
    print("  Small |Δd| + high corr → robot preserves hand-object geometry (S2 doing its job).")

    out = PROJECT_DIR / "experiments" / "archive" / "warmup_diagnosis" / "analyze_object_metrics.npz"
    np.savez(out,
             src_tip_obj=src_tip_obj,
             ho_edges=ho_edges_arr, total_edges=total_edges_arr,
             ho_share=ho_share, **results_for_save)
    print(f"\nSaved: {out.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
