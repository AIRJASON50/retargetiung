"""
EXP: Zhang 2023 edge-ratio cost vs Laplacian baseline.

Hypothesis:
  Replacing Laplacian coordinates with per-edge ratio alignment (Zhang 2023 style)
  reduces hyperextension by avoiding the "polluted mean neighbor" problem from
  long-range Delaunay edges. Normalization by ref edge length handles bone
  proportion mismatch without explicit bone scaling.

Design:
  - Both conditions use identical Delaunay topology and preprocessing.
  - Baseline: current Laplacian cost (uniform weight, fixed topology).
  - Edge-ratio: distance-filtered Delaunay edges (< threshold) + ratio cost
    + exponential distance weights. No Laplacian variable.

Multi-param sweep is supported: pass multiple --threshold / --kweight values
(comma-separated) to test several configurations in one run.

Usage:
    python experiments/exp_edge_ratio.py --frames 500
    python experiments/exp_edge_ratio.py --threshold 0.06,0.08 --kweight 20,40
    python experiments/exp_edge_ratio.py --no-cache
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).parent.parent
DATA_PKL  = Path("/home/l/ws/RL/retargeting/data/manus_for_pinch/manus1_5k.pkl")
MJCF_PATH = Path("/home/l/ws/doc/WujiRepo/wuji_retargeting_private/public/wuji_retargeting/wuji_hand_description/urdf/left.urdf")
CACHE_DIR = REPO / "experiments" / "edge_ratio_exp"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

import sys
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, "/home/l/ws/doc/WujiRepo/wuji_retargeting_private/public")

from hand_retarget.config import HandRetargetConfig
from hand_retarget.retargeter import InteractionMeshHandRetargeter
from hand_retarget.mediapipe_io import load_pkl_sequence, preprocess_sequence


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
PIP_INDICES = [2, 6, 10, 14, 18]   # PIP DOF indices in 20-DOF qpos
DIP_INDICES = [3, 7, 11, 15, 19]   # DIP DOF indices


def compute_metrics(qpos_seq: np.ndarray) -> dict:
    """Compute hyperextension rate and basic stats on a (T, 20) qpos sequence."""
    pip = qpos_seq[:, PIP_INDICES]
    dip = qpos_seq[:, DIP_INDICES]
    hyper_pip = (pip < 0).any(axis=1).mean()
    hyper_dip = (dip < 0).any(axis=1).mean()
    hyper_any = ((pip < 0) | (dip < 0)).any(axis=1).mean()
    return {
        "hyper_pip": hyper_pip,
        "hyper_dip": hyper_dip,
        "hyper_any": hyper_any,
        "pip_mean_deg": np.degrees(pip).mean(),
        "dip_mean_deg": np.degrees(dip).mean(),
    }


# ---------------------------------------------------------------------------
# retargeting runner
# ---------------------------------------------------------------------------
def run_retarget(cfg: HandRetargetConfig, landmarks_seq: np.ndarray, tag: str,
                 use_cache: bool = True) -> tuple[np.ndarray, float]:
    """Run retargeting for a config, return (qpos_seq, fps)."""
    cache_path = CACHE_DIR / f"{tag}_cache.npz"
    if use_cache and cache_path.exists():
        print(f"  [cache] loading {cache_path.name}")
        data = np.load(cache_path)
        return data["qpos"], float(data["fps"])

    retargeter = InteractionMeshHandRetargeter(cfg)
    q_prev = retargeter.hand.get_default_qpos()
    T = len(landmarks_seq)
    qpos_seq = np.zeros((T, retargeter.nq))

    t0 = time.perf_counter()
    for t, lm in enumerate(landmarks_seq):
        q_prev = retargeter.retarget_frame(
            lm, q_prev,
            is_first_frame=(t == 0),
            use_semantic_weights=True,
        )
        qpos_seq[t] = q_prev
    elapsed = time.perf_counter() - t0
    fps = T / elapsed

    np.savez(cache_path, qpos=qpos_seq, fps=fps)
    print(f"  [done ] {tag}: {fps:.1f} fps, {elapsed:.1f}s for {T} frames")
    return qpos_seq, fps


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Edge-ratio cost experiment")
    parser.add_argument("--frames", type=int, default=500)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--threshold", type=str, default="0.06",
                        help="Comma-separated edge distance thresholds (m), e.g. 0.05,0.06,0.08")
    parser.add_argument("--kweight", type=str, default="20.0",
                        help="Comma-separated exponential weight k values, e.g. 10,20,40")
    args = parser.parse_args()

    use_cache = not args.no_cache
    thresholds = [float(v) for v in args.threshold.split(",")]
    kweights   = [float(v) for v in args.kweight.split(",")]

    # Load and preprocess sequence
    print(f"Loading {DATA_PKL.name}, first {args.frames} frames ...")
    lm_seq, _ = load_pkl_sequence(str(DATA_PKL), hand_side="left")
    lm_seq    = preprocess_sequence(
        lm_seq, {"x": 0.0, "y": 0.0, "z": 15.0},
        hand_side="left", global_scale=1.0,
    )[:args.frames]
    print(f"  {len(lm_seq)} frames loaded")

    # -----------------------------------------------------------------------
    # Baseline: current Laplacian + semantic weights
    # -----------------------------------------------------------------------
    print("\n--- Baseline (Laplacian, fixed topology, semantic weights) ---")
    cfg_bl = HandRetargetConfig(
        mjcf_path=str(MJCF_PATH),
        hand_side="left",
        use_mano_rotation=True,
    )
    tag_bl = f"manus_{args.frames}f_laplacian"
    qpos_bl, fps_bl = run_retarget(cfg_bl, lm_seq, tag_bl, use_cache)
    m_bl = compute_metrics(qpos_bl)

    # -----------------------------------------------------------------------
    # Edge-ratio variants
    # -----------------------------------------------------------------------
    results = []
    for thr in thresholds:
        for kw in kweights:
            tag = f"manus_{args.frames}f_edgeratio_t{int(thr*1000)}mm_k{int(kw)}"
            print(f"\n--- Edge-ratio (threshold={thr*1000:.0f}mm, k={kw}) ---")
            cfg_er = HandRetargetConfig(
                mjcf_path=str(MJCF_PATH),
                hand_side="left",
                use_mano_rotation=True,
                use_edge_ratio_cost=True,
                edge_ratio_distance_threshold=thr,
                edge_ratio_k_weight=kw,
            )
            qpos_er, fps_er = run_retarget(cfg_er, lm_seq, tag, use_cache)
            m_er = compute_metrics(qpos_er)
            results.append((thr, kw, fps_er, m_er, tag))

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"{'Condition':<42}  {'FPS':>6}  {'Hyper%':>7}  {'PIP%':>6}  {'DIP%':>6}")
    print("-"*80)

    label_bl = f"Baseline (Laplacian)"
    print(f"{label_bl:<42}  {fps_bl:>6.1f}  {m_bl['hyper_any']*100:>6.1f}%  "
          f"{m_bl['hyper_pip']*100:>5.1f}%  {m_bl['hyper_dip']*100:>5.1f}%")

    for thr, kw, fps_er, m_er, tag in results:
        label = f"EdgeRatio t={thr*1000:.0f}mm k={kw:.0f}"
        delta_hyper = (m_er['hyper_any'] - m_bl['hyper_any']) * 100
        sign = "+" if delta_hyper >= 0 else ""
        print(f"{label:<42}  {fps_er:>6.1f}  {m_er['hyper_any']*100:>6.1f}%  "
              f"{m_er['hyper_pip']*100:>5.1f}%  {m_er['hyper_dip']*100:>5.1f}%"
              f"  (Δ={sign}{delta_hyper:.1f}pp)")

    print("="*80)

    # Quick edge count info (diagnostic)
    print("\n[Diagnostic] Edge count info at first frame:")
    from hand_retarget.mesh_utils import create_interaction_mesh, get_adjacency_list, get_edge_list
    _, simplices = create_interaction_mesh(lm_seq[0])
    adj = get_adjacency_list(simplices, 21)
    all_edges = get_edge_list(adj)
    print(f"  Total Delaunay edges: {len(all_edges)}")
    for thr in thresholds:
        dists = [np.linalg.norm(lm_seq[0][j] - lm_seq[0][i]) for i, j in all_edges]
        kept = sum(1 for d in dists if d < thr)
        print(f"  Kept with threshold={thr*1000:.0f}mm: {kept}/{len(all_edges)} edges "
              f"({100*kept/len(all_edges):.0f}%)")


if __name__ == "__main__":
    main()
