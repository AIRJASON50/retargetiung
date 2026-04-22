"""HISTORICAL: this probe monkey-patched ``_extract_cosik_targets`` — a method
that was DELETED from retargeter.py after this probe proved it was a zombie
(99.4% of frames ran 1 redundant cosine-IK iter producing ||Δq|| ≈ 3e-5).

Running this script now will fail with AttributeError. The .npz sibling file
(``probe_extract_targets.npz``) preserves the original measurements.

Original question: is `_extract_cosik_targets` doing anything, or is it a zombie?

Measured per frame:
  - q_S1  = q after warmup convergence loop
  - q_tgt = q_target from _extract_cosik_targets(q_S1)
  - ||q_tgt - q_S1||
  - extra iters used inside _extract_cosik_targets
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from hand_retarget import HandRetargetConfig, InteractionMeshHandRetargeter  # noqa: E402
from hand_retarget.mediapipe_io import load_hocap_clip  # noqa: E402
import hand_retarget.retargeter as R  # noqa: E402

HOCAP_DIR = PROJECT_DIR / "data" / "hocap" / "hocap"
CLIP_ID = "hocap__subject_1__20231025_165502__seg00"
HAND_SIDE = "left"
SCENE = PROJECT_DIR / "assets" / "scenes" / "single_hand_obj_left.xml"
FRAME_LIMIT = 200
CONV = 1e-3

records = []

_orig_extract = R.InteractionMeshHandRetargeter._extract_cosik_targets


def patched_extract(self, q_current, q_prev, landmarks_21):
    """Same body as original but counts iters and tracks ||Δq||."""
    q_S1 = q_current.copy()
    q_target = q_current.copy()
    deltas = []
    iters_used = 0
    for k in range(3):
        q_before = q_target.copy()
        q_target = self.solve_angle_warmup(q_target, q_prev, landmarks_21, n_iters=1)
        d = float(np.linalg.norm(q_target - q_before))
        deltas.append(d)
        iters_used = k + 1
        if d < CONV:
            break

    # Replicate confidence construction
    confidence = np.ones(self.nq)
    if self.config.floating_base and self.nq > 20:
        confidence[:3] = 0.0
        confidence[3:6] = 0.5
        for f in range(5):
            confidence[6 + 4 * f + 1] = 0.5
    else:
        for f in range(5):
            confidence[4 * f + 1] = 0.5

    records.append({
        "q_S1": q_S1,
        "q_target": q_target,
        "total_drift": float(np.linalg.norm(q_target - q_S1)),
        "iter_deltas": deltas,
        "iters_used": iters_used,
    })

    return q_target, confidence


R.InteractionMeshHandRetargeter._extract_cosik_targets = patched_extract


def main():
    npz = str(HOCAP_DIR / "motions" / f"{CLIP_ID}.npz")
    meta = str(HOCAP_DIR / "motions" / f"{CLIP_ID}.meta.json")
    clip = load_hocap_clip(npz, meta, str(HOCAP_DIR / "assets"),
                           hand_side=HAND_SIDE, sample_count=50)
    if FRAME_LIMIT:
        N = min(FRAME_LIMIT, len(clip["landmarks"]))
        pts_local = clip["object_pts_local"]
        clip = {k: v[:N] if isinstance(v, np.ndarray) and v.ndim > 0 and v.shape[0] >= N else v
                for k, v in clip.items()}
        clip["object_pts_local"] = pts_local

    cfg = HandRetargetConfig(mjcf_path=str(SCENE), hand_side=HAND_SIDE,
                             floating_base=True, object_sample_count=50)
    ret = InteractionMeshHandRetargeter(cfg)
    ret.retarget_hocap_sequence(clip, use_semantic_weights=False)

    drifts = np.array([r["total_drift"] for r in records])
    iters = np.array([r["iters_used"] for r in records])

    print(f"\n=== _extract_cosik_targets redundancy probe [{len(records)} frames] ===\n")

    print("[Drift ||q_target - q_S1||  — how much did 3 extra iters add after warmup?]")
    print(f"  median   {np.median(drifts):.4e}")
    print(f"  p50      {np.percentile(drifts,50):.4e}")
    print(f"  p90      {np.percentile(drifts,90):.4e}")
    print(f"  p99      {np.percentile(drifts,99):.4e}")
    print(f"  max      {drifts.max():.4e}")
    print(f"  frames with drift < 1e-4 (basically no-op):  {int((drifts<1e-4).sum())}/{len(drifts)} "
          f"({100*(drifts<1e-4).mean():.1f}%)")
    print(f"  frames with drift < 1e-3 (anchor threshold): {int((drifts<1e-3).sum())}/{len(drifts)} "
          f"({100*(drifts<1e-3).mean():.1f}%)")

    print("\n[Iterations used inside _extract_cosik_targets]")
    for k in range(1, 4):
        n = int((iters == k).sum())
        print(f"  {k} iter{'s' if k>1 else ' '}: {n:4d}  ({100*n/len(iters):5.1f}%)")
    print(f"  mean: {iters.mean():.2f}")

    # The first iter's Δ tells us: did warmup already converge?
    first_delta = np.array([r["iter_deltas"][0] for r in records])
    print("\n[First iter's ||Δq|| — tells us whether warmup already converged]")
    print(f"  median   {np.median(first_delta):.4e}")
    print(f"  p90      {np.percentile(first_delta,90):.4e}")
    print(f"  max      {first_delta.max():.4e}")
    n_conv = int((first_delta < CONV).sum())
    print(f"  warmup already converged (first iter δ < {CONV}): {n_conv}/{len(first_delta)} "
          f"({100*n_conv/len(first_delta):.1f}%)  → _extract_cosik_targets would break on iter 1")

    # Frame-0 special case
    print(f"\n[Frame 0 detail — the only frame warmup failed to converge]")
    print(f"  q_S1 → q_target drift  = {records[0]['total_drift']:.4e}")
    print(f"  iter_deltas            = {[f'{d:.2e}' for d in records[0]['iter_deltas']]}")
    print(f"  iters_used             = {records[0]['iters_used']}")

    out = PROJECT_DIR / "experiments" / "archive" / "warmup_diagnosis" / "probe_extract_targets.npz"
    np.savez(out, drifts=drifts, iters_used=iters, first_delta=first_delta)
    print(f"\nSaved: {out.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
