"""Probe: how many iterations does the cosine-IK warmup actually use per frame?

Reports: distribution of used iters (1..5), per-iter ||Δq||, tail (non-converged) rate.
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

CONVERGENCE_DELTA = 1e-3

iter_used = []
delta_trace = []


_orig_run_opt = R.InteractionMeshHandRetargeter._run_optimization


def patched(self, q_prev, landmarks, source_pts_full, adj_list, target_lap, sem_w,
            object_pts_world=None, obj_frame=None, object_pts_local=None,
            is_first_frame=False):
    n_iter = self.config.n_iter_first if is_first_frame else self.config.n_iter
    q_current = q_prev.copy()

    if self.config.use_angle_warmup:
        lm_21 = landmarks[:21] if landmarks.shape[0] > 21 else landmarks
        max_warmup = (self.config.angle_warmup_iters_first if is_first_frame
                      else self.config.angle_warmup_iters)
        deltas = []
        used = 0
        for k in range(max_warmup):
            q_before = q_current.copy()
            q_current = self.solve_angle_warmup(q_current, q_prev, lm_21, n_iters=1)
            d = float(np.linalg.norm(q_current - q_before))
            deltas.append(d)
            used = k + 1
            if d < CONVERGENCE_DELTA:
                break
        iter_used.append(used)
        delta_trace.append(deltas)

    angle_targets_pair = None
    if self.config.use_angle_warmup and self.config.angle_anchor_weight > 0:
        angle_targets_pair = (q_current.copy(), self._build_anchor_confidence())

    solve_obj_pts = object_pts_local if obj_frame is not None else object_pts_world
    s2_conv = self.config.s2_convergence_delta
    for _ in range(n_iter):
        q_before_s2 = q_current.copy()
        q_current, _cost = self.solve_single_iteration(
            q_current, q_prev, target_lap, adj_list, sem_w,
            object_pts=solve_obj_pts, obj_frame=obj_frame,
            angle_targets=angle_targets_pair,
        )
        if np.linalg.norm(q_current - q_before_s2) < s2_conv:
            break

    return q_current


R.InteractionMeshHandRetargeter._run_optimization = patched


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

    cfg = HandRetargetConfig(
        mjcf_path=str(SCENE), hand_side=HAND_SIDE,
        floating_base=True, object_sample_count=50,
    )
    ret = InteractionMeshHandRetargeter(cfg)
    ret.retarget_hocap_sequence(clip, use_semantic_weights=False)

    iters = np.array(iter_used)
    max_slot = max(int(iters.max()), 5)
    cap_first = cfg.angle_warmup_iters_first
    cap_rest = cfg.angle_warmup_iters
    print(f"\n=== Warmup convergence probe on {CLIP_ID} [{len(iters)} frames] ===")
    print(f"clip fps={clip['fps']}, threshold={CONVERGENCE_DELTA}, "
          f"cap={cap_first} (first) / {cap_rest} (rest)\n")

    print("Distribution of iterations used:")
    for k in range(1, max_slot + 1):
        n = int((iters == k).sum())
        if n == 0 and k > 5:
            continue
        pct = 100.0 * n / len(iters)
        print(f"  {k} iter{'s' if k>1 else ' '}: {n:4d} frames  ({pct:5.1f}%)")

    # First frame treated separately — cap may be larger
    print(f"\n  first frame iters used : {iters[0]}  (cap {cap_first})")
    rest = iters[1:]
    hit_cap = int((rest == cap_rest).sum())
    print(f"  rest: hit cap ({cap_rest}) : {hit_cap}/{len(rest)}  "
          f"({100*hit_cap/len(rest):.1f}%)")
    print(f"  rest: mean iters        : {rest.mean():.2f}")

    print("\nPer-iter ||Δq|| (median / p90 across frames):")
    for k in range(max_slot):
        vals = [d[k] for d in delta_trace if len(d) > k]
        if not vals:
            continue
        vals = np.asarray(vals)
        print(f"  iter {k+1:2d}: n={len(vals):4d}  median={np.median(vals):.4e}  "
              f"p90={np.percentile(vals, 90):.4e}  max={vals.max():.4e}")

    out = PROJECT_DIR / "experiments" / "archive" / "warmup_diagnosis" / "probe_convergence.npz"
    # pad delta_trace to max_slot for a rectangular array
    pad = max_slot
    np.savez(out, iter_used=iters,
             delta_trace=np.array([d + [np.nan]*(pad-len(d)) for d in delta_trace]))
    print(f"\nSaved: {out.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
