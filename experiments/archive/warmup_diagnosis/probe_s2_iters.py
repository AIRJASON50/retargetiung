"""Measure actual S2 iteration count per frame and cost-trace.

Checks:
  (1) Does first frame hit the 200 cap or early-converge?
  (2) Do subsequent frames hit the 10 cap or typically early-converge?
  (3) How does cost decay look? (is the 1e-3 cost-delta a tight or loose stop?)
  (4) Does q actually stop moving when cost-delta triggers? (ghost-convergence check)
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

s2_records = []  # list of dicts per frame: {is_first, iters_used, costs, q_deltas}


_orig_run_opt = R.InteractionMeshHandRetargeter._run_optimization


def patched(self, q_prev, landmarks, source_pts_full, adj_list, target_lap, sem_w,
            object_pts_world=None, obj_frame=None, object_pts_local=None,
            is_first_frame=False):
    n_iter = self.config.n_iter_first if is_first_frame else self.config.n_iter
    q_current = q_prev.copy()

    # S1 warmup
    if self.config.use_angle_warmup:
        lm_21 = landmarks[:21] if landmarks.shape[0] > 21 else landmarks
        max_warmup = (self.config.angle_warmup_iters_first if is_first_frame
                      else self.config.angle_warmup_iters)
        for _ in range(max_warmup):
            q_before = q_current.copy()
            q_current = self.solve_angle_warmup(q_current, q_prev, lm_21, n_iters=1)
            if np.linalg.norm(q_current - q_before) < CONVERGENCE_DELTA:
                break

    angle_targets_pair = None
    if self.config.use_angle_warmup and self.config.angle_anchor_weight > 0:
        angle_targets_pair = (q_current.copy(), self._build_anchor_confidence())

    # ======== Instrumented S2 loop (q-norm stop, matches new main code) ========
    solve_obj_pts = object_pts_local if obj_frame is not None else object_pts_world
    costs = []
    q_deltas = []
    iters_used = 0
    s2_conv = self.config.s2_convergence_delta

    for k in range(n_iter):
        q_before = q_current.copy()
        q_current, cost = self.solve_single_iteration(
            q_current, q_prev, target_lap, adj_list, sem_w,
            object_pts=solve_obj_pts, obj_frame=obj_frame,
            angle_targets=angle_targets_pair,
        )
        costs.append(float(cost))
        dq = float(np.linalg.norm(q_current - q_before))
        q_deltas.append(dq)
        iters_used = k + 1
        if dq < s2_conv:
            break

    s2_records.append({
        "is_first": is_first_frame,
        "iters_used": iters_used,
        "n_iter_cap": n_iter,
        "costs": costs,
        "q_deltas": q_deltas,
    })
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

    cfg = HandRetargetConfig(mjcf_path=str(SCENE), hand_side=HAND_SIDE,
                             floating_base=True, object_sample_count=50)
    ret = InteractionMeshHandRetargeter(cfg)
    ret.retarget_hocap_sequence(clip, use_semantic_weights=False)

    print(f"\n=== S2 iteration probe on {CLIP_ID} [{len(s2_records)} frames] ===")
    print(f"thresholds: cost_delta < {CONVERGENCE_DELTA}\n")

    first = s2_records[0]
    print(f"[First frame]  cap={first['n_iter_cap']}  iters_used={first['iters_used']}  "
          f"early_stop={'yes' if first['iters_used'] < first['n_iter_cap'] else 'NO (hit cap)'}")
    print(f"  costs[:5] = {['%.3e' % c for c in first['costs'][:5]]}")
    print(f"  costs[-5:]= {['%.3e' % c for c in first['costs'][-5:]]}")
    print(f"  q_deltas[:5]  = {['%.3e' % d for d in first['q_deltas'][:5]]}")
    print(f"  q_deltas[-5:] = {['%.3e' % d for d in first['q_deltas'][-5:]]}")

    subs = [r for r in s2_records if not r["is_first"]]
    iters_sub = np.array([r["iters_used"] for r in subs])
    caps_sub = np.array([r["n_iter_cap"] for r in subs])
    print(f"\n[Subsequent frames]  cap={int(caps_sub[0])}  n={len(iters_sub)}")
    for k in sorted(set(iters_sub.tolist())):
        n = int((iters_sub == k).sum())
        print(f"  {k} iter{'s' if k>1 else ' '}: {n:4d} frames  ({100*n/len(iters_sub):5.1f}%)")
    hit_cap = int((iters_sub == caps_sub).sum())
    print(f"\n  early stop  : {len(iters_sub) - hit_cap}/{len(iters_sub)} "
          f"({100*(len(iters_sub)-hit_cap)/len(iters_sub):.1f}%)")
    print(f"  hit cap ({caps_sub[0]}) : {hit_cap}/{len(iters_sub)} "
          f"({100*hit_cap/len(iters_sub):.1f}%)")
    print(f"  mean iters  : {iters_sub.mean():.2f}")

    # Ghost-convergence: when early-stop triggers, is q still moving?
    early_stop_final_qd = [r["q_deltas"][-1] for r in subs
                           if r["iters_used"] < r["n_iter_cap"]]
    if early_stop_final_qd:
        arr = np.array(early_stop_final_qd)
        print(f"\n[Early-stop ||Δq|| on last iter]  "
              f"median={np.median(arr):.3e}  p90={np.percentile(arr,90):.3e}  max={arr.max():.3e}")
        # 1e-3 is the S1 q-space threshold. Compare: is cost-stop matching q-stop?
        n_below_1e3 = int((arr < 1e-3).sum())
        print(f"   of {len(arr)} early-stops: {n_below_1e3} ({100*n_below_1e3/len(arr):.0f}%) "
              f"have ||Δq|| < 1e-3 (true convergence)")
        print(f"   remaining {len(arr)-n_below_1e3} have ||Δq|| ≥ 1e-3 "
              f"(GHOST convergence — cost stopped changing but q still moving)")

    # Cost trace for first frame
    print("\n[First-frame cost trace — first 20 iters]")
    for i, c in enumerate(first["costs"][:20]):
        qd = first["q_deltas"][i]
        print(f"  iter {i+1:3d}: cost={c:+.4e}  ||Δq||={qd:.3e}")

    out = PROJECT_DIR / "experiments" / "archive" / "warmup_diagnosis" / "probe_s2_iters.npz"
    save = {
        "iters_used": np.array([r["iters_used"] for r in s2_records]),
        "is_first": np.array([r["is_first"] for r in s2_records]),
        "cap": np.array([r["n_iter_cap"] for r in s2_records]),
    }
    np.savez(out, **save)
    print(f"\nSaved: {out.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
