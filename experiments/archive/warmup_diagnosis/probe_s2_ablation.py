"""S2 ablation on HO-Cap: how much does each stage contribute?

Runs 5 configurations end-to-end on one HO-Cap clip and compares outputs.

Configs:
  A_current       warmup=ON  anchor=5   S2=ON   (default joint optimization)
  B_s2_only       warmup=OFF anchor=0   S2=ON   (pure IM from q_prev)
  C_warm_no_anc   warmup=ON  anchor=0   S2=ON   (exp8's 'sequential' failure mode)
  D_warmup_only   warmup=ON  anchor=5   S2=OFF  (S1 is the final answer)
  E_strong_anchor warmup=ON  anchor=50  S2=ON   (over-anchored — S2 can barely move)

End-to-end metrics per config:
  tip_err_mm       mean fingertip-to-source-landmark distance (5 tips × N frames)
  bone_cos_err     1 - mean(cos) over 20 bones × N frames
  dip_rev_rate     fraction of frames with any DIP joint q < 0 (reversed)
  jitter           mean frame-to-frame ||Δq_t||
  ‖q − q_A‖        mean deviation from current config

Plus (on config A only): per-frame decomposition of q-space motion:
  ‖Δ_warmup‖    = ‖q_S1 − q_prev‖
  ‖Δ_S2‖        = ‖q_final − q_S1‖
  ratio         = ‖Δ_S2‖ / (‖Δ_warmup‖ + eps)
  cos(Δ_warmup, Δ_S2)  — are they aligned or perpendicular?
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from hand_retarget import HandRetargetConfig, InteractionMeshHandRetargeter  # noqa: E402
from hand_retarget.config import JOINTS_MAPPING_LEFT  # noqa: E402
from hand_retarget.mediapipe_io import load_hocap_clip, transform_object_points  # noqa: E402
import hand_retarget.retargeter as R  # noqa: E402


HOCAP_DIR = PROJECT_DIR / "data" / "hocap" / "hocap"
CLIP_ID = "hocap__subject_1__20231025_165502__seg00"
HAND_SIDE = "left"
SCENE = PROJECT_DIR / "assets" / "scenes" / "single_hand_obj_left.xml"
FRAME_LIMIT = 200

CHAINS = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12],
          [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]]
TIP_MPS = [4, 8, 12, 16, 20]


# -----------------------------------------------------------------------------
# Instrumentation: capture q_S1 and q_final per frame for config A
# -----------------------------------------------------------------------------

traces = {"q_prev": [], "q_S1": [], "q_final": []}
_trace_enabled = [False]
_orig_run_opt = R.InteractionMeshHandRetargeter._run_optimization


def traced_run_opt(self, q_prev, landmarks, source_pts_full, adj_list, target_lap,
                   sem_w, object_pts_world=None, obj_frame=None,
                   object_pts_local=None, is_first_frame=False):
    """Wraps _run_optimization to capture q_prev / q_S1 / q_final."""
    if not _trace_enabled[0]:
        return _orig_run_opt(self, q_prev, landmarks, source_pts_full, adj_list,
                             target_lap, sem_w, object_pts_world, obj_frame,
                             object_pts_local, is_first_frame)
    traces["q_prev"].append(q_prev.copy())

    n_iter = self.config.n_iter_first if is_first_frame else self.config.n_iter
    q_current = q_prev.copy()
    warmup_conv = self.config.warmup_convergence_delta
    s2_conv = self.config.s2_convergence_delta

    if self.config.use_angle_warmup:
        lm_21 = landmarks[:21] if landmarks.shape[0] > 21 else landmarks
        max_warmup = (self.config.angle_warmup_iters_first if is_first_frame
                      else self.config.angle_warmup_iters)
        for _ in range(max_warmup):
            q_before = q_current.copy()
            q_current = self.solve_angle_warmup(q_current, q_prev, lm_21, n_iters=1)
            if np.linalg.norm(q_current - q_before) < warmup_conv:
                break

    traces["q_S1"].append(q_current.copy())

    angle_targets_pair = None
    if self.config.use_angle_warmup and self.config.angle_anchor_weight > 0:
        angle_targets_pair = (q_current.copy(), self._build_anchor_confidence())

    solve_obj_pts = object_pts_local if obj_frame is not None else object_pts_world
    for _ in range(n_iter):
        q_before = q_current.copy()
        q_current, _cost = self.solve_single_iteration(
            q_current, q_prev, target_lap, adj_list, sem_w,
            object_pts=solve_obj_pts, obj_frame=obj_frame,
            angle_targets=angle_targets_pair,
        )
        if np.linalg.norm(q_current - q_before) < s2_conv:
            break

    traces["q_final"].append(q_current.copy())
    return q_current


R.InteractionMeshHandRetargeter._run_optimization = traced_run_opt


# -----------------------------------------------------------------------------
# Metric helpers
# -----------------------------------------------------------------------------


def compute_metrics(qpos_seq, landmarks_seq, ret):
    """Compute end-to-end metrics. qpos_seq: (T, nq). landmarks_seq: (T, 21, 3) aligned."""
    _jm = JOINTS_MAPPING_LEFT
    T = len(qpos_seq)

    tip_errs = []
    bone_cos_errs = []
    dip_reversed = 0

    for t in range(T):
        ret.hand.forward(qpos_seq[t])
        # Tip error
        for tip_mp in TIP_MPS:
            p_rob = ret.hand.get_body_pos(_jm[tip_mp])
            p_src = landmarks_seq[t, tip_mp]
            tip_errs.append(float(np.linalg.norm(p_rob - p_src)))
        # Bone cosine error
        for chain in CHAINS:
            for k in range(4):
                p_p = ret.hand.get_body_pos(_jm[chain[k]])
                p_c = ret.hand.get_body_pos(_jm[chain[k + 1]])
                e_r = p_c - p_p
                e_s = landmarks_seq[t, chain[k + 1]] - landmarks_seq[t, chain[k]]
                if np.linalg.norm(e_r) < 1e-8 or np.linalg.norm(e_s) < 1e-8:
                    continue
                d_r = e_r / np.linalg.norm(e_r)
                d_s = e_s / np.linalg.norm(e_s)
                bone_cos_errs.append(1.0 - float(d_r @ d_s))

    # DIP reversal (floating base: DIP joint indices 6+4f+3 for f=0..4)
    nq = qpos_seq.shape[1]
    is_floating = nq > 20
    dip_idx_off = 6 if is_floating else 0
    for t in range(T):
        for f in range(5):
            if qpos_seq[t, dip_idx_off + 4 * f + 3] < 0:
                dip_reversed += 1
                break

    # Jitter: inter-frame ||Δq|| over finger DOFs only
    finger_slice = slice(6, 26) if is_floating else slice(0, 20)
    jitter = np.mean(np.linalg.norm(np.diff(qpos_seq[:, finger_slice], axis=0), axis=1))

    return {
        "tip_err_mm": 1000 * np.mean(tip_errs),
        "bone_cos_err": float(np.mean(bone_cos_errs)),
        "dip_rev_rate": dip_reversed / T,
        "jitter": float(jitter),
    }


def make_config(floating=True, warmup=True, anchor=5.0, s2_on=True):
    cfg = HandRetargetConfig(mjcf_path=str(SCENE), hand_side=HAND_SIDE,
                             floating_base=floating, object_sample_count=50)
    cfg.use_angle_warmup = warmup
    cfg.angle_anchor_weight = anchor
    if not s2_on:
        cfg.n_iter = 0
        cfg.n_iter_first = 0
    return cfg


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def run_one(name, cfg, clip, trace=False):
    _trace_enabled[0] = trace
    if trace:
        for k in traces:
            traces[k].clear()
    ret = InteractionMeshHandRetargeter(cfg)
    import time
    t0 = time.time()
    qpos = ret.retarget_hocap_sequence(clip)
    elapsed = time.time() - t0

    # Build aligned landmarks for metric computation
    obj_pts_local = clip["object_pts_local"]
    obj_q = clip["object_q"]
    obj_t = clip["object_t"]
    landmarks_raw = clip["landmarks"]
    T = len(landmarks_raw)
    lm_aligned = np.empty((T, 21, 3))
    for t in range(T):
        obj_world = transform_object_points(obj_pts_local, obj_q[t], obj_t[t])
        lm, _ = ret._align_frame(landmarks_raw[t], None, obj_world)
        lm_aligned[t] = lm[:21]

    # IMPORTANT: must still lock wrist before metric FK, since retarget_hocap_sequence
    # restores q_lb/q_ub on exit. For metrics we need the same wrist-locked frame.
    ret.q_lb[:6] = 0.0
    ret.q_ub[:6] = 0.0

    metrics = compute_metrics(qpos, lm_aligned, ret)
    metrics["elapsed_s"] = elapsed
    metrics["fps"] = T / elapsed
    return qpos, metrics


def main():
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
    print(f"\n=== S2 ablation on {CLIP_ID} [{T} frames] ===\n")

    configs = {
        "A_current":       make_config(warmup=True,  anchor=5.0,  s2_on=True),
        "B_s2_only":       make_config(warmup=False, anchor=0.0,  s2_on=True),
        "C_warm_no_anc":   make_config(warmup=True,  anchor=0.0,  s2_on=True),
        "D_warmup_only":   make_config(warmup=True,  anchor=5.0,  s2_on=False),
        "E_strong_anchor": make_config(warmup=True,  anchor=50.0, s2_on=True),
    }

    all_qpos = {}
    all_metrics = {}
    for name, cfg in configs.items():
        qpos, metrics = run_one(name, cfg, clip, trace=(name == "A_current"))
        all_qpos[name] = qpos
        all_metrics[name] = metrics

    qA = all_qpos["A_current"]

    # ── Table 1: end-to-end metrics ──
    print("[End-to-end metrics per config]\n")
    print(f"{'config':<18} {'tip_mm':>8} {'bone_cos':>9} {'dip_rev':>9} "
          f"{'jitter':>9} {'‖q−qA‖':>9} {'fps':>7}")
    print("-" * 72)
    for name, metrics in all_metrics.items():
        dev_to_A = np.linalg.norm(all_qpos[name] - qA, axis=1).mean()
        print(f"{name:<18} {metrics['tip_err_mm']:>8.2f} "
              f"{metrics['bone_cos_err']:>9.4f} {metrics['dip_rev_rate']:>8.1%} "
              f"{metrics['jitter']:>9.4f} {dev_to_A:>9.4f} {metrics['fps']:>7.1f}")

    # ── Table 2: warmup vs S2 contribution (config A only) ──
    q_prev = np.stack(traces["q_prev"])
    q_S1 = np.stack(traces["q_S1"])
    q_final = np.stack(traces["q_final"])

    d_warmup = np.linalg.norm(q_S1 - q_prev, axis=1)
    d_s2 = np.linalg.norm(q_final - q_S1, axis=1)
    ratio = d_s2 / (d_warmup + 1e-12)

    cos_warm_s2 = []
    for t in range(len(q_prev)):
        a = q_S1[t] - q_prev[t]
        b = q_final[t] - q_S1[t]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-10 or nb < 1e-10:
            cos_warm_s2.append(np.nan)
        else:
            cos_warm_s2.append(float(a @ b) / (na * nb))
    cos_warm_s2 = np.array(cos_warm_s2)

    print("\n[q-space motion decomposition on A_current — first frame vs rest]\n")
    print(f"{'set':<10} {'‖Δ_warm‖':>10} {'‖Δ_S2‖':>10} {'S2/warm':>9} "
          f"{'cos(warm,S2)':>14}")
    print("-" * 60)
    print(f"{'frame 0':<10} {d_warmup[0]:>10.4f} {d_s2[0]:>10.4f} "
          f"{ratio[0]:>9.3f} {cos_warm_s2[0]:>+14.3f}")

    rest_slice = slice(1, None)
    for label, agg in [("median", np.nanmedian), ("p10", lambda x: np.nanpercentile(x, 10)),
                        ("p90", lambda x: np.nanpercentile(x, 90))]:
        print(f"rest {label:<5} {agg(d_warmup[rest_slice]):>10.4f} "
              f"{agg(d_s2[rest_slice]):>10.4f} "
              f"{agg(ratio[rest_slice]):>9.3f} "
              f"{agg(cos_warm_s2[rest_slice]):>+14.3f}")

    # Direction alignment distribution (rest only)
    rest_cos = cos_warm_s2[rest_slice]
    rest_cos = rest_cos[~np.isnan(rest_cos)]
    aligned = int((rest_cos > 0.5).sum())
    orth = int((np.abs(rest_cos) < 0.3).sum())
    oppose = int((rest_cos < -0.3).sum())
    print(f"\n  rest-frame cos alignment:  aligned(>0.5) {aligned}/{len(rest_cos)} "
          f"({100*aligned/len(rest_cos):.0f}%)  "
          f"orth(|·|<0.3) {orth} ({100*orth/len(rest_cos):.0f}%)  "
          f"oppose(<-0.3) {oppose} ({100*oppose/len(rest_cos):.0f}%)")

    print("\n[Interpretation cheatsheet]")
    print("  A vs D:  S2 adds on top of warmup-only  → how much IM refinement matters")
    print("  A vs C:  anchor vs no-anchor            → does anchor tame S2?")
    print("  A vs B:  with vs without warmup        → does warmup add over pure IM?")
    print("  A vs E:  current anchor vs strong       → is current anchor already tight?")
    print("  S2/warm ratio ≪ 1  →  warmup dominates q; S2 ≈ cosmetic polish")
    print("  cos(warm, S2) ≈ 0  →  S2 moves orthogonally to warmup (no cancel)")
    print("  cos < 0           →  S2 partly undoes warmup (bad: anchor too weak)")

    out = PROJECT_DIR / "experiments" / "archive" / "warmup_diagnosis" / "probe_s2_ablation.npz"
    save = {"q_prev": q_prev, "q_S1": q_S1, "q_final": q_final,
            "d_warmup": d_warmup, "d_s2": d_s2, "cos_warm_s2": cos_warm_s2}
    for name, qp in all_qpos.items():
        save[f"qpos_{name}"] = qp
    for name, m in all_metrics.items():
        for k, v in m.items():
            save[f"m_{name}_{k}"] = v
    np.savez(out, **save)
    print(f"\nSaved: {out.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
