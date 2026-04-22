"""A vs B anchor ablation on HO-Cap.

Configs:
  A   anchor_mode="l2"          (current: L2 anchor to q_S1, isotropic in q-space)
  B5  anchor_mode="cosik_live", w_rot=5   (same raw weight as warmup)
  B1  anchor_mode="cosik_live", w_rot=1   (weaker, gives IM more room)
  B05 anchor_mode="cosik_live", w_rot=0.5 (even weaker)

Metrics:
  tip_err_mm            robot tip → source landmark tip (hand correspondence)
  bone_cos_err          1 - mean cos over 20 bones (hand-shape preservation)
  dip_rev_rate          fraction of frames with any DIP angle < 0 (reversal)
  jitter                mean ||Δq_t|| (temporal smoothness)
  tip_obj_mm            robot tip → object surface distance, abs delta from source
  tip_obj_corr          temporal correlation of tip→obj distance series
  pinch_err_mm          |src pinch dist − robot pinch dist| for 4 thumb-finger pairs
  mcp_abd_range         max - min of MCP abduction angle distribution per finger
  q-deviation from A    mean ||q - q_A|| per frame
  fps                   throughput
"""

import sys
import time
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from hand_retarget import HandRetargetConfig, InteractionMeshHandRetargeter  # noqa: E402
from hand_retarget.config import JOINTS_MAPPING_LEFT  # noqa: E402
from hand_retarget.mediapipe_io import load_hocap_clip, transform_object_points  # noqa: E402


HOCAP_DIR = PROJECT_DIR / "data" / "hocap" / "hocap"
CLIP_ID = "hocap__subject_1__20231025_165502__seg00"
HAND_SIDE = "left"
SCENE = PROJECT_DIR / "assets" / "scenes" / "single_hand_obj_left.xml"

CHAINS = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12],
          [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]]
TIP_MPS = [4, 8, 12, 16, 20]
THUMB_TIP = 4
OTHER_TIPS = [8, 12, 16, 20]
FINGER_NAMES = ["thumb", "index", "mid", "ring", "pinky"]


def min_dist(p, pts):
    return float(np.linalg.norm(pts - p, axis=1).min())


def compute_metrics(qpos, lm_aligned, obj_aligned, ret):
    _jm = JOINTS_MAPPING_LEFT
    T = len(qpos)
    nq = qpos.shape[1]

    tip_errs, bone_cos_errs = [], []
    dip_rev = 0
    tip_obj_src = np.empty((T, 5))
    tip_obj_rob = np.empty((T, 5))
    pinch_errs = []  # per (frame, 4 pairs)

    for t in range(T):
        ret.hand.forward(qpos[t])
        # Tip pos and tip-to-obj distance
        rob_tips = np.empty((5, 3))
        for f, mp in enumerate(TIP_MPS):
            rob_tips[f] = ret.hand.get_body_pos(_jm[mp])
        for f, mp in enumerate(TIP_MPS):
            tip_errs.append(float(np.linalg.norm(rob_tips[f] - lm_aligned[t, mp])))
            tip_obj_src[t, f] = min_dist(lm_aligned[t, mp], obj_aligned[t])
            tip_obj_rob[t, f] = min_dist(rob_tips[f], obj_aligned[t])
        # Bone cos err
        for chain in CHAINS:
            for k in range(4):
                p_p = ret.hand.get_body_pos(_jm[chain[k]])
                p_c = ret.hand.get_body_pos(_jm[chain[k + 1]])
                e_r, e_s = p_c - p_p, lm_aligned[t, chain[k + 1]] - lm_aligned[t, chain[k]]
                nr, ns = np.linalg.norm(e_r), np.linalg.norm(e_s)
                if nr < 1e-8 or ns < 1e-8:
                    continue
                bone_cos_errs.append(1.0 - float((e_r / nr) @ (e_s / ns)))
        # Pinch distances: source vs robot
        for other in OTHER_TIPS:
            src_d = float(np.linalg.norm(lm_aligned[t, THUMB_TIP] - lm_aligned[t, other]))
            f_idx = OTHER_TIPS.index(other) + 1  # 1..4 (index..pinky)
            rob_d = float(np.linalg.norm(rob_tips[0] - rob_tips[f_idx]))
            pinch_errs.append(abs(src_d - rob_d))

    dip_off = 6 if nq > 20 else 0
    for t in range(T):
        for f in range(5):
            if qpos[t, dip_off + 4 * f + 3] < 0:
                dip_rev += 1
                break

    finger_sl = slice(6, 26) if nq > 20 else slice(0, 20)
    jitter = np.mean(np.linalg.norm(np.diff(qpos[:, finger_sl], axis=0), axis=1))

    tip_obj_err = np.abs(tip_obj_rob - tip_obj_src)  # (T, 5)
    tip_obj_corr = []
    for f in range(5):
        s, r = tip_obj_src[:, f], tip_obj_rob[:, f]
        if s.std() > 1e-8 and r.std() > 1e-8:
            tip_obj_corr.append(float(np.corrcoef(s, r)[0, 1]))

    # MCP abd range per finger
    mcp_abd_idx = [dip_off + 4 * f + 1 for f in range(5)]
    mcp_ranges = [float(qpos[:, j].max() - qpos[:, j].min()) for j in mcp_abd_idx]

    return {
        "tip_err_mm": 1000 * float(np.mean(tip_errs)),
        "bone_cos_err": float(np.mean(bone_cos_errs)),
        "dip_rev_rate": dip_rev / T,
        "jitter": float(jitter),
        "tip_obj_mm": 1000 * float(tip_obj_err.mean()),
        "tip_obj_p90_mm": 1000 * float(np.percentile(tip_obj_err, 90)),
        "tip_obj_corr": float(np.mean(tip_obj_corr)) if tip_obj_corr else float("nan"),
        "pinch_err_mm": 1000 * float(np.mean(pinch_errs)),
        "mcp_abd_range": mcp_ranges,
    }


def run_one(name, cfg, clip):
    ret = InteractionMeshHandRetargeter(cfg)
    t0 = time.time()
    qpos = ret.retarget_hocap_sequence(clip)
    elapsed = time.time() - t0

    # Aligned source + obj for metrics
    T = len(clip["landmarks"])
    lm_aligned = np.empty((T, 21, 3))
    obj_aligned = np.empty((T, len(clip["object_pts_local"]), 3))
    for t in range(T):
        obj_w = transform_object_points(clip["object_pts_local"],
                                         clip["object_q"][t], clip["object_t"][t])
        lm, obj = ret._align_frame(clip["landmarks"][t], None, obj_w)
        lm_aligned[t] = lm[:21]
        obj_aligned[t] = obj
    ret.q_lb[:6] = 0.0
    ret.q_ub[:6] = 0.0

    m = compute_metrics(qpos, lm_aligned, obj_aligned, ret)
    m["fps"] = T / elapsed
    return qpos, m


def main():
    clip = load_hocap_clip(
        str(HOCAP_DIR / "motions" / f"{CLIP_ID}.npz"),
        str(HOCAP_DIR / "motions" / f"{CLIP_ID}.meta.json"),
        str(HOCAP_DIR / "assets"),
        hand_side=HAND_SIDE, sample_count=50,
    )
    T = len(clip["landmarks"])
    print(f"\n=== A vs B (cosik_live) ablation on {CLIP_ID} [{T} frames] ===\n")

    def mk(mode, w_rot):
        cfg = HandRetargetConfig(mjcf_path=str(SCENE), hand_side=HAND_SIDE,
                                 floating_base=True, object_sample_count=50)
        cfg.anchor_mode = mode
        cfg.anchor_cosik_weight = w_rot
        return cfg

    configs = {
        "A_l2":              mk("l2",         5.0),
        "B_cosik_w5":        mk("cosik_live", 5.0),
        "B_cosik_w1":        mk("cosik_live", 1.0),
        "B_cosik_w0.5":      mk("cosik_live", 0.5),
    }

    results, qpos_all = {}, {}
    for name, cfg in configs.items():
        print(f"running {name} ...", flush=True)
        qpos, m = run_one(name, cfg, clip)
        results[name] = m
        qpos_all[name] = qpos

    qA = qpos_all["A_l2"]

    print("\n" + "─" * 100)
    print(f"{'config':<16} {'tip_mm':>7} {'bone':>7} {'dip_rev':>8} {'jit':>7} "
          f"{'obj_mm':>7} {'obj_p90':>8} {'obj_cor':>8} {'pinch_mm':>9} "
          f"{'‖q−qA‖':>9} {'fps':>6}")
    print("─" * 100)
    for name, m in results.items():
        dev = np.linalg.norm(qpos_all[name] - qA, axis=1).mean()
        print(f"{name:<16} {m['tip_err_mm']:>7.2f} {m['bone_cos_err']:>7.4f} "
              f"{m['dip_rev_rate']:>7.1%} {m['jitter']:>7.4f} "
              f"{m['tip_obj_mm']:>7.2f} {m['tip_obj_p90_mm']:>8.2f} "
              f"{m['tip_obj_corr']:>+8.3f} {m['pinch_err_mm']:>9.2f} "
              f"{dev:>9.4f} {m['fps']:>6.1f}")
    print("─" * 100)

    print("\n[MCP abduction dynamic range per finger (rad) — larger = more 'exploration']")
    print(f"{'config':<16}  " + "  ".join(f"{n:>8}" for n in FINGER_NAMES))
    for name, m in results.items():
        vals = m["mcp_abd_range"]
        print(f"{name:<16}  " + "  ".join(f"{v:>8.4f}" for v in vals))

    # Save
    out = PROJECT_DIR / "experiments" / "archive" / "warmup_diagnosis" / "probe_ab_cosik.npz"
    save = {}
    for name, qp in qpos_all.items():
        safe = name.replace(".", "_")
        save[f"qpos_{safe}"] = qp
    np.savez(out, **save)
    print(f"\nSaved: {out.relative_to(PROJECT_DIR)}")

    print("\n[How to read]")
    print("  A vs B: B should preserve bone_cos (hand shape) but change q differently")
    print("          and potentially improve pinch/obj geometry with IM's extra room.")
    print("  w_rot ↓: less bone-direction lock → IM has more influence")
    print("  Watch: tip_obj_mm and pinch_err_mm — these are IM's supposed strengths.")


if __name__ == "__main__":
    main()
