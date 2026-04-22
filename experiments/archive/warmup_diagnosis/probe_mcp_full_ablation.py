"""Full MCP surrogate ablation:
  C1 link1 (default)
  C2 link2
  C3 midpoint
  C5a/b/c link2 + palm-normal offset (3/5/8 mm)
  T1 link2 + thumb CMC link2 (thumb audit)

Metrics:
  per-bone 1-cos residual (finger × bone) — aggregate and per-finger
  tip_err_mm, tip_obj_mm, dip_rev_rate, fps
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
from hand_retarget.retargeter import FINGER_CHAINS_MP  # noqa: E402


HOCAP_DIR = PROJECT_DIR / "data" / "hocap" / "hocap"
CLIP_ID = "hocap__subject_3__20231024_161306__seg00"
SCENES = {
    "left":  PROJECT_DIR / "assets" / "scenes" / "single_hand_obj_left.xml",
    "right": PROJECT_DIR / "assets" / "scenes" / "single_hand_obj.xml",
}
FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]
NON_THUMB = [1, 2, 3, 4]


def make_cfg(side: str, surrogate: str, offset_m: float, thumb_cmc: str):
    c = HandRetargetConfig(mjcf_path=str(SCENES[side]), hand_side=side,
                            floating_base=True, object_sample_count=50)
    c.mcp_surrogate = surrogate
    c.mcp_surface_offset_m = offset_m
    c.thumb_cmc_surrogate = thumb_cmc
    return c


def analyse(cfg, clip):
    ret = InteractionMeshHandRetargeter(cfg)
    t0 = time.time()
    qpos = ret.retarget_hocap_sequence(clip)
    elapsed = time.time() - t0
    ret.q_lb[:6] = 0.0
    ret.q_ub[:6] = 0.0

    obj_pts_local = clip["object_pts_local"]
    T = len(qpos)
    cos_err = np.zeros((T, 5, 4))
    tip_err = np.zeros((T, 5))
    tip_obj = np.zeros((T, 5))
    tip_obj_src = np.zeros((T, 5))

    for t in range(T):
        obj_world = transform_object_points(obj_pts_local, clip["object_q"][t], clip["object_t"][t])
        lm, obj_aligned = ret._align_frame(clip["landmarks"][t], None, obj_world)
        lm21 = lm[:21]
        ret.hand.forward(qpos[t])
        for f_idx, chain in enumerate(FINGER_CHAINS_MP):
            for k in range(4):
                rp, _ = ret._mp_body_pos_jacp(chain[k])
                rc, _ = ret._mp_body_pos_jacp(chain[k + 1])
                e_rob, e_src = rc - rp, lm21[chain[k + 1]] - lm21[chain[k]]
                lr, ls = np.linalg.norm(e_rob), np.linalg.norm(e_src)
                if lr < 1e-8 or ls < 1e-8:
                    continue
                cos_err[t, f_idx, k] = 1.0 - float((e_rob / lr) @ (e_src / ls))
            tip_body = JOINTS_MAPPING_LEFT[chain[4]] if cfg.hand_side == "left" \
                else JOINTS_MAPPING_LEFT[chain[4]].replace("left_", "right_")
            p_rob = ret.hand.get_body_pos(tip_body)
            tip_err[t, f_idx] = np.linalg.norm(p_rob - lm21[chain[4]])
            tip_obj[t, f_idx] = np.linalg.norm(p_rob - obj_aligned, axis=1).min()
            tip_obj_src[t, f_idx] = np.linalg.norm(lm21[chain[4]] - obj_aligned, axis=1).min()

    dip_rev = 0
    for t in range(T):
        for f in range(5):
            if qpos[t, 6 + 4 * f + 3] < 0:
                dip_rev += 1
                break

    return {
        "cos_err": cos_err,
        "tip_err_mm": 1000 * tip_err.mean(),
        "tip_obj_mm": 1000 * np.abs(tip_obj - tip_obj_src).mean(),
        "dip_rev_rate": dip_rev / T,
        "fps": T / elapsed,
    }


def main():
    npz = str(HOCAP_DIR / "motions" / f"{CLIP_ID}.npz")
    meta = str(HOCAP_DIR / "motions" / f"{CLIP_ID}.meta.json")

    # label → (mcp_surrogate, offset_m, thumb_cmc_surrogate)
    SPECS = {
        "C1_link1":        ("link1",    0.000, "link1"),
        "C2_link2":        ("link2",    0.000, "link1"),
        "C3_midpoint":     ("midpoint", 0.000, "link1"),
        "C5a_link2+3mm":   ("link2",    0.003, "link1"),
        "C5b_link2+5mm":   ("link2",    0.005, "link1"),
        "C5c_link2+8mm":   ("link2",    0.008, "link1"),
        "T1_link2+thumb2": ("link2",    0.000, "link2"),
    }

    results = {}
    for label, (surr, off, thumb) in SPECS.items():
        for side in ("left", "right"):
            key = f"{label}__{side}"
            print(f">>> {key}", flush=True)
            clip = load_hocap_clip(npz, meta, str(HOCAP_DIR / "assets"),
                                    hand_side=side, sample_count=50)
            cfg = make_cfg(side, surr, off, thumb)
            results[key] = analyse(cfg, clip)

    def pair(label, metric):
        return 0.5 * (results[f"{label}__left"][metric] + results[f"{label}__right"][metric])

    def bone_mean(label, bone_idx, finger_subset):
        vals = []
        for f_idx in finger_subset:
            for side in ("left", "right"):
                vals.append(results[f"{label}__{side}"]["cos_err"][:, f_idx, bone_idx].mean())
        return np.mean(vals)

    print("\n" + "=" * 98)
    print(f"FULL MCP SURROGATE ABLATION  [{CLIP_ID}, bimanual, 297 frames, both hands]")
    print("=" * 98)

    # ── Table A: non-thumb wrist→MCP per finger ──
    print("\n[A] wrist→MCP (掌骨方向) per non-thumb finger — 1-cos")
    print(f"{'config':<20} " + " ".join(f"{FINGER_NAMES[f]:>9}" for f in NON_THUMB))
    for label in SPECS:
        row = [bone_mean(label, 0, [f]) for f in NON_THUMB]
        print(f"{label:<20} " + " ".join(f"{v:>9.4f}" for v in row))

    # ── Table B: aggregate non-thumb bones ──
    print("\n[B] Aggregate non-thumb 4 fingers (both hands)")
    print(f"{'config':<20} {'wrist→MCP':>12} {'MCP→PIP':>12} {'PIP→DIP':>12} {'DIP→TIP':>12}")
    for label in SPECS:
        row = [bone_mean(label, k, NON_THUMB) for k in range(4)]
        print(f"{label:<20} " + " ".join(f"{v:>12.4f} (≈{np.degrees(np.arccos(np.clip(1-v,-1,1))):>4.1f}°)" for v in row))

    # ── Table C: thumb bones ──
    print("\n[C] Thumb bones (per chain position) — audits thumb_cmc_surrogate")
    for k, name in enumerate(["wrist→CMC", "CMC→MCP", "MCP→IP", "IP→TIP"]):
        print(f"  {name:<12} " + "  ".join(f"{label}={bone_mean(label, k, [0]):.4f}" for label in SPECS))

    # ── Table D: end-to-end ──
    print("\n[D] End-to-end metrics")
    print(f"{'config':<20} {'tip_mm':>9} {'tip_obj':>9} {'dip_rev':>9} {'fps':>6}")
    for label in SPECS:
        print(f"{label:<20} {pair(label, 'tip_err_mm'):>9.2f} "
              f"{pair(label, 'tip_obj_mm'):>9.2f} {pair(label, 'dip_rev_rate'):>8.1%} "
              f"{pair(label, 'fps'):>6.1f}")

    # ── Table E: improvement vs C1 baseline ──
    print("\n[E] Δ vs C1_link1 (negative = better)")
    print(f"{'config':<20} {'Δtip_mm':>9} {'Δobj_mm':>9} {'Δwrist→MCP(°)':>16}")
    base_tip = pair("C1_link1", "tip_err_mm")
    base_obj = pair("C1_link1", "tip_obj_mm")
    base_wmc = bone_mean("C1_link1", 0, NON_THUMB)
    base_wmc_deg = np.degrees(np.arccos(np.clip(1 - base_wmc, -1, 1)))
    for label in SPECS:
        if label == "C1_link1":
            continue
        d_tip = pair(label, "tip_err_mm") - base_tip
        d_obj = pair(label, "tip_obj_mm") - base_obj
        wmc = bone_mean(label, 0, NON_THUMB)
        wmc_deg = np.degrees(np.arccos(np.clip(1 - wmc, -1, 1)))
        d_wmc = wmc_deg - base_wmc_deg
        print(f"{label:<20} {d_tip:>+9.3f} {d_obj:>+9.3f} {d_wmc:>+16.3f}")

    out = PROJECT_DIR / "experiments" / "archive" / "warmup_diagnosis" / "probe_mcp_full_ablation.npz"
    save = {}
    for key, r in results.items():
        for k, v in r.items():
            save[f"{key}__{k}"] = np.asarray(v)
    np.savez(out, **save)
    print(f"\nSaved: {out.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
