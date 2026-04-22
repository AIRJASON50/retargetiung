"""MCP surrogate candidate shootout: link1 (C1, current) vs link2 (C2) vs midpoint (C3).

For each candidate, retargets the full HO-Cap subject_3 clip (both hands) and
reports per-bone residual metrics plus end-to-end quality indicators.

Primary axis: ``wrist→MCP`` bone residual (should drop if candidate matches
MediaPipe's ball-joint MCP semantics better than the flex-pivot link1 default).

Secondary:
  MCP→PIP bone residual         (proximal phalanx direction — should be cleaner
                                 when compound offset isn't baked in)
  src/rob length ratio variance (bone-length consistency across fingers)
  tip_err                        (end-to-end Cartesian tip alignment)
  tip_obj_mm                     (hand-object relative geometry)
  dip_rev_rate                   (reversal as proxy for IM stress)
  fps                            (C3 pays one extra FK + 1 Jacobian per MCP)
"""

import sys
import time
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from hand_retarget import HandRetargetConfig, InteractionMeshHandRetargeter  # noqa: E402
from hand_retarget.config import JOINTS_MAPPING_LEFT, JOINTS_MAPPING_RIGHT  # noqa: E402
from hand_retarget.mediapipe_io import load_hocap_clip, transform_object_points  # noqa: E402
from hand_retarget.retargeter import FINGER_CHAINS_MP  # noqa: E402


HOCAP_DIR = PROJECT_DIR / "data" / "hocap" / "hocap"
CLIP_ID = "hocap__subject_3__20231024_161306__seg00"

SCENES = {
    "left":  PROJECT_DIR / "assets" / "scenes" / "single_hand_obj_left.xml",
    "right": PROJECT_DIR / "assets" / "scenes" / "single_hand_obj.xml",
}

CANDIDATES = ("link1", "link2", "midpoint")
FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]
NON_THUMB_FINGER_IDX = [1, 2, 3, 4]  # only these have MCP surrogate choice


def analyse(hand_side: str, clip: dict, mcp_mode: str):
    cfg = HandRetargetConfig(
        mjcf_path=str(SCENES[hand_side]), hand_side=hand_side,
        floating_base=True, object_sample_count=50,
    )
    cfg.mcp_surrogate = mcp_mode
    ret = InteractionMeshHandRetargeter(cfg)

    t0 = time.time()
    qpos = ret.retarget_hocap_sequence(clip)
    elapsed = time.time() - t0

    ret.q_lb[:6] = 0.0
    ret.q_ub[:6] = 0.0

    _jm = JOINTS_MAPPING_LEFT if hand_side == "left" else JOINTS_MAPPING_RIGHT
    obj_pts_local = clip["object_pts_local"]
    T = len(qpos)

    cos_err = np.zeros((T, 5, 4))          # finger × bone
    bone_len_rob = np.zeros((T, 5, 4))
    bone_len_src = np.zeros((T, 5, 4))
    tip_err = np.zeros((T, 5))             # m
    tip_obj = np.zeros((T, 5))             # rob tip to obj surface
    tip_obj_src = np.zeros((T, 5))         # src tip to obj surface

    for t in range(T):
        obj_world = transform_object_points(obj_pts_local, clip["object_q"][t], clip["object_t"][t])
        lm, obj_aligned = ret._align_frame(clip["landmarks"][t], None, obj_world)
        lm21 = lm[:21]

        ret.hand.forward(qpos[t])
        for f_idx, chain in enumerate(FINGER_CHAINS_MP):
            for k in range(4):
                # Note: for the residual metric we use the CURRENT candidate's
                # body choice (same logic that the solver saw), so we're asking
                # "given this MCP surrogate, how well does the solver hit it?"
                rp, _ = ret._mp_body_pos_jacp(chain[k])
                rc, _ = ret._mp_body_pos_jacp(chain[k + 1])
                e_rob = rc - rp
                lr = np.linalg.norm(e_rob)
                e_src = lm21[chain[k + 1]] - lm21[chain[k]]
                ls = np.linalg.norm(e_src)
                if lr < 1e-8 or ls < 1e-8:
                    continue
                d_rob = e_rob / lr
                d_src = e_src / ls
                cos_err[t, f_idx, k] = 1.0 - float(d_rob @ d_src)
                bone_len_rob[t, f_idx, k] = lr
                bone_len_src[t, f_idx, k] = ls

            # Tip position error (always uses tip_link mapping, not MCP surrogate)
            tip_body = _jm[chain[4]]
            p_rob = ret.hand.get_body_pos(tip_body)
            tip_err[t, f_idx] = np.linalg.norm(p_rob - lm21[chain[4]])
            tip_obj[t, f_idx] = np.linalg.norm(p_rob - obj_aligned, axis=1).min()
            tip_obj_src[t, f_idx] = np.linalg.norm(lm21[chain[4]] - obj_aligned, axis=1).min()

    # DIP reversal rate
    dip_off = 6  # floating-base wrist DOFs
    dip_rev = 0
    for t in range(T):
        for f in range(5):
            if qpos[t, dip_off + 4 * f + 3] < 0:
                dip_rev += 1
                break

    tip_obj_err = np.abs(tip_obj - tip_obj_src)

    return {
        "cos_err": cos_err,                     # (T, 5, 4)
        "bone_len_rob": bone_len_rob,
        "bone_len_src": bone_len_src,
        "tip_err_mm": 1000 * tip_err.mean(),
        "tip_obj_mm": 1000 * tip_obj_err.mean(),
        "tip_obj_p90_mm": 1000 * float(np.percentile(tip_obj_err, 90)),
        "dip_rev_rate": dip_rev / T,
        "fps": T / elapsed,
        "qpos": qpos,
    }


def main():
    npz = str(HOCAP_DIR / "motions" / f"{CLIP_ID}.npz")
    meta = str(HOCAP_DIR / "motions" / f"{CLIP_ID}.meta.json")

    results = {}
    for mode in CANDIDATES:
        for side in ("left", "right"):
            key = f"{mode}_{side}"
            print(f"\n>>> retargeting {side} hand with mcp_surrogate={mode!r}", flush=True)
            clip = load_hocap_clip(npz, meta, str(HOCAP_DIR / "assets"),
                                    hand_side=side, sample_count=50)
            results[key] = analyse(side, clip, mode)

    # === Per-bone residual table (averaged across both hands, per mode) ===
    print("\n" + "=" * 80)
    print(f"SUMMARY: per-bone 1-cos residual (avg over both hands, {CLIP_ID})")
    print("=" * 80)
    print(f"{'finger':<10} {'bone':<14}  " + "  ".join(f"{c:>10}" for c in CANDIDATES))
    print("-" * 80)

    bone_names = ["wrist→MCP", "MCP→PIP", "PIP→DIP", "DIP→TIP"]
    for f_idx in range(5):
        fname = FINGER_NAMES[f_idx]
        if f_idx == 0:
            bone_disp = ["wrist→CMC", "CMC→MCP", "MCP→IP", "IP→TIP"]
        else:
            bone_disp = bone_names
        for k in range(4):
            row_vals = []
            for mode in CANDIDATES:
                both = np.concatenate([
                    results[f"{mode}_left"]["cos_err"][:, f_idx, k],
                    results[f"{mode}_right"]["cos_err"][:, f_idx, k],
                ])
                row_vals.append(both.mean())
            # Highlight the non-thumb wrist→MCP and MCP→PIP rows
            mark = ""
            if f_idx in NON_THUMB_FINGER_IDX and k in (0, 1):
                mark = " ←"
            print(f"{fname:<10} {bone_disp[k]:<14}  " +
                  "  ".join(f"{v:>10.4f}" for v in row_vals) + mark)
        print()

    # === Aggregate key bones ===
    print("=" * 80)
    print("AGGREGATE (non-thumb 4 fingers only, both hands)")
    print("=" * 80)
    for k, label in [(0, "wrist→MCP"), (1, "MCP→PIP"), (2, "PIP→DIP"), (3, "DIP→TIP")]:
        row = []
        for mode in CANDIDATES:
            vals = []
            for f_idx in NON_THUMB_FINGER_IDX:
                for side in ("left", "right"):
                    vals.append(results[f"{mode}_{side}"]["cos_err"][:, f_idx, k].mean())
            v_mean = np.mean(vals)
            row.append(v_mean)
        print(f"{label:<14}  " + "  ".join(f"1-cos={v:>.4f} (≈{np.degrees(np.arccos(np.clip(1-v,-1,1))):>5.2f}°)"
                                            for v in row))

    # === End-to-end metrics ===
    print("\n" + "=" * 80)
    print("End-to-end metrics (averaged over both hands)")
    print("=" * 80)
    print(f"{'metric':<20}  " + "  ".join(f"{c:>12}" for c in CANDIDATES))
    print("-" * 80)
    for metric in ("tip_err_mm", "tip_obj_mm", "tip_obj_p90_mm", "dip_rev_rate", "fps"):
        row = []
        for mode in CANDIDATES:
            v = 0.5 * (results[f"{mode}_left"][metric] + results[f"{mode}_right"][metric])
            row.append(v)
        unit = "%" if metric == "dip_rev_rate" else ""
        fmt = "{:>12.2%}" if metric == "dip_rev_rate" else "{:>12.2f}"
        print(f"{metric:<20}  " + "  ".join(fmt.format(v) for v in row))

    # q-diff between candidates (non-thumb finger DOFs)
    print()
    qA = results["link1_left"]["qpos"]
    qB = results["link2_left"]["qpos"]
    qC = results["midpoint_left"]["qpos"]
    d_AB = np.linalg.norm(qA - qB, axis=1).mean()
    d_AC = np.linalg.norm(qA - qC, axis=1).mean()
    d_BC = np.linalg.norm(qB - qC, axis=1).mean()
    print(f"q-space separation (left hand): A↔B {d_AB:.4f} rad   "
          f"A↔C {d_AC:.4f} rad   B↔C {d_BC:.4f} rad")

    out = PROJECT_DIR / "experiments" / "archive" / "warmup_diagnosis" / "probe_mcp_candidate.npz"
    save = {}
    for key, r in results.items():
        for k, v in r.items():
            if isinstance(v, np.ndarray):
                save[f"{key}__{k}"] = v
            else:
                save[f"{key}__{k}"] = np.array(v)
    np.savez(out, **save)
    print(f"\nSaved: {out.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
