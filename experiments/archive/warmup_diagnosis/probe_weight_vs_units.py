"""A/B test: is the 'bone direction' term in warmup actually doing anything?

Runs warmup with different weight configs on the same frames and compares final q.

Configs:
  A  current      w_rot=5,   w_tip=100                                (baseline)
  B  tip-only     w_rot=0,   w_tip=100                                (drop bone term)
  C  bone-only    w_rot=5,   w_tip=0                                  (drop tip term)
  D  equal raw    w_rot=1,   w_tip=1                                  (no weight bias → shows natural unit bias)
  E  reversed     w_rot=100, w_tip=5                                  (flip current 20:1 ratio)
  F  unit-match   w_rot=5,   w_tip=100,  bone residual × bone_length  (convert bone err to meters)

Key metrics:
  ||q_A - q_B||     small → tip term alone reproduces current
  ||q_A - q_C||     small → bone term alone reproduces current  (can't both be small)
  cos(Δq_A, Δq_B)   close to 1 → bone term only tweaks magnitude, not direction
  config D vs A     tells us whether current weight ratio is needed
  config F vs A     tells us whether the issue is units (dimensionality) or weight
"""

import sys
from pathlib import Path

import numpy as np
import qpsolvers

PROJECT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from hand_retarget import HandRetargetConfig, InteractionMeshHandRetargeter  # noqa: E402
from hand_retarget.config import JOINTS_MAPPING_LEFT, JOINTS_MAPPING_RIGHT  # noqa: E402
from hand_retarget.mediapipe_io import load_hocap_clip, transform_object_points  # noqa: E402

HOCAP_DIR = PROJECT_DIR / "data" / "hocap" / "hocap"
CLIP_ID = "hocap__subject_1__20231025_165502__seg00"
HAND_SIDE = "left"
SCENE = PROJECT_DIR / "assets" / "scenes" / "single_hand_obj_left.xml"
FRAME_LIMIT = 200

CHAINS = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12],
          [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]]


def warmup_custom(ret, q_start, q_prev, landmarks_21,
                  w_rot, w_tip, unit_match=False,
                  n_iters=5, convergence_delta=1e-3):
    """Same structure as solve_angle_warmup + outer convergence loop."""
    _jm = JOINTS_MAPPING_LEFT if ret.config.hand_side == "left" else JOINTS_MAPPING_RIGHT
    eps = 1e-8
    q = q_start.copy()

    for _ in range(n_iters):
        q_before = q.copy()
        ret.hand.forward(q)

        residuals, J_rows = [], []
        for chain in CHAINS:
            for k in range(4):
                rp = ret.hand.get_body_pos(_jm[chain[k]])
                rc = ret.hand.get_body_pos(_jm[chain[k + 1]])
                e_rob = rc - rp
                e_len = np.linalg.norm(e_rob)
                if e_len < eps or w_rot == 0:
                    continue
                d_rob = e_rob / e_len

                e_src = landmarks_21[chain[k + 1]] - landmarks_21[chain[k]]
                s_len = np.linalg.norm(e_src)
                if s_len < eps:
                    continue
                d_src = e_src / s_len

                Jp = ret.hand.get_body_jacp(_jm[chain[k]])
                Jc = ret.hand.get_body_jacp(_jm[chain[k + 1]])
                Je = Jc - Jp

                if unit_match:
                    # Multiply bone residual by bone length → residual is in meters
                    # r' = e_rob - (s_len/s_len)*d_src*e_len ... actually let's just scale r and J by e_len
                    res = (d_rob - d_src) * e_len  # now in meters
                    J_dir = (np.eye(3) - np.outer(d_rob, d_rob)) @ Je  # no 1/e_len
                else:
                    res = d_rob - d_src
                    J_dir = (np.eye(3) - np.outer(d_rob, d_rob)) / e_len @ Je

                residuals.append(np.sqrt(w_rot) * res)
                J_rows.append(np.sqrt(w_rot) * J_dir)

            if w_tip > 0:
                tip_body = _jm[chain[4]]
                res_tip = ret.hand.get_body_pos(tip_body) - landmarks_21[chain[4]]
                J_tip = ret.hand.get_body_jacp(tip_body)
                residuals.append(np.sqrt(w_tip) * res_tip)
                J_rows.append(np.sqrt(w_tip) * J_tip)

        if not residuals:
            break

        nq = ret.nq
        dq_smooth = q_prev - q
        H = ret.config.smooth_weight * np.eye(nq) + 1e-12 * np.eye(nq)
        c = -ret.config.smooth_weight * dq_smooth
        for r, Jr in zip(residuals, J_rows):
            H += Jr.T @ Jr
            c += Jr.T @ r

        lb = np.maximum(ret.q_lb - q, -ret.config.step_size)
        ub = np.minimum(ret.q_ub - q, ret.config.step_size)

        problem = qpsolvers.Problem(H, c, lb=lb, ub=ub)
        sol = qpsolvers.solve_problem(problem, solver="daqp")
        if sol.found:
            q = np.clip(q + sol.x, ret.q_lb, ret.q_ub)

        if np.linalg.norm(q - q_before) < convergence_delta:
            break

    return q


def tip_error(ret, q, lm):
    """Mean ||robot_tip - source_tip|| over 5 fingertips."""
    _jm = JOINTS_MAPPING_LEFT if ret.config.hand_side == "left" else JOINTS_MAPPING_RIGHT
    ret.hand.forward(q)
    errs = []
    for chain in CHAINS:
        rob = ret.hand.get_body_pos(_jm[chain[4]])
        errs.append(np.linalg.norm(rob - lm[chain[4]]))
    return float(np.mean(errs))


def bone_cos_err(ret, q, lm):
    """Mean (1 - cos) over 20 bone directions."""
    _jm = JOINTS_MAPPING_LEFT if ret.config.hand_side == "left" else JOINTS_MAPPING_RIGHT
    ret.hand.forward(q)
    errs = []
    for chain in CHAINS:
        for k in range(4):
            e_r = ret.hand.get_body_pos(_jm[chain[k + 1]]) - ret.hand.get_body_pos(_jm[chain[k]])
            e_s = lm[chain[k + 1]] - lm[chain[k]]
            if np.linalg.norm(e_r) < 1e-8 or np.linalg.norm(e_s) < 1e-8:
                continue
            d_r = e_r / np.linalg.norm(e_r)
            d_s = e_s / np.linalg.norm(e_s)
            errs.append(1.0 - float(d_r @ d_s))
    return float(np.mean(errs))


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
    ret.q_lb[:6] = 0.0
    ret.q_ub[:6] = 0.0

    CONFIGS = {
        "A_current":   dict(w_rot=5,   w_tip=100, unit_match=False),
        "B_tip_only":  dict(w_rot=0,   w_tip=100, unit_match=False),
        "C_bone_only": dict(w_rot=5,   w_tip=0,   unit_match=False),
        "D_equal_raw": dict(w_rot=1,   w_tip=1,   unit_match=False),
        "E_reversed":  dict(w_rot=100, w_tip=5,   unit_match=False),
        "F_unitmatch": dict(w_rot=5,   w_tip=100, unit_match=True),
    }

    T = len(clip["landmarks"])
    results = {k: {"q": [], "tip_err": [], "bone_err": [], "dq_norm": []} for k in CONFIGS}

    q_prev_chain = ret.hand.get_default_qpos()

    for t in range(T):
        obj_world = transform_object_points(clip["object_pts_local"],
                                             clip["object_q"][t], clip["object_t"][t])
        lm, _ = ret._align_frame(clip["landmarks"][t], None, obj_world)
        lm21 = lm[:21]

        q_starts_from = q_prev_chain  # all configs start from same warm start

        for name, kwargs in CONFIGS.items():
            q_out = warmup_custom(ret, q_starts_from, q_prev_chain, lm21, **kwargs)
            results[name]["q"].append(q_out)
            results[name]["tip_err"].append(tip_error(ret, q_out, lm21))
            results[name]["bone_err"].append(bone_cos_err(ret, q_out, lm21))
            results[name]["dq_norm"].append(float(np.linalg.norm(q_out - q_starts_from)))

        # Advance q_prev_chain using config A (the current production behavior)
        q_prev_chain = results["A_current"]["q"][-1]

    # Convert to arrays
    for name in results:
        for k in results[name]:
            results[name][k] = np.array(results[name][k]) if k != "q" else np.stack(results[name][k])

    qA = results["A_current"]["q"]

    print(f"\n=== Warmup weight-vs-units probe on {CLIP_ID} [{T} frames] ===\n")

    # Table header
    print(f"{'config':<14} {'||Δq||':>10} {'tip_err_mm':>12} {'bone_cos_err':>14} "
          f"{'||q-qA||':>10} {'cos(Δq,ΔqA)':>14}")
    for name in CONFIGS:
        r = results[name]
        diff_to_A = np.linalg.norm(r["q"] - qA, axis=1)
        dqA = qA - qA[[0] + list(range(len(qA) - 1))]  # only for orientation; use dq_norm direction
        # direction alignment: cos(q_out - q_start, qA_out - q_start) per frame
        cos_vs_A = []
        q_start_seq = [ret.hand.get_default_qpos()]
        for i in range(len(qA) - 1):
            q_start_seq.append(qA[i])
        for i in range(len(qA)):
            v1 = r["q"][i] - q_start_seq[i]
            v2 = qA[i] - q_start_seq[i]
            denom = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12
            cos_vs_A.append(float(v1 @ v2) / denom)
        cos_vs_A = np.array(cos_vs_A)

        print(f"{name:<14} {r['dq_norm'].mean():>10.4f} {1000*r['tip_err'].mean():>12.2f} "
              f"{r['bone_err'].mean():>14.4f} {diff_to_A.mean():>10.4f} "
              f"{np.median(cos_vs_A):>+14.3f}")

    print("\n[Interpretation]")
    dB = np.linalg.norm(results["B_tip_only"]["q"] - qA, axis=1)
    dC = np.linalg.norm(results["C_bone_only"]["q"] - qA, axis=1)
    print(f"  ||qA - qB (tip only)||   median={np.median(dB):.4f}   max={dB.max():.4f}")
    print(f"  ||qA - qC (bone only)||  median={np.median(dC):.4f}   max={dC.max():.4f}")
    ratio = dC / (dB + 1e-12)
    print(f"  dropping tip moves q ~{np.median(ratio):.1f}× more than dropping bone")

    out = PROJECT_DIR / "experiments" / "archive" / "warmup_diagnosis" / "probe_weight_vs_units.npz"
    save_dict = {}
    for name in results:
        for k in results[name]:
            save_dict[f"{name}_{k}"] = results[name][k]
    np.savez(out, **save_dict)
    print(f"\nSaved: {out.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
