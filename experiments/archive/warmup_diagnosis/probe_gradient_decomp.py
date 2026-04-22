"""Decompose warmup-iteration gradient into bone-direction vs tip-position contributions.

For each frame, reconstruct the exact H, c that solve_angle_warmup builds, but
keep the two sub-terms separated so we can compare their magnitudes.

Reports per frame (and aggregated):
  ||r_bone||, ||r_tip||              — residual magnitudes
  ||c_bone||, ||c_tip||              — linear term = J^T r (drives dq direction)
  ||H_bone||, ||H_tip||  (Frobenius) — quadratic term = J^T J
  cos(c_bone, c_tip)                 — alignment of the two pulls on dq
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from hand_retarget import HandRetargetConfig, InteractionMeshHandRetargeter  # noqa: E402
from hand_retarget.config import JOINTS_MAPPING_LEFT, JOINTS_MAPPING_RIGHT  # noqa: E402
from hand_retarget.mediapipe_io import load_hocap_clip  # noqa: E402


HOCAP_DIR = PROJECT_DIR / "data" / "hocap" / "hocap"
CLIP_ID = "hocap__subject_1__20231025_165502__seg00"
HAND_SIDE = "left"
SCENE = PROJECT_DIR / "assets" / "scenes" / "single_hand_obj_left.xml"
FRAME_LIMIT = 200


def decompose_at(ret, q_current, landmarks_21):
    """Replicate the first iter of solve_angle_warmup, split into bone/tip."""
    chains_mp = [
        [0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12],
        [0, 13, 14, 15, 16], [0, 17, 18, 19, 20],
    ]
    _jm = JOINTS_MAPPING_LEFT if ret.config.hand_side == "left" else JOINTS_MAPPING_RIGHT
    w_rot = ret.config.angle_warmup_weight  # 5.0
    w_tip = 100.0
    eps = 1e-8

    ret.hand.forward(q_current)
    nq = ret.nq

    r_bone_list, J_bone_list = [], []
    r_tip_list, J_tip_list = [], []

    for chain in chains_mp:
        for k in range(4):
            parent_mp, child_mp = chain[k], chain[k + 1]
            rp = ret.hand.get_body_pos(_jm[parent_mp])
            rc = ret.hand.get_body_pos(_jm[child_mp])
            e_rob = rc - rp
            e_len = np.linalg.norm(e_rob)
            if e_len < eps:
                continue
            d_rob = e_rob / e_len

            e_src = landmarks_21[child_mp] - landmarks_21[parent_mp]
            s_len = np.linalg.norm(e_src)
            if s_len < eps:
                continue
            d_src = e_src / s_len
            res = d_rob - d_src

            Jp = ret.hand.get_body_jacp(_jm[parent_mp])
            Jc = ret.hand.get_body_jacp(_jm[child_mp])
            P = (np.eye(3) - np.outer(d_rob, d_rob)) / e_len
            J_dir = P @ (Jc - Jp)

            r_bone_list.append(np.sqrt(w_rot) * res)
            J_bone_list.append(np.sqrt(w_rot) * J_dir)

        tip_mp = chain[4]
        tip_body = _jm[tip_mp]
        rob_tip = ret.hand.get_body_pos(tip_body)
        src_tip = landmarks_21[tip_mp]
        res_tip = rob_tip - src_tip
        J_tip = ret.hand.get_body_jacp(tip_body)

        r_tip_list.append(np.sqrt(w_tip) * res_tip)
        J_tip_list.append(np.sqrt(w_tip) * J_tip)

    r_bone = np.concatenate(r_bone_list)
    J_bone = np.vstack(J_bone_list)
    r_tip = np.concatenate(r_tip_list)
    J_tip = np.vstack(J_tip_list)

    c_bone = J_bone.T @ r_bone
    c_tip = J_tip.T @ r_tip
    H_bone = J_bone.T @ J_bone
    H_tip = J_tip.T @ J_tip

    # raw residual magnitudes (unweighted, for reporting)
    r_bone_raw = r_bone / np.sqrt(w_rot)
    r_tip_raw = r_tip / np.sqrt(w_tip)

    return {
        "r_bone_raw": np.linalg.norm(r_bone_raw),   # dimensionless
        "r_tip_raw": np.linalg.norm(r_tip_raw),     # meters
        "c_bone": np.linalg.norm(c_bone),
        "c_tip": np.linalg.norm(c_tip),
        "H_bone_fro": np.linalg.norm(H_bone, "fro"),
        "H_tip_fro": np.linalg.norm(H_tip, "fro"),
        "cos_c": float(c_bone @ c_tip / (np.linalg.norm(c_bone) * np.linalg.norm(c_tip) + eps)),
        "J_bone_fro": np.linalg.norm(J_bone, "fro"),
        "J_tip_fro": np.linalg.norm(J_tip, "fro"),
    }


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

    # Use the same alignment pipeline as retarget_hocap_sequence
    from hand_retarget.mediapipe_io import transform_object_points
    landmarks_raw = clip["landmarks"]
    obj_pts_local = clip["object_pts_local"]
    obj_q = clip["object_q"]
    obj_t = clip["object_t"]

    # Lock wrist DOFs (HO-Cap uses SVD+MANO alignment which puts hand in wrist frame)
    ret.q_lb[:6] = 0.0
    ret.q_ub[:6] = 0.0

    q_prev = ret.hand.get_default_qpos()
    recs = []
    q_current = q_prev.copy()

    for t in range(len(landmarks_raw)):
        obj_world = transform_object_points(obj_pts_local, obj_q[t], obj_t[t])
        lm, _ = ret._align_frame(landmarks_raw[t], None, obj_world)

        # Analyze at the start of warmup (q_current = q_prev for first iter)
        rec = decompose_at(ret, q_current, lm[:21])
        recs.append(rec)

        # Advance q_current as if we ran the full pipeline (cheap: just do one warmup step)
        q_current = ret.solve_angle_warmup(q_current, q_prev, lm[:21], n_iters=5)
        q_prev = q_current

    keys = list(recs[0].keys())
    agg = {k: np.array([r[k] for r in recs]) for k in keys}

    print(f"\n=== Warmup gradient decomposition on {CLIP_ID} [{len(recs)} frames] ===\n")

    def stats(name, arr, fmt="{:.3e}"):
        p = np.percentile(arr, [10, 50, 90])
        print(f"  {name:20s}  median={fmt.format(p[1])}  p10={fmt.format(p[0])}  p90={fmt.format(p[2])}")

    print("[Residual magnitudes — UNWEIGHTED, shows raw mismatch]")
    stats("||r_bone|| (no unit)", agg["r_bone_raw"])
    stats("||r_tip||  (meters)",  agg["r_tip_raw"])

    print("\n[Jacobian magnitude (Frobenius, WITH sqrt(w))]")
    stats("||√w·J_bone||", agg["J_bone_fro"])
    stats("||√w·J_tip||", agg["J_tip_fro"])

    print("\n[Linear term c = J^T r   — this is what drives the QP solution dq]")
    stats("||c_bone||", agg["c_bone"])
    stats("||c_tip||",  agg["c_tip"])
    ratio = agg["c_bone"] / (agg["c_tip"] + 1e-12)
    print(f"  ratio bone/tip:     median={np.median(ratio):.2f}  p10={np.percentile(ratio,10):.2f}  "
          f"p90={np.percentile(ratio,90):.2f}")
    print(f"  frames where c_tip > c_bone: {int((ratio < 1).sum())}/{len(ratio)} "
          f"({100*(ratio<1).mean():.1f}%)")

    print("\n[Quadratic term H = J^T J  — shapes the QP Hessian]")
    stats("||H_bone||_F", agg["H_bone_fro"])
    stats("||H_tip||_F",  agg["H_tip_fro"])

    print("\n[Direction alignment cos(c_bone, c_tip)]")
    stats("cos(c_b, c_t)", agg["cos_c"], fmt="{:+.3f}")
    n_aligned = int((agg["cos_c"] > 0.5).sum())
    n_orth = int((np.abs(agg["cos_c"]) < 0.3).sum())
    n_oppose = int((agg["cos_c"] < -0.3).sum())
    N = len(agg["cos_c"])
    print(f"  aligned (>0.5):   {n_aligned}/{N} ({100*n_aligned/N:.1f}%)  "
          f"orth (<0.3): {n_orth}/{N} ({100*n_orth/N:.1f}%)  "
          f"oppose (<-0.3): {n_oppose}/{N} ({100*n_oppose/N:.1f}%)")

    out = PROJECT_DIR / "experiments" / "archive" / "warmup_diagnosis" / "probe_gradient_decomp.npz"
    np.savez(out, **agg)
    print(f"\nSaved: {out.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
