"""Break per-bone direction error down to pinpoint the MCP issue the user saw.

Runs current (cosik_live default) retarget on the same bimanual clip the user
viewed and reports the unit-vector direction error for each of the 20 bones
AND the gradient reachability (||J_dir||) for each.

If WRIST→MCP bones have large residuals but J ≈ 0 (wrist locked), the mismatch
is a **mapping artifact** (MediaPipe surface landmark vs robot joint pivot),
not a solver bug — cos IK literally cannot fix it.

If MCP→PIP bones have large residuals AND J > 0, that's a **real joint-angle
mismatch** and would be a candidate bug.
"""

import sys
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
FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]
BONE_NAMES = ["wrist→base", "base→mid1", "mid1→mid2", "mid2→tip"]
# bone 0 = WRIST→CMC/MCP, bone 1 = CMC→MCP / MCP→PIP, etc.

SCENES = {
    "left": PROJECT_DIR / "assets" / "scenes" / "single_hand_obj_left.xml",
    "right": PROJECT_DIR / "assets" / "scenes" / "single_hand_obj.xml",
}


def analyse_side(hand_side: str, clip: dict):
    """Per-bone error analysis for one hand."""
    cfg = HandRetargetConfig(
        mjcf_path=str(SCENES[hand_side]), hand_side=hand_side,
        floating_base=True, object_sample_count=50,
    )
    ret = InteractionMeshHandRetargeter(cfg)
    qpos = ret.retarget_hocap_sequence(clip)

    # Must lock wrist before FK (retarget_hocap_sequence restores on exit)
    ret.q_lb[:6] = 0.0
    ret.q_ub[:6] = 0.0

    _jm = JOINTS_MAPPING_LEFT if hand_side == "left" else JOINTS_MAPPING_RIGHT
    obj_pts_local = clip["object_pts_local"]

    T = len(qpos)
    n_bones = 20
    cos_err = np.zeros((T, n_bones))       # 1 - cos(d_rob, d_src)
    j_norm = np.zeros((T, n_bones))        # ||J_dir||_F (controllability)
    bone_len_rob = np.zeros((T, n_bones))
    bone_len_src = np.zeros((T, n_bones))

    for t in range(T):
        obj_world = transform_object_points(obj_pts_local, clip["object_q"][t], clip["object_t"][t])
        lm, _ = ret._align_frame(clip["landmarks"][t], None, obj_world)
        lm21 = lm[:21]

        ret.hand.forward(qpos[t])
        b = 0
        for chain in FINGER_CHAINS_MP:
            for k in range(4):
                p_mp, c_mp = chain[k], chain[k + 1]
                rp = ret.hand.get_body_pos(_jm[p_mp])
                rc = ret.hand.get_body_pos(_jm[c_mp])
                e_rob = rc - rp
                lr = np.linalg.norm(e_rob)
                e_src = lm21[c_mp] - lm21[p_mp]
                ls = np.linalg.norm(e_src)
                if lr < 1e-8 or ls < 1e-8:
                    b += 1
                    continue
                d_rob = e_rob / lr
                d_src = e_src / ls
                cos_err[t, b] = 1.0 - float(d_rob @ d_src)

                # Controllability: Jacobian of d_rob wrt q
                Jp = ret.hand.get_body_jacp(_jm[p_mp])
                Jc = ret.hand.get_body_jacp(_jm[c_mp])
                P = (np.eye(3) - np.outer(d_rob, d_rob)) / lr
                J_dir = P @ (Jc - Jp)
                # Exclude wrist DOFs (locked): their Jacobian rows will be 0-d constrained by box
                # Compute norm over finger DOFs only (slice 6:)
                j_norm[t, b] = float(np.linalg.norm(J_dir[:, 6:]))

                bone_len_rob[t, b] = lr
                bone_len_src[t, b] = ls
                b += 1

    # Aggregate per bone
    mean_err = cos_err.mean(axis=0)                    # (20,)
    mean_deg = np.degrees(np.arccos(np.clip(1 - mean_err, -1, 1)))  # approx angle
    mean_jn = j_norm.mean(axis=0)
    mean_lr = bone_len_rob.mean(axis=0)
    mean_ls = bone_len_src.mean(axis=0)
    len_ratio = mean_ls / np.clip(mean_lr, 1e-8, None)

    print(f"\n=== {hand_side.upper()} hand — per-bone diagnosis ===\n")
    print(f"{'bone':<20} {'1-cos':>8} {'≈deg':>7} {'‖J_f‖':>8} {'L_rob':>7} {'L_src':>7} {'src/rob':>8}")
    print("-" * 72)
    for f_idx, fname in enumerate(FINGER_NAMES):
        for k in range(4):
            b = 4 * f_idx + k
            label = f"{fname:<7} {BONE_NAMES[k]}"
            flag = " ←" if mean_err[b] > 0.03 else ""
            print(f"{label:<20} {mean_err[b]:>8.4f} {mean_deg[b]:>6.2f}° {mean_jn[b]:>8.3f} "
                  f"{1000*mean_lr[b]:>6.1f}mm {1000*mean_ls[b]:>6.1f}mm {len_ratio[b]:>7.3f}{flag}")
        print()

    # Flag bones whose residual is large AND J is near-zero (uncontrollable)
    uncontrollable = (mean_err > 0.03) & (mean_jn < 0.5)
    if uncontrollable.any():
        print("⚠  Bones with LARGE residual but NEAR-ZERO Jacobian (uncontrollable — mapping artifact):")
        for b in np.where(uncontrollable)[0]:
            f, k = b // 4, b % 4
            print(f"   {FINGER_NAMES[f]} {BONE_NAMES[k]}: 1-cos={mean_err[b]:.4f}, ‖J_f‖={mean_jn[b]:.3f}")

    real_errors = (mean_err > 0.03) & (mean_jn >= 0.5)
    if real_errors.any():
        print("\n⚠  Bones with LARGE residual AND non-zero Jacobian (solver isn't fixing):")
        for b in np.where(real_errors)[0]:
            f, k = b // 4, b % 4
            print(f"   {FINGER_NAMES[f]} {BONE_NAMES[k]}: 1-cos={mean_err[b]:.4f}, ‖J_f‖={mean_jn[b]:.3f}")

    return {
        "cos_err": cos_err, "j_norm": j_norm,
        "bone_len_rob": bone_len_rob, "bone_len_src": bone_len_src,
    }


def main():
    npz = str(HOCAP_DIR / "motions" / f"{CLIP_ID}.npz")
    meta = str(HOCAP_DIR / "motions" / f"{CLIP_ID}.meta.json")
    results = {}
    for side in ("left", "right"):
        clip = load_hocap_clip(npz, meta, str(HOCAP_DIR / "assets"),
                                hand_side=side, sample_count=50)
        results[side] = analyse_side(side, clip)

    out = PROJECT_DIR / "experiments" / "archive" / "warmup_diagnosis" / "probe_mcp_bone_error.npz"
    save = {}
    for side, r in results.items():
        for k, v in r.items():
            save[f"{side}_{k}"] = v
    np.savez(out, **save)
    print(f"\nSaved: {out.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
