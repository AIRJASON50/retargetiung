"""Correctness checks for the cosik_live anchor mode.

Tests:
  1. At q = q_S1 (warmup converged), bone residuals ≈ 0, so c_cosik ≈ 0.
     → cosik_live at warmup endpoint produces NO extra pull (initial iter).
  2. H_cosik contribution is PSD (J^T J) and not NaN.
  3. Full HO-Cap retarget with anchor_mode="cosik_live" produces finite q
     with no NaN, no joint-limit violations.
  4. Compare A (l2) vs B (cosik_live) on the default clip: q trajectories
     should differ but both be stable.
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from hand_retarget import HandRetargetConfig, InteractionMeshHandRetargeter  # noqa: E402
from hand_retarget.mediapipe_io import load_hocap_clip, transform_object_points  # noqa: E402

HOCAP_DIR = PROJECT_DIR / "data" / "hocap" / "hocap"
CLIP_ID = "hocap__subject_1__20231025_165502__seg00"
HAND_SIDE = "left"
SCENE = PROJECT_DIR / "assets" / "scenes" / "single_hand_obj_left.xml"


def main():
    clip = load_hocap_clip(
        str(HOCAP_DIR / "motions" / f"{CLIP_ID}.npz"),
        str(HOCAP_DIR / "motions" / f"{CLIP_ID}.meta.json"),
        str(HOCAP_DIR / "assets"),
        hand_side=HAND_SIDE, sample_count=50,
    )
    # 50 frames for fast smoke test
    N = 50
    pts_local = clip["object_pts_local"]
    clip = {k: v[:N] if isinstance(v, np.ndarray) and v.ndim > 0 and v.shape[0] >= N else v
            for k, v in clip.items()}
    clip["object_pts_local"] = pts_local

    # ── Check 1: residuals at warmup-converged q ──
    print("=" * 72)
    print("Check 1: bone residual at warmup converged q_S1 (on frame 0)")
    print("=" * 72)
    cfg = HandRetargetConfig(mjcf_path=str(SCENE), hand_side=HAND_SIDE,
                             floating_base=True, object_sample_count=50)
    ret = InteractionMeshHandRetargeter(cfg)
    ret.q_lb[:6] = 0.0
    ret.q_ub[:6] = 0.0

    # Align first-frame landmarks
    obj_world_0 = transform_object_points(
        clip["object_pts_local"], clip["object_q"][0], clip["object_t"][0])
    lm_aligned, _ = ret._align_frame(clip["landmarks"][0], None, obj_world_0)
    lm_21 = lm_aligned[:21]

    # Run warmup to convergence on frame 0
    q_prev = ret.hand.get_default_qpos()
    q_S1 = q_prev.copy()
    for it in range(20):
        q_before = q_S1.copy()
        q_S1 = ret.solve_angle_warmup(q_S1, q_prev, lm_21, n_iters=1)
        if np.linalg.norm(q_S1 - q_before) < 1e-3:
            break
    print(f"  warmup used {it+1} iters, final ||Δq|| = {np.linalg.norm(q_S1 - q_before):.2e}")

    # Compute bone residual at q_S1
    ret.hand.forward(q_S1)
    r_bones, J_bones = ret._compute_bone_dir_residuals_and_jac(lm_21)
    print(f"  r_bones shape: {r_bones.shape}  (expect (60,))")
    print(f"  J_bones shape: {J_bones.shape}  (expect (60, 26))")
    print(f"  ||r_bones|| at q_S1 = {np.linalg.norm(r_bones):.4e}  "
          f"(if warmup converged, ≪ 1)")
    print(f"  ||J_bones^T @ r_bones|| = {np.linalg.norm(J_bones.T @ r_bones):.4e}  "
          f"(c_cosik at q_S1, expect ≪ 1)")

    # ── Check 2: H contribution is PSD and finite ──
    print("\n" + "=" * 72)
    print("Check 2: H_cosik = J^T @ J is PSD and finite")
    print("=" * 72)
    H_cosik = 5.0 * (J_bones.T @ J_bones)
    eigs = np.linalg.eigvalsh(H_cosik)
    print(f"  min eigenvalue = {eigs.min():+.4e}   (PSD iff ≥ 0, up to float noise)")
    print(f"  max eigenvalue = {eigs.max():+.4e}")
    print(f"  condition ratio = {eigs.max() / max(eigs.min(), 1e-20):.2e}  "
          f"(high ratio = anisotropic valley — expected)")
    print(f"  any NaN: {np.isnan(H_cosik).any()}  any Inf: {np.isinf(H_cosik).any()}")

    # ── Check 3: Full retarget runs cleanly with cosik_live ──
    print("\n" + "=" * 72)
    print("Check 3: full retarget in cosik_live mode (50 frames)")
    print("=" * 72)
    cfg_B = HandRetargetConfig(mjcf_path=str(SCENE), hand_side=HAND_SIDE,
                               floating_base=True, object_sample_count=50)
    cfg_B.anchor_mode = "cosik_live"
    ret_B = InteractionMeshHandRetargeter(cfg_B)

    import time; t0 = time.time()
    qpos_B = ret_B.retarget_hocap_sequence(clip)
    dt_B = time.time() - t0
    print(f"  elapsed: {dt_B*1000:.0f}ms  ({N/dt_B:.1f} fps)")
    print(f"  qpos shape: {qpos_B.shape}")
    print(f"  any NaN: {np.isnan(qpos_B).any()}  any Inf: {np.isinf(qpos_B).any()}")
    print(f"  q range: [{qpos_B.min():.3f}, {qpos_B.max():.3f}]")
    # Joint-limit check (use finger slice 6:26)
    q_lb_f = ret_B.hand.q_lb[6:26]
    q_ub_f = ret_B.hand.q_ub[6:26]
    violations = ((qpos_B[:, 6:26] < q_lb_f - 1e-6) | (qpos_B[:, 6:26] > q_ub_f + 1e-6)).sum()
    print(f"  joint-limit violations: {violations}  (expect 0)")

    # ── Check 4: A vs B comparison ──
    print("\n" + "=" * 72)
    print("Check 4: A (l2) vs B (cosik_live) comparison on 50 frames")
    print("=" * 72)
    cfg_A = HandRetargetConfig(mjcf_path=str(SCENE), hand_side=HAND_SIDE,
                               floating_base=True, object_sample_count=50)
    ret_A = InteractionMeshHandRetargeter(cfg_A)
    t0 = time.time(); qpos_A = ret_A.retarget_hocap_sequence(clip); dt_A = time.time() - t0

    diff = qpos_A - qpos_B
    per_frame_diff = np.linalg.norm(diff, axis=1)
    print(f"  A fps: {N/dt_A:.1f}   B fps: {N/dt_B:.1f}  "
          f"(B overhead: {100*(dt_B - dt_A)/dt_A:+.0f}%)")
    print(f"  ‖q_A − q_B‖ per frame: mean={per_frame_diff.mean():.4f}  "
          f"median={np.median(per_frame_diff):.4f}  max={per_frame_diff.max():.4f}")
    finger = slice(6, 26)
    finger_diff = np.linalg.norm(qpos_A[:, finger] - qpos_B[:, finger], axis=1)
    print(f"  finger DOFs only ‖Δ‖:  mean={finger_diff.mean():.4f}  "
          f"max={finger_diff.max():.4f}")

    # MCP abduction (j = 6 + 4f + 1 for f=0..4) — the direction B should unlock
    mcp_abd_idx = [6 + 4*f + 1 for f in range(5)]
    a_mcp = qpos_A[:, mcp_abd_idx]
    b_mcp = qpos_B[:, mcp_abd_idx]
    mcp_diff = np.abs(a_mcp - b_mcp).mean(axis=0)
    print(f"  MCP abd mean |Δ| per finger (rad): {mcp_diff}")
    print(f"  (larger = B is exploring MCP abd differently than A)")

    print("\n" + "=" * 72)
    print("All checks passed if: no NaN, residual ≈ 0 at q_S1, min_eig ≥ 0,")
    print("q_A and q_B differ (non-trivially) but both in joint limits.")
    print("=" * 72)


if __name__ == "__main__":
    main()
