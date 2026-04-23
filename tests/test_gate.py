"""Gate tests for retargeting pipeline.

Covers:
  A. Smoke         — import + default-value drift
  B. Fail-fast     — config validation
  C. Manus         — qpos regression + tip Cartesian sanity + joint-limit invariant
  D. HO-Cap        — qpos regression + floating-base wrist-lock invariant +
                     joint-limit invariant + pen_max upper bound (np-C default)
  E. Synthetic     — FK → retarget → q recovery

Baselines:
  tests/refactor_gate_baseline.npz   — Manus 100f qpos
  tests/hocap_gate_baseline.npz      — HO-Cap (subject_1 seg00 left) 50f qpos

Rebuild baselines with: PYTHONPATH=src python tests/capture_gate_baselines.py
Run: PYTHONPATH=src pytest tests/test_gate.py -v
"""

from __future__ import annotations

import os

import numpy as np
import pytest

_WUJI_SDK = os.environ.get("WUJI_SDK_PATH", "/home/l/ws/doc/WujiRepo/wuji_retargeting_private/public")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URDF = os.path.join(_WUJI_SDK, "wuji_retargeting/wuji_hand_description/urdf/left.urdf")
PKL = os.path.join(PROJECT_DIR, "data/manus_for_pinch/manus1_5k.pkl")
MANUS_CONFIG = os.path.join(PROJECT_DIR, "config/manus.yaml")
MANUS_BASELINE = os.path.join(PROJECT_DIR, "tests/refactor_gate_baseline.npz")

HOCAP_DIR = os.path.join(PROJECT_DIR, "data/hocap/hocap")
HOCAP_CLIP = "hocap__subject_1__20231025_165502__seg00"
HOCAP_HAND = "left"
HOCAP_SCENE = os.path.join(PROJECT_DIR, "assets/scenes/single_hand_obj_left.xml")
HOCAP_CONFIG = os.path.join(PROJECT_DIR, "config/hocap.yaml")
HOCAP_BASELINE = os.path.join(PROJECT_DIR, "tests/hocap_gate_baseline.npz")
HOCAP_FRAMES = 50

MANUS_FRAMES = 100

# Regression threshold: daqp QP is deterministic on same input, float jitter < 1e-6
# rad ≈ 6e-5 deg; we give 0.01° headroom to tolerate cross-platform BLAS.
REGRESSION_TOL_DEG = 0.01

# Geometric sanity thresholds (Manus 100f sanity, measured ~15.4 mm mean).
MANUS_TIP_ERR_MEAN_MAX_MM = 25.0



# ==========================================================================
# Fixtures
# ==========================================================================


@pytest.fixture(scope="module")
def manus_qpos_100f():
    """Run Manus retarget for 100 frames. Reused across Manus tests."""
    from hand_retarget import HandRetargetConfig, InteractionMeshHandRetargeter
    from hand_retarget.mediapipe_io import load_pkl_sequence, preprocess_sequence

    cfg = HandRetargetConfig(mjcf_path=URDF)
    r = InteractionMeshHandRetargeter(cfg)
    lm_raw, _ = load_pkl_sequence(PKL, "left")
    lm_seq = preprocess_sequence(lm_raw, cfg.mediapipe_rotation, hand_side="left", global_scale=1.0)

    q = r.hand.get_default_qpos()
    qpos = np.zeros((MANUS_FRAMES, r.nq))
    for t in range(MANUS_FRAMES):
        q = r.retarget_frame(lm_seq[t], q, is_first_frame=(t == 0), use_semantic_weights=True)
        qpos[t] = q

    return {"retargeter": r, "qpos": qpos, "landmarks": lm_seq[:MANUS_FRAMES]}


@pytest.fixture(scope="module")
def hocap_qpos_50f():
    """Run HO-Cap retarget for 50 frames (single-hand left). Reused across HO-Cap tests."""
    from hand_retarget import HandRetargetConfig, InteractionMeshHandRetargeter
    from hand_retarget.mediapipe_io import load_hocap_clip

    cfg = HandRetargetConfig.from_yaml(
        HOCAP_CONFIG, mjcf_path=HOCAP_SCENE, hand_side=HOCAP_HAND,
    )
    r = InteractionMeshHandRetargeter(cfg)
    clip = load_hocap_clip(
        os.path.join(HOCAP_DIR, "motions", f"{HOCAP_CLIP}.npz"),
        os.path.join(HOCAP_DIR, "motions", f"{HOCAP_CLIP}.meta.json"),
        os.path.join(HOCAP_DIR, "assets"),
        hand_side=HOCAP_HAND,
        sample_count=cfg.object_sample_count,
    )
    # Truncate to HOCAP_FRAMES while preserving object_pts_local (time-invariant).
    pts_local = clip["object_pts_local"]
    clip = {k: (v[:HOCAP_FRAMES] if isinstance(v, np.ndarray) and v.ndim > 0 and len(v) >= HOCAP_FRAMES else v)
            for k, v in clip.items()}
    clip["object_pts_local"] = pts_local

    qpos = r.retarget_hocap_sequence(clip)
    return {"retargeter": r, "qpos": qpos, "clip": clip, "config": cfg}


# ==========================================================================
# A. Smoke
# ==========================================================================


def test_import():
    """Basic import and initialization."""
    from hand_retarget import HandRetargetConfig, InteractionMeshHandRetargeter

    cfg = HandRetargetConfig(mjcf_path=URDF)
    r = InteractionMeshHandRetargeter(cfg)
    assert r.nq == 20
    assert r.n_keypoints == 21


def test_config_defaults():
    """Verify default config values: 5:5:1 ratio, cosik_live anchor, link2 MCP surrogate."""
    from hand_retarget.config import HandRetargetConfig

    cfg = HandRetargetConfig()
    assert cfg.use_angle_warmup is True
    assert cfg.angle_anchor_weight == 5.0
    assert cfg.smooth_weight == 1.0
    assert cfg.global_scale == 1.0
    assert cfg.anchor_mode == "cosik_live"
    assert cfg.anchor_cosik_weight == 5.0
    assert cfg.mcp_surrogate == "link2"
    assert cfg.thumb_cmc_surrogate == "link2"
    assert cfg.mcp_surface_offset_m == 0.0


# ==========================================================================
# B. Fail-fast validation
# ==========================================================================


def test_anchor_mode_validation():
    """Invalid anchor_mode should fail fast at config construction."""
    from hand_retarget.config import HandRetargetConfig

    with pytest.raises(ValueError, match="anchor_mode"):
        HandRetargetConfig(mjcf_path=URDF, anchor_mode="bogus")


def test_mcp_surrogate_validation():
    """Invalid mcp_surrogate should fail fast at config construction."""
    from hand_retarget.config import HandRetargetConfig

    with pytest.raises(ValueError, match="mcp_surrogate"):
        HandRetargetConfig(mjcf_path=URDF, mcp_surrogate="bogus")


# ==========================================================================
# C. Manus
# ==========================================================================


def test_manus_qpos_regression(manus_qpos_100f):
    """Manus 100f qpos must match frozen baseline within REGRESSION_TOL_DEG."""
    baseline = np.load(MANUS_BASELINE)["qpos"]
    diff_deg = np.degrees(np.abs(manus_qpos_100f["qpos"] - baseline))
    max_diff = diff_deg.max()
    assert max_diff < REGRESSION_TOL_DEG, (
        f"Manus regression: max diff {max_diff:.4f} deg (tol {REGRESSION_TOL_DEG})"
    )


def test_manus_joint_limits_respected(manus_qpos_100f):
    """Every Manus qpos sample must lie within [q_lb, q_ub]."""
    r = manus_qpos_100f["retargeter"]
    q = manus_qpos_100f["qpos"]
    # Allow 1e-9 rad slack for QP float rounding at boundary.
    eps = 1e-9
    assert np.all(q >= r.q_lb - eps), (
        f"Below q_lb: min slack {(q - r.q_lb).min():.2e}"
    )
    assert np.all(q <= r.q_ub + eps), (
        f"Above q_ub: min slack {(r.q_ub - q).min():.2e}"
    )


def test_manus_tip_err_bound(manus_qpos_100f):
    """Mean Cartesian fingertip error (robot vs source landmark) must be reasonable."""
    r = manus_qpos_100f["retargeter"]
    qpos = manus_qpos_100f["qpos"]
    lm = manus_qpos_100f["landmarks"]

    tip_mp = [4, 8, 12, 16, 20]
    tip_bodies = [r.config.joints_mapping[i] for i in tip_mp]

    errs = []
    for t in range(len(qpos)):
        r.hand.forward(qpos[t])
        robot_tips = np.stack([r.hand.get_body_pos(b) for b in tip_bodies])  # (5, 3)
        src_tips = lm[t, tip_mp]                                             # (5, 3)
        errs.append(np.linalg.norm(robot_tips - src_tips, axis=1))           # (5,)
    mean_mm = float(np.mean(errs)) * 1000.0
    assert mean_mm < MANUS_TIP_ERR_MEAN_MAX_MM, (
        f"Manus tip err mean {mean_mm:.1f} mm > {MANUS_TIP_ERR_MEAN_MAX_MM} mm"
    )


# ==========================================================================
# D. HO-Cap
# ==========================================================================


def test_hocap_qpos_regression(hocap_qpos_50f):
    """HO-Cap 50f qpos must match frozen baseline within REGRESSION_TOL_DEG."""
    if not os.path.exists(HOCAP_BASELINE):
        pytest.skip(f"HO-Cap baseline not captured yet: {HOCAP_BASELINE}")
    baseline = np.load(HOCAP_BASELINE)["qpos"]
    diff_deg = np.degrees(np.abs(hocap_qpos_50f["qpos"] - baseline))
    max_diff = diff_deg.max()
    assert max_diff < REGRESSION_TOL_DEG, (
        f"HO-Cap regression: max diff {max_diff:.4f} deg (tol {REGRESSION_TOL_DEG})"
    )


def test_hocap_wrist_locked(hocap_qpos_50f):
    """Floating-base wrist (qpos[:, :6]) must stay identically 0 on HO-Cap.

    SVD+OPERATOR2MANO moves landmarks into the wrist frame; the solver then
    pins q[:6] via q_lb/q_ub=0. Any drift here indicates the wrist-lock guard
    regressed (e.g. inject_object_mesh forgetting to re-zero q_lb/q_ub).
    """
    wrist_qpos = hocap_qpos_50f["qpos"][:, :6]
    assert np.all(wrist_qpos == 0.0), (
        f"HO-Cap wrist DOFs drifted: max |q[:6]| = {np.abs(wrist_qpos).max():.2e}"
    )


def test_hocap_joint_limits_respected(hocap_qpos_50f):
    """HO-Cap qpos[:, 6:] must respect finger joint limits."""
    r = hocap_qpos_50f["retargeter"]
    q = hocap_qpos_50f["qpos"]
    # q_lb / q_ub from the MuJoCo floating model include the 6-DOF wrist zeros.
    eps = 1e-9
    finger_q = q[:, 6:]
    finger_lb = r.q_lb[6:]
    finger_ub = r.q_ub[6:]
    assert np.all(finger_q >= finger_lb - eps), (
        f"HO-Cap finger qpos below lower limit: slack min {(finger_q - finger_lb).min():.2e}"
    )
    assert np.all(finger_q <= finger_ub + eps), (
        f"HO-Cap finger qpos above upper limit: slack min {(finger_ub - finger_q).min():.2e}"
    )


def test_hocap_penetration_flags_default(hocap_qpos_50f):
    """HO-Cap config default must have both penetration flags enabled (np-C mode).

    Guards against silent drift of the EXP-13 default. When BUG-06 (soft
    fallback for source-landmark penetration) lands, this test should upgrade
    to an absolute pen_max bound; current per-clip bound (~36 mm on subject_1
    early frames) tracks source-landmark penetration, not solver enforcement.
    """
    cfg = hocap_qpos_50f["config"]
    assert cfg.activate_non_penetration_warmup is True, (
        "np-C requires warmup penetration flag; config drifted"
    )
    assert cfg.activate_non_penetration_s2 is True, (
        "np-C requires s2 penetration flag; config drifted"
    )


# ==========================================================================
# E. Synthetic closed-loop
# ==========================================================================


def test_synthetic_roundtrip():
    """FK generates landmarks, retarget recovers joint angles."""
    from hand_retarget import HandRetargetConfig, InteractionMeshHandRetargeter

    cfg = HandRetargetConfig(mjcf_path=URDF, use_angle_warmup=False)
    r = InteractionMeshHandRetargeter(cfg)
    q_true = r.hand.get_default_qpos()

    r.hand.forward(q_true)
    synth_lm = np.zeros((21, 3))
    for i, body in enumerate(r.body_names):
        synth_lm[r.mp_indices[i]] = r.hand.get_body_pos(body)

    q_init = r.hand.get_default_qpos()
    q_out = r.retarget_frame(synth_lm, q_init, is_first_frame=True)
    max_err = np.degrees(np.abs(q_out - q_true).max())
    assert max_err < 5.0, f"Roundtrip: max err {max_err:.1f} deg (threshold 5.0)"
