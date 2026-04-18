"""Gate tests for retargeting pipeline.

Run: PYTHONPATH=src pytest tests/test_gate.py -v
"""

import os
import sys

import numpy as np
import pytest

_WUJI_SDK = os.environ.get("WUJI_SDK_PATH", "/home/l/ws/doc/WujiRepo/wuji_retargeting_private/public")
sys.path.insert(0, _WUJI_SDK)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URDF = os.path.join(_WUJI_SDK, "wuji_retargeting/wuji_hand_description/urdf/left.urdf")
PKL = os.path.join(PROJECT_DIR, "data/manus_for_pinch/manus1_5k.pkl")
BASELINE = os.path.join(PROJECT_DIR, "tests/refactor_gate_baseline.npz")


def test_import():
    """Basic import and initialization."""
    from hand_retarget import InteractionMeshHandRetargeter, HandRetargetConfig

    cfg = HandRetargetConfig(mjcf_path=URDF)
    r = InteractionMeshHandRetargeter(cfg)
    assert r.nq == 20
    assert r.n_keypoints == 21
    assert cfg.use_angle_warmup is True
    assert cfg.smooth_weight == 1.0


def test_retarget_regression():
    """100 frames must match frozen baseline within 0.01 degrees."""
    from hand_retarget import InteractionMeshHandRetargeter, HandRetargetConfig
    from hand_retarget.mediapipe_io import load_pkl_sequence, preprocess_sequence

    cfg = HandRetargetConfig(mjcf_path=URDF)
    r = InteractionMeshHandRetargeter(cfg)
    lm_raw, ts = load_pkl_sequence(PKL, "left")
    lm_seq = preprocess_sequence(lm_raw, cfg.mediapipe_rotation, hand_side="left", global_scale=1.0)

    q = r.hand.get_default_qpos()
    qpos = np.zeros((100, r.nq))
    for t in range(100):
        q = r.retarget_frame(lm_seq[t], q, is_first_frame=(t == 0), use_semantic_weights=True)
        qpos[t] = q

    baseline = np.load(BASELINE)["qpos"]
    max_diff_deg = np.degrees(np.abs(qpos - baseline).max())
    assert max_diff_deg < 0.01, f"Regression: max diff {max_diff_deg:.4f} deg"


def test_config_defaults():
    """Verify default config values for 5:5:1 ratio."""
    from hand_retarget.config import HandRetargetConfig

    cfg = HandRetargetConfig()
    assert cfg.use_angle_warmup is True
    assert cfg.angle_anchor_weight == 5.0
    assert cfg.smooth_weight == 1.0
    assert cfg.global_scale == 1.0


def test_synthetic_roundtrip():
    """FK generates landmarks, retarget recovers joint angles."""
    from hand_retarget import InteractionMeshHandRetargeter, HandRetargetConfig

    cfg = HandRetargetConfig(mjcf_path=URDF, use_angle_warmup=False)
    r = InteractionMeshHandRetargeter(cfg)
    q_true = r.hand.get_default_qpos()

    # Generate landmarks from FK
    r.hand.forward(q_true)
    synth_lm = np.zeros((21, 3))
    for i, body in enumerate(r.body_names):
        synth_lm[r.mp_indices[i]] = r.hand.get_body_pos(body)

    # Retarget from default should converge close
    q_init = r.hand.get_default_qpos()
    q_out = r.retarget_frame(synth_lm, q_init, is_first_frame=True)
    max_err = np.degrees(np.abs(q_out - q_true).max())
    assert max_err < 5.0, f"Roundtrip: max err {max_err:.1f} deg (threshold 5.0)"
