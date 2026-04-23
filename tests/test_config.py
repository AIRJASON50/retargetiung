"""Unit tests for ``HandRetargetConfig.from_yaml``.

Covers:
  * YAML -> dataclass field mapping for the two shipped configs
    (``config/manus.yaml``, ``config/hocap.yaml``).
  * ``**overrides`` precedence over YAML values.
  * ``mjcf_path`` default when the caller omits it.
  * Unknown-YAML-key handling (currently: silently dropped).
  * ``make_stamp`` sensitivity: different hashes for differing configs,
    identical hashes for byte-identical configs.

Fast by design: no MuJoCo/Pinocchio model is loaded, only YAML parsing
and dataclass construction. Target runtime < 100 ms for the whole file.

Run: ``PYTHONPATH=src pytest tests/test_config.py -v``
"""

from __future__ import annotations

import os

import pytest
import yaml

from hand_retarget.config import HandRetargetConfig

# ==========================================================================
# Paths (module-level, like tests/test_gate.py)
# ==========================================================================

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANUS_CONFIG = os.path.join(PROJECT_DIR, "config/manus.yaml")
HOCAP_CONFIG = os.path.join(PROJECT_DIR, "config/hocap.yaml")


# ==========================================================================
# YAML -> field mapping
# ==========================================================================


def test_from_yaml_manus():
    """Loading ``config/manus.yaml`` populates dataclass with expected values."""
    cfg = HandRetargetConfig.from_yaml(MANUS_CONFIG)

    # Fields encoded in the YAML.
    assert cfg.hand_side == "left"
    assert cfg.step_size == 0.1
    assert cfg.smooth_weight == 1.0
    assert cfg.n_iter_first == 20
    assert cfg.n_iter == 10
    assert cfg.activate_joint_limits is True
    assert cfg.warmup_convergence_delta == 0.001
    assert cfg.s2_convergence_delta == 0.001
    assert cfg.anchor_mode == "cosik_live"
    assert cfg.anchor_cosik_weight == 5.0

    # angle_warmup section.
    assert cfg.use_angle_warmup is True
    assert cfg.angle_warmup_weight == 5.0
    assert cfg.angle_warmup_iters_first == 20
    assert cfg.angle_warmup_iters == 5
    assert cfg.angle_anchor_weight == 5.0

    # retarget section — nested dict preserved verbatim.
    assert cfg.mediapipe_rotation == {"x": 0.0, "y": 0.0, "z": 15.0}

    # Fields NOT in manus.yaml should keep dataclass defaults.
    assert cfg.floating_base is False
    assert cfg.object_sample_count == 100
    assert cfg.mcp_surrogate == "link2"


def test_from_yaml_hocap():
    """Loading ``config/hocap.yaml`` populates floating-base + penetration flags."""
    cfg = HandRetargetConfig.from_yaml(HOCAP_CONFIG)

    assert cfg.floating_base is True
    assert cfg.activate_non_penetration_warmup is True
    assert cfg.activate_non_penetration_s2 is True
    assert cfg.penetration_tolerance == 0.001
    assert cfg.penetration_max_trust_shrinks == 3
    assert cfg.object_sample_count == 50

    # Top-level YAML keys that land on ``cfg`` directly.
    assert cfg.delaunay_edge_threshold == 0.06
    assert cfg.laplacian_distance_weight_k == 20.0

    # hand_side is explicitly not in hocap.yaml (resolved per clip in Python);
    # ``from_yaml`` must leave the dataclass default untouched.
    assert cfg.hand_side == "left"


# ==========================================================================
# Override precedence
# ==========================================================================


def test_kwarg_override():
    """``**overrides`` must win over YAML values (highest precedence)."""
    cfg = HandRetargetConfig.from_yaml(
        MANUS_CONFIG,
        smooth_weight=42.0,
        hand_side="right",
    )
    assert cfg.smooth_weight == 42.0
    assert cfg.hand_side == "right"


def test_unknown_override_raises_typeerror():
    """Unknown kwargs must fail fast to catch typos at call sites."""
    with pytest.raises(TypeError, match="unknown override"):
        HandRetargetConfig.from_yaml(MANUS_CONFIG, not_a_real_field=123)


# ==========================================================================
# mjcf_path default
# ==========================================================================


def test_required_kwarg_mjcf_path():
    """Omitting ``mjcf_path`` yields an empty-string default, not an error.

    ``HandRetargetConfig`` declares ``mjcf_path: str = ""`` and ``from_yaml``
    signature ``mjcf_path: str = ""``; no model file is opened at config
    construction, so an empty path round-trips fine.
    """
    cfg = HandRetargetConfig.from_yaml(MANUS_CONFIG)
    assert cfg.mjcf_path == ""

    cfg_explicit = HandRetargetConfig.from_yaml(MANUS_CONFIG, mjcf_path="/tmp/x.xml")
    assert cfg_explicit.mjcf_path == "/tmp/x.xml"


# ==========================================================================
# Unknown-YAML-key handling
# ==========================================================================


def test_unknown_yaml_key_handling(tmp_path):
    """Unknown YAML keys (both inside and outside known sections) are silently
    dropped -- ``from_yaml`` only consults ``_YAML_FIELD_MAP``/``_YAML_ENABLED_MAP``.

    This test pins the *current* behavior; it should flip to ``pytest.warns``
    or ``pytest.raises`` if BUG-19 decides to tighten YAML parsing later.
    """
    data = {
        "hand_side": "right",
        "optimization": {
            "smooth_weight": 2.5,
            "foobar_unknown_key": 999,  # unknown key in a known section
        },
        "totally_unknown_section": {"x": 1, "y": 2},  # unknown section
    }
    yaml_path = tmp_path / "weird.yaml"
    yaml_path.write_text(yaml.safe_dump(data))

    cfg = HandRetargetConfig.from_yaml(str(yaml_path))

    # Known keys still applied.
    assert cfg.hand_side == "right"
    assert cfg.smooth_weight == 2.5
    # Unknown YAML key must NOT materialize as an attribute on the dataclass.
    assert not hasattr(cfg, "foobar_unknown_key")
    assert not hasattr(cfg, "totally_unknown_section")


# ==========================================================================
# make_stamp determinism
# ==========================================================================


def test_make_stamp_changes_with_field():
    """``make_stamp`` must be deterministic across equal configs and
    distinguish configs that differ on a hashed field."""
    cfg_a = HandRetargetConfig(mjcf_path="", laplacian_distance_weight_k=20.0)
    cfg_a_copy = HandRetargetConfig(mjcf_path="", laplacian_distance_weight_k=20.0)
    cfg_b = HandRetargetConfig(mjcf_path="", laplacian_distance_weight_k=10.0)

    stamp_a = cfg_a.make_stamp()
    stamp_a_copy = cfg_a_copy.make_stamp()
    stamp_b = cfg_b.make_stamp()

    # Determinism on identical inputs.
    assert stamp_a == stamp_a_copy
    # Sensitivity to a distinguishing field (k goes into the prefix AND the hash).
    assert stamp_a != stamp_b

    # ``mjcf_path`` is explicitly excluded from the stamp so cache keys are
    # portable across environments with different absolute model paths.
    cfg_a_other_path = HandRetargetConfig(
        mjcf_path="/elsewhere/model.xml", laplacian_distance_weight_k=20.0,
    )
    assert cfg_a_other_path.make_stamp() == stamp_a
