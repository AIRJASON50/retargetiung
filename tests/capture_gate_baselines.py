"""One-shot helper to (re)build gate baselines.

Run after any code change that legitimately shifts the numerical output
(new cost formulation, LM damping, unit normalization, etc.). Gate tests
then compare current qpos against these frozen snapshots.

Usage:
    PYTHONPATH=src python tests/capture_gate_baselines.py            # both
    PYTHONPATH=src python tests/capture_gate_baselines.py --manus    # only Manus
    PYTHONPATH=src python tests/capture_gate_baselines.py --hocap    # only HO-Cap

Produces:
    tests/refactor_gate_baseline.npz   — Manus 100f qpos (20 DOF)
    tests/hocap_gate_baseline.npz      — HO-Cap 50f qpos (26 DOF)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
_WUJI_SDK = os.environ.get(
    "WUJI_SDK_PATH", "/home/l/ws/doc/WujiRepo/wuji_retargeting_private/public"
)
sys.path.insert(0, _WUJI_SDK)

from hand_retarget import HandRetargetConfig, InteractionMeshHandRetargeter  # noqa: E402
from hand_retarget.mediapipe_io import (  # noqa: E402
    load_hocap_clip,
    load_pkl_sequence,
    preprocess_sequence,
)

URDF = Path(_WUJI_SDK) / "wuji_retargeting/wuji_hand_description/urdf/left.urdf"
PKL = PROJECT_DIR / "data/manus_for_pinch/manus1_5k.pkl"
MANUS_BASELINE = PROJECT_DIR / "tests/refactor_gate_baseline.npz"
MANUS_FRAMES = 100

HOCAP_DIR = PROJECT_DIR / "data/hocap/hocap"
HOCAP_CLIP = "hocap__subject_1__20231025_165502__seg00"
HOCAP_HAND = "left"
HOCAP_SCENE = PROJECT_DIR / "assets/scenes/single_hand_obj_left.xml"
HOCAP_CONFIG = PROJECT_DIR / "config/hocap.yaml"
HOCAP_BASELINE = PROJECT_DIR / "tests/hocap_gate_baseline.npz"
HOCAP_FRAMES = 50


def capture_manus() -> None:
    print(f"[manus] retargeting {MANUS_FRAMES} frames...")
    cfg = HandRetargetConfig(mjcf_path=str(URDF))
    r = InteractionMeshHandRetargeter(cfg)
    lm_raw, _ = load_pkl_sequence(str(PKL), "left")
    lm_seq = preprocess_sequence(
        lm_raw, cfg.mediapipe_rotation, hand_side="left", global_scale=1.0,
    )
    q = r.hand.get_default_qpos()
    qpos = np.zeros((MANUS_FRAMES, r.nq))
    for t in range(MANUS_FRAMES):
        q = r.retarget_frame(
            lm_seq[t], q, is_first_frame=(t == 0), use_semantic_weights=True,
        )
        qpos[t] = q
    np.savez(str(MANUS_BASELINE), qpos=qpos)
    print(f"[manus] saved {MANUS_BASELINE}  shape={qpos.shape}  "
          f"range=[{qpos.min():.4f}, {qpos.max():.4f}]")


def capture_hocap() -> None:
    print(f"[hocap] retargeting {HOCAP_FRAMES} frames of {HOCAP_CLIP} ({HOCAP_HAND})...")
    cfg = HandRetargetConfig.from_yaml(
        str(HOCAP_CONFIG), mjcf_path=str(HOCAP_SCENE), hand_side=HOCAP_HAND,
    )
    r = InteractionMeshHandRetargeter(cfg)
    clip = load_hocap_clip(
        str(HOCAP_DIR / "motions" / f"{HOCAP_CLIP}.npz"),
        str(HOCAP_DIR / "motions" / f"{HOCAP_CLIP}.meta.json"),
        str(HOCAP_DIR / "assets"),
        hand_side=HOCAP_HAND,
        sample_count=cfg.object_sample_count,
    )
    pts_local = clip["object_pts_local"]
    clip = {
        k: (v[:HOCAP_FRAMES] if isinstance(v, np.ndarray) and v.ndim > 0 and len(v) >= HOCAP_FRAMES else v)
        for k, v in clip.items()
    }
    clip["object_pts_local"] = pts_local
    qpos = r.retarget_hocap_sequence(clip)
    np.savez(str(HOCAP_BASELINE), qpos=qpos)
    print(f"[hocap] saved {HOCAP_BASELINE}  shape={qpos.shape}  "
          f"range=[{qpos.min():.4f}, {qpos.max():.4f}]  "
          f"wrist_abs_max={np.abs(qpos[:, :6]).max():.2e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manus", action="store_true", help="Only rebuild Manus baseline")
    parser.add_argument("--hocap", action="store_true", help="Only rebuild HO-Cap baseline")
    args = parser.parse_args()

    do_manus = args.manus or not args.hocap
    do_hocap = args.hocap or not args.manus

    if do_manus:
        capture_manus()
    if do_hocap:
        capture_hocap()
