"""
Baseline (wuji_retargeting) replay with precomputed cache, pause/rewind, keypoint overlay.

Usage:
    python demos/legacy/play_manus.py
    python demos/legacy/play_manus.py --live
    python demos/legacy/play_manus.py --collision
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from wuji_retargeting import Retargeter

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
WUJI_DEMO = SCRIPT_DIR / "wuji_manus_demo"

sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR))
import os; _WUJI_SDK = os.environ.get("WUJI_SDK_PATH", "/home/l/ws/doc/WujiRepo/wuji_retargeting_private/public"); sys.path.insert(0, _WUJI_SDK)
from hand_retarget.mediapipe_io import preprocess_landmarks  # noqa: E402

from demos.shared.cache import load_or_compute  # noqa: E402
from demos.shared.overlay import KEY_DOWN, KEY_UP, add_sphere, set_geom_alpha  # noqa: E402
from demos.shared.playback import PlaybackController  # noqa: E402

DEFAULT_PKL = PROJECT_DIR / "data" / "manus_for_pinch" / "manus1_5k.pkl"
DEFAULT_CONFIG = WUJI_DEMO / "config" / "retarget_manus_left.yaml"
DEFAULT_MJCF = Path("/home/l/ws/doc/WujiRepo/urdf_cali/reference/result/xml/left.xml")
HAND_SIDE = "left"

# Colors (match play_interaction_mesh.py)
COL_SOURCE = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)   # red
COL_ROBOT = np.array([0.0, 0.9, 0.9, 0.8], dtype=np.float32)    # cyan
SPHERE_SIZE = 0.006

# Baseline uses these MediaPipe indices for tip tracking
MP_TIP_INDICES = [4, 8, 12, 16, 20]
MP_ALL_INDICES = list(range(21))


def main():
    parser = argparse.ArgumentParser(description="Baseline retargeting replay")
    parser.add_argument("--pkl", type=str, default=str(DEFAULT_PKL))
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--mjcf", type=str, default=str(DEFAULT_MJCF))
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--collision", action="store_true")
    parser.add_argument("--live", action="store_true", help="Live retarget (no precompute)")
    args = parser.parse_args()

    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path(args.mjcf)
    data = mujoco.MjData(model)

    # Semi-transparent hand, keep ground opaque
    set_geom_alpha(model, alpha=0.25)

    if not args.collision:
        model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT

    data.qpos[:] = 0.0
    mujoco.mj_forward(model, data)

    # Load data
    with open(args.pkl, "rb") as f:
        recording = pickle.load(f)
    frames = [f for f in recording if not np.allclose(f[f"{HAND_SIDE}_fingers"], 0)]
    total_frames = len(frames)
    timestamps = np.array([f["t"] for f in frames])

    if total_frames > 1 and (timestamps[-1] - timestamps[0]) > 0.01:
        avg_dt = (timestamps[-1] - timestamps[0]) / (total_frames - 1)
    else:
        avg_dt = 1.0 / 30.0

    # Preprocess source landmarks for visualization
    from hand_retarget.config import HandRetargetConfig
    im_config = HandRetargetConfig.from_yaml(
        str(PROJECT_DIR / "config" / "interaction_mesh_left.yaml"))
    proc_landmarks = np.zeros((total_frames, 21, 3))
    for t in range(total_frames):
        proc_landmarks[t] = preprocess_landmarks(
            frames[t][f"{HAND_SIDE}_fingers"],
            im_config.mediapipe_rotation,
            hand_side=HAND_SIDE,
            global_scale=1.0,
        )

    # Retargeter
    retargeter = Retargeter.from_yaml(args.config, HAND_SIDE)

    # --- Precompute or live ---
    CACHE_DIR = PROJECT_DIR / "data" / "cache"
    cache_path = CACHE_DIR / f"{Path(args.pkl).stem}_bl_cache.npz"

    qpos_cache = None
    if not args.live:
        def _compute_baseline():
            print(f"No cache found, pre-retargeting {total_frames} frames...")
            qpos = np.zeros((total_frames, model.nq))
            t0 = time.time()
            for t in range(total_frames):
                qpos[t] = retargeter.retarget(frames[t][f"{HAND_SIDE}_fingers"])
                if (t + 1) % 500 == 0:
                    fps = (t + 1) / (time.time() - t0)
                    print(f"  {t + 1}/{total_frames} ({fps:.0f} fps)")
            print(f"Pre-retargeting done in {time.time() - t0:.1f}s")
            return qpos

        qpos_cache = load_or_compute(cache_path, total_frames, _compute_baseline)

    # --- Visibility state ---
    vis_state = {"source": True, "robot": True}

    def _custom_key_handler(keycode: int) -> bool:
        """Handle UP/DOWN for visibility toggles. Returns True if consumed."""
        if keycode == KEY_UP:
            vis_state["source"] = not vis_state["source"]
            print(f"  [SOURCE {'ON' if vis_state['source'] else 'OFF'}]")
            return True
        if keycode == KEY_DOWN:
            vis_state["robot"] = not vis_state["robot"]
            print(f"  [ROBOT {'ON' if vis_state['robot'] else 'OFF'}]")
            return True
        return False

    # --- Playback controller ---
    playback = PlaybackController(
        total_frames=total_frames,
        avg_dt=avg_dt,
        speed=args.speed,
        loop=True,
        custom_key_handler=_custom_key_handler,
    )

    viewer = mujoco.viewer.launch_passive(model, data, key_callback=playback.key_callback)
    viewer.cam.azimuth = 180
    viewer.cam.elevation = -20
    viewer.cam.distance = 0.5
    viewer.cam.lookat[:] = [0, 0, 0.05]

    mode_str = "LIVE" if args.live else "PRECOMPUTED"
    print(f"Playing: {args.pkl} ({total_frames} frames)")
    print(f"Mode:    {mode_str} | Baseline (NLopt IK)")
    print(f"Speed:   {args.speed}x")
    if not args.live:
        print("Keys:    SPACE=pause  LEFT=step/reverse  UP=source  DOWN=robot")
    print("=" * 50)

    frame_count = 0
    fps_start = time.time()
    live_frame_idx = 0
    last_live_time = time.time()

    try:
        while viewer.is_running():
            if args.live:
                # --- LIVE MODE ---
                now = time.time()
                dt = now - last_live_time
                adv = int(dt / (avg_dt / args.speed))
                if adv < 1:
                    time.sleep(0.001)
                    viewer.sync()
                    continue
                last_live_time = now
                live_frame_idx += adv
                if live_frame_idx >= total_frames:
                    live_frame_idx = 0
                current_idx = live_frame_idx
                qpos = retargeter.retarget(frames[current_idx][f"{HAND_SIDE}_fingers"])
            else:
                # --- PRECOMPUTE MODE (via PlaybackController) ---
                current_idx, need_update = playback.advance()
                if not need_update:
                    viewer.sync()
                    continue
                current_idx = max(0, min(current_idx, total_frames - 1))
                qpos = qpos_cache[current_idx]

            # Apply qpos
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)

            # Draw keypoint overlay
            any_vis = vis_state["source"] or vis_state["robot"]
            if any_vis:
                # Robot tip positions from MuJoCo FK
                robot_tip_names = [f"left_finger{i}_link4" for i in range(1, 6)]
                robot_tips = np.array([data.xpos[model.body(n).id].copy() for n in robot_tip_names])

                # Source landmarks
                source_pts = proc_landmarks[current_idx]

                with viewer.lock():
                    viewer.user_scn.ngeom = 0
                    if vis_state["source"]:
                        for idx in MP_ALL_INDICES:
                            add_sphere(viewer.user_scn, source_pts[idx], COL_SOURCE, size=SPHERE_SIZE)
                    if vis_state["robot"]:
                        for pt in robot_tips:
                            add_sphere(viewer.user_scn, pt, COL_ROBOT, size=SPHERE_SIZE)
            else:
                with viewer.lock():
                    viewer.user_scn.ngeom = 0

            viewer.sync()

            frame_count += 1
            if frame_count % 500 == 0:
                fps = frame_count / (time.time() - fps_start)
                print(f"Frame {current_idx}/{total_frames}, FPS: {fps:.1f}")

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
