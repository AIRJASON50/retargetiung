"""
Side-by-side comparison: Baseline (NLopt IK) vs Interaction Mesh (Laplacian SOCP).
Two MuJoCo viewers playing the same .pkl data simultaneously.

Usage:
    python scripts/play_compare.py
    python scripts/play_compare.py --live            # live retarget both
    python scripts/play_compare.py --speed 0.5
"""

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from hand_retarget import InteractionMeshHandRetargeter, HandRetargetConfig
from hand_retarget.mediapipe_io import load_pkl_sequence, preprocess_sequence

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[2]
DEFAULT_PKL = PROJECT_DIR / "data" / "manus1_5k.pkl"
DEFAULT_IM_CONFIG = PROJECT_DIR / "config" / "interaction_mesh_left.yaml"
DEFAULT_BL_CONFIG = PROJECT_DIR / "config" / "baseline_left.yaml"
DEFAULT_MJCF = Path("/home/l/ws/doc/WujiRepo/urdf_cali/reference/result/xml/left.xml")
HAND_SIDE = "left"


def setup_viewer(model, data, title_hint="", x_offset=0):
    """Create and configure a MuJoCo passive viewer."""
    viewer = mujoco.viewer.launch_passive(model, data)
    viewer.cam.azimuth = 180
    viewer.cam.elevation = -20
    viewer.cam.distance = 0.5
    viewer.cam.lookat[:] = [0, 0, 0.05]
    return viewer


def main():
    parser = argparse.ArgumentParser(description="Side-by-side retargeting comparison")
    parser.add_argument("--pkl", type=str, default=str(DEFAULT_PKL))
    parser.add_argument("--im-config", type=str, default=str(DEFAULT_IM_CONFIG))
    parser.add_argument("--bl-config", type=str, default=str(DEFAULT_BL_CONFIG))
    parser.add_argument("--mjcf", type=str, default=str(DEFAULT_MJCF))
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--live", action="store_true", help="Live retarget (no precompute)")
    args = parser.parse_args()

    # --- Load data ---
    import pickle
    with open(args.pkl, "rb") as f:
        recording = pickle.load(f)
    frames = [f for f in recording if not np.allclose(f[f"{HAND_SIDE}_fingers"], 0)]
    total_frames = len(frames)

    landmarks_seq, timestamps = load_pkl_sequence(args.pkl, HAND_SIDE)
    if total_frames > 1 and (timestamps[-1] - timestamps[0]) > 0.01:
        avg_dt = (timestamps[-1] - timestamps[0]) / (total_frames - 1)
    else:
        avg_dt = 1.0 / 30.0

    # --- Baseline setup ---
    from wuji_retargeting import Retargeter
    retargeter_bl = Retargeter.from_yaml(args.bl_config, HAND_SIDE)

    model_bl = mujoco.MjModel.from_xml_path(args.mjcf)
    data_bl = mujoco.MjData(model_bl)
    data_bl.qpos[:] = (model_bl.jnt_range[:, 0] + model_bl.jnt_range[:, 1]) / 2
    mujoco.mj_forward(model_bl, data_bl)

    # --- Interaction Mesh setup ---
    im_config = HandRetargetConfig.from_yaml(args.im_config, mjcf_path=args.mjcf)
    retargeter_im = InteractionMeshHandRetargeter(im_config)

    model_im = mujoco.MjModel.from_xml_path(args.mjcf)
    data_im = mujoco.MjData(model_im)
    data_im.qpos[:] = (model_im.jnt_range[:, 0] + model_im.jnt_range[:, 1]) / 2
    mujoco.mj_forward(model_im, data_im)

    # Preprocess for IM
    proc_seq = preprocess_sequence(
        landmarks_seq, im_config.mediapipe_rotation, hand_side=HAND_SIDE, global_scale=retargeter_im.global_scale
    )

    # --- Precompute if not live (with disk cache) ---
    CACHE_DIR = PROJECT_DIR / "data" / "cache"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pkl_stem = Path(args.pkl).stem
    bl_cache_path = CACHE_DIR / f"{pkl_stem}_bl_cache.npz"
    im_cache_path = CACHE_DIR / f"{pkl_stem}_im_cache.npz"

    bl_cache = None
    im_cache = None
    if not args.live:
        # Baseline cache
        if bl_cache_path.exists():
            cached = np.load(bl_cache_path)
            if len(cached["qpos"]) == total_frames:
                bl_cache = cached["qpos"]
                print(f"Loaded baseline cache: {bl_cache_path}")
        if bl_cache is None:
            print(f"Computing baseline ({total_frames} frames)...")
            bl_cache = np.zeros((total_frames, model_bl.nq))
            for t in range(total_frames):
                bl_cache[t] = retargeter_bl.retarget(frames[t][f"{HAND_SIDE}_fingers"])
                if (t + 1) % 3000 == 0:
                    print(f"  Baseline: {t + 1}/{total_frames}")
            np.savez(bl_cache_path, qpos=bl_cache)
            print(f"  Saved: {bl_cache_path}")

        # Interaction Mesh cache
        if im_cache_path.exists():
            cached = np.load(im_cache_path)
            if len(cached["qpos"]) == total_frames:
                im_cache = cached["qpos"]
                print(f"Loaded IM cache: {im_cache_path}")
        if im_cache is None:
            print(f"Computing interaction mesh ({total_frames} frames)...")
            im_cache = np.zeros((total_frames, retargeter_im.nq))
            q_prev = retargeter_im.hand.get_default_qpos()
            for t in range(total_frames):
                q = retargeter_im.retarget_frame(proc_seq[t], q_prev, is_first_frame=(t == 0))
                im_cache[t] = q
                q_prev = q
                if (t + 1) % 3000 == 0:
                    print(f"  IM: {t + 1}/{total_frames}")
            np.savez(im_cache_path, qpos=im_cache)
            print(f"  Saved: {im_cache_path}")

    # --- Playback state ---
    KEY_SPACE = 32
    KEY_LEFT = 263
    KEY_RIGHT = 262

    state = {
        "paused": False,
        "direction": 1,
        "step_request": 0,
        "frame_idx": 0,
        "resume_flag": False,
    }

    def key_callback(keycode: int):
        if args.live:
            return
        if keycode == KEY_SPACE:
            was_paused = state["paused"]
            state["paused"] = not was_paused
            if was_paused:
                state["resume_flag"] = True
            status = "PAUSED" if state["paused"] else "PLAYING"
            print(f"  [{status}] frame {state['frame_idx']}/{total_frames}")
        elif keycode == KEY_LEFT:
            if state["paused"]:
                state["step_request"] = -1
            else:
                state["direction"] = -1
                print("  [REVERSE <<<]")
        elif keycode == KEY_RIGHT:
            if state["paused"]:
                state["step_request"] = 1
            else:
                state["direction"] = 1
                print("  [FORWARD >>>]")

    # --- Launch viewers ---
    viewer_bl = mujoco.viewer.launch_passive(model_bl, data_bl, key_callback=key_callback)
    viewer_bl.cam.azimuth = 180
    viewer_bl.cam.elevation = -20
    viewer_bl.cam.distance = 0.5
    viewer_bl.cam.lookat[:] = [0, 0, 0.05]

    viewer_im = mujoco.viewer.launch_passive(model_im, data_im, key_callback=key_callback)
    viewer_im.cam.azimuth = 180
    viewer_im.cam.elevation = -20
    viewer_im.cam.distance = 0.5
    viewer_im.cam.lookat[:] = [0, 0, 0.05]

    mode_str = "LIVE" if args.live else "PRECOMPUTED"
    print(f"PKL:    {args.pkl} ({total_frames} frames)")
    print(f"Mode:   {mode_str}")
    print(f"Speed:  {args.speed}x")
    print(f"Left viewer:  Baseline (NLopt IK)")
    print(f"Right viewer: Interaction Mesh (Laplacian SOCP)")
    if not args.live:
        print(f"Keys:   SPACE=pause  LEFT/RIGHT=step/direction")
    print("=" * 50)

    frame_count = 0
    fps_start = time.time()
    last_frame_time = time.time()

    # Live state
    live_q_prev = retargeter_im.hand.get_default_qpos()

    try:
        while viewer_bl.is_running() and viewer_im.is_running():
            if args.live:
                # --- LIVE MODE ---
                now = time.time()
                dt = now - last_frame_time
                frames_to_advance = int(dt / (avg_dt / args.speed))
                if frames_to_advance < 1:
                    time.sleep(0.001)
                    viewer_bl.sync()
                    viewer_im.sync()
                    continue
                last_frame_time = now

                idx = state["frame_idx"] + frames_to_advance
                if idx >= total_frames:
                    idx = 0
                    live_q_prev = retargeter_im.hand.get_default_qpos()
                state["frame_idx"] = idx

                # Baseline
                qpos_bl = retargeter_bl.retarget(frames[idx][f"{HAND_SIDE}_fingers"])
                data_bl.qpos[:] = qpos_bl
                mujoco.mj_forward(model_bl, data_bl)

                # Interaction Mesh
                qpos_im = retargeter_im.retarget_frame(
                    proc_seq[idx], live_q_prev, is_first_frame=(frame_count == 0)
                )
                live_q_prev = qpos_im
                data_im.qpos[:] = qpos_im
                mujoco.mj_forward(model_im, data_im)

            else:
                # --- PRECOMPUTE MODE ---
                now = time.time()
                idx = state["frame_idx"]
                need_update = False

                if state["paused"]:
                    step = state["step_request"]
                    if step != 0:
                        idx = max(0, min(idx + step, total_frames - 1))
                        state["step_request"] = 0
                        state["frame_idx"] = idx
                        need_update = True
                    else:
                        time.sleep(0.01)
                else:
                    if state["resume_flag"]:
                        last_frame_time = now
                        state["resume_flag"] = False
                    dt = now - last_frame_time
                    adv = int(dt / (avg_dt / args.speed))
                    if adv >= 1:
                        idx += state["direction"] * adv
                        last_frame_time = now
                        if idx >= total_frames:
                            idx = idx % total_frames
                        elif idx < 0:
                            idx = total_frames + (idx % total_frames)
                        state["frame_idx"] = idx
                        need_update = True
                    else:
                        time.sleep(0.001)

                if not need_update:
                    viewer_bl.sync()
                    viewer_im.sync()
                    continue

                data_bl.qpos[:] = bl_cache[idx]
                mujoco.mj_forward(model_bl, data_bl)

                data_im.qpos[:] = im_cache[idx]
                mujoco.mj_forward(model_im, data_im)

            viewer_bl.sync()
            viewer_im.sync()

            frame_count += 1
            if frame_count % 500 == 0:
                fps = frame_count / (time.time() - fps_start)
                print(f"Frame {state['frame_idx']}/{total_frames}, FPS: {fps:.1f}")

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        viewer_bl.close()
        viewer_im.close()


if __name__ == "__main__":
    main()
