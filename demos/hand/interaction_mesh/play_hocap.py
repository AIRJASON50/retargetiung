"""
HO-Cap object interaction retargeting viewer.

Shows floating hand (26 DOF) retargeted with object surface anchors.
  - Semi-transparent hand mesh
  - Red spheres: source MediaPipe keypoints
  - Blue spheres: object surface sample points
  - Cyan lines: Delaunay mesh edges (hand + object combined)
  - Green spheres: robot FK keypoints

Controls:
  SPACE       pause/resume
  LEFT        step backward (paused) / reverse (playing)
  UP          toggle mesh edges
  DOWN        toggle source points
  RIGHT       toggle robot points

Usage:
    python demos/hand/interaction_mesh/play_hocap.py
    python demos/hand/interaction_mesh/play_hocap.py --clip data/hocap/hocap/motions/xxx.npz
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
from hand_retarget.mediapipe_io import (
    load_hocap_clip, preprocess_landmarks, transform_object_points,
)
from hand_retarget.mesh_utils import create_interaction_mesh, get_adjacency_list

PROJECT_DIR = Path(__file__).resolve().parents[3]
HOCAP_DIR = PROJECT_DIR / "data" / "hocap" / "hocap"
DEFAULT_CLIP = "hocap__subject_1__20231025_165502__seg00"
DEFAULT_SCENE = PROJECT_DIR / "assets" / "scenes" / "single_hand_obj_left.xml"

# Colors
COL_SOURCE = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
COL_ROBOT = np.array([0.0, 1.0, 0.3, 1.0], dtype=np.float32)
COL_OBJ_PTS = np.array([0.3, 0.3, 1.0, 0.8], dtype=np.float32)
COL_MESH = np.array([0.0, 0.7, 0.7, 0.4], dtype=np.float32)
SPHERE_HAND = 0.005
SPHERE_OBJ = 0.003


def _add_sphere(scene, pos, rgba, size):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                         np.array([size, 0, 0], dtype=np.float64),
                         pos.astype(np.float64),
                         np.eye(3, dtype=np.float64).flatten(), rgba)
    scene.ngeom += 1


def _add_line(scene, p1, p2, rgba, width=1.0):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_LINE,
                         np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64),
                         np.eye(3, dtype=np.float64).flatten(), rgba)
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, width,
                          p1.astype(np.float64), p2.astype(np.float64))
    scene.ngeom += 1


def main():
    parser = argparse.ArgumentParser(description="HO-Cap object interaction retargeting viewer")
    parser.add_argument("--clip", type=str, default=DEFAULT_CLIP, help="HO-Cap clip ID or npz path")
    parser.add_argument("--scene", type=str, default=str(DEFAULT_SCENE))
    parser.add_argument("--hand", type=str, default="left")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--obj-samples", type=int, default=50)
    parser.add_argument("--semantic-weight", action="store_true")
    args = parser.parse_args()

    # Resolve clip path
    clip_id = args.clip
    if not clip_id.endswith(".npz"):
        npz_path = str(HOCAP_DIR / "motions" / f"{clip_id}.npz")
        meta_path = str(HOCAP_DIR / "motions" / f"{clip_id}.meta.json")
    else:
        npz_path = clip_id
        meta_path = clip_id.replace(".npz", ".meta.json")

    # Load clip
    clip = load_hocap_clip(npz_path, meta_path, str(HOCAP_DIR / "assets"),
                           hand_side=args.hand, sample_count=args.obj_samples)
    total_frames = len(clip["landmarks"])

    # Config
    config = HandRetargetConfig(
        mjcf_path=args.scene, hand_side=args.hand, floating_base=True,
        object_sample_count=args.obj_samples,
    )
    retargeter = InteractionMeshHandRetargeter(config)

    # Precompute retargeting
    print(f"Clip: {clip['asset_name']}, {total_frames} frames, {clip['fps']} fps")
    print(f"Object samples: {args.obj_samples}")
    print("Pre-retargeting...")

    qpos_cache = retargeter.retarget_hocap_sequence(clip, use_semantic_weights=args.semantic_weight)

    # Also precompute per-frame object points in wrist-relative frame (for visualization)
    from wuji_retargeting.mediapipe import apply_mediapipe_transformations
    obj_pts_viz = []
    source_pts_viz = []
    for t in range(total_frames):
        lm_raw = clip["landmarks"][t]
        wrist = lm_raw[0]
        # Object in wrist-relative + same transform as hand
        obj_world = transform_object_points(clip["object_pts_local"], clip["object_q"][t], clip["object_t"][t])
        obj_rel = obj_world - wrist
        raw_centered = lm_raw - wrist
        transformed = apply_mediapipe_transformations(lm_raw.copy(), args.hand)
        R_t, _, _, _ = np.linalg.lstsq(raw_centered[1:6], transformed[1:6], rcond=None)
        obj_t = obj_rel @ R_t
        angles = [config.mediapipe_rotation.get(k, 0) for k in "xyz"]
        if any(a != 0 for a in angles):
            from scipy.spatial.transform import Rotation
            R_extra = Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
            obj_t = obj_t @ R_extra.T
        obj_pts_viz.append(obj_t)
        source_pts_viz.append(transformed)

    avg_dt = 1.0 / clip["fps"]

    # MuJoCo viewer (use the retargeter's model for rendering)
    model = retargeter.hand.model
    data = retargeter.hand.data

    # Semi-transparent hand
    for i in range(model.ngeom):
        if model.geom_type[i] != mujoco.mjtGeom.mjGEOM_PLANE:
            model.geom_rgba[i, 3] = 0.25
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT

    data.qpos[:retargeter.nq] = qpos_cache[0]
    mujoco.mj_forward(model, data)

    # Playback state
    KEY_SPACE, KEY_LEFT, KEY_UP, KEY_DOWN, KEY_RIGHT = 32, 263, 265, 264, 262
    state = {"paused": False, "direction": 1, "step_request": 0, "frame_idx": 0, "resume_flag": False}
    vis = {"mesh": True, "source": True, "robot": True}

    def key_callback(keycode):
        if keycode == KEY_UP:
            vis["mesh"] = not vis["mesh"]
            print(f"  [MESH {'ON' if vis['mesh'] else 'OFF'}]")
        elif keycode == KEY_DOWN:
            vis["source"] = not vis["source"]
            print(f"  [SOURCE {'ON' if vis['source'] else 'OFF'}]")
        elif keycode == KEY_RIGHT:
            vis["robot"] = not vis["robot"]
            print(f"  [ROBOT {'ON' if vis['robot'] else 'OFF'}]")
        elif keycode == KEY_SPACE:
            was = state["paused"]
            state["paused"] = not was
            if was: state["resume_flag"] = True
            print(f"  [{'PAUSED' if state['paused'] else 'PLAYING'}] frame {state['frame_idx']}/{total_frames}")
        elif keycode == KEY_LEFT:
            if state["paused"]:
                state["step_request"] = -1
            else:
                state["direction"] *= -1

    viewer = mujoco.viewer.launch_passive(model, data, key_callback=key_callback)
    viewer.cam.azimuth = 180
    viewer.cam.elevation = -25
    viewer.cam.distance = 0.4
    viewer.cam.lookat[:] = [0, 0, 0.08]

    print(f"Keys: SPACE=pause LEFT=step/reverse UP=mesh DOWN=source RIGHT=robot")
    print("=" * 50)

    last_frame_time = time.time()
    mp_indices = retargeter.mp_indices
    body_names = retargeter.body_names

    try:
        while viewer.is_running():
            now = time.time()
            idx = state["frame_idx"]
            need_update = False

            if state["paused"]:
                s = state["step_request"]
                if s != 0:
                    idx = max(0, min(idx + s, total_frames - 1))
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
                    idx = idx % total_frames
                    state["frame_idx"] = idx
                    need_update = True
                else:
                    time.sleep(0.001)

            if not need_update:
                viewer.sync()
                continue

            idx = max(0, min(idx, total_frames - 1))

            # Apply qpos
            data.qpos[:retargeter.nq] = qpos_cache[idx]
            mujoco.mj_forward(model, data)

            # Draw overlay
            any_vis = vis["mesh"] or vis["source"] or vis["robot"]
            if any_vis:
                source_lm = source_pts_viz[idx]
                source_mapped = source_lm[mp_indices]
                obj_pts = obj_pts_viz[idx]

                robot_pts = np.array([
                    data.xpos[retargeter.hand.get_body_id(n)].copy()
                    if retargeter.hand.get_body_id(n) >= 0
                    else data.site_xpos[-(retargeter.hand.get_body_id(n) + 1)].copy()
                    for n in body_names
                ])

                with viewer.lock():
                    viewer.user_scn.ngeom = 0

                    if vis["source"]:
                        for pt in source_mapped:
                            _add_sphere(viewer.user_scn, pt, COL_SOURCE, SPHERE_HAND)
                        for pt in obj_pts:
                            _add_sphere(viewer.user_scn, pt, COL_OBJ_PTS, SPHERE_OBJ)

                    if vis["robot"]:
                        for pt in robot_pts:
                            _add_sphere(viewer.user_scn, pt, COL_ROBOT, SPHERE_HAND)

                    if vis["mesh"]:
                        all_pts = np.vstack([source_mapped, obj_pts])
                        _, simp = create_interaction_mesh(all_pts)
                        adj = get_adjacency_list(simp, len(all_pts))
                        edges = {(min(i, j), max(i, j)) for i, nb in enumerate(adj) for j in nb}
                        for i, j in edges:
                            _add_line(viewer.user_scn, all_pts[i], all_pts[j], COL_MESH)
            else:
                with viewer.lock():
                    viewer.user_scn.ngeom = 0

            viewer.sync()

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
