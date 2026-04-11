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

    # Precompute per-frame transforms (wrist-relative frame, matching retargeting)
    from wuji_retargeting.mediapipe import apply_mediapipe_transformations
    from scipy.spatial.transform import Rotation

    obj_pts_viz = []     # (T, M, 3) object surface points in viz frame
    source_pts_viz = []  # (T, 21, 3) hand landmarks in viz frame
    obj_pose_viz = []    # (T, (pos, quat_wxyz)) object center pose in viz frame

    for t in range(total_frames):
        lm_raw = clip["landmarks"][t]
        wrist = lm_raw[0]

        # Compute the SVD+MANO rotation from raw→preprocessed
        raw_centered = lm_raw - wrist
        transformed = apply_mediapipe_transformations(lm_raw.copy(), args.hand)
        R_t, _, _, _ = np.linalg.lstsq(raw_centered[1:6], transformed[1:6], rcond=None)

        angles = [config.mediapipe_rotation.get(k, 0) for k in "xyz"]
        if any(a != 0 for a in angles):
            R_extra = Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
            R_full = R_t @ R_extra.T
        else:
            R_full = R_t

        # Object surface points → viz frame
        obj_world = transform_object_points(clip["object_pts_local"], clip["object_q"][t], clip["object_t"][t])
        obj_rel = obj_world - wrist
        obj_viz = obj_rel @ R_full
        obj_pts_viz.append(obj_viz)

        # Object center pose → viz frame
        obj_center_rel = clip["object_t"][t] - wrist
        obj_center_viz = obj_center_rel @ R_full
        # Object rotation in viz frame
        R_obj_world = Rotation.from_quat(clip["object_q"][t]).as_matrix()
        R_obj_viz = R_full.T @ R_obj_world  # compose rotations
        q_obj_viz = Rotation.from_matrix(R_obj_viz).as_quat()  # xyzw
        q_obj_wxyz = [q_obj_viz[3], q_obj_viz[0], q_obj_viz[1], q_obj_viz[2]]  # MuJoCo uses wxyz
        obj_pose_viz.append((obj_center_viz, np.array(q_obj_wxyz)))

        source_pts_viz.append(transformed)

    avg_dt = 1.0 / clip["fps"]

    # Build MuJoCo model with object mesh for visualization
    # Rebuild from scene XML with MjSpec to inject the HO-Cap object mesh
    from scene_builder.hand_builder import load_scene_model
    obj_mesh_path = clip["mesh_path"]

    # Load base scene model (hand only)
    spec = mujoco.MjSpec.from_file(args.scene)

    # Inject object as a mocap body (position updated each frame)
    obj_body = spec.worldbody.add_body()
    obj_body.name = "hocap_object"
    obj_body.mocap = True

    # Add object mesh asset and geom
    obj_mesh_asset = spec.add_mesh()
    obj_mesh_asset.name = "hocap_obj_mesh"
    obj_mesh_asset.file = str(Path(obj_mesh_path).resolve())

    obj_geom = obj_body.add_geom()
    obj_geom.name = "hocap_obj_geom"
    obj_geom.type = mujoco.mjtGeom.mjGEOM_MESH
    obj_geom.meshname = "hocap_obj_mesh"
    obj_geom.rgba = [0.6, 0.4, 0.2, 0.5]  # semi-transparent brown
    obj_geom.contype = 0
    obj_geom.conaffinity = 0

    # Inject wrist6dof + fingertip sites via hand_builder
    from scene_builder.hand_builder import _inject_wrist6dof_mode, _inject_fingertip_sites
    _inject_fingertip_sites(spec, args.hand)
    _inject_wrist6dof_mode(spec)

    model = spec.compile()
    data = mujoco.MjData(model)

    # Find mocap body id for object
    obj_mocap_id = None
    for i in range(model.nbody):
        if model.body(i).name == "hocap_object":
            obj_mocap_id = i
            break
    # Mocap body index (for data.mocap_pos / data.mocap_quat)
    mocap_idx = 0  # should be the only mocap body

    # Semi-transparent hand, keep object visible
    for i in range(model.ngeom):
        geom_name = model.geom(i).name
        geom_type = model.geom_type[i]
        is_obj = "hocap" in geom_name
        is_ground = geom_type == mujoco.mjtGeom.mjGEOM_PLANE
        if not is_obj and not is_ground:
            model.geom_rgba[i, 3] = 0.25

    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT

    # Map retargeter body names to this new model's body ids
    # (retargeter used a separately compiled model, body ids may differ)
    viz_body_ids = {}
    viz_site_ids = {}
    for i in range(model.nbody):
        viz_body_ids[model.body(i).name] = i
    for i in range(model.nsite):
        viz_site_ids[model.site(i).name] = i

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

            # Apply hand qpos + object mocap pose
            data.qpos[:retargeter.nq] = qpos_cache[idx]
            obj_pos_t, obj_quat_t = obj_pose_viz[idx]
            data.mocap_pos[mocap_idx] = obj_pos_t
            data.mocap_quat[mocap_idx] = obj_quat_t
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
