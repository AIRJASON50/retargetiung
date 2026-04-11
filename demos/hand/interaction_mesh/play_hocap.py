"""
HO-Cap retargeting viewer. Auto-detects single/bimanual hand clips.

Usage:
    python demos/hand/interaction_mesh/play_hocap.py
    python demos/hand/interaction_mesh/play_hocap.py --clip hocap__subject_2__20231022_200657__seg00
    python demos/hand/interaction_mesh/play_hocap.py --obj-samples 100
"""

import argparse
import json
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
from scene_builder.hand_builder import (
    load_scene_model, _inject_wrist6dof_mode, _inject_fingertip_sites,
)

PROJECT_DIR = Path(__file__).resolve().parents[3]
HOCAP_DIR = PROJECT_DIR / "data" / "hocap" / "hocap"
DEFAULT_CLIP = "hocap__subject_1__20231025_165502__seg00"

SCENE_LEFT = PROJECT_DIR / "assets" / "scenes" / "single_hand_obj_left.xml"
SCENE_RIGHT = PROJECT_DIR / "assets" / "scenes" / "single_hand_obj.xml"
SCENE_BIMANUAL = PROJECT_DIR / "assets" / "scenes" / "bimanual_hand_obj.xml"

COL_SOURCE = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
COL_ROBOT = np.array([0.0, 1.0, 0.3, 1.0], dtype=np.float32)
COL_OBJ_PTS = np.array([0.3, 0.3, 1.0, 0.8], dtype=np.float32)
COL_MESH = np.array([0.0, 0.7, 0.7, 0.4], dtype=np.float32)


def _add_sphere(scene, pos, rgba, size=0.005):
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


def detect_handedness(npz_path, meta_path):
    """Detect which hands have data in the clip."""
    with open(meta_path) as f:
        meta = json.load(f)
    handedness = meta.get("handedness", "unknown")

    # Also verify data availability
    data = np.load(npz_path, allow_pickle=True)
    has_left = "mediapipe_l_world" in data and data["mediapipe_l_world"].dtype != object
    has_right = "mediapipe_r_world" in data and data["mediapipe_r_world"].dtype != object

    hands = []
    if has_left:
        hands.append("left")
    if has_right:
        hands.append("right")
    return hands, handedness


def retarget_hand(clip, hand_side, scene_xml, obj_samples, semantic_weight):
    """Retarget one hand and return qpos + viz data."""
    from wuji_retargeting.mediapipe import apply_mediapipe_transformations
    from scipy.spatial.transform import Rotation

    config = HandRetargetConfig(
        mjcf_path=str(scene_xml), hand_side=hand_side,
        floating_base=True, object_sample_count=obj_samples,
    )
    retargeter = InteractionMeshHandRetargeter(config)
    qpos = retargeter.retarget_hocap_sequence(clip, use_semantic_weights=semantic_weight)

    # Precompute viz data
    T = len(clip["landmarks"])
    obj_pts_viz = []
    source_pts_viz = []
    obj_pose_viz = []

    for t in range(T):
        lm_raw = clip["landmarks"][t]
        wrist = lm_raw[0]
        raw_centered = lm_raw - wrist
        transformed = apply_mediapipe_transformations(lm_raw.copy(), hand_side)
        R_t, _, _, _ = np.linalg.lstsq(raw_centered[1:6], transformed[1:6], rcond=None)

        angles = [config.mediapipe_rotation.get(k, 0) for k in "xyz"]
        R_extra = Rotation.from_euler("xyz", angles, degrees=True).as_matrix() if any(a != 0 for a in angles) else np.eye(3)
        R_full = R_t @ R_extra.T

        obj_world = transform_object_points(clip["object_pts_local"], clip["object_q"][t], clip["object_t"][t])
        obj_viz = (obj_world - wrist) @ R_full
        obj_pts_viz.append(obj_viz)

        obj_center_viz = (clip["object_t"][t] - wrist) @ R_full
        R_obj_world = Rotation.from_quat(clip["object_q"][t]).as_matrix()
        R_obj_viz = R_full.T @ R_obj_world
        q_xyzw = Rotation.from_matrix(R_obj_viz).as_quat()
        obj_pose_viz.append((obj_center_viz, np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])))

        source_pts_viz.append(transformed)

    return {
        "qpos": qpos,
        "retargeter": retargeter,
        "obj_pts_viz": obj_pts_viz,
        "source_pts_viz": source_pts_viz,
        "obj_pose_viz": obj_pose_viz,
        "config": config,
    }


def main():
    parser = argparse.ArgumentParser(description="HO-Cap retargeting viewer (auto single/bimanual)")
    parser.add_argument("--clip", type=str, default=DEFAULT_CLIP)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--obj-samples", type=int, default=50)
    parser.add_argument("--semantic-weight", action="store_true")
    parser.add_argument("--frames", type=int, default=None, help="Limit number of frames")
    args = parser.parse_args()

    # Resolve paths
    clip_id = args.clip
    if not clip_id.endswith(".npz"):
        npz_path = str(HOCAP_DIR / "motions" / f"{clip_id}.npz")
        meta_path = str(HOCAP_DIR / "motions" / f"{clip_id}.meta.json")
    else:
        npz_path = clip_id
        meta_path = clip_id.replace(".npz", ".meta.json")

    # Detect handedness
    hands, handedness = detect_handedness(npz_path, meta_path)
    bimanual = len(hands) == 2
    print(f"Clip: {clip_id}")
    print(f"Handedness: {handedness} → active hands: {hands}")

    # Get object info
    with open(meta_path) as f:
        meta = json.load(f)
    asset_name = meta["objects"][0]["asset_name"]
    mesh_path = str((HOCAP_DIR / "assets" / asset_name / "mesh_med.stl").resolve())
    print(f"Object: {asset_name}")

    # Retarget each hand
    hand_data = {}
    for hand_side in hands:
        scene = SCENE_LEFT if hand_side == "left" else SCENE_RIGHT
        print(f"\nRetargeting {hand_side} hand...")
        clip = load_hocap_clip(npz_path, meta_path, str(HOCAP_DIR / "assets"),
                               hand_side=hand_side, sample_count=args.obj_samples)
        # Limit frames if requested
        if args.frames:
            N = min(args.frames, len(clip["landmarks"]))
            clip = {k: v[:N] if isinstance(v, np.ndarray) and v.ndim > 0 and v.shape[0] >= N else v
                    for k, v in clip.items()}
            clip["object_pts_local"] = load_hocap_clip(npz_path, meta_path, str(HOCAP_DIR / "assets"),
                                                        hand_side=hand_side, sample_count=args.obj_samples)["object_pts_local"]
        hand_data[hand_side] = retarget_hand(clip, hand_side, scene, args.obj_samples, args.semantic_weight)

    total_frames = len(next(iter(hand_data.values()))["qpos"])
    fps = clip["fps"]
    avg_dt = 1.0 / fps

    # Build MuJoCo viz model
    if bimanual:
        spec = mujoco.MjSpec.from_file(str(SCENE_BIMANUAL))
        _inject_fingertip_sites(spec, "right", name_prefix="rh_", body_prefix="rh_")
        _inject_fingertip_sites(spec, "left", name_prefix="lh_", body_prefix="lh_")
        _inject_wrist6dof_mode(spec, wrist_body_name="wuji_rh_wrist", joint_prefix="rh_")
        _inject_wrist6dof_mode(spec, wrist_body_name="wuji_lh_wrist", joint_prefix="lh_")
    else:
        hand_side = hands[0]
        scene = SCENE_LEFT if hand_side == "left" else SCENE_RIGHT
        spec = mujoco.MjSpec.from_file(str(scene))
        _inject_fingertip_sites(spec, hand_side)
        _inject_wrist6dof_mode(spec)

    # Inject object mesh
    obj_body = spec.worldbody.add_body()
    obj_body.name = "hocap_object"
    obj_body.mocap = True
    obj_mesh = spec.add_mesh()
    obj_mesh.name = "hocap_obj_mesh"
    obj_mesh.file = mesh_path
    obj_geom = obj_body.add_geom()
    obj_geom.name = "hocap_obj_geom"
    obj_geom.type = mujoco.mjtGeom.mjGEOM_MESH
    obj_geom.meshname = "hocap_obj_mesh"
    obj_geom.rgba = [0.6, 0.4, 0.2, 0.5]
    obj_geom.group = 0
    obj_geom.contype = 0
    obj_geom.conaffinity = 0

    model = spec.compile()
    data = mujoco.MjData(model)

    # Semi-transparent hands
    for i in range(model.ngeom):
        geom_name = model.geom(i).name
        is_obj = "hocap" in geom_name
        is_ground = model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE
        if not is_obj and not is_ground:
            model.geom_rgba[i, 3] = 0.25

    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT

    # Map qpos indices for each hand in the combined model
    # Bimanual: [0:26] = right, [26:52] = left
    # Single: [0:26] = that hand
    qpos_slices = {}
    if bimanual:
        qpos_slices["right"] = slice(0, 26)
        qpos_slices["left"] = slice(26, 52)
    else:
        qpos_slices[hands[0]] = slice(0, 26)

    mujoco.mj_forward(model, data)

    # Playback state
    KEY_SPACE, KEY_LEFT, KEY_UP, KEY_DOWN, KEY_RIGHT = 32, 263, 265, 264, 262
    state = {"paused": False, "direction": 1, "step_request": 0, "frame_idx": 0, "resume_flag": False}
    vis = {"mesh": False, "source": True, "robot": True}

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
    viewer.cam.distance = 0.6
    viewer.cam.lookat[:] = [0, 0, 0.15]

    mode_str = "bimanual" if bimanual else f"single ({hands[0]})"
    print(f"\nPlaying: {mode_str}, {total_frames} frames, {fps} fps")
    print(f"Keys: SPACE=pause LEFT=step/reverse UP=mesh DOWN=source RIGHT=robot")
    print("=" * 50)

    last_frame_time = time.time()

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

            # Set qpos for each hand
            for hand_side in hands:
                hd = hand_data[hand_side]
                q = hd["qpos"][idx].copy()

                # For bimanual: offset wrist translation by world-frame wrist position
                # (each hand was retargeted in its own wrist-relative frame)
                if bimanual and hasattr(hd["retargeter"], "_source_wrist_world"):
                    wrist_world = hd["retargeter"]._source_wrist_world[idx]
                    # Apply the same SVD+MANO rotation to get viz-frame offset
                    # Use the precomputed R_full from source_pts_viz
                    # Simpler: just use raw world offset (approximate but separates hands)
                    q[0] += wrist_world[0]  # tx
                    q[1] += wrist_world[1]  # ty
                    q[2] += wrist_world[2]  # tz

                data.qpos[qpos_slices[hand_side]] = q

            # Set object mocap — use world-frame object position for bimanual
            if bimanual:
                # Object in world frame (not wrist-relative, since hands are now in world)
                raw_data = np.load(npz_path, allow_pickle=True)
                obj_t_frame = raw_data["object_t"][idx, 0]
                obj_q_frame = raw_data["object_q"][idx, 0]
                data.mocap_pos[0] = obj_t_frame
                # Convert xyzw → wxyz for MuJoCo
                data.mocap_quat[0] = [obj_q_frame[3], obj_q_frame[0], obj_q_frame[1], obj_q_frame[2]]
            else:
                first_hand = hand_data[hands[0]]
                obj_pos, obj_quat = first_hand["obj_pose_viz"][idx]
                data.mocap_pos[0] = obj_pos
                data.mocap_quat[0] = obj_quat
            mujoco.mj_forward(model, data)

            # Draw overlay
            any_vis = vis["source"] or vis["robot"] or vis["mesh"]
            if any_vis:
                with viewer.lock():
                    viewer.user_scn.ngeom = 0

                    for hand_side in hands:
                        hd = hand_data[hand_side]
                        ret = hd["retargeter"]
                        source_lm = hd["source_pts_viz"][idx]
                        source_mapped = source_lm[ret.mp_indices]
                        obj_pts = hd["obj_pts_viz"][idx]

                        if vis["source"]:
                            for pt in source_mapped:
                                _add_sphere(viewer.user_scn, pt, COL_SOURCE)
                            for pt in obj_pts:
                                _add_sphere(viewer.user_scn, pt, COL_OBJ_PTS, 0.003)

                        if vis["robot"]:
                            ret.hand.forward(hd["qpos"][idx])
                            for name in ret.body_names:
                                pt = ret.hand.get_body_pos(name)
                                _add_sphere(viewer.user_scn, pt, COL_ROBOT)

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
