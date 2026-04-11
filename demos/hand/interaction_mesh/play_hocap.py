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

    # Precompute per-frame inverse transforms for world-frame visualization
    # retarget works in SVD+MANO rotated wrist-relative frame
    # viz needs world frame: world_pos = R_full_inv @ local_pos + wrist_world
    T = len(clip["landmarks"])
    R_inv_list = []    # per-frame inverse rotation
    wrist_list = []    # per-frame wrist world position

    for t in range(T):
        lm_raw = clip["landmarks"][t]
        wrist = lm_raw[0]
        raw_centered = lm_raw - wrist
        transformed = apply_mediapipe_transformations(lm_raw.copy(), hand_side)
        R_t, _, _, _ = np.linalg.lstsq(raw_centered[1:6], transformed[1:6], rcond=None)

        angles = [config.mediapipe_rotation.get(k, 0) for k in "xyz"]
        R_extra = Rotation.from_euler("xyz", angles, degrees=True).as_matrix() if any(a != 0 for a in angles) else np.eye(3)
        R_full = R_t @ R_extra.T

        # R_full transforms: world_centered → preprocessed
        # R_full_inv transforms: preprocessed → world_centered
        R_inv_list.append(np.linalg.inv(R_full))
        wrist_list.append(wrist.copy())

    return {
        "qpos": qpos,
        "retargeter": retargeter,
        "R_inv_list": R_inv_list,
        "wrist_list": wrist_list,
        "clip": clip,
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

    # For bimanual: hide hand mesh (FK is in wrong frame for viz, overlay shows world-frame)
    # For single hand: semi-transparent
    for i in range(model.ngeom):
        geom_name = model.geom(i).name
        is_obj = "hocap" in geom_name
        is_ground = model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE
        if not is_obj and not is_ground:
            model.geom_rgba[i, 3] = 0.0 if bimanual else 0.25

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

    # Preload bimanual object world-frame data (avoid per-frame np.load)
    bimanual_obj_t = None
    bimanual_obj_q_wxyz = None
    if bimanual:
        raw_data = np.load(npz_path, allow_pickle=True)
        obj_t_all = raw_data["object_t"][:total_frames, 0, :]  # (T, 3)
        obj_q_all = raw_data["object_q"][:total_frames, 0, :]  # (T, 4) xyzw
        bimanual_obj_t = obj_t_all
        bimanual_obj_q_wxyz = np.stack([obj_q_all[:, 3], obj_q_all[:, 0], obj_q_all[:, 1], obj_q_all[:, 2]], axis=1)

    viewer = mujoco.viewer.launch_passive(model, data, key_callback=key_callback)
    viewer.cam.azimuth = 180
    viewer.cam.elevation = -25
    viewer.cam.distance = 0.8 if bimanual else 0.6
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

            # Set qpos — retarget qpos is in wrist-relative frame, no offset needed for MuJoCo
            # (MuJoCo model is wrist-relative too — wrist body at origin)
            for hand_side in hands:
                hd = hand_data[hand_side]
                data.qpos[qpos_slices[hand_side]] = hd["qpos"][idx]

            # Object mocap in world frame
            data.mocap_pos[0] = bimanual_obj_t[idx] if bimanual else [0, 0, 0]
            data.mocap_quat[0] = bimanual_obj_q_wxyz[idx] if bimanual else [1, 0, 0, 0]
            mujoco.mj_forward(model, data)

            # Draw overlay — transform everything to world frame for visualization
            any_vis = vis["source"] or vis["robot"] or vis["mesh"]
            if any_vis:
                with viewer.lock():
                    viewer.user_scn.ngeom = 0

                    for hand_side in hands:
                        hd = hand_data[hand_side]
                        ret = hd["retargeter"]
                        R_inv = hd["R_inv_list"][idx]
                        wrist_w = hd["wrist_list"][idx]

                        # Source: preprocessed (wrist-relative rotated) → world
                        lm_raw = hd["clip"]["landmarks"][idx]
                        source_world = lm_raw[ret.mp_indices]  # direct world coords from data

                        # Object points in world frame
                        obj_world = transform_object_points(
                            hd["clip"]["object_pts_local"],
                            hd["clip"]["object_q"][idx],
                            hd["clip"]["object_t"][idx],
                        )

                        # Robot FK: retarget qpos → FK in wrist-relative → inverse rotate → world
                        ret.hand.forward(hd["qpos"][idx])
                        robot_local = ret._get_robot_keypoints()  # wrist-relative, rotated
                        robot_world = (robot_local @ R_inv.T) + wrist_w  # back to world

                        if vis["source"]:
                            for pt in source_world:
                                _add_sphere(viewer.user_scn, pt, COL_SOURCE)
                            for pt in obj_world:
                                _add_sphere(viewer.user_scn, pt, COL_OBJ_PTS, 0.003)

                        if vis["robot"]:
                            for pt in robot_world:
                                _add_sphere(viewer.user_scn, pt, COL_ROBOT)

                        if vis["mesh"]:
                            all_pts = np.vstack([source_world, obj_world])
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
