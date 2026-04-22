"""HO-Cap retargeting viewer with physics-based collision.

Auto-detects single/bimanual hand clips. Renders hand mesh + overlay
keypoints + object mesh with contact visualization.

Usage:
    python demos/hocap/play_hocap.py
    python demos/hocap/play_hocap.py --clip hocap__subject_2__20231022_200657__seg00
    python demos/hocap/play_hocap.py --obj-samples 100
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as RotLib

# Add src and project root to path
PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR))

from hand_retarget import InteractionMeshHandRetargeter, HandRetargetConfig  # noqa: E402
from hand_retarget.mediapipe_io import load_hocap_clip, transform_object_points  # noqa: E402
from hand_retarget.mesh_utils import create_interaction_mesh, get_adjacency_list, get_edge_list  # noqa: E402
from scene_builder.hand_builder import _inject_wrist6dof_mode, _inject_fingertip_sites  # noqa: E402

from demos.shared.overlay import (  # noqa: E402
    KEY_DOWN,
    KEY_RIGHT,
    KEY_UP,
    add_line,
    add_sphere,
    set_geom_alpha,
)
from demos.shared.playback import PlaybackController  # noqa: E402

# ============================================================
# Constants
# ============================================================

HOCAP_DIR = PROJECT_DIR / "data" / "hocap" / "hocap"
DEFAULT_CLIP = "hocap__subject_1__20231025_165502__seg00"

SCENE_LEFT = PROJECT_DIR / "assets" / "scenes" / "single_hand_obj_left.xml"
SCENE_RIGHT = PROJECT_DIR / "assets" / "scenes" / "single_hand_obj.xml"
SCENE_BIMANUAL = PROJECT_DIR / "assets" / "scenes" / "bimanual_hand_obj.xml"

COL_SOURCE = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
COL_ROBOT = np.array([0.0, 1.0, 0.3, 1.0], dtype=np.float32)
COL_OBJ_PTS = np.array([0.3, 0.3, 1.0, 0.8], dtype=np.float32)
COL_EDGE_KEPT = np.array([0.0, 1.0, 0.2, 0.8], dtype=np.float32)   # green: kept (< threshold)
COL_EDGE_LONG = np.array([1.0, 0.2, 0.2, 0.25], dtype=np.float32)  # red translucent: filtered


# ============================================================
# Collision setup helper
# ============================================================


def _enable_hand_collision(body):
    """Recursively set collision groups on hand geoms (contype=1, conaffinity=2)."""
    for geom in body.geoms:
        geom.contype = 1
        geom.conaffinity = 2
    for child in body.bodies:
        _enable_hand_collision(child)


# ============================================================
# Data loading / retargeting
# ============================================================


def detect_handedness(npz_path: str, meta_path: str) -> tuple[list[str], str]:
    """Detect which hands have data in the clip."""
    with open(meta_path) as f:
        meta = json.load(f)
    handedness = meta.get("handedness", "unknown")

    data = np.load(npz_path, allow_pickle=True)
    has_left = "mediapipe_l_world" in data and data["mediapipe_l_world"].dtype != object
    has_right = "mediapipe_r_world" in data and data["mediapipe_r_world"].dtype != object

    hands = []
    if has_left:
        hands.append("left")
    if has_right:
        hands.append("right")
    return hands, handedness


HOCAP_CONFIG_YAML = PROJECT_DIR / "config" / "hocap.yaml"


def retarget_hand(clip: dict, hand_side: str, scene_xml: Path,
                  obj_samples: int, semantic_weight: bool,
                  link_midpoint: bool = False,
                  angle_warmup: bool = False) -> dict:
    """Retarget one hand and return qpos + per-frame wrist transforms."""
    config = HandRetargetConfig.from_yaml(
        str(HOCAP_CONFIG_YAML),
        mjcf_path=str(scene_xml),
        hand_side=hand_side,
        object_sample_count=obj_samples,  # CLI override
    )
    if link_midpoint:
        config.use_link_midpoints = True
    if angle_warmup:
        config.use_angle_warmup = True
    retargeter = InteractionMeshHandRetargeter(config)
    qpos = retargeter.retarget_hocap_sequence(clip, use_semantic_weights=semantic_weight)

    # Per-frame wrist transform for world-frame visualization
    # R_align = R_wrist.T @ OPERATOR2MANO (same as retargeter._align_frame)
    # R_inv = R_align.T converts robot-local -> world-centered
    T = len(clip["landmarks"])
    R_inv_list = []
    wrist_list = []

    for t in range(T):
        wrist_list.append(clip["landmarks"][t, 0].copy())
        # Use SVD alignment (consistent with retarget_hocap_sequence which forces SVD)
        from wuji_retargeting.mediapipe import estimate_frame_from_hand_points
        lm_centered = clip["landmarks"][t] - clip["landmarks"][t, 0]
        R_svd = estimate_frame_from_hand_points(lm_centered)
        if retargeter.config.hand_side == "left":
            from wuji_retargeting.mediapipe import OPERATOR2MANO_LEFT
            R_align = R_svd @ np.array(OPERATOR2MANO_LEFT)
        else:
            from wuji_retargeting.mediapipe import OPERATOR2MANO_RIGHT
            R_align = R_svd @ np.array(OPERATOR2MANO_RIGHT)
        R_inv_list.append(R_align.T)

    return {
        "qpos": qpos,
        "retargeter": retargeter,
        "R_inv_list": R_inv_list,
        "wrist_list": wrist_list,
        "clip": clip,
    }


# ============================================================
# MuJoCo model building
# ============================================================


def build_viz_model(
    hands: list[str], bimanual: bool, mesh_path: str,
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Build MuJoCo visualization model with hand(s) + object + collision."""
    if bimanual:
        spec = mujoco.MjSpec.from_file(str(SCENE_BIMANUAL))
        _inject_fingertip_sites(spec, "right", name_prefix="rh_", body_prefix="rh_")
        _inject_fingertip_sites(spec, "left", name_prefix="lh_", body_prefix="lh_")
        _inject_wrist6dof_mode(spec, wrist_body_name="wuji_rh_wrist", joint_prefix="rh_")
        _inject_wrist6dof_mode(spec, wrist_body_name="wuji_lh_wrist", joint_prefix="lh_")
        _enable_hand_collision(spec.body("wuji_rh_wrist"))
        _enable_hand_collision(spec.body("wuji_lh_wrist"))
    else:
        hand_side = hands[0]
        scene = SCENE_LEFT if hand_side == "left" else SCENE_RIGHT
        spec = mujoco.MjSpec.from_file(str(scene))
        _inject_fingertip_sites(spec, hand_side)
        _inject_wrist6dof_mode(spec)
        _enable_hand_collision(spec.body("wuji_wrist"))

    # Object mesh (mocap body, collision with hand)
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
    obj_geom.contype = 2
    obj_geom.conaffinity = 1

    model = spec.compile()
    data = mujoco.MjData(model)

    # Semi-transparent hands (skip object and ground)
    set_geom_alpha(model, alpha=0.25, skip_names=["hocap"])

    model.opt.gravity[:] = 0
    return model, data


def build_joint_mapping(hands: list[str], bimanual: bool) -> tuple[dict, dict, dict]:
    """Build qpos/ctrl slice mappings per hand.

    Returns:
        qpos_slices, ctrl_finger_slices, ctrl_wrist_slices: per-hand slice dicts
    """
    qpos_slices = {}
    ctrl_finger_slices = {}
    ctrl_wrist_slices = {}
    if bimanual:
        qpos_slices["right"] = slice(0, 26)
        qpos_slices["left"] = slice(26, 52)
        ctrl_finger_slices["right"] = slice(0, 20)
        ctrl_finger_slices["left"] = slice(20, 40)
        ctrl_wrist_slices["right"] = slice(40, 46)
        ctrl_wrist_slices["left"] = slice(46, 52)
    else:
        qpos_slices[hands[0]] = slice(0, 26)
        ctrl_finger_slices[hands[0]] = slice(0, 20)
        ctrl_wrist_slices[hands[0]] = slice(20, 26)
    return qpos_slices, ctrl_finger_slices, ctrl_wrist_slices


def qpos_to_world(q: np.ndarray, R_inv: np.ndarray, wrist_w: np.ndarray) -> np.ndarray:
    """Transform retarget qpos from robot-local to world frame."""
    q = q.copy()
    q[:3] = q[:3] @ R_inv + wrist_w
    R_hinge = RotLib.from_euler("XYZ", q[3:6]).as_matrix()
    q[3:6] = RotLib.from_matrix(R_inv.T @ R_hinge).as_euler("XYZ")
    return q


# ============================================================
# Overlay rendering
# ============================================================


def draw_overlay(viewer, mj_data, hand_data, hands, bimanual, vis, idx):
    """Draw source/robot keypoint spheres and interaction mesh lines."""
    with viewer.lock():
        viewer.user_scn.ngeom = 0

        for hand_side in hands:
            hd = hand_data[hand_side]
            ret = hd["retargeter"]

            # Source: direct world coords
            source_world = hd["clip"]["landmarks"][idx][ret.mp_indices]

            # Object points in world
            obj_world = transform_object_points(
                hd["clip"]["object_pts_local"],
                hd["clip"]["object_q"][idx],
                hd["clip"]["object_t"][idx],
            )

            # Robot: physics-resolved positions
            bp = ("lh_" if hand_side == "left" else "rh_") if bimanual else ""
            robot_world = np.array([
                mj_data.site(f"{bp}finger{n.split('_finger')[1].split('_')[0]}_tip").xpos.copy()
                if "tip_link" in n else mj_data.body(f"{bp}{n}").xpos.copy()
                for n in ret.body_names
            ])

            if vis["source"]:
                for pt in source_world:
                    add_sphere(viewer.user_scn, pt, COL_SOURCE)
                for pt in obj_world:
                    add_sphere(viewer.user_scn, pt, COL_OBJ_PTS, 0.003)

            if vis["robot"]:
                for pt in robot_world:
                    add_sphere(viewer.user_scn, pt, COL_ROBOT)

            if vis["mesh"]:
                all_pts = np.vstack([source_world, obj_world])
                _, simp = create_interaction_mesh(all_pts)
                adj = get_adjacency_list(simp, len(all_pts))
                threshold = getattr(ret.config, "delaunay_edge_threshold", None)
                decay_k = getattr(ret.config, "laplacian_distance_weight_k", None)
                for i, j in get_edge_list(adj):
                    dist = np.linalg.norm(all_pts[j] - all_pts[i])
                    if threshold is None or dist < threshold:
                        if decay_k is not None:
                            alpha = float(np.exp(-decay_k * dist))
                            color = np.array([COL_EDGE_KEPT[0], COL_EDGE_KEPT[1],
                                             COL_EDGE_KEPT[2], max(0.1, COL_EDGE_KEPT[3] * alpha)],
                                            dtype=np.float32)
                        else:
                            color = COL_EDGE_KEPT
                    else:
                        color = COL_EDGE_LONG
                    add_line(viewer.user_scn, all_pts[i], all_pts[j], color)


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="HO-Cap retargeting viewer")
    parser.add_argument("--clip", type=str, default=DEFAULT_CLIP)
    parser.add_argument("--cache", type=str, default=None,
                        help="Explicit path to a stamped .npz cache file "
                             "(overrides auto-discovery; format written by scripts/retarget_hocap.py)")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--obj-samples", type=int, default=50)
    parser.add_argument("--semantic-weight", action="store_true")
    parser.add_argument("--link-midpoint", action="store_true", help="Use 20 link midpoints")
    parser.add_argument("--angle-warmup", action="store_true", help="Two-stage: angle warmup before Laplacian")
    parser.add_argument("--frames", type=int, default=None, help="Limit number of frames")
    args = parser.parse_args()

    # Resolve paths -- support numeric ID (e.g. "42") or full clip_id
    clip_id = args.clip
    if clip_id.isdigit():
        # Numeric shorthand -> look up in clip_index.txt
        index_path = PROJECT_DIR / "data" / "cache" / "hocap" / "clip_index.txt"
        num = int(clip_id)
        clip_id = None
        with open(index_path) as f:
            for line in f:
                if line.startswith(f"{num:03d} "):
                    clip_id = line.split()[1]
                    break
        if clip_id is None:
            print(f"Error: clip #{num} not found in {index_path}")
            sys.exit(1)
        print(f"Clip #{num} -> {clip_id}")
    if not clip_id.endswith(".npz"):
        npz_path = str(HOCAP_DIR / "motions" / f"{clip_id}.npz")
        meta_path = str(HOCAP_DIR / "motions" / f"{clip_id}.meta.json")
    else:
        npz_path = clip_id
        meta_path = clip_id.replace(".npz", ".meta.json")

    hands, handedness = detect_handedness(npz_path, meta_path)
    bimanual = len(hands) == 2
    print(f"Clip: {clip_id}")
    print(f"Handedness: {handedness} -> active hands: {hands}")

    with open(meta_path) as f:
        meta = json.load(f)
    asset_name = meta["objects"][0]["asset_name"]
    mesh_path = str((HOCAP_DIR / "assets" / asset_name / "mesh_med.stl").resolve())
    print(f"Object: {asset_name}")

    # ── Cache discovery ──────────────────────────────────────────
    # Priority: --cache flag  >  stamped files (newest)  >  index-based  >  flat
    CACHE_DIR = PROJECT_DIR / "data" / "cache" / "hocap"
    cache_path = None

    if args.cache:
        # Explicit override (accepts relative or absolute path)
        cache_path = Path(args.cache)
        if not cache_path.is_absolute():
            cache_path = PROJECT_DIR / cache_path
        if not cache_path.exists():
            print(f"WARNING: --cache path not found: {cache_path}")
            cache_path = None
        else:
            cached_tmp = np.load(str(cache_path), allow_pickle=True)
            stamp_info = str(cached_tmp.get("stamp", b"unknown"), "utf-8") \
                if "stamp" in cached_tmp else "unknown"
            print(f"Cache:   {cache_path.name}  [stamp={stamp_info}]")

    if cache_path is None:
        # Stamped files: {clip_id}__{stamp}.npz — pick the most recently modified
        stamped = sorted(
            CACHE_DIR.glob(f"{clip_id}__*.npz"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if stamped:
            cache_path = stamped[0]
            cached_tmp = np.load(str(cache_path), allow_pickle=True)
            stamp_info = str(cached_tmp.get("stamp", b"unknown"), "utf-8") \
                if "stamp" in cached_tmp else "unknown"
            print(f"Cache:   {cache_path.name}  [stamp={stamp_info}]")
            if len(stamped) > 1:
                print(f"         ({len(stamped)-1} older stamped file(s) also available)")

    if cache_path is None:
        # Index-based subdirs (legacy batch_retarget_hocap.py output)
        index_path = CACHE_DIR / "clip_index.txt"
        if index_path.exists():
            with open(index_path) as f:
                for line in f:
                    parts_idx = line.split()
                    if len(parts_idx) >= 5 and parts_idx[1] == clip_id:
                        num, hand_dir = parts_idx[0], parts_idx[4]
                        candidate = CACHE_DIR / hand_dir / f"{num}.npz"
                        if candidate.exists():
                            cache_path = candidate
                        break

    if cache_path is None:
        # Flat file (legacy, same name as clip)
        flat = CACHE_DIR / f"{clip_id}.npz"
        if flat.exists():
            cache_path = flat
            print(f"Cache:   {flat.name}  [legacy flat, no stamp]")

    # Retarget each hand (load from cache if available)
    hand_data = {}
    for hand_side in hands:
        scene = SCENE_LEFT if hand_side == "left" else SCENE_RIGHT
        clip = load_hocap_clip(npz_path, meta_path, str(HOCAP_DIR / "assets"),
                               hand_side=hand_side, sample_count=args.obj_samples)
        if args.frames:
            N = min(args.frames, len(clip["landmarks"]))
            pts_local = clip["object_pts_local"]
            clip = {k: v[:N] if isinstance(v, np.ndarray) and v.ndim > 0 and v.shape[0] >= N else v
                    for k, v in clip.items()}
            clip["object_pts_local"] = pts_local

        # Try loading from cache (written by scripts/batch_retarget_hocap.py)
        # Skip old cache when using experimental modes
        use_cache = cache_path is not None and cache_path.exists() and not args.frames
        if getattr(args, "link_midpoint", False) or getattr(args, "angle_warmup", False):
            use_cache = False
        if use_cache:
            cached = np.load(cache_path, allow_pickle=True)
            if f"qpos_{hand_side}" in cached:
                print(f"\n  {hand_side} hand: loaded from cache")
                config = HandRetargetConfig.from_yaml(
                    str(HOCAP_CONFIG_YAML),
                    mjcf_path=str(scene),
                    hand_side=hand_side,
                    object_sample_count=args.obj_samples,
                )
                retargeter = InteractionMeshHandRetargeter(config)
                hand_data[hand_side] = {
                    "qpos": cached[f"qpos_{hand_side}"],
                    "retargeter": retargeter,
                    "R_inv_list": cached[f"R_inv_{hand_side}"],
                    "wrist_list": cached[f"wrist_{hand_side}"],
                    "clip": clip,
                }
                continue

        print(f"\nRetargeting {hand_side} hand...")
        hand_data[hand_side] = retarget_hand(
            clip, hand_side, scene, args.obj_samples, args.semantic_weight,
            link_midpoint=getattr(args, "link_midpoint", False),
            angle_warmup=getattr(args, "angle_warmup", False),
        )

    total_frames = len(next(iter(hand_data.values()))["qpos"])
    fps = clip["fps"]
    avg_dt = 1.0 / fps

    # Build visualization model
    model, data = build_viz_model(hands, bimanual, mesh_path)
    qpos_slices, ctrl_finger_slices, ctrl_wrist_slices = build_joint_mapping(hands, bimanual)
    n_substeps = max(1, int(np.ceil(avg_dt / model.opt.timestep)))

    # Initialize to first frame
    for hand_side in hands:
        hd = hand_data[hand_side]
        q = qpos_to_world(hd["qpos"][0], hd["R_inv_list"][0], hd["wrist_list"][0])
        data.qpos[qpos_slices[hand_side]] = q
        data.ctrl[ctrl_finger_slices[hand_side]] = q[6:26]
        data.ctrl[ctrl_wrist_slices[hand_side]] = q[0:6]
    mujoco.mj_forward(model, data)

    # Visibility state
    vis = {"mesh": False, "source": True, "robot": True}

    def _custom_key_handler(keycode: int) -> bool:
        """Handle UP/DOWN/RIGHT for visibility toggles."""
        if keycode == KEY_UP:
            vis["mesh"] = not vis["mesh"]
            print(f"  [MESH {'ON' if vis['mesh'] else 'OFF'}]")
            return True
        if keycode == KEY_DOWN:
            vis["source"] = not vis["source"]
            print(f"  [SOURCE {'ON' if vis['source'] else 'OFF'}]")
            return True
        if keycode == KEY_RIGHT:
            vis["robot"] = not vis["robot"]
            print(f"  [ROBOT {'ON' if vis['robot'] else 'OFF'}]")
            return True
        return False

    playback = PlaybackController(
        total_frames=total_frames,
        avg_dt=avg_dt,
        speed=args.speed,
        loop=True,
        custom_key_handler=_custom_key_handler,
    )

    viewer = mujoco.viewer.launch_passive(model, data, key_callback=playback.key_callback)
    viewer.cam.azimuth = 180
    viewer.cam.elevation = -25
    viewer.cam.distance = 0.8 if bimanual else 0.6
    viewer.cam.lookat[:] = [0, 0, 0.15]
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

    mode_str = "bimanual" if bimanual else f"single ({hands[0]})"
    print(f"\nPlaying: {mode_str}, {total_frames} frames, {fps} fps")
    print("Keys: SPACE=pause LEFT=step/reverse UP=mesh DOWN=source RIGHT=robot")
    print("=" * 50)

    try:
        while viewer.is_running():
            idx, need_update = playback.advance()
            if not need_update:
                viewer.sync()
                continue

            idx = max(0, min(idx, total_frames - 1))

            # Physics: set targets, step
            data.qvel[:] = 0
            for hand_side in hands:
                hd = hand_data[hand_side]
                q = qpos_to_world(hd["qpos"][idx], hd["R_inv_list"][idx], hd["wrist_list"][idx])
                data.ctrl[ctrl_finger_slices[hand_side]] = q[6:26]
                data.ctrl[ctrl_wrist_slices[hand_side]] = q[0:6]
                data.qpos[qpos_slices[hand_side]] = q

            hd0 = hand_data[hands[0]]
            clip0 = hd0["clip"]
            data.mocap_pos[0] = clip0["object_t"][idx]
            oq = clip0["object_q"][idx]
            data.mocap_quat[0] = [oq[3], oq[0], oq[1], oq[2]]

            for _ in range(n_substeps):
                mujoco.mj_step(model, data)

            # Overlay
            if vis["source"] or vis["robot"] or vis["mesh"]:
                draw_overlay(viewer, data, hand_data, hands, bimanual, vis, idx)
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
