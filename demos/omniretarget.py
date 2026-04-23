"""
Visualize OmniRetarget's humanoid retargeting with interaction mesh overlay.
Supports three modes: robot_only, box (object interaction), slope (terrain).

Shows:
  - G1 robot (semi-transparent): retargeted qpos
  - Red spheres: original SMPLH human keypoints
  - Red spheres (large): 15 mapped human keypoints
  - Cyan spheres: 15 robot keypoints (FK)
  - Cyan lines: Delaunay mesh edges
  - Green lines: human skeleton
  - Blue spheres: object/terrain sampled points (box/slope modes)
  - Yellow lines: source-target match lines

Controls:
  SPACE       pause/resume
  LEFT/RIGHT  step/direction

Usage:
    PYTHONPATH=src python demos/omniretarget.py                    # robot_only (default)
    PYTHONPATH=src python demos/omniretarget.py --mode box         # with largebox
    PYTHONPATH=src python demos/omniretarget.py --mode slope       # with terrain slope
    PYTHONPATH=src python demos/omniretarget.py --speed 0.3
"""

import argparse
import sys
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch
import trimesh

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from hand_retarget.mesh_utils import create_interaction_mesh, get_adjacency_list  # noqa: E402
from hand_retarget_viz.overlay import add_line, add_sphere, set_geom_alpha  # noqa: E402
from hand_retarget_viz.playback import PlaybackController  # noqa: E402

OMNI_BASE = PROJECT_DIR / "lib" / "25_OmniRetarget" / "code" / "src"
RETARGET_DIR = OMNI_BASE / "holosoma_retargeting" / "holosoma_retargeting"
MOTION_DIR = OMNI_BASE / "holosoma" / "holosoma" / "data" / "motions" / "g1_29dof" / "whole_body_tracking"

# Model paths
G1_XML = RETARGET_DIR / "models" / "g1" / "g1_29dof.xml"
HEIGHT_DICT = RETARGET_DIR / "demo_data" / "height_dict.pkl"

# Data per mode
MODE_CONFIG = {
    # robot_only: 15 human keypoints + 225 ground plane points (15x15 grid, [-1,1])
    "robot_only": {
        "retarget_npz": MOTION_DIR / "sub3_largebox_003_mj.npz",
        "smplh_pt": RETARGET_DIR / "demo_data" / "OMOMO_new" / "sub3_largebox_003.pt",
        "subject": "sub3",
        "object_mesh": None,
        "has_object_traj": False,
        "scene_xml": None,
        "use_robot_body_pos": False,
        "ground_grid_size": 15,        # 15x15 = 225 points
        "ground_range": (-1.0, 1.0),
    },
    # box: 15 human keypoints + 100 uniform box surface samples
    "box": {
        "retarget_npz": MOTION_DIR / "sub3_largebox_003_mj_w_obj.npz",
        "smplh_pt": RETARGET_DIR / "demo_data" / "OMOMO_new" / "sub3_largebox_003.pt",
        "subject": "sub3",
        "object_mesh": MOTION_DIR / "largebox.obj",
        "has_object_traj": True,
        "scene_xml": RETARGET_DIR / "models" / "g1" / "g1_29dof_w_largebox.xml",
        "use_robot_body_pos": False,
        "obj_sample_count": 100,       # uniform surface sampling
        "obj_sample_seed": 0,
    },
    # slope: 15 robot keypoints + 100 weighted terrain samples (20x top bias) + 64 ground points
    "slope": {
        "retarget_npz": MOTION_DIR / "motion_crawl_slope.npz",
        "smplh_pt": None,
        "subject": None,
        "object_mesh": MOTION_DIR / "terrain_slope.obj",
        "has_object_traj": False,
        "scene_xml": None,
        "use_robot_body_pos": True,
        "obj_sample_count": 100,
        "obj_sample_seed": 0,
        "obj_weight_threshold": 0.9,   # z > 0.9 gets 20x weight
        "obj_weight_high": 20,
        "obj_weight_low": 1,
        "ground_grid_size": 8,         # 8x8 = 64 ground points
        "ground_range": (-2.0, 2.0),
    },
}

# SMPLH 52 joint names
SMPLH_JOINTS = [
    "Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe",
    "R_Hip", "R_Knee", "R_Ankle", "R_Toe",
    "Torso", "Spine", "Chest", "Neck", "Head",
    "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist",
    "L_Index1", "L_Index2", "L_Index3",
    "L_Middle1", "L_Middle2", "L_Middle3",
    "L_Pinky1", "L_Pinky2", "L_Pinky3",
    "L_Ring1", "L_Ring2", "L_Ring3",
    "L_Thumb1", "L_Thumb2", "L_Thumb3",
    "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist",
    "R_Index1", "R_Index2", "R_Index3",
    "R_Middle1", "R_Middle2", "R_Middle3",
    "R_Pinky1", "R_Pinky2", "R_Pinky3",
    "R_Ring1", "R_Ring2", "R_Ring3",
    "R_Thumb1", "R_Thumb2", "R_Thumb3",
]

SMPLH_G1_MAPPING = {
    "Pelvis": "pelvis_contour_link",
    "L_Hip": "left_hip_pitch_link", "R_Hip": "right_hip_pitch_link",
    "L_Knee": "left_knee_link", "R_Knee": "right_knee_link",
    "L_Shoulder": "left_shoulder_roll_link", "R_Shoulder": "right_shoulder_roll_link",
    "L_Elbow": "left_elbow_link", "R_Elbow": "right_elbow_link",
    "L_Ankle": "left_ankle_intermediate_1_link", "R_Ankle": "right_ankle_intermediate_1_link",
    "L_Toe": "left_ankle_roll_sphere_5_link", "R_Toe": "right_ankle_roll_sphere_5_link",
    "L_Wrist": "left_rubber_hand_link", "R_Wrist": "right_rubber_hand_link",
}

HUMAN_SKELETON_CHAINS = [
    ["Pelvis", "Torso", "Spine", "Chest", "Neck", "Head"],
    ["Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe"],
    ["Pelvis", "R_Hip", "R_Knee", "R_Ankle", "R_Toe"],
    ["Chest", "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist"],
    ["Chest", "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist"],
]

# Colors
COL_HUMAN_ALL = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
COL_HUMAN_MAPPED = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
COL_ROBOT_KP = np.array([0.0, 1.0, 0.3, 1.0], dtype=np.float32)      # bright green
COL_MESH_EDGE = np.array([0.0, 0.7, 0.7, 0.5], dtype=np.float32)
COL_SKELETON = np.array([0.3, 0.9, 0.3, 0.25], dtype=np.float32)
COL_MATCH_LINE = np.array([1.0, 1.0, 0.0, 0.3], dtype=np.float32)
COL_OBJ_PTS = np.array([0.3, 0.3, 1.0, 0.8], dtype=np.float32)       # blue (object surface)
COL_GROUND_PTS = np.array([0.6, 0.6, 0.6, 0.5], dtype=np.float32)   # gray (ground grid)

SPHERE_BIG = 0.030
SPHERE_MED = 0.024
SPHERE_SMALL = 0.012
SPHERE_OBJ = 0.008


def weighted_surface_sampling(mesh_path, n_samples=100, seed=0,
                              weight_func=None):
    """
    Sample points from mesh surface with per-face weighting.
    Matches OmniRetarget's weighted_surface_sampling() from utils.py.

    Args:
        mesh_path: path to .obj mesh
        n_samples: number of points to sample
        seed: random seed (OmniRetarget uses 0)
        weight_func: callable(face_center) -> weight. Default: uniform.
    """
    mesh = trimesh.load(str(mesh_path))
    rng = np.random.RandomState(seed)

    # Face areas and centers
    triangles = mesh.vertices[mesh.faces]  # (F, 3, 3)
    v0, v1, v2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    face_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    face_centers = triangles.mean(axis=1)  # (F, 3)

    # Apply weights
    if weight_func is not None:
        weights = np.array([weight_func(c) for c in face_centers])
    else:
        weights = np.ones(len(face_areas))

    weighted_areas = face_areas * weights
    probs = weighted_areas / weighted_areas.sum()

    # Sample face indices
    face_indices = rng.choice(len(mesh.faces), size=n_samples, p=probs)

    # Sample points within each triangle using barycentric coordinates
    pts = np.zeros((n_samples, 3))
    for i, fi in enumerate(face_indices):
        r1, r2 = rng.random(), rng.random()
        if r1 + r2 > 1:
            r1, r2 = 1 - r1, 1 - r2
        pts[i] = (1 - r1 - r2) * v0[fi] + r1 * v1[fi] + r2 * v2[fi]

    return pts


def create_ground_points(x_range, y_range, grid_size=8):
    """
    Create uniform ground plane points.
    Matches OmniRetarget's create_ground_points() from robot_retarget.py.
    """
    x = np.linspace(x_range[0], x_range[1], grid_size)
    y = np.linspace(y_range[0], y_range[1], grid_size)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    return np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])


def transform_points(pts, quat_wxyz, trans):
    """Transform points by quaternion (wxyz) + translation."""
    from scipy.spatial.transform import Rotation as R
    quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
    rot = R.from_quat(quat_xyzw).as_matrix()
    return (rot @ pts.T).T + trans


def main():
    parser = argparse.ArgumentParser(description="OmniRetarget humanoid demo viewer")
    parser.add_argument("--mode", choices=["robot_only", "box", "slope"], default="robot_only")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--no-loop", action="store_true")
    parser.add_argument("--no-mesh", action="store_true")
    parser.add_argument("--no-human", action="store_true")
    parser.add_argument("--obj-samples", type=int, default=50, help="Object surface sample count")
    args = parser.parse_args()

    loop = not args.no_loop
    show_mesh = not args.no_mesh
    show_human = not args.no_human
    cfg = MODE_CONFIG[args.mode]

    # --- Load retargeted data ---
    retarget_data = np.load(str(cfg["retarget_npz"]))
    qpos_seq = retarget_data["joint_pos"]
    body_names = list(retarget_data["body_names"])
    retarget_fps = int(retarget_data["fps"])
    T_robot = len(qpos_seq)

    # Object trajectory (box mode only)
    obj_pos_seq = retarget_data.get("object_pos_w", None)
    obj_quat_seq = retarget_data.get("object_quat_w", None)

    # --- Load human data (SMPLH modes) or use robot body_pos_w as fallback ---
    human_joints_all = None
    robot_body_pos_all = None
    T_human = T_robot
    scale = 1.0

    if cfg["smplh_pt"] is not None:
        smplh_raw = torch.load(str(cfg["smplh_pt"]), weights_only=False).detach().numpy()
        human_joints_all = smplh_raw[:, 162:162 + 52 * 3].reshape(-1, 52, 3)
        import pickle
        with open(str(HEIGHT_DICT), "rb") as f:
            height_dict = pickle.load(f)
        human_height = height_dict[cfg["subject"]]
        robot_height = 1.32
        scale = robot_height / human_height
        human_joints_all = human_joints_all * scale
        T_human = len(human_joints_all)
    elif cfg.get("use_robot_body_pos"):
        # No SMPLH data -- use retarget body_pos_w to extract robot keypoints for mesh
        robot_body_pos_all = retarget_data["body_pos_w"]  # (T, nbody, 3)
        T_human = T_robot

    # --- Object/terrain surface points + ground plane points ---
    # Sampling strategy per mode matches OmniRetarget exactly:
    #   robot_only: 225 ground grid points (15x15, [-1,1]), no object
    #   box: 100 uniform box surface samples, no ground
    #   slope: 100 weighted terrain samples (20x top bias) + 64 ground grid points
    obj_surface_pts_local = None
    ground_pts = None

    if cfg.get("object_mesh") is not None:
        # Weighted or uniform surface sampling
        wt = cfg.get("obj_weight_threshold")
        if wt is not None:
            # Weighted sampling (slope/climb mode)
            w_high = cfg.get("obj_weight_high", 20)
            w_low = cfg.get("obj_weight_low", 1)
            weight_func = lambda p: w_high if p[2] > wt else w_low
            obj_surface_pts_local = weighted_surface_sampling(
                cfg["object_mesh"],
                n_samples=cfg.get("obj_sample_count", 100),
                seed=cfg.get("obj_sample_seed", 0),
                weight_func=weight_func,
            )
        else:
            # Uniform sampling (box mode)
            obj_surface_pts_local = weighted_surface_sampling(
                cfg["object_mesh"],
                n_samples=cfg.get("obj_sample_count", 100),
                seed=cfg.get("obj_sample_seed", 0),
                weight_func=None,
            )

    if cfg.get("ground_grid_size"):
        ground_pts = create_ground_points(
            cfg["ground_range"], cfg["ground_range"], cfg["ground_grid_size"]
        )

    # Mapped keypoints
    mapped_names = list(SMPLH_G1_MAPPING.keys())
    mapped_indices = [SMPLH_JOINTS.index(n) for n in mapped_names]
    robot_body_names = list(SMPLH_G1_MAPPING.values())

    skel_idx_chains = [[SMPLH_JOINTS.index(n) for n in chain] for chain in HUMAN_SKELETON_CHAINS]

    # --- MuJoCo model ---
    scene_xml = cfg.get("scene_xml")
    model_path = str(scene_xml) if scene_xml else str(G1_XML)

    # For slope mode: inject terrain mesh into the G1 XML
    if cfg["object_mesh"] is not None and scene_xml is None:
        terrain_obj_path = str(cfg["object_mesh"].resolve())
        import tempfile
        import xml.etree.ElementTree as ET
        tree = ET.parse(model_path)
        root = tree.getroot()
        # Add mesh asset
        asset = root.find("asset")
        if asset is None:
            asset = ET.SubElement(root, "asset")
        ET.SubElement(asset, "mesh", name="terrain_mesh",
                      file=terrain_obj_path, scale="1 1 1")
        # Add terrain body
        worldbody = root.find("worldbody")
        terrain_body = ET.SubElement(worldbody, "body", name="terrain",
                                     pos="0 0 0")
        ET.SubElement(terrain_body, "geom", name="terrain_geom",
                      type="mesh", mesh="terrain_mesh",
                      rgba="0.5 0.5 0.5 0.5", contype="0", conaffinity="0")
        # Write temp XML in same directory as original so relative mesh paths resolve
        model_dir = str(Path(model_path).parent)
        tmp = tempfile.NamedTemporaryFile(suffix=".xml", delete=False, mode="wb", dir=model_dir)
        tree.write(tmp, xml_declaration=True)
        tmp.close()
        model_path = tmp.name

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    robot_body_ids = [model.body(n).id for n in robot_body_names]

    # Semi-transparent robot, keep box/ground opaque
    set_geom_alpha(model, alpha=0.25, skip_names=["largebox", "box"])

    # Set initial robot qpos (first 36 dims), box qpos handled per-frame
    data.qpos[:36] = qpos_seq[0]
    mujoco.mj_forward(model, data)

    # --- Playback ---
    total_display = min(T_human, T_robot)
    avg_dt = 1.0 / retarget_fps

    playback = PlaybackController(
        total_frames=total_display,
        avg_dt=avg_dt,
        speed=args.speed,
        loop=loop,
    )

    viewer = mujoco.viewer.launch_passive(model, data, key_callback=playback.key_callback)
    viewer.cam.azimuth = 135
    viewer.cam.elevation = -20
    viewer.cam.distance = 3.0
    viewer.cam.lookat[:] = [0.5, 0.8, 0.7]

    print(f"OmniRetarget Demo: mode={args.mode}")
    print(f"  Robot: {T_robot} frames (G1, 29 DOF)")
    if human_joints_all is not None:
        print(f"  Human: {T_human} frames (SMPLH, 52 joints, scale={scale:.3f})")
    if obj_surface_pts_local is not None:
        print(f"  Object: {len(obj_surface_pts_local)} surface samples from {cfg['object_mesh'].name}")
    if cfg["has_object_traj"]:
        print(f"  Object trajectory: {len(obj_pos_seq)} frames")
    print(f"  Display: {total_display} frames, FPS: {retarget_fps}")
    print(f"  Keys: SPACE=pause  LEFT/RIGHT=step/direction")
    print("=" * 60)

    frame_count = 0

    try:
        while viewer.is_running():
            idx, need_update = playback.advance()
            if not need_update:
                viewer.sync()
                continue

            # Frame index mapping
            h_idx = min(int(idx * T_human / total_display), T_human - 1)
            r_idx = min(int(idx * T_robot / total_display), T_robot - 1)

            # Robot FK + object pose
            data.qpos[:36] = qpos_seq[r_idx]
            if cfg["has_object_traj"] and obj_pos_seq is not None and model.nq > 36:
                # Box freejoint qpos: [x, y, z, qw, qx, qy, qz]
                data.qpos[36:39] = obj_pos_seq[r_idx]
                data.qpos[39:43] = obj_quat_seq[r_idx]  # already wxyz
            mujoco.mj_forward(model, data)
            robot_kp_pts = np.array([data.xpos[bid].copy() for bid in robot_body_ids])

            # Human keypoints (or robot body positions as fallback)
            human_pts = None
            mapped_human_pts = None
            if human_joints_all is not None:
                human_pts = human_joints_all[h_idx]
                mapped_human_pts = human_pts[mapped_indices]
            elif robot_body_pos_all is not None:
                # Use robot body positions from retarget result as mesh source
                mapped_human_pts = robot_kp_pts.copy()  # use FK positions as "source"

            # Object surface points in world frame
            obj_pts_world = None
            if obj_surface_pts_local is not None:
                if cfg["has_object_traj"] and obj_pos_seq is not None:
                    obj_pts_world = transform_points(
                        obj_surface_pts_local, obj_quat_seq[r_idx], obj_pos_seq[r_idx]
                    )
                else:
                    obj_pts_world = obj_surface_pts_local.copy()

            # Build Delaunay: human/robot points + object points + ground points
            # Matches OmniRetarget: np.vstack([human_mapped_joints, object_points_local])
            if mapped_human_pts is not None:
                parts = [mapped_human_pts]
                if obj_pts_world is not None:
                    parts.append(obj_pts_world)
                if ground_pts is not None:
                    parts.append(ground_pts)
                all_mesh_pts = np.vstack(parts)
                _, simplices = create_interaction_mesh(all_mesh_pts)
                adj = get_adjacency_list(simplices, len(all_mesh_pts))
                current_edges = frozenset((min(i, j), max(i, j)) for i, nb in enumerate(adj) for j in nb)
            else:
                all_mesh_pts = None
                current_edges = frozenset()

            # --- Draw ---
            with viewer.lock():
                viewer.user_scn.ngeom = 0

                if show_human and human_pts is not None:
                    # Skeleton chains
                    for chain in skel_idx_chains:
                        for k in range(len(chain) - 1):
                            add_line(viewer.user_scn, human_pts[chain[k]], human_pts[chain[k + 1]],
                                     COL_SKELETON, 1.0)
                    # All 52 joints
                    for pt in human_pts:
                        add_sphere(viewer.user_scn, pt, COL_HUMAN_ALL, SPHERE_SMALL)
                    # 15 mapped
                    for pt in mapped_human_pts:
                        add_sphere(viewer.user_scn, pt, COL_HUMAN_MAPPED, SPHERE_MED)
                    # Match lines
                    for hp, rp in zip(mapped_human_pts, robot_kp_pts):
                        add_line(viewer.user_scn, hp, rp, COL_MATCH_LINE, 0.5)

                # Robot keypoints
                for pt in robot_kp_pts:
                    add_sphere(viewer.user_scn, pt, COL_ROBOT_KP, SPHERE_MED)

                # Object surface points (blue)
                if obj_pts_world is not None:
                    for pt in obj_pts_world:
                        add_sphere(viewer.user_scn, pt, COL_OBJ_PTS, SPHERE_OBJ)

                # Ground grid points (gray)
                if ground_pts is not None:
                    for pt in ground_pts:
                        add_sphere(viewer.user_scn, pt, COL_GROUND_PTS, SPHERE_OBJ * 0.6)

                # Mesh edges
                if show_mesh and all_mesh_pts is not None:
                    for i, j in current_edges:
                        add_line(viewer.user_scn, all_mesh_pts[i], all_mesh_pts[j],
                                 COL_MESH_EDGE, 1.5)

            viewer.sync()
            frame_count += 1

            if frame_count % 200 == 0:
                n_e = len(current_edges)
                n_pts = len(all_mesh_pts) if all_mesh_pts is not None else 0
                print(f"Frame {idx}/{total_display}, mesh: {n_pts} pts, {n_e} edges")

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
