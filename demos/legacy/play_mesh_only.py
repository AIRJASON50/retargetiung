"""
Visualize interaction mesh topology on MediaPipe keypoints only.
No robot hand, no retargeting -- pure Delaunay + Laplacian mesh inspection.

Draws:
  - Orange spheres: all 21 MediaPipe landmarks
  - Cyan spheres: 16 mapped keypoints (used for Delaunay)
  - Cyan lines: Delaunay mesh edges between mapped keypoints
  - Green lines: finger bone chains (ground truth skeleton)

Controls:
  SPACE       pause/resume
  LEFT/RIGHT  step backward/forward (paused) or reverse/forward (playing)

Usage:
    python scripts/play_mesh_only.py
    python scripts/play_mesh_only.py --speed 0.5
"""

import argparse
import sys
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

# Add src and project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR))

from hand_retarget.mediapipe_io import load_pkl_sequence, preprocess_sequence  # noqa: E402
from hand_retarget.mesh_utils import create_interaction_mesh, get_adjacency_list  # noqa: E402
from hand_retarget.config import HandRetargetConfig  # noqa: E402

from demos.shared.overlay import add_line, add_sphere  # noqa: E402
from demos.shared.playback import PlaybackController  # noqa: E402

DEFAULT_PKL = PROJECT_DIR / "data" / "manus_for_pinch" / "manus1_5k.pkl"
DEFAULT_CONFIG = PROJECT_DIR / "config" / "manus.yaml"
HAND_SIDE = "left"

# Colors
COL_MP_ALL = np.array([1.0, 0.5, 0.0, 0.4], dtype=np.float32)      # orange dim (non-mapped)
COL_MP_MAPPED = np.array([0.0, 0.9, 0.9, 0.9], dtype=np.float32)   # cyan (mapped keypoints)
COL_MESH_EDGE = np.array([0.0, 0.7, 0.7, 0.6], dtype=np.float32)   # cyan lines
COL_SKELETON = np.array([0.3, 0.9, 0.3, 0.3], dtype=np.float32)    # green (bone chains)
COL_WRIST = np.array([1.0, 1.0, 0.0, 0.9], dtype=np.float32)       # yellow (wrist)

SPHERE_SIZE = 0.003
SPHERE_SIZE_SMALL = 0.002
LINE_WIDTH = 1.5
SKEL_LINE_WIDTH = 1.0

# MediaPipe skeleton chains (for ground truth bone drawing)
MP_SKELETON_CHAINS = [
    [0, 1, 2, 3, 4],       # thumb
    [0, 5, 6, 7, 8],       # index
    [0, 9, 10, 11, 12],    # middle
    [0, 13, 14, 15, 16],   # ring
    [0, 17, 18, 19, 20],   # pinky
]


def draw_frame(scene, landmarks_21, mapped_indices, adj_list, edges):
    """Draw one frame of mesh visualization."""
    mapped_set = set(mapped_indices)
    mapped_pts = landmarks_21[mapped_indices]

    # 1. Skeleton bone chains (green)
    for chain in MP_SKELETON_CHAINS:
        for i in range(len(chain) - 1):
            add_line(scene, landmarks_21[chain[i]], landmarks_21[chain[i + 1]],
                     COL_SKELETON, SKEL_LINE_WIDTH)

    # 2. Delaunay mesh edges (cyan)
    for i, j in edges:
        add_line(scene, mapped_pts[i], mapped_pts[j], COL_MESH_EDGE, LINE_WIDTH)

    # 3. All 21 MediaPipe points
    for idx in range(21):
        if idx == 0:
            add_sphere(scene, landmarks_21[idx], COL_WRIST, SPHERE_SIZE * 1.3)
        elif idx in mapped_set:
            add_sphere(scene, landmarks_21[idx], COL_MP_MAPPED, SPHERE_SIZE)
        else:
            add_sphere(scene, landmarks_21[idx], COL_MP_ALL, SPHERE_SIZE_SMALL)


def main():
    parser = argparse.ArgumentParser(description="Interaction mesh topology viewer (MediaPipe only)")
    parser.add_argument("--pkl", type=str, default=str(DEFAULT_PKL))
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--no-loop", action="store_true")
    args = parser.parse_args()

    loop = not args.no_loop
    config = HandRetargetConfig.from_yaml(args.config)
    mapped_indices = sorted(config.joints_mapping.keys())

    # Load and preprocess
    landmarks_seq, timestamps = load_pkl_sequence(args.pkl, HAND_SIDE)
    proc_seq = preprocess_sequence(
        landmarks_seq, config.mediapipe_rotation, hand_side=HAND_SIDE, global_scale=1.0
    )
    total_frames = len(proc_seq)

    if total_frames > 1 and (timestamps[-1] - timestamps[0]) > 0.01:
        avg_dt = (timestamps[-1] - timestamps[0]) / (total_frames - 1)
    else:
        avg_dt = 1.0 / 30.0

    # Minimal MuJoCo model (empty world with ground plane for camera reference)
    xml = """
    <mujoco>
      <worldbody>
        <light pos="0 0 0.5" dir="0 0 -1"/>
        <geom type="plane" size="0.3 0.3 0.01" rgba="0.15 0.15 0.15 1" pos="0 0 -0.05"/>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Playback controller
    playback = PlaybackController(
        total_frames=total_frames,
        avg_dt=avg_dt,
        speed=args.speed,
        loop=loop,
    )

    viewer = mujoco.viewer.launch_passive(model, data, key_callback=playback.key_callback)
    viewer.cam.azimuth = 180
    viewer.cam.elevation = -30
    viewer.cam.distance = 0.4
    viewer.cam.lookat[:] = [0, 0, 0.08]

    print(f"PKL:     {args.pkl} ({total_frames} frames)")
    print(f"Mapped:  {len(mapped_indices)} / 21 keypoints")
    print(f"Speed:   {args.speed}x")
    print(f"Colors:  yellow=wrist  cyan=mapped  orange=unmapped  green=skeleton")
    print(f"Keys:    SPACE=pause  LEFT/RIGHT=step/direction")
    print("=" * 50)

    frame_count = 0

    try:
        while viewer.is_running():
            idx, need_update = playback.advance()
            if not need_update:
                viewer.sync()
                continue

            # Build Delaunay for this frame
            lm = proc_seq[idx]  # (21, 3)
            mapped_pts = lm[mapped_indices]  # (16, 3)
            _, simplices = create_interaction_mesh(mapped_pts)
            adj_list = get_adjacency_list(simplices, len(mapped_indices))

            # Collect unique edges
            edges = set()
            for i, neighbors in enumerate(adj_list):
                for j in neighbors:
                    edges.add((min(i, j), max(i, j)))
            edges = sorted(edges)

            # Draw
            with viewer.lock():
                viewer.user_scn.ngeom = 0
                draw_frame(viewer.user_scn, lm, mapped_indices, adj_list, edges)

            viewer.sync()
            frame_count += 1

            if frame_count % 500 == 0:
                print(f"Frame {idx}/{total_frames}, edges: {len(edges)}")

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
