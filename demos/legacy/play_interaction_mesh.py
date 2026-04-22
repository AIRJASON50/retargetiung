"""
Live interaction mesh retargeting replay in MuJoCo viewer
with interaction mesh topology visualization.

Draws:
  - Cyan spheres: robot keypoints (16 mapped joints)
  - Cyan lines: Delaunay mesh edges between keypoints
  - Orange spheres: source MediaPipe keypoints (for comparison)

Usage:
    python scripts/play_interaction_mesh.py
    python scripts/play_interaction_mesh.py --speed 0.5
    python scripts/play_interaction_mesh.py --no-mesh          # hide mesh overlay
    python scripts/play_interaction_mesh.py --no-source        # hide source keypoints
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

# Add src and project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR))
# editable install fallback: wuji_retargeting source
import os; _WUJI_SDK = os.environ.get("WUJI_SDK_PATH", "/home/l/ws/doc/WujiRepo/wuji_retargeting_private/public"); sys.path.insert(0, _WUJI_SDK)

from hand_retarget import InteractionMeshHandRetargeter, HandRetargetConfig  # noqa: E402
from hand_retarget.mediapipe_io import load_pkl_sequence  # noqa: E402
from hand_retarget.mesh_utils import create_interaction_mesh, get_adjacency_list  # noqa: E402

from demos.shared.overlay import (  # noqa: E402
    KEY_DOWN,
    KEY_RIGHT,
    KEY_UP,
    add_line,
    add_sphere,
    set_geom_alpha,
)
from demos.shared.playback import PlaybackController  # noqa: E402

DEFAULT_PKL = PROJECT_DIR / "data" / "manus_for_pinch" / "manus1_5k.pkl"
DEFAULT_CONFIG = PROJECT_DIR / "config" / "manus.yaml"
DEFAULT_MJCF = Path("/home/l/ws/doc/WujiRepo/urdf_cali/reference/result/xml/left.xml")  # MuJoCo XML for visualization
DEFAULT_URDF = Path(_WUJI_SDK) / "wuji_retargeting" / "wuji_hand_description" / "urdf" / "left.urdf"  # Pinocchio URDF for retargeting (has tip_link)
HAND_SIDE = "left"

# Visualization colors [R, G, B, A] as float32
COLOR_ROBOT_NODE = np.array([0.0, 0.9, 0.9, 0.8], dtype=np.float32)   # cyan
COLOR_EDGE_KEPT = np.array([0.0, 1.0, 0.2, 0.8], dtype=np.float32)    # green: kept (< threshold)
COLOR_EDGE_LONG = np.array([1.0, 0.2, 0.2, 0.25], dtype=np.float32)   # red translucent: filtered
COLOR_SOURCE_NODE = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)   # red, opaque
COLOR_PROBE_ROBOT = np.array([0.0, 1.0, 0.3, 0.9], dtype=np.float32)   # green, robot probe

SPHERE_SIZE = 0.003   # keypoint sphere radius (m)
LINE_WIDTH = 1.5      # mesh edge line width (pixels)


def _collect_mesh_edges(adj_list: list[list[int]]) -> list[tuple[int, int]]:
    """Extract unique edges from adjacency list."""
    edges = set()
    for i, neighbors in enumerate(adj_list):
        for j in neighbors:
            edge = (min(i, j), max(i, j))
            edges.add(edge)
    return sorted(edges)


def draw_interaction_mesh(
    scene: mujoco.MjvScene,
    robot_positions: np.ndarray,
    adj_list: list[list[int]],
    edges: list[tuple[int, int]],
    source_positions: np.ndarray | None = None,
    show_mesh: bool = True,
    show_source: bool = True,
    edge_threshold: float | None = None,
    distance_decay_k: float | None = None,
    hyperext_mask: np.ndarray | None = None,
):
    """
    Draw the interaction mesh overlay in the MuJoCo scene.

    Args:
        scene: MjvScene to add custom geoms to
        robot_positions: (N, 3) robot keypoint world positions
        adj_list: mesh adjacency list
        edges: list of (i, j) edge tuples
        source_positions: (N, 3) source MediaPipe keypoints (optional)
        show_mesh: whether to draw mesh edges and robot nodes
        show_source: whether to draw source keypoint spheres
        edge_threshold: if set, color edges green (< threshold) or red (>= threshold)
        distance_decay_k: if set, alpha of kept edges scales as exp(-k * dist)
        hyperext_mask: (N,) bool array, True for hyperextending keypoints (red spheres)
    """
    if show_mesh and source_positions is not None:
        # Mesh edges on SOURCE keypoints colored by threshold, alpha by distance decay
        for i, j in edges:
            if edge_threshold is not None:
                dist = np.linalg.norm(source_positions[j] - source_positions[i])
                if dist < edge_threshold:
                    if distance_decay_k is not None:
                        alpha = float(np.exp(-distance_decay_k * dist))
                        color = np.array([COLOR_EDGE_KEPT[0], COLOR_EDGE_KEPT[1],
                                         COLOR_EDGE_KEPT[2], max(0.1, COLOR_EDGE_KEPT[3] * alpha)],
                                        dtype=np.float32)
                    else:
                        color = COLOR_EDGE_KEPT
                else:
                    color = COLOR_EDGE_LONG
            else:
                color = COLOR_EDGE_KEPT
            add_line(scene, source_positions[i], source_positions[j], color)

    if show_source and source_positions is not None:
        # Source MediaPipe keypoints (red)
        for pos in source_positions:
            add_sphere(scene, pos, COLOR_SOURCE_NODE, size=SPHERE_SIZE)

    # Robot keypoints -- green normally, red if hyperextending
    for idx, pos in enumerate(robot_positions):
        color = COLOR_ROBOT_NODE
        if hyperext_mask is not None and idx < len(hyperext_mask) and hyperext_mask[idx]:
            color = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
        add_sphere(scene, pos, color)


def main():
    parser = argparse.ArgumentParser(description="Interaction mesh retargeting live viewer")
    parser.add_argument("--pkl", type=str, default=str(DEFAULT_PKL))
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--mjcf", type=str, default=str(DEFAULT_MJCF), help="MuJoCo XML for visualization")
    parser.add_argument("--urdf", type=str, default=str(DEFAULT_URDF), help="URDF for retargeting (has tip_link)")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed")
    parser.add_argument("--no-loop", action="store_true")
    parser.add_argument("--no-mesh", action="store_true", help="Hide mesh topology overlay")
    parser.add_argument("--no-source", action="store_true", help="Hide source MediaPipe keypoints")
    parser.add_argument("--live", action="store_true", help="Live retargeting (no precompute, no rewind)")
    parser.add_argument("--semantic-weight", action="store_true", help="Use pinch-aware semantic weights on Laplacian loss")
    parser.add_argument("--collision", action="store_true", help="Enable MuJoCo collision detection and contact rendering")
    parser.add_argument("--skeleton", action="store_true", help="Use hand skeleton topology instead of Delaunay")
    parser.add_argument("--link-midpoint", action="store_true", help="Use 20 link midpoints instead of 21 joint origins")
    parser.add_argument("--angle-warmup", action="store_true", help="Two-stage: angle warmup before Laplacian")
    parser.add_argument("--cache", type=str, default=None, help="Load precomputed cache directly from this .npz path")
    args = parser.parse_args()

    loop = not args.no_loop
    show_mesh = not args.no_mesh
    show_source = not args.no_source

    # Load config with URDF for retargeting (Pinocchio, has tip_link)
    config = HandRetargetConfig.from_yaml(args.config, mjcf_path=args.urdf)
    if args.skeleton:
        config.use_skeleton_topology = True
    if args.link_midpoint:
        config.use_link_midpoints = True
    if args.angle_warmup:
        config.use_angle_warmup = True

    # Load MuJoCo model for visualization (separate from retargeting model)
    model = mujoco.MjModel.from_xml_path(args.mjcf)
    data = mujoco.MjData(model)

    # Disable collision by default (pure kinematics playback)
    if not args.collision:
        model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT

    # Make hand mesh semi-transparent (keep ground plane opaque)
    set_geom_alpha(model, alpha=0.25)

    # Set initial pose (mid-range) and run FK (no physics sim)
    data.qpos[:] = (model.jnt_range[:, 0] + model.jnt_range[:, 1]) / 2
    mujoco.mj_forward(model, data)

    # Load trajectory
    landmarks_seq, timestamps = load_pkl_sequence(args.pkl, HAND_SIDE)
    total_frames = len(landmarks_seq)

    # Initialize retargeter
    retargeter = InteractionMeshHandRetargeter(config)

    # Preprocess all landmarks
    from hand_retarget.mediapipe_io import preprocess_sequence
    proc_seq = preprocess_sequence(landmarks_seq, config.mediapipe_rotation, hand_side=HAND_SIDE, global_scale=retargeter.global_scale)

    # --- Precompute or live mode ---
    live_mode = args.live
    qpos_cache = None
    cached_adj_list = None
    cached_edges = []

    # Cache file path: data/<pkl_stem>_{mode}_cache.npz
    CACHE_DIR = PROJECT_DIR / "data" / "cache"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_suffix = "im"
    if args.semantic_weight:
        cache_suffix += "_semw"
    if args.skeleton:
        cache_suffix += "_skel"
    if args.link_midpoint:
        cache_suffix += "_midpt"
    if args.angle_warmup:
        cache_suffix += "_aw"
    # --cache overrides the auto-derived path
    if args.cache:
        cache_path = Path(args.cache)
    else:
        cache_path = CACHE_DIR / f"{Path(args.pkl).stem}_{cache_suffix}_cache.npz"

    if not live_mode:
        if cache_path.exists():
            cached = np.load(cache_path)
            qpos_cache = cached["qpos"]
            if len(qpos_cache) == total_frames:
                print(f"Loaded cache: {cache_path} ({total_frames} frames)")
            else:
                print(f"Cache frame count mismatch ({len(qpos_cache)} vs {total_frames}), re-computing...")
                qpos_cache = None

        if qpos_cache is None:
            import time as _time
            parts = []
            if args.semantic_weight: parts.append("semantic weight")
            mode_label = " + ".join(parts) if parts else "default"
            print(f"No cache found, pre-retargeting {total_frames} frames ({mode_label})...")
            qpos_cache = np.zeros((total_frames, retargeter.nq))
            q_prev = retargeter.hand.get_default_qpos()
            t0 = _time.time()

            for t in range(total_frames):
                q_opt = retargeter.retarget_frame(proc_seq[t], q_prev, is_first_frame=(t == 0),
                                                  use_semantic_weights=args.semantic_weight)
                qpos_cache[t] = q_opt
                q_prev = q_opt
                if (t + 1) % 500 == 0:
                    fps = (t + 1) / (_time.time() - t0)
                    print(f"  {t + 1}/{total_frames} ({fps:.0f} fps)")

            print(f"Pre-retargeting done in {_time.time() - t0:.1f}s")
            np.savez(cache_path, qpos=qpos_cache)
            print(f"Saved cache: {cache_path}")

    # --- Runtime visibility toggles ---
    vis_state = {
        "mesh": show_mesh,
        "source": show_source,
        "robot": True,
    }

    def _custom_key_handler(keycode: int) -> bool:
        """Handle UP/DOWN/RIGHT for visibility toggles. Returns True if consumed."""
        if keycode == KEY_UP:
            vis_state["mesh"] = not vis_state["mesh"]
            print(f"  [MESH {'ON' if vis_state['mesh'] else 'OFF'}]")
            return True
        if keycode == KEY_DOWN:
            vis_state["source"] = not vis_state["source"]
            print(f"  [SOURCE {'ON' if vis_state['source'] else 'OFF'}]")
            return True
        if keycode == KEY_RIGHT:
            vis_state["robot"] = not vis_state["robot"]
            print(f"  [ROBOT {'ON' if vis_state['robot'] else 'OFF'}]")
            return True
        return False

    # Frame interval for playback timing
    if total_frames > 1 and (timestamps[-1] - timestamps[0]) > 0.01:
        avg_dt = (timestamps[-1] - timestamps[0]) / (total_frames - 1)
    else:
        avg_dt = 1.0 / 30.0

    # --- Playback controller (precompute mode) ---
    playback = PlaybackController(
        total_frames=total_frames,
        avg_dt=avg_dt,
        speed=args.speed,
        loop=loop,
        custom_key_handler=_custom_key_handler,
    )

    # Launch viewer
    if live_mode:
        # Live mode: custom key handler only (no playback controls)
        viewer = mujoco.viewer.launch_passive(
            model, data, key_callback=lambda kc: _custom_key_handler(kc),
        )
    else:
        viewer = mujoco.viewer.launch_passive(model, data, key_callback=playback.key_callback)
    viewer.cam.azimuth = 180
    viewer.cam.elevation = -20
    viewer.cam.distance = 0.5
    viewer.cam.lookat[:] = [0, 0, 0.05]

    mode_str = "LIVE" if live_mode else "PRECOMPUTED"
    print(f"Playing: {args.pkl}")
    print(f"Config:  {args.config}")
    print(f"MJCF:    {args.mjcf}")
    print(f"Mode:    {mode_str}")
    print(f"Frames:  {total_frames}")
    print(f"Speed:   {args.speed}x")
    print(f"Loop:    {loop}")
    print(f"Mesh:    {'ON' if show_mesh else 'OFF'}  |  Source: {'ON' if show_source else 'OFF'}")
    if not live_mode:
        print(f"Keys:    SPACE=pause  LEFT=step/reverse  UP=mesh  DOWN=source  RIGHT=robot")
    print("=" * 50)

    if cached_adj_list:
        print(f"Mesh topology: {retargeter.n_keypoints} nodes, {len(cached_edges)} edges")

    frame_count = 0
    fps_start = time.time()
    last_frame_time = time.time()

    # Live mode state
    live_frame_idx = 0
    live_q_prev = retargeter.hand.get_default_qpos()

    try:
        while viewer.is_running():
            if live_mode:
                # === LIVE MODE: retarget each frame on the fly ===
                # Advance exactly 1 frame at a time (no skipping) to match precompute quality.
                # Speed is controlled by sleep, not by frame skipping.
                now = time.time()
                dt = now - last_frame_time
                target_dt = avg_dt / args.speed
                if dt < target_dt:
                    time.sleep(0.001)
                    viewer.sync()
                    continue
                last_frame_time = now

                live_frame_idx += 1
                if live_frame_idx >= total_frames:
                    if loop:
                        live_frame_idx = 0
                        live_q_prev = retargeter.hand.get_default_qpos()
                    else:
                        break

                is_first = (frame_count == 0)
                qpos = retargeter.retarget_frame(
                    proc_seq[live_frame_idx], live_q_prev, is_first_frame=is_first
                )
                live_q_prev = qpos
                current_idx = live_frame_idx

                # Cache topology on first frame
                if cached_adj_list is None and retargeter._adj_list is not None:
                    cached_adj_list = retargeter._adj_list
                    cached_edges = _collect_mesh_edges(cached_adj_list)
                    print(f"Mesh topology: {retargeter.n_keypoints} nodes, {len(cached_edges)} edges")

            else:
                # === PRECOMPUTE MODE (via PlaybackController) ===
                current_idx, need_update = playback.advance()
                if not need_update:
                    viewer.sync()
                    continue

                current_idx = max(0, min(current_idx, total_frames - 1))
                qpos = qpos_cache[current_idx]

            # Apply qpos (pure kinematics)
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)

            # Draw overlay only if any visualization is enabled
            any_vis = vis_state["mesh"] or vis_state["source"] or vis_state["robot"]
            if any_vis:
                retargeter.hand.forward(qpos)
                robot_pts = retargeter._get_robot_keypoints() if vis_state["robot"] else np.empty((0, 3))
                lm_frame = proc_seq[current_idx]
                source_pts = retargeter._extract_source_keypoints(lm_frame)

                if vis_state["mesh"]:
                    if args.link_midpoint and args.skeleton:
                        from hand_retarget.mesh_utils import get_midpoint_skeleton_adjacency
                        frame_adj = get_midpoint_skeleton_adjacency(len(source_pts))
                    elif args.skeleton:
                        from hand_retarget.mesh_utils import get_skeleton_adjacency
                        frame_adj = get_skeleton_adjacency(len(source_pts))
                    else:
                        _, simplices = create_interaction_mesh(source_pts)
                        frame_adj = get_adjacency_list(simplices, len(source_pts))
                        edge_thr = getattr(retargeter.config, "delaunay_edge_threshold", None)
                        if edge_thr is not None:
                            from hand_retarget.mesh_utils import filter_adjacency_by_distance
                            frame_adj = filter_adjacency_by_distance(frame_adj, source_pts, edge_thr)
                    frame_edges = _collect_mesh_edges(frame_adj)
                else:
                    frame_adj = []
                    frame_edges = []

                # Build hyperextension mask for robot keypoints
                hyper_mask = None
                if qpos_cache is not None:
                    cur_qpos = qpos_cache[current_idx]
                    n_kp = len(robot_pts)
                    hyper_mask = np.zeros(n_kp, dtype=bool)
                    pip_q = [2, 6, 10, 14, 18]
                    dip_q = [3, 7, 11, 15, 19]
                    if retargeter.config.use_link_midpoints:
                        for f in range(5):
                            if cur_qpos[pip_q[f]] < 0:
                                hyper_mask[4 * f + 1] = True
                            if cur_qpos[dip_q[f]] < 0:
                                hyper_mask[4 * f + 2] = True
                    else:
                        mp_pip = [6, 10, 14, 18]
                        mp_dip = [3, 7, 11, 15, 19]
                        for i, qi in enumerate(pip_q[1:]):
                            if cur_qpos[qi] < 0:
                                mp_idx = mp_pip[i]
                                kp_idx = retargeter.mp_indices.index(mp_idx)
                                hyper_mask[kp_idx] = True
                        for i, qi in enumerate(dip_q):
                            if cur_qpos[qi] < 0:
                                mp_idx = mp_dip[i]
                                kp_idx = retargeter.mp_indices.index(mp_idx)
                                hyper_mask[kp_idx] = True

                with viewer.lock():
                    viewer.user_scn.ngeom = 0
                    draw_interaction_mesh(
                        viewer.user_scn,
                        robot_pts,
                        frame_adj,
                        frame_edges,
                        source_positions=source_pts if vis_state["source"] else None,
                        show_mesh=vis_state["mesh"],
                        show_source=vis_state["source"],
                        edge_threshold=getattr(retargeter.config, "delaunay_edge_threshold", None),
                        distance_decay_k=getattr(retargeter.config, "laplacian_distance_weight_k", None),
                        hyperext_mask=hyper_mask,
                    )

                    # Wrist reference sphere (always visible, at origin) in link-midpoint mode
                    if retargeter.config.use_link_midpoints:
                        wrist_pos = lm_frame[0]
                        add_sphere(
                            viewer.user_scn,
                            wrist_pos,
                            np.array([1.0, 1.0, 0.0, 0.6], dtype=np.float32),
                            size=SPHERE_SIZE * 1.5,
                        )


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
        total_time = time.time() - fps_start
        if frame_count > 0:
            print(f"Total: {frame_count} frames, {frame_count / total_time:.1f} avg FPS")


if __name__ == "__main__":
    main()
