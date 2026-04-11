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

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from hand_retarget import InteractionMeshHandRetargeter, HandRetargetConfig
from hand_retarget.mediapipe_io import load_pkl_sequence
from hand_retarget.mesh_utils import create_interaction_mesh, get_adjacency_list

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[2]
DEFAULT_PKL = PROJECT_DIR / "data" / "manus1_5k.pkl"
DEFAULT_CONFIG = PROJECT_DIR / "config" / "interaction_mesh_left.yaml"
DEFAULT_MJCF = Path("/home/l/ws/doc/WujiRepo/urdf_cali/reference/result/xml/left.xml")  # MuJoCo XML for visualization
DEFAULT_URDF = Path("/home/l/ws/doc/WujiRepo/wuji_retargeting_private/public/wuji_retargeting/wuji_hand_description/urdf/left.urdf")  # Pinocchio URDF for retargeting (has tip_link)
HAND_SIDE = "left"

# Visualization colors [R, G, B, A] as float32
COLOR_ROBOT_NODE = np.array([0.0, 0.9, 0.9, 0.8], dtype=np.float32)   # cyan
COLOR_MESH_EDGE = np.array([0.0, 0.7, 0.7, 0.5], dtype=np.float32)    # cyan translucent
COLOR_SOURCE_NODE = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)   # red, opaque

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


def _add_sphere(scene: mujoco.MjvScene, pos: np.ndarray, rgba: np.ndarray, size: float = SPHERE_SIZE):
    """Add a sphere geom to the scene."""
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g,
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([size, 0, 0], dtype=np.float64),
        pos=pos.astype(np.float64),
        mat=np.eye(3, dtype=np.float64).flatten(),
        rgba=rgba,
    )
    scene.ngeom += 1


def _add_line(scene: mujoco.MjvScene, p1: np.ndarray, p2: np.ndarray,
              rgba: np.ndarray, width: float = LINE_WIDTH):
    """Add a line geom (connector) to the scene."""
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g,
        type=mujoco.mjtGeom.mjGEOM_LINE,
        size=np.zeros(3, dtype=np.float64),
        pos=np.zeros(3, dtype=np.float64),
        mat=np.eye(3, dtype=np.float64).flatten(),
        rgba=rgba,
    )
    mujoco.mjv_connector(
        g,
        type=mujoco.mjtGeom.mjGEOM_LINE,
        width=width,
        from_=p1.astype(np.float64),
        to=p2.astype(np.float64),
    )
    scene.ngeom += 1


def draw_interaction_mesh(
    scene: mujoco.MjvScene,
    robot_positions: np.ndarray,
    adj_list: list[list[int]],
    edges: list[tuple[int, int]],
    source_positions: np.ndarray | None = None,
    show_mesh: bool = True,
    show_source: bool = True,
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
    """
    if show_mesh and source_positions is not None:
        # Mesh edges on SOURCE keypoints (this is the actual interaction mesh)
        for i, j in edges:
            _add_line(scene, source_positions[i], source_positions[j], COLOR_MESH_EDGE)

    if show_source and source_positions is not None:
        # Source MediaPipe keypoints (red)
        for pos in source_positions:
            _add_sphere(scene, pos, COLOR_SOURCE_NODE, size=SPHERE_SIZE)

    # Robot keypoints (green) — optimization result, no mesh of its own
    for pos in robot_positions:
        _add_sphere(scene, pos, COLOR_ROBOT_NODE)


def main():
    parser = argparse.ArgumentParser(description="Interaction mesh retargeting live viewer")
    parser.add_argument("--pkl", type=str, default=str(DEFAULT_PKL))
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--mjcf", type=str, default=str(DEFAULT_MJCF), help="MuJoCo XML for visualization")
    parser.add_argument("--urdf", type=str, default=str(DEFAULT_URDF), help="URDF for retargeting (has tip_link)")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed")
    parser.add_argument("--loop", action="store_true", default=True)
    parser.add_argument("--no-loop", action="store_true")
    parser.add_argument("--no-mesh", action="store_true", help="Hide mesh topology overlay")
    parser.add_argument("--no-source", action="store_true", help="Hide source MediaPipe keypoints")
    parser.add_argument("--live", action="store_true", help="Live retargeting (no precompute, no rewind)")
    parser.add_argument("--fixed-topology", action="store_true", help="Use fixed topology from first frame (Ho 2010)")
    parser.add_argument("--distance-weight", action="store_true", help="Use distance-based Laplacian weights (Ho 2010)")
    parser.add_argument("--semantic-weight", action="store_true", help="Use pinch-aware semantic weights on Laplacian loss")
    parser.add_argument("--collision", action="store_true", help="Enable MuJoCo collision detection and contact rendering")
    args = parser.parse_args()

    loop = not args.no_loop
    show_mesh = not args.no_mesh
    show_source = not args.no_source

    # Load config with URDF for retargeting (Pinocchio, has tip_link)
    config = HandRetargetConfig.from_yaml(args.config, mjcf_path=args.urdf)

    # Load MuJoCo model for visualization (separate from retargeting model)
    model = mujoco.MjModel.from_xml_path(args.mjcf)
    data = mujoco.MjData(model)

    # Disable collision by default (pure kinematics playback)
    if not args.collision:
        model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT

    # Make hand mesh semi-transparent (keep ground plane opaque)
    for i in range(model.ngeom):
        geom_type = model.geom_type[i]
        if geom_type != mujoco.mjtGeom.mjGEOM_PLANE:
            model.geom_rgba[i, 3] = 0.25

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
    if args.fixed_topology:
        cache_suffix += "_fixed"
    if args.distance_weight:
        cache_suffix += "_distw"
    if args.semantic_weight:
        cache_suffix += "_semw"
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
            use_uniform = not args.distance_weight
            parts = []
            if args.fixed_topology: parts.append("fixed topology")
            if args.distance_weight: parts.append("distance weight")
            if args.semantic_weight: parts.append("semantic weight")
            mode_label = " + ".join(parts) if parts else "per-frame Delaunay"
            print(f"No cache found, pre-retargeting {total_frames} frames ({mode_label})...")
            qpos_cache = np.zeros((total_frames, retargeter.nq))
            q_prev = retargeter.hand.get_default_qpos()
            t0 = _time.time()

            if args.fixed_topology:
                from hand_retarget.mesh_utils import calculate_laplacian_coordinates, calculate_laplacian_matrix
                from scipy import sparse as sp
                import cvxpy as cp

                source_pts_0 = retargeter._extract_source_keypoints(proc_seq[0])
                _, simplices = create_interaction_mesh(source_pts_0)
                fixed_adj = get_adjacency_list(simplices, retargeter.n_keypoints)

                for t in range(total_frames):
                    source_pts = retargeter._extract_source_keypoints(proc_seq[t])
                    target_lap = calculate_laplacian_coordinates(source_pts, fixed_adj, uniform_weight=use_uniform)

                    n_iter = 50 if t == 0 else 10
                    q_current = q_prev.copy()
                    last_cost = float("inf")
                    for _ in range(n_iter):
                        # Custom solve with weight scheme
                        retargeter.hand.forward(q_current)
                        robot_pts = retargeter._get_robot_keypoints()
                        J_V = retargeter._get_robot_jacobians()

                        L = calculate_laplacian_matrix(robot_pts, fixed_adj, uniform_weight=use_uniform)
                        L_sp = sp.csr_matrix(L)
                        Kron = sp.kron(L_sp, sp.eye(3, format="csr"), format="csr")
                        J_L = Kron @ J_V

                        lap0 = L_sp @ robot_pts
                        lap0_vec = lap0.reshape(-1)
                        target_lap_vec = target_lap.reshape(-1)

                        V = retargeter.n_keypoints
                        if args.semantic_weight:
                            sem_w = retargeter._compute_semantic_weights(source_pts)
                            w_v = retargeter.laplacian_weight * sem_w
                        else:
                            w_v = retargeter.laplacian_weight * np.ones(V)
                        sqrt_w3 = np.sqrt(np.repeat(w_v, 3))

                        dq = cp.Variable(retargeter.nq, name="dq")
                        lap_var = cp.Variable(3 * V, name="lap")

                        constraints = [
                            cp.Constant(J_L) @ dq - lap_var == -lap0_vec,
                            dq >= retargeter.q_lb - q_current,
                            dq <= retargeter.q_ub - q_current,
                            cp.SOC(retargeter.config.step_size, dq),
                        ]
                        obj_terms = [
                            cp.sum_squares(cp.multiply(sqrt_w3, lap_var - target_lap_vec)),
                            retargeter.config.smooth_weight * cp.sum_squares(dq - (q_prev - q_current)),
                        ]
                        problem = cp.Problem(cp.Minimize(cp.sum(obj_terms)), constraints)
                        problem.solve(solver=cp.CLARABEL, verbose=False)
                        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                            break
                        q_current = q_current + dq.value
                        q_current = np.clip(q_current, retargeter.q_lb, retargeter.q_ub)
                        if abs(last_cost - problem.value) < 1e-8:
                            break
                        last_cost = problem.value

                    qpos_cache[t] = q_current
                    q_prev = q_current
                    if (t + 1) % 500 == 0:
                        fps = (t + 1) / (_time.time() - t0)
                        print(f"  {t + 1}/{total_frames} ({fps:.0f} fps)")
                retargeter._adj_list = fixed_adj
            else:
                # Per-frame Delaunay (default)
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

    # --- Playback state (used in precompute mode) ---
    KEY_SPACE = 32
    KEY_LEFT = 263   # step backward / reverse
    KEY_UP = 265     # toggle interaction mesh edges
    KEY_DOWN = 264   # toggle source points
    KEY_RIGHT = 262  # toggle robot points

    playback_state = {
        "paused": False,
        "direction": 1,
        "step_request": 0,
        "frame_idx": 0,
        "resume_flag": False,
    }

    # Runtime visibility toggles (override CLI flags)
    vis_state = {
        "mesh": show_mesh,
        "source": show_source,
        "robot": True,
    }

    def key_callback(keycode: int):
        # Visibility toggles (work in all modes)
        if keycode == KEY_UP:
            vis_state["mesh"] = not vis_state["mesh"]
            print(f"  [MESH {'ON' if vis_state['mesh'] else 'OFF'}]")
            return
        if keycode == KEY_DOWN:
            vis_state["source"] = not vis_state["source"]
            print(f"  [SOURCE {'ON' if vis_state['source'] else 'OFF'}]")
            return
        if keycode == KEY_RIGHT:
            vis_state["robot"] = not vis_state["robot"]
            print(f"  [ROBOT {'ON' if vis_state['robot'] else 'OFF'}]")
            return
        if live_mode:
            return
        # Playback controls (precompute mode only)
        if keycode == KEY_SPACE:
            was_paused = playback_state["paused"]
            playback_state["paused"] = not was_paused
            if was_paused:
                playback_state["resume_flag"] = True
            status = "PAUSED" if playback_state["paused"] else "PLAYING"
            print(f"  [{status}] frame {playback_state['frame_idx']}/{total_frames}")
        elif keycode == KEY_LEFT:
            if playback_state["paused"]:
                playback_state["step_request"] = -1
            else:
                playback_state["direction"] *= -1
                dir_str = ">>>" if playback_state["direction"] == 1 else "<<<"
                print(f"  [{dir_str}]")

    # Launch viewer
    viewer = mujoco.viewer.launch_passive(model, data, key_callback=key_callback)
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

    # Frame interval for playback timing
    if total_frames > 1 and (timestamps[-1] - timestamps[0]) > 0.01:
        avg_dt = (timestamps[-1] - timestamps[0]) / (total_frames - 1)
    else:
        avg_dt = 1.0 / 30.0

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
                # === PRECOMPUTE MODE: random access with pause/rewind ===
                now = time.time()
                current_idx = playback_state["frame_idx"]
                need_update = False

                if playback_state["paused"]:
                    step = playback_state["step_request"]
                    if step != 0:
                        current_idx += step
                        current_idx = max(0, min(current_idx, total_frames - 1))
                        playback_state["step_request"] = 0
                        playback_state["frame_idx"] = current_idx
                        need_update = True
                    else:
                        time.sleep(0.01)
                else:
                    if playback_state["resume_flag"]:
                        last_frame_time = now
                        playback_state["resume_flag"] = False
                    dt = now - last_frame_time
                    frames_to_advance = int(dt / (avg_dt / args.speed))
                    if frames_to_advance >= 1:
                        current_idx += playback_state["direction"] * frames_to_advance
                        last_frame_time = now

                        if loop:
                            current_idx = current_idx % total_frames
                        else:
                            current_idx = max(0, min(current_idx, total_frames - 1))

                        playback_state["frame_idx"] = current_idx
                        need_update = True
                    else:
                        time.sleep(0.001)

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
                source_pts = retargeter._extract_source_keypoints(proc_seq[current_idx])

                if vis_state["mesh"]:
                    _, simplices = create_interaction_mesh(source_pts)
                    frame_adj = get_adjacency_list(simplices, len(source_pts))
                    frame_edges = _collect_mesh_edges(frame_adj)
                else:
                    frame_adj = []
                    frame_edges = []

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
