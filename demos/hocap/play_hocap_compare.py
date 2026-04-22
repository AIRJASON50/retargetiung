"""Generic A/B overlay comparator for retargeting experiments.

Ghost-hand pattern (borrowed from
``playground/bh_motion_track/tools/replay_motiongen.py``): inject transparent
``ghost_`` hand bodies into the HO-Cap scene alongside the primary hands.
At playback, primary hands get cache A's qpos, ghost hands get cache B's.
Both are overlaid at the same world position, with the same object — the
visual delta is exactly the change cache B introduced vs cache A.

Convention (strict, all presets obey):
  * **Primary** (opaque, full STL meshes, default MuJoCo rendering)
    = **current main pipeline** (``np-C``: warmup+S2 hard non-penetration
    on, default HandRetargetConfig). This is the reference "ground truth"
    for what the system produces today.
  * **Ghost** (translucent green, alpha ≈ 0.35, STL only — collision
    primitives hidden) = **experimental variant** being evaluated. Could
    be an ablation (disable something), a knob sweep (IM weight), or a
    future change under test.

Two ways to specify which caches:

1. ``--preset NAME`` — convenience preset, picks two stamp suffixes.
   Current presets:
     * ``penetration``  → primary = np-C, ghost = np-D
       (main with hard constraint vs no-constraint ablation; ghost shows
       what would happen without the constraint)
     * ``im-boost``     → primary = np-C, ghost = np-C-IMx1111
       (main vs IM weight boosted by ×1111 via L_char=0.03)

   New experiments should add an entry in the ``PRESETS`` dict below,
   always with ``primary = np-C`` and ghost = the experimental variant.

2. ``--primary-cache PATH --ghost-cache PATH`` — explicit paths.
   Overrides ``--preset`` entirely. Useful for ad-hoc comparisons with
   non-standard cache names.

Usage:
    # Default: penetration compare
    PYTHONPATH=src python demos/hocap/play_hocap_compare.py \
        --clip hocap__subject_3__20231024_161306__seg00

    # Compare current main against IM-boosted experiment
    PYTHONPATH=src python demos/hocap/play_hocap_compare.py \
        --clip hocap__subject_3__20231024_161306__seg00 \
        --preset im-boost

    # Fully explicit
    PYTHONPATH=src python demos/hocap/play_hocap_compare.py \
        --clip ... \
        --primary-cache <path> --ghost-cache <path>

Requires the referenced caches to exist (generate via
experiments/exp_penetration_ablation.py with the matching --tag-suffix).
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR))

from scene_builder.hand_builder import (  # noqa: E402
    _inject_fingertip_sites,
    _inject_wrist6dof_mode,
)
from demos.hocap.play_hocap import qpos_to_world  # noqa: E402
from demos.shared.playback import PlaybackController  # noqa: E402
from demos.shared.overlay import (  # noqa: E402
    KEY_DOWN, KEY_RIGHT, KEY_UP,
    add_line, add_sphere,
)
from hand_retarget.mediapipe_io import load_hocap_clip, transform_object_points  # noqa: E402
from hand_retarget.mesh_utils import (  # noqa: E402
    create_interaction_mesh, get_adjacency_list, get_edge_list,
)
from hand_retarget.config import JOINTS_MAPPING_LEFT, JOINTS_MAPPING_RIGHT  # noqa: E402

# Overlay colors (match play_hocap.py palette)
COL_SOURCE    = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)  # red
COL_ROBOT     = np.array([0.0, 1.0, 0.3, 1.0], dtype=np.float32)  # green
COL_OBJ_PTS   = np.array([0.3, 0.3, 1.0, 0.8], dtype=np.float32)  # blue
COL_EDGE_KEPT = np.array([0.0, 1.0, 0.2, 0.8], dtype=np.float32)  # green translucent
COL_EDGE_LONG = np.array([1.0, 0.2, 0.2, 0.25], dtype=np.float32) # red filtered

HOCAP_DIR  = PROJECT_DIR / "data" / "hocap" / "hocap"
CACHE_DIR  = PROJECT_DIR / "data" / "cache" / "hocap"
SCENE_LEFT     = PROJECT_DIR / "assets" / "scenes" / "single_hand_obj_left.xml"
SCENE_RIGHT    = PROJECT_DIR / "assets" / "scenes" / "single_hand_obj.xml"
SCENE_BIMANUAL = PROJECT_DIR / "assets" / "scenes" / "bimanual_hand_obj.xml"

GHOST_ALPHA = 0.35


# ---------------------------------------------------------------------------
# Experiment presets
# ---------------------------------------------------------------------------
# Each preset picks two cache stamp suffixes (stored under data/cache/hocap/
# as <clip_id>__<suffix>.npz). Primary = "baseline" or "reference", Ghost =
# "experiment variant" being evaluated.
#
# Add new presets for future experiments. Naming convention for suffixes:
#   np-D    : no non-penetration constraint
#   np-A    : warmup-only non-penetration
#   np-B    : S2-only non-penetration
#   np-C    : warmup + S2 non-penetration (current main pipeline)
#   np-C-<tag> : C with additional experimental knob (e.g. -IMx1111)
# ---------------------------------------------------------------------------

PRESETS: dict[str, dict] = {
    # Convention (all presets): primary = current main pipeline (np-C),
    # ghost = experimental variant under evaluation.
    "penetration": {
        "primary": "np-C",
        "ghost":   "np-D",
        "description": "Main pipeline (primary, warmup+S2 hard non-penetration) "
                       "vs no-constraint ablation (ghost, np-D). Shows what "
                       "the hard constraint fixes — ghost should penetrate "
                       "more, primary should sit on the object surface.",
    },
    "im-boost": {
        "primary": "np-C",
        "ghost":   "np-C-IMx1111",
        "description": "Main pipeline (primary) vs IM weight boosted ~1111× "
                       "via interaction_mesh_length_scale=0.03 (ghost). Shows "
                       "the effect of IM term dominance on fingertip tracking.",
    },
    # Add future experiments here, always primary=np-C (= main pipeline):
    # "layer0-preprocess": {"primary": "np-C", "ghost": "np-C-L0clean", ...},
    # "tol-sweep":         {"primary": "np-C", "ghost": "np-C-tol5mm", ...},
}


# ---------------------------------------------------------------------------
# Ghost XML builders (mirror replay_motiongen.py's pattern)
# ---------------------------------------------------------------------------

_WRIST_JOINT_DEFS = [
    ("1 0 0", "slide", "tx"), ("0 1 0", "slide", "ty"), ("0 0 1", "slide", "tz"),
    ("1 0 0", "hinge", "rx"), ("0 1 0", "hinge", "ry"), ("0 0 1", "hinge", "rz"),
]


def _build_ghost_bimanual_xml() -> str:
    """XML for ghost_rh + ghost_lh bodies (kinematic, same attach as primary)."""
    parts = []
    for side, pref in [("right", "ghost_rh_"), ("left", "ghost_lh_")]:
        joints = ""
        for ax, jtype, sfx in _WRIST_JOINT_DEFS:
            rng = "-1.5 1.5" if jtype == "slide" else "-6.28 6.28"
            joints += (
                f'      <joint name="{pref}wrist_{sfx}" type="{jtype}" '
                f'axis="{ax}" range="{rng}"/>\n'
            )
        model_name = "wuji_rh" if side == "right" else "wuji_lh"
        attach_body = f"{side}_palm_link"
        parts.append(
            f'    <body name="{pref}wrist" pos="0 0 0">\n'
            f'      <inertial mass="0.05" pos="0 0 0" diaginertia="1e-6 1e-6 1e-6"/>\n'
            f"{joints}"
            f'      <attach model="{model_name}" body="{attach_body}" prefix="{pref}"/>\n'
            f'    </body>\n'
        )
    return "\n".join(parts)


def _build_ghost_single_xml(hand_side: str) -> str:
    """XML for one ghost hand."""
    pref = f"ghost_{hand_side[0]}h_"
    joints = ""
    for ax, jtype, sfx in _WRIST_JOINT_DEFS:
        rng = "-1.5 1.5" if jtype == "slide" else "-6.28 6.28"
        joints += (
            f'      <joint name="{pref}wrist_{sfx}" type="{jtype}" '
            f'axis="{ax}" range="{rng}"/>\n'
        )
    model_name = "wuji_rh" if hand_side == "right" else "wuji_lh"
    attach_body = f"{hand_side}_palm_link"
    return (
        f'    <body name="{pref}wrist" pos="0 0 0">\n'
        f'      <inertial mass="0.05" pos="0 0 0" diaginertia="1e-6 1e-6 1e-6"/>\n'
        f"{joints}"
        f'      <attach model="{model_name}" body="{attach_body}" prefix="{pref}"/>\n'
        f'    </body>\n'
    )


def _apply_ghost_visual(model: mujoco.MjModel, alpha: float = GHOST_ALPHA) -> None:
    """Post-compile pass: turn any body with name starting 'ghost_' into a
    pure kinematic overlay — no collision, only the STL visual meshes render.

    Classification by TYPE, not name:
      * geom_type == MESH  → visual STL → transparent green overlay
      * geom_type != MESH  → collision primitive (box / capsule / sphere) → hide

    This is more robust than name-based ``_col`` matching because palm boxes
    on this model (``left_palm_main`` / ``left_palm_lower`` / ...) are
    collision primitives WITHOUT a ``_col`` suffix in their names.
    """
    mesh_t = int(mujoco.mjtGeom.mjGEOM_MESH)
    for bi in range(model.nbody):
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bi) or ""
        if not bname.startswith("ghost_"):
            continue
        start = model.body_geomadr[bi]
        count = model.body_geomnum[bi]
        if start < 0 or count == 0:
            continue
        for gi in range(start, start + count):
            model.geom_contype[gi] = 0
            model.geom_conaffinity[gi] = 0
            if int(model.geom_type[gi]) == mesh_t:
                # Visual STL → transparent green overlay
                model.geom_rgba[gi] = [0.2, 1.0, 0.3, alpha]
                model.geom_group[gi] = 5
            else:
                # Collision primitive (box, capsule, sphere, ...) → fully hidden
                model.geom_rgba[gi][3] = 0.0

    # Zero any actuators belonging to ghost (inherited from attached hand model)
    for ai in range(model.nu):
        name = model.actuator(ai).name
        if name.startswith("ghost_"):
            model.actuator_gainprm[ai, :] = 0
            model.actuator_biasprm[ai, :] = 0


# ---------------------------------------------------------------------------
# Model assembly: primary scene + injected ghost XML
# ---------------------------------------------------------------------------


def build_overlay_model(hands: list[str], bimanual: bool, mesh_path: str,
                        ) -> tuple[mujoco.MjModel, mujoco.MjData, dict]:
    """Compile HO-Cap scene with primary hand(s) + ghost overlay hand(s) +
    one shared object mocap body.

    Strategy (borrows from playground/bh_motion_track/tools/replay_motiongen.py):
      1. Read the raw scene XML text (preserves `<model>` asset declarations).
      2. Text-inject ghost `<body>` + `<attach>` before the closing `</worldbody>`.
      3. Use a temp XML file as the base for MjSpec.from_file, so ghost's
         `<attach model="wuji_rh">` resolves to the same asset the primary uses.
      4. Apply primary injectors (_inject_fingertip_sites, _inject_wrist6dof_mode)
         on the spec. Do the SAME for ghost prefixes (they have their own
         `ghost_rh_wrist` / `ghost_lh_wrist` bodies waiting for wrist6dof).
      5. Add the shared object mocap body programmatically.
      6. Compile. Post-compile: make ghost geoms transparent via
         `_apply_ghost_visual`.
    """
    scene = (SCENE_BIMANUAL if bimanual
             else (SCENE_LEFT if hands[0] == "left" else SCENE_RIGHT))
    xml_text = scene.read_text()

    ghost_block = _build_ghost_bimanual_xml() if bimanual else _build_ghost_single_xml(hands[0])
    idx = xml_text.rfind("</worldbody>")
    if idx < 0:
        raise RuntimeError(f"Could not find </worldbody> in {scene}")
    xml_with_ghost = xml_text[:idx] + ghost_block + xml_text[idx:]

    with tempfile.NamedTemporaryFile(
        "w", suffix=".xml", delete=False, dir=str(scene.parent)
    ) as tf:
        tf.write(xml_with_ghost)
        tmp_path = Path(tf.name)

    try:
        spec = mujoco.MjSpec.from_file(str(tmp_path))
    finally:
        tmp_path.unlink(missing_ok=True)

    # Primary hand injectors
    if bimanual:
        _inject_fingertip_sites(spec, "right", name_prefix="rh_", body_prefix="rh_")
        _inject_fingertip_sites(spec, "left",  name_prefix="lh_", body_prefix="lh_")
        _inject_wrist6dof_mode(spec, wrist_body_name="wuji_rh_wrist", joint_prefix="rh_")
        _inject_wrist6dof_mode(spec, wrist_body_name="wuji_lh_wrist", joint_prefix="lh_")
    else:
        _inject_fingertip_sites(spec, hands[0])
        _inject_wrist6dof_mode(spec)
    # (Ghost wrist 6-DOF joints are already built into the ghost XML block.
    #  We do NOT run _inject_fingertip_sites on ghost bodies since we don't
    #  need the ghost tips for visualization, just the overlay geometry.)

    # Shared object mocap body (collides with PRIMARY only; ghost is contype=0)
    obj_body = spec.worldbody.add_body()
    obj_body.name = "hocap_object"
    obj_body.mocap = True
    obj_mesh = spec.add_mesh()
    obj_mesh.name = "hocap_obj_mesh"
    obj_mesh.file = mesh_path
    og = obj_body.add_geom()
    og.name = "hocap_obj_geom"
    og.type = mujoco.mjtGeom.mjGEOM_MESH
    og.meshname = "hocap_obj_mesh"
    og.rgba = [0.6, 0.4, 0.2, 0.5]
    og.group = 0
    og.contype = 2
    og.conaffinity = 1

    model = spec.compile()
    data = mujoco.MjData(model)
    _apply_ghost_visual(model)

    # Build qpos slice info for both primary and ghost
    def _qpos_slices(prefix: str) -> dict:
        """For a given side prefix ('rh_', 'lh_', or '' for single-hand), return
        {'wrist': (start, end), 'finger': (start, end)} into model.qpos."""
        def jid(name):
            return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)

        # Wrist: the 6 injected/defined joints, suffixes tx/ty/tz/rx/ry/rz
        # for both primary (rh_/lh_) and ghost (ghost_rh_/ghost_lh_).
        wrist_names = [f"{prefix}wrist_{s}" for s in ("tx", "ty", "tz", "rx", "ry", "rz")]
        wrist_ids = [jid(n) for n in wrist_names]
        if any(i < 0 for i in wrist_ids):
            return {}
        w_adr = [model.jnt_qposadr[i] for i in wrist_ids]
        wrist = (min(w_adr), max(w_adr) + 1)

        # Finger joints: {prefix}{side_pref}finger{f}_joint{k}
        # For bimanual primary/ghost the model includes side_pref (right_/left_) from <attach model="wuji_rh">
        # For single-hand primary, side_pref is left_/right_ depending on scene.
        # Enumerate by scanning all joints with this prefix.
        f_adr = []
        for f in range(1, 6):
            for k in range(1, 5):
                # Try both styles
                for side_pref in ("right_", "left_"):
                    jn = f"{prefix}{side_pref}finger{f}_joint{k}"
                    fid = jid(jn)
                    if fid >= 0:
                        f_adr.append(model.jnt_qposadr[fid])
                        break
        if not f_adr:
            return {"wrist": wrist}
        finger = (min(f_adr), max(f_adr) + 1)
        return {"wrist": wrist, "finger": finger}

    info: dict = {
        "hands": hands,
        "bimanual": bimanual,
        "obj_mocap": model.body("hocap_object").mocapid[0],
    }
    if bimanual:
        info["prim_slices"]  = {"right": _qpos_slices("rh_"), "left": _qpos_slices("lh_")}
        info["ghost_slices"] = {"right": _qpos_slices("ghost_rh_"), "left": _qpos_slices("ghost_lh_")}
    else:
        info["prim_slices"]  = {hands[0]: _qpos_slices("")}
        info["ghost_slices"] = {hands[0]: _qpos_slices(f"ghost_{hands[0][0]}h_")}

    return model, data, info


# ---------------------------------------------------------------------------
# Per-frame apply
# ---------------------------------------------------------------------------


def draw_primary_overlay(viewer, model, data, clips, hands, bimanual, vis,
                         frame_idx, delaunay_threshold, decay_k):
    """Draw source landmarks + primary robot keypoints + interaction mesh,
    based on the PRIMARY hand's body/site positions in the viz model.

    ``clips`` is a dict: hand_side → clip dict (each carries that hand's
    landmarks). For bimanual we iterate both hands; each hand reads its
    own clip for landmarks, shares the (identical) object pose.

    Ghost hands are NOT overlaid (only primary's retargeting geometry is
    shown, matching play_hocap.py behavior).
    """
    with viewer.lock():
        viewer.user_scn.ngeom = 0

        for hand_side in hands:
            clip = clips[hand_side]  # per-hand landmarks
            # Source landmarks (world frame, per clip)
            jm = JOINTS_MAPPING_LEFT if hand_side == "left" else JOINTS_MAPPING_RIGHT
            mp_indices = sorted(jm.keys())
            source_world = clip["landmarks"][frame_idx][mp_indices]

            # Object sample points (world frame)
            obj_world = transform_object_points(
                clip["object_pts_local"],
                clip["object_q"][frame_idx],
                clip["object_t"][frame_idx],
            )

            # Robot keypoints (from primary hand in the viz model).
            # Body names carry an extra attach-prefix: rh_/lh_ (bimanual) or "" (single).
            bp = ("lh_" if hand_side == "left" else "rh_") if bimanual else ""
            robot_world = []
            for mp_idx in mp_indices:
                body_name = jm[mp_idx]
                if "tip_link" in body_name:
                    # Fingertip site (injected by _inject_fingertip_sites for primary)
                    f = body_name.split("_finger")[1].split("_")[0]
                    site_name = f"{bp}finger{f}_tip"
                    try:
                        robot_world.append(data.site(site_name).xpos.copy())
                    except KeyError:
                        # Fallback to body if site missing
                        robot_world.append(data.body(f"{bp}{body_name}").xpos.copy())
                else:
                    robot_world.append(data.body(f"{bp}{body_name}").xpos.copy())
            robot_world = np.array(robot_world)

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
                for i, j in get_edge_list(adj):
                    dist = np.linalg.norm(all_pts[j] - all_pts[i])
                    if delaunay_threshold is None or dist < delaunay_threshold:
                        if decay_k is not None:
                            alpha = float(np.exp(-decay_k * dist))
                            color = np.array([
                                COL_EDGE_KEPT[0], COL_EDGE_KEPT[1], COL_EDGE_KEPT[2],
                                max(0.1, COL_EDGE_KEPT[3] * alpha),
                            ], dtype=np.float32)
                        else:
                            color = COL_EDGE_KEPT
                    else:
                        color = COL_EDGE_LONG
                    add_line(viewer.user_scn, all_pts[i], all_pts[j], color)


def apply_frame(model, data, info, cache_d, cache_c, clip, frame_idx):
    for hand_side in info["hands"]:
        # Primary (D cache)
        qraw_d = cache_d[f"qpos_{hand_side}"][frame_idx]
        R_inv_d = cache_d[f"R_inv_{hand_side}"][frame_idx]
        wrist_d = cache_d[f"wrist_{hand_side}"][frame_idx]
        q_d = qpos_to_world(qraw_d, R_inv_d, wrist_d)

        pri = info["prim_slices"][hand_side]
        ws, we = pri["wrist"]
        data.qpos[ws:ws + 3] = q_d[:3]
        data.qpos[ws + 3:ws + 6] = q_d[3:6]
        if "finger" in pri:
            fs, fe = pri["finger"]
            data.qpos[fs:fe] = q_d[6:6 + (fe - fs)]

        # Ghost (C cache)
        qraw_c = cache_c[f"qpos_{hand_side}"][frame_idx]
        R_inv_c = cache_c[f"R_inv_{hand_side}"][frame_idx]
        wrist_c = cache_c[f"wrist_{hand_side}"][frame_idx]
        q_c = qpos_to_world(qraw_c, R_inv_c, wrist_c)

        ghs = info["ghost_slices"][hand_side]
        ws, we = ghs["wrist"]
        data.qpos[ws:ws + 3] = q_c[:3]
        data.qpos[ws + 3:ws + 6] = q_c[3:6]
        if "finger" in ghs:
            fs, fe = ghs["finger"]
            data.qpos[fs:fe] = q_c[6:6 + (fe - fs)]

    # Shared object
    data.mocap_pos[info["obj_mocap"]] = clip["object_t"][frame_idx]
    oq = clip["object_q"][frame_idx]
    data.mocap_quat[info["obj_mocap"]] = [oq[3], oq[0], oq[1], oq[2]]

    mujoco.mj_forward(model, data)


# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Overlay A/B viz for HO-Cap retargeting experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "\nAvailable presets:\n"
            + "\n".join(
                f"  {name:<15} primary={p['primary']:<15} ghost={p['ghost']:<20} "
                f"— {p['description'][:60]}..."
                for name, p in PRESETS.items()
            )
        ),
    )
    parser.add_argument("--clip", type=str, required=True)
    parser.add_argument("--preset", choices=list(PRESETS.keys()),
                        default="penetration",
                        help="Which A/B pair to overlay. See epilog.")
    parser.add_argument("--primary-cache", type=str, default=None,
                        help="Explicit path for primary (opaque). Overrides --preset.")
    parser.add_argument("--ghost-cache", type=str, default=None,
                        help="Explicit path for ghost (translucent). Overrides --preset.")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--ghost-alpha", type=float, default=GHOST_ALPHA)
    args = parser.parse_args()

    clip_id = args.clip
    npz = HOCAP_DIR / "motions" / f"{clip_id}.npz"
    meta = HOCAP_DIR / "motions" / f"{clip_id}.meta.json"
    with open(meta) as f:
        mt = json.load(f)
    mesh_path = str((HOCAP_DIR / "assets" / mt["objects"][0]["asset_name"] / "mesh_med.stl").resolve())

    data_npz = np.load(str(npz), allow_pickle=True)
    hands = []
    if "mediapipe_l_world" in data_npz and data_npz["mediapipe_l_world"].ndim > 0:
        hands.append("left")
    if "mediapipe_r_world" in data_npz and data_npz["mediapipe_r_world"].ndim > 0:
        hands.append("right")
    bimanual = len(hands) == 2

    # Resolve cache paths:
    #   explicit --primary-cache / --ghost-cache > --preset suffixes
    preset = PRESETS[args.preset]
    if args.primary_cache:
        primary_path = Path(args.primary_cache)
        primary_label = f"<{primary_path.name}>"
    else:
        primary_path = CACHE_DIR / f"{clip_id}__{preset['primary']}.npz"
        primary_label = preset["primary"]
    if args.ghost_cache:
        ghost_path = Path(args.ghost_cache)
        ghost_label = f"<{ghost_path.name}>"
    else:
        ghost_path = CACHE_DIR / f"{clip_id}__{preset['ghost']}.npz"
        ghost_label = preset["ghost"]

    # Helpful failure if cache missing
    missing = [p for p in (primary_path, ghost_path) if not p.exists()]
    if missing:
        print("ERROR: cache(s) not found:")
        for p in missing:
            print(f"  {p}")
        print("\nGenerate with:")
        print(f"  PYTHONPATH=src python experiments/exp_penetration_ablation.py "
              f"--clips {clip_id}")
        if args.preset == "im-boost":
            print(f"  PYTHONPATH=src python experiments/exp_penetration_ablation.py "
                  f"--clips {clip_id} --im-scale 0.03 --tag-suffix IMx1111")
        sys.exit(1)

    print(f"Preset:  {args.preset}  ({preset['description']})")
    print(f"Primary (opaque):    {primary_label}  [{primary_path.name}]")
    print(f"Ghost   (green):     {ghost_label}  [{ghost_path.name}]")
    cache_d = np.load(primary_path, allow_pickle=True)
    cache_c = np.load(ghost_path, allow_pickle=True)
    T = min(len(cache_d[f"qpos_{hands[0]}"]), len(cache_c[f"qpos_{hands[0]}"]))
    print(f"Clip:  {clip_id}")
    print(f"Hands: {hands}  (bimanual={bimanual})  Frames: {T}")

    model, data, info = build_overlay_model(hands, bimanual, mesh_path)
    _apply_ghost_visual(model, alpha=args.ghost_alpha)
    print(f"Model nq={model.nq}  ngeom={model.ngeom}  nbody={model.nbody}")

    # Per-hand clip dicts — crucial for bimanual: load_hocap_clip(hand_side="left")
    # returns left-hand landmarks only. We need BOTH hands' landmarks for the
    # overlay to show, and they come from separate loads. The object-side data
    # is identical across both loads (derived from clip meta) so using either
    # for mocap pose is fine.
    clips = {
        hs: load_hocap_clip(str(npz), str(meta), str(HOCAP_DIR / "assets"),
                             hand_side=hs, sample_count=50)
        for hs in hands
    }
    clip_obj = clips[hands[0]]  # object data reference for apply_frame/mocap

    apply_frame(model, data, info, cache_d, cache_c, clip_obj, 0)

    # Delaunay / Laplacian decay parameters for the mesh overlay. Read from
    # HandRetargetConfig defaults (matches play_hocap.py). Hard-coded here to
    # avoid importing config just for 2 constants.
    DELAUNAY_THRESHOLD = 0.06
    DECAY_K = 20.0

    # Overlay visibility state — toggled by UP / DOWN / RIGHT arrows.
    # Same semantics as play_hocap.py: overlays apply to PRIMARY hand only,
    # ghost never receives the source / robot / mesh annotations.
    vis = {"mesh": False, "source": False, "robot": False}

    def _custom_key_handler(keycode: int) -> bool:
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

    fps = clip_obj.get("fps", 30)
    playback = PlaybackController(total_frames=T, avg_dt=1.0 / fps, speed=args.speed, loop=True,
                                  custom_key_handler=_custom_key_handler)
    viewer = mujoco.viewer.launch_passive(model, data, key_callback=playback.key_callback)
    viewer.cam.azimuth = 180
    viewer.cam.elevation = -25
    viewer.cam.distance = 0.7
    viewer.cam.lookat[:] = [0, 0, 0.15]
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    # Enable group 5 (ghost visuals) by default
    viewer.opt.geomgroup[5] = 1

    print("\n" + "=" * 60)
    print(f"OVERLAY COMPARE  (primary = {primary_label},  ghost = {ghost_label})")
    print(f"  {preset['description']}")
    print("Keys: SPACE=pause  ←/→=step (SHIFT=reverse)  [/]=speed")
    print("      UP=mesh overlay  DOWN=source keypoints  RIGHT=primary robot keypoints")
    print("      (overlays show PRIMARY's data; ghost is kept visually clean)")
    print("=" * 60)

    try:
        while viewer.is_running():
            idx, need_update = playback.advance()
            if not need_update:
                viewer.sync()
                continue
            idx = max(0, min(idx, T - 1))
            apply_frame(model, data, info, cache_d, cache_c, clip_obj, idx)
            # Overlay on primary only (if any toggle on)
            if vis["source"] or vis["robot"] or vis["mesh"]:
                draw_primary_overlay(
                    viewer, model, data, clips, hands, bimanual, vis, idx,
                    DELAUNAY_THRESHOLD, DECAY_K,
                )
            else:
                with viewer.lock():
                    viewer.user_scn.ngeom = 0
            viewer.sync()
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
