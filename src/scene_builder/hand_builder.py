"""Runtime model builder for WujiHand scenes.

Uses MjSpec API (MuJoCo 3.2.0+) to inject non-native elements into scene XML
that uses <model> + <attach> for hand and cube loading.

Injected elements:
  - Fingertip sites on link4 bodies (for sensor + reward computation)
  - palm_frame_body debug markers (for z_warn reference)
  - Physics mode elements (mocap body + freejoint + weld constraint)
  - Wrist 6DOF mode (3 slide + 3 hinge joints + 6 PD actuators on wrist body)
  - Home keyframe re-indexing (MjSpec doesn't re-index when joints are added)

Architecture:
  Scene XML (<attach>)  ->  MjSpec (inject sites/markers/physics)  ->  MjModel
  right_mjx.xml / left_mjx.xml loaded from disk, never modified.

Multi-hand support:
  load_scene_model(..., hands=[...]) injects elements for each hand with
  name_prefix to avoid naming collisions. Single-hand (hands=None) uses
  legacy unprefixed names for backward compatibility.
"""

import mujoco
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Fingertip site positions (in link4 local frame).
# From right_mjx.xml / left_mjx.xml tip_link mesh geom positions (rubber fingertip cap).
# Left/right hands share the same offsets (only thumb Y sign differs, ~0.2mm, negligible).
_FINGERTIP_SITES = {
    "finger1": [-0.00105, -0.0002, 0.0273],
    "finger2": [-0.00105, 0.0002, 0.0267],
    "finger3": [-0.00105, 0.0002, 0.0267],
    "finger4": [-0.00106, 0.0002, 0.0267],
    "finger5": [-0.00104, 0.0002, 0.0267],
}

# Physics mode: wuji_wrist inertial for freejoint body
_WRIST_INERTIAL = {"mass": 0.05, "pos": [0, 0, 0], "inertia": [0.0001, 0.0001, 0.0001]}

# Weld constraint parameters
_WELD_SOLREF = [0.02, 1.0]
_WELD_SOLIMP = [0.9, 0.95, 0.001, 0.5, 2]

# 6-DOF wrist virtual joint parameters (IsaacLab-style PD control)
# Slides first (outermost), then hinges (innermost) -> intrinsic XYZ Euler.
_WRIST6DOF_SLIDE = {
    "range": [-0.5, 0.5],  # m (trajectory motion range)
    "kp": 1600.0,  # N/m (stiff position tracking)
    "kv": 60.0,  # N*s/m (damping)
    "forcerange": [-100, 100],  # N
    "armature": 0.01,
}
_WRIST6DOF_HINGE = {
    "range": [-3.14159, 3.14159],  # rad (full rotation)
    "kp": 15.0,  # N*m/rad (compliant orientation tracking, reduced from 100)
    "kv": 0.5,  # N*m*s/rad (reduced from 4.0, zeta ~0.4 for I~0.015 kg*m^2)
    "forcerange": [-50, 50],  # N*m
    "armature": 0.01,
}
_WRIST_SLIDE_NAMES = ["wrist_tx", "wrist_ty", "wrist_tz"]
_WRIST_HINGE_ORDERS = {
    "XYZ": (["wrist_rx", "wrist_ry", "wrist_rz"], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    "ZXY": (["wrist_rz", "wrist_rx", "wrist_ry"], [[0, 0, 1], [1, 0, 0], [0, 1, 0]]),
}
_WRIST_HINGE_NAMES, _WRIST_AXES = _WRIST_HINGE_ORDERS["XYZ"]  # default


# ---------------------------------------------------------------------------
# MjSpec injection helpers
# ---------------------------------------------------------------------------


def _inject_fingertip_sites(
    spec: mujoco.MjSpec, hand_side: str = "right", name_prefix: str = "", body_prefix: str = ""
) -> None:
    """Inject fingertip sites into link4 bodies.

    Args:
        spec: MjSpec to modify
        hand_side: "right" or "left" (body naming from XML)
        name_prefix: Prefix for site names (e.g. "rh_" for bimanual).
                     Empty string for backward-compatible single-hand.
        body_prefix: Extra prefix prepended to body names (from <attach prefix>).
                     E.g. "rh_" when <attach prefix="rh_"/> makes bodies "rh_right_*".
    """
    full_body_prefix = f"{body_prefix}{hand_side}_"
    for finger, pos in _FINGERTIP_SITES.items():
        link4 = spec.body(f"{full_body_prefix}{finger}_link4")
        site = link4.add_site()
        site.name = f"{name_prefix}{finger}_tip"
        site.pos = pos
        site.group = 4


def _inject_wrist_frame(
    spec: mujoco.MjSpec, hand_side: str = "right", name_prefix: str = "", body_prefix: str = ""
) -> None:
    """Inject palm_frame_body with debug frame markers.

    Used for z_warn reward reference (palm_frame site Z coordinate).

    Args:
        spec: MjSpec to modify
        hand_side: "right" or "left" (body naming from XML)
        name_prefix: Prefix for body/site/geom names (e.g. "rh_" for bimanual)
        body_prefix: Extra prefix from <attach prefix>. See _inject_fingertip_sites.
    """
    full_body_prefix = f"{body_prefix}{hand_side}_"
    hand_root = spec.body(f"{full_body_prefix}palm_link")

    frame_body = hand_root.add_body()
    frame_body.name = f"{name_prefix}palm_frame_body"
    frame_body.pos = [0, 0, 0.05]

    # Yellow sphere marker at palm_frame origin
    frame_site = frame_body.add_site()
    frame_site.name = f"{name_prefix}palm_frame"
    frame_site.pos = [0, 0, 0]
    frame_site.size = [0.005, 0, 0]
    frame_site.rgba = [1, 1, 0, 1]
    frame_site.group = 0

    # RGB axis cylinders (precomputed quaternions for 90° rotations, wxyz)
    # Default cylinder axis is Z. Rotate to X/Y via quaternions.
    _SQRT2_2 = 0.7071068
    for axis_name, pos, quat, rgba in [
        ("x", [0.02, 0, 0], [_SQRT2_2, 0, _SQRT2_2, 0], [1, 0, 0, 0.9]),
        ("y", [0, 0.02, 0], [_SQRT2_2, _SQRT2_2, 0, 0], [0, 1, 0, 0.9]),
        ("z", [0, 0, 0.02], [1, 0, 0, 0], [0, 0, 1, 0.9]),
    ]:
        g = frame_body.add_geom()
        g.name = f"{name_prefix}palm_frame_{axis_name}"
        g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        g.pos = pos
        g.quat = quat
        g.size = [0.003, 0.02, 0]
        g.rgba = rgba
        g.contype = 0
        g.conaffinity = 0
        g.group = 0
        g.mass = 0


def _inject_physics_mode(
    spec: mujoco.MjSpec,
    mount_pos: list,
    mount_quat: list,
    wrist_body_name: str = "wuji_wrist",
    name_prefix: str = "",
) -> None:
    """Inject Adroit-style physics elements for mocap-driven wrist tracking.

    Adds:
      1. freejoint + explicit inertial on wrist body
      2. mocap body (invisible, no collision)
      3. weld equality constraint (mocap → wrist)

    Args:
        spec: MjSpec to modify
        mount_pos: unused (kept for API compatibility)
        mount_quat: unused (kept for API compatibility)
        wrist_body_name: Name of the wrist body in the XML
        name_prefix: Prefix for element names (e.g. "rh_" for bimanual)
    """
    # 1. Freejoint + inertial on wrist body
    wrist_body = spec.body(wrist_body_name)
    jnt = wrist_body.add_freejoint()
    jnt.name = f"{name_prefix}wrist_freejoint"
    wrist_body.mass = _WRIST_INERTIAL["mass"]
    wrist_body.ipos = _WRIST_INERTIAL["pos"]
    wrist_body.inertia = _WRIST_INERTIAL["inertia"]
    wrist_body.explicitinertial = True

    # 2. Mocap body at SAME position as wrist body (zero weld reference offset).
    mocap = spec.worldbody.add_body()
    mocap.name = f"{name_prefix}wrist_mocap"
    mocap.mocap = True
    mocap.pos = list(wrist_body.pos)
    mocap.quat = list(wrist_body.quat)
    g = mocap.add_geom()
    g.type = mujoco.mjtGeom.mjGEOM_SPHERE
    g.size = [0.015, 0, 0]
    g.rgba = [0.9, 0.3, 0.3, 0.2]
    g.contype = 0
    g.conaffinity = 0
    g.group = 2

    # 3. Weld equality constraint
    eq = spec.add_equality()
    eq.type = mujoco.mjtEq.mjEQ_WELD
    eq.name = f"{name_prefix}wrist_weld"
    eq.objtype = mujoco.mjtObj.mjOBJ_BODY
    eq.name1 = f"{name_prefix}wrist_mocap"
    eq.name2 = wrist_body_name
    eq.solref = _WELD_SOLREF
    eq.solimp = _WELD_SOLIMP


def _inject_wrist6dof_mode(
    spec: mujoco.MjSpec,
    wrist_body_name: str = "wuji_wrist",
    joint_prefix: str = "",
    hinge_order: str = "XYZ",
) -> None:
    """Inject 6-DOF virtual wrist joints with PD position actuators.

    Adds on the specified wrist body:
      - 3 slide joints (tx, ty, tz) for translation
      - 3 hinge joints for rotation (order set by hinge_order)
      - 6 PD position actuators tracking ctrl targets

    Args:
        spec: MjSpec to modify
        wrist_body_name: Name of the wrist body to add joints to
        joint_prefix: Prefix for joint/actuator names (e.g. "rh_" for bimanual).
                      Empty string for backward-compatible single-hand.
        hinge_order: Rotation hinge order, "XYZ" (default) or "ZXY" (play_dynamic style).
    """
    wrist_body = spec.body(wrist_body_name)

    # Explicit inertial for wrist body (dynamics stability)
    wrist_body.mass = _WRIST_INERTIAL["mass"]
    wrist_body.ipos = _WRIST_INERTIAL["pos"]
    wrist_body.inertia = _WRIST_INERTIAL["inertia"]
    wrist_body.explicitinertial = True

    # --- Slide joints (translation) ---
    for name, axis in zip(_WRIST_SLIDE_NAMES, _WRIST_AXES):
        jnt = wrist_body.add_joint()
        jnt.name = f"{joint_prefix}{name}"
        jnt.type = mujoco.mjtJoint.mjJNT_SLIDE
        jnt.axis = axis
        jnt.range = _WRIST6DOF_SLIDE["range"]
        jnt.damping = 0
        jnt.armature = _WRIST6DOF_SLIDE["armature"]

    # --- Hinge joints (rotation, intrinsic Euler — order set by hinge_order) ---
    hinge_names, hinge_axes = _WRIST_HINGE_ORDERS[hinge_order]
    for name, axis in zip(hinge_names, hinge_axes):
        jnt = wrist_body.add_joint()
        jnt.name = f"{joint_prefix}{name}"
        jnt.type = mujoco.mjtJoint.mjJNT_HINGE
        jnt.axis = axis
        jnt.range = _WRIST6DOF_HINGE["range"]
        jnt.damping = 0
        jnt.armature = _WRIST6DOF_HINGE["armature"]

    # --- PD position actuators (appended after XML finger actuators -> ctrl[20:26]) ---
    # force = kp * (ctrl - qpos) - kv * qvel
    joint_cfgs = [(n, _WRIST6DOF_SLIDE) for n in _WRIST_SLIDE_NAMES] + [
        (n, _WRIST6DOF_HINGE) for n in hinge_names
    ]
    for joint_name, cfg in joint_cfgs:
        act = spec.add_actuator()
        act.name = f"{joint_prefix}{joint_name}_act"
        act.target = f"{joint_prefix}{joint_name}"
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        act.gainprm[0] = cfg["kp"]
        act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        act.biasprm[1] = -cfg["kp"]
        act.biasprm[2] = -cfg["kv"]
        act.forcerange = cfg["forcerange"]
        act.ctrlrange = cfg["range"]


def _reindex_home_keyframe(
    spec: mujoco.MjSpec,
    pre_qpos: np.ndarray,
    pre_ctrl: np.ndarray,
    *,
    physics_mode: bool = False,
    wrist_mode: str | None = None,
    n_wrist_sets: int = 1,
) -> None:
    """Re-index home keyframe after joint injection.

    MjSpec does NOT re-index keyframe arrays when joints are added via API.
    This function remaps pre-injection values to the post-injection layout.

    wrist6dof (single): qpos [20 finger + 7 cube] -> [6 wrist + 20 finger + 7 cube]
                         ctrl [20 finger]           -> [20 finger + 6 wrist]
    wrist6dof (N sets): qpos [N*20 finger + 7 cube] -> [N*(6 wrist + 20 finger) + 7 cube]
                         ctrl [N*20 finger]           -> [N*(20 finger + 6 wrist)]
    physics:             qpos [20 finger + 7 cube] -> [7 freejoint + 20 finger + 7 cube]
                         ctrl unchanged (no new actuators)

    Args:
        spec: MjSpec to modify
        pre_qpos: Keyframe qpos before injection
        pre_ctrl: Keyframe ctrl before injection
        physics_mode: Freejoint mode
        wrist_mode: "wrist6dof" for 6-DOF virtual joints
        n_wrist_sets: Number of wrist joint sets injected (1 for single-hand, 2 for bimanual)
    """
    if wrist_mode == "wrist6dof":
        # Split pre_qpos into per-hand finger blocks + cube
        n_finger_per_hand = 20
        n_cube = 7  # freejoint qpos
        n_hands = n_wrist_sets
        total_finger = n_hands * n_finger_per_hand

        # Pre-injection: [hand1_finger(20), hand2_finger(20), ..., cube(7)]
        finger_blocks = []
        for h in range(n_hands):
            start = h * n_finger_per_hand
            end = start + n_finger_per_hand
            finger_blocks.append(pre_qpos[start:end])
        cube_qpos = pre_qpos[total_finger : total_finger + n_cube]

        # Post-injection: [wrist1(6)+finger1(20), wrist2(6)+finger2(20), ..., cube(7)]
        new_qpos_parts = []
        for block in finger_blocks:
            new_qpos_parts.append(np.zeros(6))  # wrist qpos = 0
            new_qpos_parts.append(block)
        new_qpos_parts.append(cube_qpos)
        new_qpos = np.concatenate(new_qpos_parts)

        # Pre-ctrl: [hand1_finger(20), hand2_finger(20), ...]
        # Post-ctrl: MuJoCo orders XML-defined actuators first, then MjSpec-injected.
        # Compiled layout: [hand1_finger(20), hand2_finger(20), ..., wrist1(6), wrist2(6), ...]
        new_ctrl_parts = []
        for h in range(n_hands):
            start = h * n_finger_per_hand
            end = start + n_finger_per_hand
            new_ctrl_parts.append(pre_ctrl[start:end])
        for _ in range(n_hands):
            new_ctrl_parts.append(np.zeros(6))
        new_ctrl = np.concatenate(new_ctrl_parts)
    elif physics_mode:
        # Freejoint: [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z]
        freejoint_init = np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float64)
        new_qpos = np.concatenate([freejoint_init, pre_qpos])
        new_ctrl = pre_ctrl
    else:
        return

    for k in spec.keys:
        if k.name == "home":
            k.qpos = new_qpos
            k.ctrl = new_ctrl
            return


def _set_collision_group(spec: mujoco.MjSpec, body_name: str, contype: int, conaffinity: int) -> None:
    """Set contype/conaffinity for all collision geoms in a body subtree.

    Only modifies geoms that currently participate in collision (contype > 0
    or conaffinity > 0). Visual-only geoms (contype=0, conaffinity=0) are
    left unchanged.

    Args:
        spec: MjSpec to modify
        body_name: Root body name (subtree is traversed recursively)
        contype: New contype value
        conaffinity: New conaffinity value
    """
    def _recurse(body):
        for geom in body.geoms:
            if geom.contype > 0 or geom.conaffinity > 0:
                geom.contype = contype
                geom.conaffinity = conaffinity
        for child in body.bodies:
            _recurse(child)

    body = spec.body(body_name)
    if body is not None:
        _recurse(body)


def _set_cube_collision_group(spec: mujoco.MjSpec, body_name: str = "cube") -> None:
    """Set cube collision geoms to collide with all hand groups and floor.

    Bit layout: bit 0 = RH, bit 1 = floor, bit 2 = LH.
    Cube needs contype=5 (bit 0+2) so hands detect it,
    conaffinity=7 (bit 0+1+2) so it responds to all.

    Args:
        spec: MjSpec to modify
        body_name: Cube root body name
    """
    def _recurse(body):
        for geom in body.geoms:
            if geom.contype > 0 or geom.conaffinity > 0:
                geom.contype = 5   # bit 0 + bit 2 (RH + LH)
                geom.conaffinity = 7  # bit 0 + bit 1 + bit 2 (RH + floor + LH)
        for child in body.bodies:
            _recurse(child)

    body = spec.body(body_name)
    if body is not None:
        _recurse(body)


def _parse_pos_str(pos_str: str) -> list[float]:
    """Parse space-separated position string to float list."""
    return [float(x) for x in pos_str.split()]


def _parse_quat_str(quat_str: str) -> list[float]:
    """Parse space-separated quaternion string to float list."""
    return [float(x) for x in quat_str.split()]


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

# Type alias for multi-hand configuration.
# Required keys: "side", "wrist_body", "prefix"
# Optional key: "attach_prefix" — the prefix used in <attach prefix="..."/> in scene XML.
#   When non-empty, body names become "{attach_prefix}{side}_palm_link" etc.
#   Solves <custom><numeric> name collision when attaching multiple hand models.
HandSpec = dict


def load_scene_model(
    scene_xml_path: str,
    hand_side: str = "right",
    mount_pos: str = "0 0 0",
    mount_quat: str = "1 0 0 0",
    physics_mode: bool = False,
    wrist_mode: str | None = None,
    assets: dict[str, bytes] | None = None,
    hands: list[HandSpec] | None = None,
    hinge_order: str = "XYZ",
) -> mujoco.MjModel:
    """Load a scene with MjSpec injection for non-native elements.

    Scene XML uses <model> + <attach> to load hand and cube from disk.
    MjSpec injects fingertip sites, debug markers, and physics/wrist elements.

    Two usage modes:
      1. Single-hand (hands=None): Uses hand_side/mount_pos/etc parameters.
         Element names are unprefixed for backward compatibility.
      2. Multi-hand (hands=[...]): Each hand dict specifies side, wrist_body,
         prefix, and attach_prefix. Scene XML must use matching <attach prefix>.

    Args:
        scene_xml_path: Absolute path to scene.xml
        hand_side: "right" or "left" (single-hand mode only)
        mount_pos: wrist body position string (for physics_mode)
        mount_quat: wrist body quaternion wxyz string (for physics_mode)
        physics_mode: If True, add mocap + freejoint + weld constraint
        wrist_mode: "wrist6dof" for 6-DOF virtual joints with PD actuators
        assets: Unused (kept for API compatibility)
        hands: List of hand specs for multi-hand mode. Each dict:
               {"side": "right"|"left", "wrist_body": str, "prefix": str,
                "attach_prefix": str}
               "attach_prefix" must match <attach prefix="..."/> in scene XML
               so body lookups use the correct prefixed names.
               When provided, hand_side/mount_pos/mount_quat are ignored.

    Returns:
        Compiled MjModel ready for simulation or mjx.put_model().

    Raises:
        ValueError: If both physics_mode and wrist_mode are specified.
    """
    if physics_mode and wrist_mode:
        raise ValueError("Cannot use both physics_mode and wrist_mode simultaneously.")

    # Load scene spec from file (all referenced files resolved from disk)
    spec = mujoco.MjSpec.from_file(scene_xml_path)

    # Save pre-injection keyframe for re-indexing after joint injection
    pre_qpos, pre_ctrl = None, None
    for k in spec.keys:
        if k.name == "home":
            pre_qpos = np.array(k.qpos, dtype=np.float64)
            pre_ctrl = np.array(k.ctrl, dtype=np.float64)
            break

    if wrist_mode is not None and wrist_mode != "wrist6dof":
        raise ValueError(f"Unsupported wrist_mode: {wrist_mode!r}. Valid: None, 'wrist6dof'.")

    if hands is not None:
        # --- Multi-hand mode ---
        for hand in hands:
            side = hand["side"]
            wrist_body = hand["wrist_body"]
            prefix = hand.get("prefix", "")
            attach_prefix = hand.get("attach_prefix", "")

            _inject_fingertip_sites(spec, side, name_prefix=prefix, body_prefix=attach_prefix)
            _inject_wrist_frame(spec, side, name_prefix=prefix, body_prefix=attach_prefix)

            if physics_mode:
                _inject_physics_mode(
                    spec,
                    mount_pos=[0, 0, 0],
                    mount_quat=[1, 0, 0, 0],
                    wrist_body_name=wrist_body,
                    name_prefix=prefix,
                )
            elif wrist_mode == "wrist6dof":
                _inject_wrist6dof_mode(spec, wrist_body_name=wrist_body, joint_prefix=prefix, hinge_order=hinge_order)

        # --- Collision groups: prevent left-right hand collision ---
        # Bit layout: bit 0 (1) = RH, bit 1 (2) = floor, bit 2 (4) = LH
        # RH vs LH: (1&4)|(4&1) = 0 → no collision
        # Both vs Cube: cube contype=5, conaffinity=7 → collides with all
        if len(hands) == 2:
            _set_collision_group(spec, hands[0]["wrist_body"], contype=1, conaffinity=1)
            _set_collision_group(spec, hands[1]["wrist_body"], contype=4, conaffinity=4)
            _set_cube_collision_group(spec)

        # Re-index home keyframe
        if pre_qpos is not None and (physics_mode or wrist_mode):
            _reindex_home_keyframe(
                spec,
                pre_qpos,
                pre_ctrl,
                physics_mode=physics_mode,
                wrist_mode=wrist_mode,
                n_wrist_sets=len(hands),
            )
    else:
        # --- Single-hand mode (backward compatible, no prefix) ---
        _inject_fingertip_sites(spec, hand_side)
        _inject_wrist_frame(spec, hand_side)

        if physics_mode:
            _inject_physics_mode(
                spec,
                mount_pos=_parse_pos_str(mount_pos),
                mount_quat=_parse_quat_str(mount_quat),
            )
        elif wrist_mode == "wrist6dof":
            _inject_wrist6dof_mode(spec, hinge_order=hinge_order)

        # Re-index home keyframe
        if pre_qpos is not None and (physics_mode or wrist_mode):
            _reindex_home_keyframe(
                spec,
                pre_qpos,
                pre_ctrl,
                physics_mode=physics_mode,
                wrist_mode=wrist_mode,
            )

    return spec.compile()
