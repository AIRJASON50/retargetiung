"""MuJoCo viewer overlay utilities for keypoint and mesh visualization.

Provides reusable primitives for drawing spheres, lines, and configuring
model transparency in the MuJoCo passive viewer. Shared by demos (manus,
hocap, omniretarget) and scripts (compare_hocap).
"""
from __future__ import annotations

import mujoco
import numpy as np

# ============================================================
# Key codes (GLFW constants used by mujoco.viewer)
# ============================================================
KEY_SPACE = 32
KEY_LEFT = 263
KEY_RIGHT = 262
KEY_UP = 265
KEY_DOWN = 264


# ============================================================
# Drawing primitives
# ============================================================


def add_sphere(
    scene: mujoco.MjvScene,
    pos: np.ndarray,
    rgba: np.ndarray,
    size: float = 0.003,
) -> None:
    """Add a sphere geom to the viewer scene.

    Args:
        scene: MjvScene to append the geom to (typically ``viewer.user_scn``).
        pos: World-frame position, shape ``(3,)``.
        rgba: Color as ``[R, G, B, A]`` float32 array.
        size: Sphere radius in metres.
    """
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


def add_line(
    scene: mujoco.MjvScene,
    p1: np.ndarray,
    p2: np.ndarray,
    rgba: np.ndarray,
    width: float = 1.5,
) -> None:
    """Add a line-segment geom (connector) to the viewer scene.

    Args:
        scene: MjvScene to append the geom to.
        p1: Start position, shape ``(3,)``.
        p2: End position, shape ``(3,)``.
        rgba: Color as ``[R, G, B, A]`` float32 array.
        width: Line width in pixels.
    """
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


# ============================================================
# Model helpers
# ============================================================


def set_geom_alpha(
    model: mujoco.MjModel,
    alpha: float = 0.25,
    skip_planes: bool = True,
    skip_names: list[str] | None = None,
) -> None:
    """Make model geoms semi-transparent (e.g. hand meshes).

    Iterates over all geoms in the model and sets ``geom_rgba[:, 3]``
    to *alpha*, optionally skipping ground planes and geoms whose names
    contain any string in *skip_names*.

    Args:
        model: MjModel whose geom alpha values to modify.
        alpha: Target alpha value (0 = invisible, 1 = opaque).
        skip_planes: If True, geoms of type ``mjGEOM_PLANE`` keep
            their original alpha.
        skip_names: List of substrings.  Any geom whose name contains
            one of these substrings keeps its original alpha.
    """
    skip_names = skip_names or []
    for i in range(model.ngeom):
        if skip_planes and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        name = model.geom(i).name
        if any(s in name for s in skip_names):
            continue
        model.geom_rgba[i, 3] = alpha
