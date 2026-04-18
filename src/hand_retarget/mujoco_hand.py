"""
Hand model wrappers for retargeting FK/Jacobian computation.

Two classes:
  MuJoCoHandModel -- Pinocchio-based, fixed base, 20 DOF (robot_only mode)
  MuJoCoFloatingHandModel -- MuJoCo-based, 6DOF wrist + 20 finger = 26 DOF (object mode)
"""

from __future__ import annotations

# ==============================================================================
# Imports
# ==============================================================================
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import pinocchio as pin

# ==============================================================================
# Constants
# ==============================================================================

NUM_FINGERS: int = 5
"""Number of fingers on the hand."""

WRIST_DOF: int = 6
"""Number of wrist DOF (3 slide + 3 hinge) for floating base mode."""

# ==============================================================================
# Protocols
# ==============================================================================


@runtime_checkable
class HandModelProtocol(Protocol):
    """Structural interface for hand FK/Jacobian models.

    Both ``MuJoCoHandModel`` (Pinocchio, fixed base) and
    ``MuJoCoFloatingHandModel`` (MuJoCo, floating base) satisfy this protocol.
    New hand models (e.g. Allegro, Shadow) should implement these members.
    """

    nq: int
    nv: int
    q_lb: np.ndarray
    q_ub: np.ndarray

    def forward(self, q: np.ndarray) -> None: ...

    def get_body_pos(self, body_name: str) -> np.ndarray: ...

    def get_body_positions(self, body_names: list[str]) -> np.ndarray: ...

    def get_body_jacp(self, body_name: str) -> np.ndarray: ...

    def get_body_jacobians(self, body_names: list[str]) -> np.ndarray: ...

    def get_default_qpos(self) -> np.ndarray: ...


# ==============================================================================
# Classes
# ==============================================================================


class PinocchioHandModel:
    """Pinocchio-based model wrapper for a fixed-base dexterous hand.

    Name kept as MuJoCoHandModel for backward compatibility with retargeter.py imports.
    """

    # ==========================================================================
    # Dunder Methods
    # ==========================================================================

    def __init__(self, urdf_path: str, probe_offset: float = 0.0) -> None:
        """Initialize the fixed-base hand model from a URDF.

        Args:
            urdf_path: Path to the hand URDF file.
            probe_offset: If > 0, add virtual probe frames offset from each
                fingertip along its local z-axis. Set to 0 to disable probes.
        """
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.nq = self.model.nq  # 20
        self.nv = self.model.nv  # 20

        # Joint limits
        self.q_lb = self.model.lowerPositionLimit.copy()
        self.q_ub = self.model.upperPositionLimit.copy()

        # Cache frame name -> id mapping
        self._frame_ids = {}
        for i in range(self.model.nframes):
            name = self.model.frames[i].name
            if self.model.frames[i].type == pin.BODY:
                self._frame_ids[name] = i

        # Add orientation probe frames if requested
        if probe_offset > 0:
            self._add_probe_frames(probe_offset)

    # ==========================================================================
    # Public Methods
    # ==========================================================================

    def forward(self, q: np.ndarray) -> None:
        """Set qpos and run forward kinematics."""
        q = np.asarray(q, dtype=np.float64)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        # Also compute joint Jacobians for later use
        pin.computeJointJacobians(self.model, self.data, q)

    def get_body_id(self, body_name: str) -> int:
        """Get frame id by name."""
        if body_name not in self._frame_ids:
            raise ValueError(f"Body '{body_name}' not found. Available: {list(self._frame_ids.keys())}")
        return self._frame_ids[body_name]

    def get_body_pos(self, body_name: str) -> np.ndarray:
        """Get world position of a body. Returns (3,) array."""
        fid = self.get_body_id(body_name)
        return self.data.oMf[fid].translation.copy()

    def get_body_positions(self, body_names: list[str]) -> np.ndarray:
        """Get world positions for multiple bodies. Returns (N, 3) array."""
        positions = np.zeros((len(body_names), 3))
        for i, name in enumerate(body_names):
            positions[i] = self.get_body_pos(name)
        return positions

    def get_body_jacp(self, body_name: str) -> np.ndarray:
        """Get translational Jacobian for a body in world frame.

        Args:
            body_name: Name of the body frame to query.

        Returns:
            (3, nq) matrix: dp/dq in world frame.
        """
        fid = self.get_body_id(body_name)
        J_local = pin.getFrameJacobian(self.model, self.data, fid, pin.LOCAL)
        R = self.data.oMf[fid].rotation
        return R @ J_local[:3, :]  # (3, nq) in world frame

    def get_body_jacobians(self, body_names: list[str]) -> np.ndarray:
        """Get stacked translational Jacobians for multiple bodies.

        Args:
            body_names: List of body frame names to query.

        Returns:
            (3*N, nq) matrix of stacked translational Jacobians.
        """
        N = len(body_names)
        J = np.zeros((3 * N, self.nv))
        for i, name in enumerate(body_names):
            J[3 * i : 3 * (i + 1), :] = self.get_body_jacp(name)
        return J

    def get_default_qpos(self) -> np.ndarray:
        """Get default (mid-range) joint positions for fixed-base hand."""
        return (self.q_lb + self.q_ub) / 2.0

    # ==========================================================================
    # Private Methods
    # ==========================================================================

    def _add_probe_frames(self, probe_offset: float) -> None:
        """Add virtual probe frames offset from each tip_link along its local z-axis.

        These frames encode fingertip orientation as position, enabling
        the Laplacian optimizer to distinguish normal flexion from hyperextension.
        """
        side = "left" if "left_palm_link" in self._frame_ids else "right"
        for f in range(1, NUM_FINGERS + 1):
            tip_name = f"{side}_finger{f}_tip_link"
            probe_name = f"{side}_finger{f}_tip_probe"
            if tip_name not in self._frame_ids:
                continue
            tip_fid = self._frame_ids[tip_name]
            tip_frame = self.model.frames[tip_fid]
            # Offset along local z-axis of the tip_link frame
            offset_placement = pin.SE3(np.eye(3), np.array([0.0, 0.0, probe_offset]))
            probe_placement = tip_frame.placement * offset_placement
            probe_frame = pin.Frame(
                probe_name,
                tip_frame.parentJoint,
                tip_fid,
                probe_placement,
                pin.BODY,
            )
            probe_fid = self.model.addFrame(probe_frame)
            self._frame_ids[probe_name] = probe_fid

        # Rebuild data to account for new frames
        self.data = self.model.createData()


class MuJoCoFloatingHandModel:
    """MuJoCo-based hand model with 6DOF wrist (3 slide + 3 hinge) + 20 finger joints.

    Uses hand_builder.load_scene_model() to inject wrist6dof joints at runtime.
    All joints are slide/hinge: nq = nv = 26, no quaternion handling needed.

    Body names use the MuJoCo model's names (e.g. 'right_palm_link', 'right_finger1_link4').
    The wrist body is 'wuji_wrist' (parent of all hand bodies).
    """

    # ==========================================================================
    # Dunder Methods
    # ==========================================================================

    def __init__(self, scene_xml: str, hand_side: str = "right") -> None:
        """Initialize the floating-base hand model from a MuJoCo scene XML.

        Args:
            scene_xml: Path to the MuJoCo scene XML file.
            hand_side: Which hand to load, ``"left"`` or ``"right"``.
        """
        import mujoco as mj

        from scene_builder.hand_builder import load_scene_model

        self._scene_xml = scene_xml
        self._hand_side = hand_side
        self._has_object = False
        self.model = load_scene_model(scene_xml, hand_side=hand_side, wrist_mode="wrist6dof")
        self.data = mj.MjData(self.model)
        self.nq = self.model.nq  # 26
        self.nv = self.model.nv  # 26

        # Joint limits
        self.q_lb = self.model.jnt_range[:, 0].copy()
        self.q_ub = self.model.jnt_range[:, 1].copy()

        self._body_ids = {}
        self._site_ids = {}
        self._rebuild_caches()

    # ==========================================================================
    # Public Methods
    # ==========================================================================

    def forward(self, q: np.ndarray) -> None:
        """Set qpos and run forward kinematics."""
        import mujoco as mj

        self.data.qpos[: self.nq] = q
        mj.mj_forward(self.model, self.data)

    def get_body_id(self, body_name: str) -> int:
        """Get body id by name."""
        if body_name not in self._body_ids:
            raise ValueError(f"Body '{body_name}' not found. Available: {list(self._body_ids.keys())}")
        return self._body_ids[body_name]

    def get_body_pos(self, body_name: str) -> np.ndarray:
        """Get world position of a body (or site for tip_link). Returns (3,) array."""
        bid = self.get_body_id(body_name)
        if bid < 0:
            # Negative = site-backed virtual body (tip_link)
            sid = -(bid + 1)
            return self.data.site_xpos[sid].copy()
        return self.data.xpos[bid].copy()

    def get_body_positions(self, body_names: list[str]) -> np.ndarray:
        """Get world positions for multiple bodies. Returns (N, 3) array."""
        positions = np.zeros((len(body_names), 3))
        for i, name in enumerate(body_names):
            positions[i] = self.get_body_pos(name)
        return positions

    def get_body_jacp(self, body_name: str) -> np.ndarray:
        """Get translational Jacobian for a body (or site) in world frame.

        Args:
            body_name: Name of the body (or site-backed tip_link) to query.

        Returns:
            (3, nq) matrix: dp/dq in world frame.
        """
        import mujoco as mj

        bid = self.get_body_id(body_name)
        jacp = np.zeros((3, self.nv))
        if bid < 0:
            # Site-backed: use mj_jacSite
            sid = -(bid + 1)
            mj.mj_jacSite(self.model, self.data, jacp, None, sid)
        else:
            mj.mj_jacBody(self.model, self.data, jacp, None, bid)
        return jacp

    def get_body_jacobians(self, body_names: list[str]) -> np.ndarray:
        """Get stacked translational Jacobians for multiple bodies.

        Args:
            body_names: List of body names to query.

        Returns:
            (3*N, nq) matrix of stacked translational Jacobians.
        """
        N = len(body_names)
        J = np.zeros((3 * N, self.nv))
        for i, name in enumerate(body_names):
            J[3 * i : 3 * (i + 1), :] = self.get_body_jacp(name)
        return J

    def get_default_qpos(self) -> np.ndarray:
        """Get default joint positions (wrist at zero, fingers at mid-range)."""
        q = np.zeros(self.nq)
        q[WRIST_DOF:] = (self.q_lb[WRIST_DOF:] + self.q_ub[WRIST_DOF:]) / 2.0
        return q

    def inject_object_mesh(self, mesh_path: str, hand_side: str = "left") -> None:
        """Rebuild model with object mesh for collision queries.

        Injects object as a mocap body with collision geom (contype=2, conaffinity=1).
        Hand geoms are set to contype=1, conaffinity=2 before compile.

        Args:
            mesh_path: Absolute path to object STL file.
            hand_side: ``"left"`` or ``"right"``.

        Raises:
            FileNotFoundError: If ``mesh_path`` does not exist (raised by MuJoCo at compile).
        """
        import mujoco as mj

        from scene_builder.hand_builder import (
            _inject_fingertip_sites,
            _inject_wrist6dof_mode,
        )

        scene_xml = self._scene_xml
        spec = mj.MjSpec.from_file(scene_xml)
        _inject_fingertip_sites(spec, hand_side)
        _inject_wrist6dof_mode(spec)

        # Enable hand collision geoms
        def _enable_col(body):
            for g in body.geoms:
                g.contype = 1
                g.conaffinity = 2
            for c in body.bodies:
                _enable_col(c)

        _enable_col(spec.body("wuji_wrist"))

        # Add object as mocap body
        obj_body = spec.worldbody.add_body()
        obj_body.name = "retarget_object"
        obj_body.mocap = True
        obj_mesh = spec.add_mesh()
        obj_mesh.name = "retarget_obj_mesh"
        obj_mesh.file = str(Path(mesh_path).resolve())
        obj_geom = obj_body.add_geom()
        obj_geom.name = "retarget_obj_geom"
        obj_geom.type = mj.mjtGeom.mjGEOM_MESH
        obj_geom.meshname = "retarget_obj_mesh"
        obj_geom.contype = 2
        obj_geom.conaffinity = 1

        self.model = spec.compile()
        self.data = mj.MjData(self.model)

        # Rebuild caches (joint limits unchanged, but body/site ids may shift)
        self.q_lb = self.model.jnt_range[:, 0].copy()
        self.q_ub = self.model.jnt_range[:, 1].copy()
        self._rebuild_caches()

        # Cache fingertip collision geom ids (link4_col geoms)
        self._tip_col_geom_ids = []
        for f in range(1, NUM_FINGERS + 1):
            gname = f"{hand_side}_finger{f}_link4_col"
            for i in range(self.model.ngeom):
                if self.model.geom(i).name == gname:
                    self._tip_col_geom_ids.append(i)
                    break

        # Object geom id
        self._obj_geom_id = self.model.geom("retarget_obj_geom").id
        self._has_object = True

    def set_object_pose(self, pos: np.ndarray, quat_xyzw: np.ndarray) -> None:
        """Set object mocap pose (call before forward/collision queries).

        Args:
            pos: (3,) world position.
            quat_xyzw: (4,) quaternion in xyzw (scipy) convention.
        """
        self.data.mocap_pos[0] = pos
        self.data.mocap_quat[0] = [
            quat_xyzw[3],
            quat_xyzw[0],
            quat_xyzw[1],
            quat_xyzw[2],
        ]

    def query_tip_penetration(self, threshold: float = 0.05) -> list[tuple[np.ndarray, float, int]]:
        """Query signed distances between fingertip geoms and object.

        Must call forward() first to update kinematics.

        Args:
            threshold: max distance to report (meters). Pairs farther are skipped.

        Returns:
            List of (J_contact, phi, tip_idx) for each close pair:
              J_contact: (nq,) contact normal projected Jacobian
              phi: signed distance (positive=separated, negative=penetrating)
              tip_idx: fingertip index 0-4 (thumb..pinky)
        """
        import mujoco as mj

        if not getattr(self, "_has_object", False):
            return []

        results = []
        fromto = np.zeros(6)
        jacp = np.zeros((3, self.nv))

        for tip_idx, gid in enumerate(self._tip_col_geom_ids):
            phi = mj.mj_geomDistance(self.model, self.data, gid, self._obj_geom_id, threshold, fromto)
            if phi > threshold:
                continue

            # Contact normal: from object surface point to hand surface point
            p_hand = fromto[:3]
            p_obj = fromto[3:]
            diff = p_hand - p_obj
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                continue
            nhat = diff / dist

            # Jacobian at the hand contact point
            body_id = self.model.geom_bodyid[gid]
            mj.mj_jac(self.model, self.data, jacp, None, p_hand, body_id)

            # Object is mocap (J=0), so J_rel = nhat @ J_hand
            J_contact = nhat @ jacp  # (nq,)

            results.append((J_contact.copy(), phi, tip_idx))

        return results

    # ==========================================================================
    # Private Methods
    # ==========================================================================

    def _rebuild_caches(self) -> None:
        """Rebuild body/site ID caches from the current MuJoCo model.

        Populates ``_body_ids`` and ``_site_ids`` dictionaries, and registers
        tip_link names as virtual bodies backed by fingertip sites for
        compatibility with ``JOINTS_MAPPING``.

        Called after initial load and after ``inject_object_mesh`` recompiles.
        """
        self._body_ids.clear()
        for i in range(self.model.nbody):
            name = self.model.body(i).name
            if name:
                self._body_ids[name] = i

        self._site_ids.clear()
        for i in range(self.model.nsite):
            name = self.model.site(i).name
            if name:
                self._site_ids[name] = i

        # Map tip_link names to site names for compatibility with JOINTS_MAPPING
        # URDF has "right_finger1_tip_link" etc, MuJoCo has "finger1_tip" site
        side_prefix = f"{self._hand_side}_"
        for f in range(1, NUM_FINGERS + 1):
            tip_link_name = f"{side_prefix}finger{f}_tip_link"
            site_name = f"finger{f}_tip"
            if site_name in self._site_ids:
                # Register tip_link as a "virtual body" backed by a site
                self._body_ids[tip_link_name] = -(self._site_ids[site_name] + 1)  # negative = site


# Backward compatibility alias
MuJoCoHandModel = PinocchioHandModel
