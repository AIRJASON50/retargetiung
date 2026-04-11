"""
Hand model wrappers for retargeting FK/Jacobian computation.

Two classes:
  MuJoCoHandModel — Pinocchio-based, fixed base, 20 DOF (robot_only mode)
  MuJoCoFloatingHandModel — MuJoCo-based, 6DOF wrist + 20 finger = 26 DOF (object mode)
"""

import numpy as np
import pinocchio as pin


class MuJoCoHandModel:
    """Pinocchio-based model wrapper for a fixed-base dexterous hand.

    Name kept as MuJoCoHandModel for backward compatibility with retargeter.py imports.
    """

    def __init__(self, urdf_path: str):
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

    def forward(self, q: np.ndarray):
        """Set qpos and run forward kinematics."""
        q = np.asarray(q, dtype=np.float64)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        # Also compute joint Jacobians for later use
        pin.computeJointJacobians(self.model, self.data, q)
        self._last_q = q.copy()

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
        """
        Get translational Jacobian for a body in world frame.
        Returns (3, nq) matrix: dp/dq.
        """
        fid = self.get_body_id(body_name)
        J_local = pin.getFrameJacobian(self.model, self.data, fid, pin.LOCAL)
        R = self.data.oMf[fid].rotation
        return R @ J_local[:3, :]  # (3, nq) in world frame

    def get_body_jacobians(self, body_names: list[str]) -> np.ndarray:
        """
        Get stacked translational Jacobians for multiple bodies.
        Returns (3*N, nq) matrix.
        """
        N = len(body_names)
        J = np.zeros((3 * N, self.nv))
        for i, name in enumerate(body_names):
            J[3 * i : 3 * (i + 1), :] = self.get_body_jacp(name)
        return J


class MuJoCoFloatingHandModel:
    """MuJoCo-based hand model with 6DOF wrist (3 slide + 3 hinge) + 20 finger joints.

    Uses hand_builder.load_scene_model() to inject wrist6dof joints at runtime.
    All joints are slide/hinge: nq = nv = 26, no quaternion handling needed.

    Body names use the MuJoCo model's names (e.g. 'right_palm_link', 'right_finger1_link4').
    The wrist body is 'wuji_wrist' (parent of all hand bodies).
    """

    def __init__(self, scene_xml: str, hand_side: str = "right"):
        import mujoco as mj
        from scene_builder.hand_builder import load_scene_model

        self.model = load_scene_model(
            scene_xml, hand_side=hand_side, wrist_mode="wrist6dof"
        )
        self.data = mj.MjData(self.model)
        self.nq = self.model.nq  # 26
        self.nv = self.model.nv  # 26

        # Joint limits
        self.q_lb = self.model.jnt_range[:, 0].copy()
        self.q_ub = self.model.jnt_range[:, 1].copy()

        # Cache body name -> id mapping
        self._body_ids = {}
        for i in range(self.model.nbody):
            name = self.model.body(i).name
            if name:
                self._body_ids[name] = i

        # Cache site name -> id mapping (fingertip sites injected by hand_builder)
        self._site_ids = {}
        for i in range(self.model.nsite):
            name = self.model.site(i).name
            if name:
                self._site_ids[name] = i

        # Map tip_link names to site names for compatibility with JOINTS_MAPPING
        # URDF has "right_finger1_tip_link" etc, MuJoCo has "finger1_tip" site
        side_prefix = f"{hand_side}_"
        for f in range(1, 6):
            tip_link_name = f"{side_prefix}finger{f}_tip_link"
            site_name = f"finger{f}_tip"
            if site_name in self._site_ids:
                # Register tip_link as a "virtual body" backed by a site
                self._body_ids[tip_link_name] = -(self._site_ids[site_name] + 1)  # negative = site

    def forward(self, q: np.ndarray):
        """Set qpos and run forward kinematics."""
        import mujoco as mj
        self.data.qpos[:self.nq] = q
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
        """Get translational Jacobian for a body (or site). Returns (3, nq) matrix."""
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
        """Get stacked translational Jacobians. Returns (3*N, nq) matrix."""
        N = len(body_names)
        J = np.zeros((3 * N, self.nv))
        for i, name in enumerate(body_names):
            J[3 * i : 3 * (i + 1), :] = self.get_body_jacp(name)
        return J

    def get_default_qpos(self) -> np.ndarray:
        """Get default joint positions (wrist at zero, fingers at mid-range)."""
        q = np.zeros(self.nq)
        # Wrist (first 6) at zero, fingers at mid-range
        q[6:] = (self.q_lb[6:] + self.q_ub[6:]) / 2.0
        return q

    def get_default_qpos(self) -> np.ndarray:
        """Get the default (mid-range) joint positions."""
        return (self.q_lb + self.q_ub) / 2.0
