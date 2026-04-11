"""
Pinocchio wrapper for fixed-base robot hand (WujiHand).

Uses the same URDF as baseline (wuji_retargeting), which includes tip_link frames.
Replaces the previous MuJoCo-based FK/Jacobian with Pinocchio to match baseline's
description asset exactly.

All joints are hinge: nq = nv, Jacobian is directly usable.
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

    def get_default_qpos(self) -> np.ndarray:
        """Get the default (mid-range) joint positions."""
        return (self.q_lb + self.q_ub) / 2.0
