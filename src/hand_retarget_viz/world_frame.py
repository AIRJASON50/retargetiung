"""Convert retarget qpos (robot-local) back to world frame for overlay rendering."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as RotLib


def qpos_to_world(q: np.ndarray, R_inv: np.ndarray, wrist_w: np.ndarray) -> np.ndarray:
    """Transform retarget qpos from robot-local to world frame."""
    q = q.copy()
    q[:3] = q[:3] @ R_inv + wrist_w
    R_hinge = RotLib.from_euler("XYZ", q[3:6]).as_matrix()
    q[3:6] = RotLib.from_matrix(R_inv.T @ R_hinge).as_euler("XYZ")
    return q
