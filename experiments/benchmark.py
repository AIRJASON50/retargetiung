"""
Retargeting benchmark metrics.

Computes per-frame and aggregate metrics from source landmarks and robot qpos.
Metrics based on GeoRT, DexFlow, and baseline evaluation conventions.

Usage:
    from experiments.benchmark import RetargetBenchmark
    bench = RetargetBenchmark(urdf_path, hand_side="left")
    results = bench.evaluate(source_landmarks, qpos_seq)
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pinocchio as pin

# ==============================================================================
# Constants
# ==============================================================================

DEFAULT_FPS: float = 156.0
"""Default data frame rate for HO-Cap sequences (Hz)."""

PINCH_THRESHOLD_MM: float = 30.0
"""Thumb-index distance below which a frame is classified as near-pinch (mm)."""

DIRECTION_NORM_EPS: float = 1e-8
"""Epsilon for direction vector normalization to avoid division by zero."""

JOINT_LIMIT_TOL: float = 1e-6
"""Tolerance for joint limit violation detection (rad)."""


# ==============================================================================
# Classes
# ==============================================================================


class RetargetBenchmark:
    """Compute retargeting quality metrics."""

    # ==========================================================================
    # ClassVar Constants
    # ==========================================================================

    TIP_INDICES: ClassVar[list[int]] = [4, 8, 12, 16, 20]
    """MediaPipe fingertip landmark indices."""

    DIP_INDICES: ClassVar[list[int]] = [3, 7, 11, 15, 19]
    """MediaPipe DIP landmark indices (for direction computation)."""

    FINGER_NAMES: ClassVar[list[str]] = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    """Human-readable finger names."""

    TIP_PAIRS: ClassVar[list[tuple[int, int]]] = [
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 3),
        (2, 4),
        (3, 4),
    ]
    """All 10 pairwise fingertip combinations for inter-fingertip distance."""

    # ==========================================================================
    # Dunder Methods
    # ==========================================================================

    def __init__(self, urdf_path: str, hand_side: str = "left") -> None:
        """Initialize benchmark with a robot URDF model.

        Args:
            urdf_path: Path to the hand URDF file.
            hand_side: "left" or "right" hand side prefix for frame names.
        """
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.nq = self.model.nq
        self.hand_side = hand_side

        # Cache frame ids for tips
        prefix = f"{hand_side}_"
        self._tip_frame_ids = []
        self._dip_frame_ids = []
        for i in range(1, 6):
            tip_name = f"{prefix}finger{i}_tip_link"
            # DIP: use link4 (closest to tip with a joint)
            dip_name = f"{prefix}finger{i}_link4"
            self._tip_frame_ids.append(self._find_frame(tip_name))
            self._dip_frame_ids.append(self._find_frame(dip_name))

        self._palm_frame_id = self._find_frame(f"{prefix}palm_link")

    # ==========================================================================
    # Public Methods
    # ==========================================================================

    def evaluate(self, source_landmarks: np.ndarray, qpos_seq: np.ndarray, fps: float = DEFAULT_FPS) -> dict:
        """Run all metrics.

        Args:
            source_landmarks: (T, 21, 3) preprocessed MediaPipe landmarks.
            qpos_seq: (T, nq) retargeted joint angles.
            fps: Data frame rate in Hz (for jerk computation).

        Returns:
            Dict with the following top-level keys:

            - ``tip_pos_error_mm``: Per-finger and overall fingertip position error (mm).
            - ``tip_dir_error_deg``: Per-finger and overall fingertip direction error (deg).
            - ``inter_tip_distance_error_mm``: Mean/max pairwise inter-fingertip distance error (mm).
            - ``smoothness``: Jerk (rad/s^3) and temporal consistency.
            - ``cspace_coverage``: Mean and per-joint configuration space utilization ratio.
            - ``joint_limit_violation_rate``: Fraction of frames with at least one limit violation.
            - ``pinch``: Pinch-specific metrics (frame counts, thumb-index error by phase, per-pair).
            - ``num_frames``: Total number of evaluated frames.
        """
        T = len(qpos_seq)
        if len(source_landmarks) != T:
            raise ValueError(f"source_landmarks length ({len(source_landmarks)}) != qpos_seq length ({T})")

        # Per-frame arrays
        tip_pos_errors = np.zeros((T, 5))  # mm
        tip_dir_errors = np.zeros((T, 5))  # degrees
        inter_tip_errors = np.zeros((T, 10))  # mm (10 pairs)
        joint_limit_violations = np.zeros(T)  # count per frame

        for t in range(T):
            tips_robot, dips_robot, palm_robot = self._fk(qpos_seq[t])
            lm = source_landmarks[t]

            # Source tips and DIPs
            tips_source = lm[self.TIP_INDICES]
            dips_source = lm[self.DIP_INDICES]

            # 1. Tip position error (mm)
            tip_pos_errors[t] = np.linalg.norm(tips_robot - tips_source, axis=1) * 1000

            # 2. Tip direction error (degrees)
            for f in range(5):
                dir_robot = tips_robot[f] - dips_robot[f]
                dir_source = tips_source[f] - dips_source[f]
                nr = np.linalg.norm(dir_robot)
                ns = np.linalg.norm(dir_source)
                if nr > DIRECTION_NORM_EPS and ns > DIRECTION_NORM_EPS:
                    cos_angle = np.clip(np.dot(dir_robot, dir_source) / (nr * ns), -1, 1)
                    tip_dir_errors[t, f] = np.degrees(np.arccos(cos_angle))

            # 3. Inter-fingertip distance error (mm)
            for p, (i, j) in enumerate(self.TIP_PAIRS):
                d_robot = np.linalg.norm(tips_robot[i] - tips_robot[j])
                d_source = np.linalg.norm(tips_source[i] - tips_source[j])
                inter_tip_errors[t, p] = abs(d_robot - d_source) * 1000

            # 4. Joint limit violations
            q = qpos_seq[t]
            violations = np.sum(q < self.model.lowerPositionLimit - JOINT_LIMIT_TOL) + np.sum(
                q > self.model.upperPositionLimit + JOINT_LIMIT_TOL
            )
            joint_limit_violations[t] = violations

        # 5. Smoothness: jerk (3rd derivative of joint angles)
        dt = 1.0 / fps
        if T >= 4:
            # 3rd finite difference
            d1 = np.diff(qpos_seq, axis=0) / dt
            d2 = np.diff(d1, axis=0) / dt
            d3 = np.diff(d2, axis=0) / dt
            jerk = np.mean(np.abs(d3))  # rad/s^3
        else:
            jerk = 0.0

        # 6. Temporal consistency (mean frame-to-frame delta)
        if T >= 2:
            dq = np.diff(qpos_seq, axis=0)
            temporal_consistency = np.mean(np.linalg.norm(dq, axis=1))
        else:
            temporal_consistency = 0.0

        # 7. C-space coverage per joint
        q_range_used = np.ptp(qpos_seq, axis=0)  # max - min per joint
        q_range_total = self.model.upperPositionLimit - self.model.lowerPositionLimit
        cspace_coverage = np.where(q_range_total > 1e-6, q_range_used / q_range_total, 0.0)

        # 8. Pinch-specific: thumb-to-each-finger distance preservation
        # TIP_PAIRS[0] = (0,1) = thumb-index, [1]=(0,2)=thumb-middle, etc.
        pinch_pair_names = ["Th-Ix", "Th-Md", "Th-Rg", "Th-Pk"]
        pinch_pairs_idx = [0, 1, 2, 3]  # first 4 of TIP_PAIRS are thumb-to-others

        # Classify frames: source thumb-index dist < 30mm = "near pinch"
        source_thumb_index_dists = (
            np.array([np.linalg.norm(source_landmarks[t, 4] - source_landmarks[t, 8]) for t in range(T)]) * 1000
        )  # mm
        pinch_mask = source_thumb_index_dists < PINCH_THRESHOLD_MM
        n_pinch = int(pinch_mask.sum())
        n_normal = T - n_pinch

        return {
            "tip_pos_error_mm": {
                "per_finger": {n: float(tip_pos_errors[:, i].mean()) for i, n in enumerate(self.FINGER_NAMES)},
                "mean": float(tip_pos_errors.mean()),
                "std": float(tip_pos_errors.std()),
                "max": float(tip_pos_errors.max()),
            },
            "tip_dir_error_deg": {
                "per_finger": {n: float(tip_dir_errors[:, i].mean()) for i, n in enumerate(self.FINGER_NAMES)},
                "mean": float(tip_dir_errors.mean()),
            },
            "inter_tip_distance_error_mm": {
                "mean": float(inter_tip_errors.mean()),
                "max": float(inter_tip_errors.max()),
            },
            "smoothness": {
                "jerk_rad_s3": float(jerk),
                "temporal_consistency": float(temporal_consistency),
            },
            "cspace_coverage": {
                "mean": float(cspace_coverage.mean()),
                "per_joint": cspace_coverage.tolist(),
            },
            "joint_limit_violation_rate": float(np.mean(joint_limit_violations > 0)),
            "pinch": {
                "n_pinch_frames": n_pinch,
                "n_normal_frames": n_normal,
                "thumb_index_dist_error_mm": {
                    "all": float(inter_tip_errors[:, 0].mean()),
                    "pinch": float(inter_tip_errors[pinch_mask, 0].mean()) if n_pinch > 0 else 0.0,
                    "normal": float(inter_tip_errors[~pinch_mask, 0].mean()) if n_normal > 0 else 0.0,
                },
                "per_pair": {
                    name: float(inter_tip_errors[:, idx].mean()) for name, idx in zip(pinch_pair_names, pinch_pairs_idx)
                },
            },
            "num_frames": T,
        }

    @staticmethod
    def print_results(results: dict, label: str = ""):
        """Pretty-print benchmark results."""
        if label:
            print(f"\n{'=' * 60}")
            print(f"  {label}")
            print(f"{'=' * 60}")

        tp = results["tip_pos_error_mm"]
        print("\nTip Position Error (mm):")
        print(f"  {'Finger':<10} {'Mean':>8}")
        print(f"  {'-' * 18}")
        for name, val in tp["per_finger"].items():
            print(f"  {name:<10} {val:>8.2f}")
        print(f"  {'Overall':<10} {tp['mean']:>8.2f}  (std={tp['std']:.2f}, max={tp['max']:.2f})")

        td = results["tip_dir_error_deg"]
        print("\nTip Direction Error (deg):")
        for name, val in td["per_finger"].items():
            print(f"  {name:<10} {val:>8.2f}")
        print(f"  {'Overall':<10} {td['mean']:>8.2f}")

        it = results["inter_tip_distance_error_mm"]
        print(f"\nInter-Fingertip Distance Error (mm): mean={it['mean']:.2f}, max={it['max']:.2f}")

        sm = results["smoothness"]
        print(f"\nSmoothness: jerk={sm['jerk_rad_s3']:.1f} rad/s^3, temporal={sm['temporal_consistency']:.5f}")

        cs = results["cspace_coverage"]
        print(f"\nC-Space Coverage: mean={cs['mean']:.3f}")

        jl = results["joint_limit_violation_rate"]
        print(f"\nJoint Limit Violation Rate: {jl * 100:.1f}%")

        pc = results["pinch"]
        print(f"\nPinch Analysis ({pc['n_pinch_frames']} pinch / {pc['n_normal_frames']} normal frames):")
        tid = pc["thumb_index_dist_error_mm"]
        print(
            f"  Thumb-Index dist error (mm): all={tid['all']:.2f}, pinch={tid['pinch']:.2f}, normal={tid['normal']:.2f}"
        )
        print("  Per-pair mean error (mm):")
        for pair_name, val in pc["per_pair"].items():
            print(f"    {pair_name:<8} {val:>8.2f}")

    # ==========================================================================
    # Private Methods
    # ==========================================================================

    def _find_frame(self, name: str) -> int:
        """Find the index of a named body frame in the Pinocchio model.

        Args:
            name: Frame name to search for.

        Returns:
            Frame index.

        Raises:
            ValueError: If no body frame with the given name exists.
        """
        for i in range(self.model.nframes):
            if self.model.frames[i].name == name and self.model.frames[i].type == pin.BODY:
                return i
        raise ValueError(f"Frame '{name}' not found")

    def _fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run forward kinematics and return key positions.

        Args:
            q: Joint configuration vector of shape (nq,).

        Returns:
            Tuple of (tips, dips, palm) where:
              - tips: (5, 3) fingertip positions in world frame.
              - dips: (5, 3) DIP joint positions in world frame.
              - palm: (3,) palm link position in world frame.
        """
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        tips = np.array([self.data.oMf[fid].translation.copy() for fid in self._tip_frame_ids])
        dips = np.array([self.data.oMf[fid].translation.copy() for fid in self._dip_frame_ids])
        palm = self.data.oMf[self._palm_frame_id].translation.copy()
        return tips, dips, palm
