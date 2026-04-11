"""
MediaPipe data loading and preprocessing for hand retargeting.
Handles .pkl files in the same format as wuji_retargeting baseline.

Uses the baseline's apply_mediapipe_transformations to ensure identical
coordinate frame alignment (SVD frame estimation + OPERATOR2MANO rotation).
"""

import pickle
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from wuji_retargeting.mediapipe import apply_mediapipe_transformations


# apply_segment_scaling removed: interaction mesh uses global scale only (like OmniRetarget's smpl_scale)
# Per-segment scaling is a baseline-specific workaround for direct IK position matching


def load_pkl_sequence(pkl_path: str, hand_side: str = "left") -> tuple[np.ndarray, np.ndarray]:
    """
    Load a .pkl trajectory file and return landmarks array.

    Args:
        pkl_path: path to .pkl file
        hand_side: "left" or "right"

    Returns:
        landmarks: (T, 21, 3) array of MediaPipe hand landmarks
        timestamps: (T,) array of timestamps
    """
    with open(pkl_path, "rb") as f:
        recording = pickle.load(f)

    key = f"{hand_side}_fingers"
    landmarks = []
    timestamps = []

    for frame in recording:
        lm = frame[key]
        if not np.allclose(lm, 0):
            landmarks.append(lm)
            timestamps.append(frame["t"])

    return np.array(landmarks, dtype=np.float64), np.array(timestamps, dtype=np.float64)


def preprocess_landmarks(
    landmarks: np.ndarray,
    mediapipe_rotation: dict,
    hand_side: str = "left",
    global_scale: float = 1.0,
) -> np.ndarray:
    """
    Full preprocessing pipeline for a single frame.
    Matches OmniRetarget robot_only: coordinate transform + global scale only.

    Pipeline:
        1. apply_mediapipe_transformations (center + SVD frame + OPERATOR2MANO)
        2. apply additional rotation from config (e.g. 15 deg Z for manus data)
        3. apply global uniform scale (like OmniRetarget's smpl_scale)

    Args:
        landmarks: (21, 3) raw MediaPipe landmarks
        mediapipe_rotation: rotation dict {'x': ..., 'y': ..., 'z': ...} in degrees
        hand_side: "left" or "right"
        global_scale: uniform scale factor (robot_size / human_size)

    Returns:
        (21, 3) preprocessed landmarks in robot hand frame
    """
    # Step 1: baseline's full transform (center + SVD + OPERATOR2MANO)
    lm = apply_mediapipe_transformations(landmarks.copy(), hand_side)

    # Step 2: additional rotation from config (manus-specific adjustment)
    angles = [mediapipe_rotation.get("x", 0), mediapipe_rotation.get("y", 0), mediapipe_rotation.get("z", 0)]
    if any(a != 0 for a in angles):
        R = Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
        lm = lm @ R.T

    # Step 3: global uniform scale (matches OmniRetarget's smpl_scale)
    # No per-segment manual calibration — Laplacian + FK handles proportion differences
    if global_scale != 1.0:
        lm *= global_scale

    return lm


def preprocess_sequence(
    landmarks_seq: np.ndarray,
    mediapipe_rotation: dict,
    hand_side: str = "left",
    global_scale: float = 1.0,
) -> np.ndarray:
    """
    Preprocess a full sequence of MediaPipe landmarks.

    Args:
        landmarks_seq: (T, 21, 3) raw landmarks
        mediapipe_rotation: rotation config
        hand_side: "left" or "right"
        global_scale: uniform scale factor

    Returns:
        (T, 21, 3) preprocessed landmarks
    """
    result = np.zeros_like(landmarks_seq)
    for t in range(len(landmarks_seq)):
        result[t] = preprocess_landmarks(
            landmarks_seq[t], mediapipe_rotation, hand_side, global_scale
        )
    return result
