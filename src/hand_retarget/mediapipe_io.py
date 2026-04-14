"""
MediaPipe data loading and preprocessing for hand retargeting.
Handles .pkl files in the same format as wuji_retargeting baseline.

Uses the baseline's apply_mediapipe_transformations to ensure identical
coordinate frame alignment (SVD frame estimation + OPERATOR2MANO rotation).
"""
from __future__ import annotations

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
    use_mano_rotation: bool = True,
) -> np.ndarray:
    """
    Full preprocessing pipeline for a single frame.

    Pipeline:
        1a. If use_mano_rotation: apply_mediapipe_transformations (center + SVD + OPERATOR2MANO)
        1b. Else: center to wrist only (no rotation)
        2. apply additional rotation from config (e.g. 15 deg Z for manus data)
        3. apply global uniform scale

    Args:
        landmarks: (21, 3) raw MediaPipe landmarks
        mediapipe_rotation: rotation dict {'x': ..., 'y': ..., 'z': ...} in degrees
        hand_side: "left" or "right"
        global_scale: uniform scale factor (robot_size / human_size)
        use_mano_rotation: if True, apply SVD + OPERATOR2MANO; if False, only center to wrist

    Returns:
        (21, 3) preprocessed landmarks
    """
    if use_mano_rotation:
        # Full transform: center + SVD frame + OPERATOR2MANO
        lm = apply_mediapipe_transformations(landmarks.copy(), hand_side)
    else:
        # Simple center to wrist, no rotation
        lm = landmarks.copy() - landmarks[0:1, :]

    # Step 2: additional rotation from config
    angles = [mediapipe_rotation.get("x", 0), mediapipe_rotation.get("y", 0), mediapipe_rotation.get("z", 0)]
    if any(a != 0 for a in angles):
        R = Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
        lm = lm @ R.T

    # Step 3: global uniform scale
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


# --- HO-Cap data loading ---

def sample_object_surface(mesh_path: str, count: int = 100, seed: int = 0) -> np.ndarray:
    """
    Sample points from an object mesh surface.

    Args:
        mesh_path: path to STL/OBJ mesh file
        count: number of points to sample
        seed: random seed for reproducibility

    Returns:
        (count, 3) array of surface points in mesh local frame
    """
    import trimesh
    mesh = trimesh.load(str(mesh_path))
    pts, _ = trimesh.sample.sample_surface(mesh, count, seed=seed)
    return pts.astype(np.float64)


def transform_object_points(pts_local: np.ndarray, quat: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """
    Transform object points from local frame to world frame.

    Args:
        pts_local: (M, 3) points in object local frame
        quat: (4,) quaternion in xyzw format (scipy convention, matches HO-Cap)
        trans: (3,) translation

    Returns:
        (M, 3) points in world frame
    """
    from scipy.spatial.transform import Rotation as R
    rot = R.from_quat(quat).as_matrix()
    return (rot @ pts_local.T).T + trans


def load_hocap_clip(npz_path: str, meta_path: str, assets_dir: str,
                    hand_side: str = "left",
                    sample_count: int = 100) -> dict:
    """
    Load a HO-Cap clip with hand landmarks + object data.

    Args:
        npz_path: path to motion .npz file
        meta_path: path to .meta.json file
        assets_dir: path to assets/ directory containing object meshes
        hand_side: "left" or "right"
        sample_count: number of surface points to sample from object mesh

    Returns:
        dict with keys:
          landmarks: (T, 21, 3) MediaPipe hand landmarks (world frame)
          object_pts_local: (M, 3) object surface samples (mesh local frame)
          object_t: (T, 3) object translation per frame
          object_q: (T, 4) object quaternion per frame
          fps: float
          asset_name: str
          mesh_path: str
    """
    import json
    from pathlib import Path

    # Load motion data
    data = np.load(npz_path, allow_pickle=True)

    # Hand landmarks
    mp_key = f"mediapipe_{hand_side[0]}_world"
    if mp_key not in data or data[mp_key].dtype == object:
        raise ValueError(f"No {hand_side} hand data in {npz_path}")
    landmarks = data[mp_key].reshape(-1, 21, 3).astype(np.float64)

    # Object pose
    object_t = data["object_t"][:, 0, :].astype(np.float64)  # (T, 3)
    object_q = data["object_q"][:, 0, :].astype(np.float64)  # (T, 4)
    fps = float(data["fps"])

    # Wrist pose (for direct assignment if needed)
    wrist_t_key = f"wrist_t_{hand_side[0]}"
    wrist_q_key = f"wrist_q_{hand_side[0]}"
    wrist_t = data[wrist_t_key].astype(np.float64) if wrist_t_key in data and data[wrist_t_key].dtype != object else None
    wrist_q = data[wrist_q_key].astype(np.float64) if wrist_q_key in data and data[wrist_q_key].dtype != object else None

    # Load meta to get asset name
    with open(meta_path) as f:
        meta = json.load(f)
    asset_name = meta["objects"][0]["asset_name"]

    # Sample object surface
    mesh_path = str(Path(assets_dir) / asset_name / "mesh_med.stl")
    object_pts_local = sample_object_surface(mesh_path, count=sample_count)

    return {
        "landmarks": landmarks,
        "object_pts_local": object_pts_local,
        "object_t": object_t,
        "object_q": object_q,
        "wrist_t": wrist_t,
        "wrist_q": wrist_q,
        "fps": fps,
        "asset_name": asset_name,
        "mesh_path": mesh_path,
    }
