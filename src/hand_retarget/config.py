"""
Configuration for interaction-mesh-based hand retargeting.
"""

from __future__ import annotations

# ==============================================================================
# Imports
# ==============================================================================
import dataclasses
import hashlib
import json
from dataclasses import dataclass, field

import yaml

# ==============================================================================
# Constants
# ==============================================================================

JOINTS_MAPPING_LEFT = {
    0: "left_palm_link",  # WRIST
    1: "left_finger1_link1",  # THUMB_CMC  -> link1
    2: "left_finger1_link3",  # THUMB_MCP  -> link3 (skip link2, thumb has 16mm gap so OK)
    3: "left_finger1_link4",  # THUMB_IP   -> link4
    4: "left_finger1_tip_link",  # THUMB_TIP  -> tip_link
    5: "left_finger2_link1",  # INDEX_MCP  -> link1
    6: "left_finger2_link3",  # INDEX_PIP  -> link3 (skip link2, only 4mm from link1)
    7: "left_finger2_link4",  # INDEX_DIP  -> link4
    8: "left_finger2_tip_link",  # INDEX_TIP  -> tip_link
    9: "left_finger3_link1",  # MIDDLE_MCP -> link1
    10: "left_finger3_link3",  # MIDDLE_PIP -> link3
    11: "left_finger3_link4",  # MIDDLE_DIP -> link4
    12: "left_finger3_tip_link",  # MIDDLE_TIP -> tip_link
    13: "left_finger4_link1",  # RING_MCP   -> link1
    14: "left_finger4_link3",  # RING_PIP   -> link3
    15: "left_finger4_link4",  # RING_DIP   -> link4
    16: "left_finger4_tip_link",  # RING_TIP   -> tip_link
    17: "left_finger5_link1",  # PINKY_MCP  -> link1
    18: "left_finger5_link3",  # PINKY_PIP  -> link3
    19: "left_finger5_link4",  # PINKY_DIP  -> link4
    20: "left_finger5_tip_link",  # PINKY_TIP  -> tip_link
}
"""MediaPipe index -> WujiHand MuJoCo body name (left hand).

21 points: palm + 4 per finger (link1=MCP, link2=PIP, link3=DIP, link4=near-tip).
Matches baseline AdaptiveOptimizerAnalytical's full link set:
  palm_link, finger{1-5}_link1, link2(=link3 for base), link3, link4, tip_link.
"""

JOINTS_MAPPING_RIGHT = {
    0: "right_palm_link",
    1: "right_finger1_link1",
    2: "right_finger1_link3",
    3: "right_finger1_link4",
    4: "right_finger1_tip_link",
    5: "right_finger2_link1",
    6: "right_finger2_link3",
    7: "right_finger2_link4",
    8: "right_finger2_tip_link",
    9: "right_finger3_link1",
    10: "right_finger3_link3",
    11: "right_finger3_link4",
    12: "right_finger3_tip_link",
    13: "right_finger4_link1",
    14: "right_finger4_link3",
    15: "right_finger4_link4",
    16: "right_finger4_tip_link",
    17: "right_finger5_link1",
    18: "right_finger5_link3",
    19: "right_finger5_link4",
    20: "right_finger5_tip_link",
}
"""MediaPipe index -> WujiHand MuJoCo body name (right hand)."""

# Link-midpoint keypoint definition: 20 points (4 per finger, wrist excluded).
# Each entry: (mp_parent, mp_child) for midpoint, or (mp_tip, None) for raw TIP.
MIDPOINT_SEGMENTS: list[tuple[int, int | None]] = [
    # Thumb: mid(CMC,MCP), mid(MCP,IP), mid(IP,TIP), TIP
    (1, 2),
    (2, 3),
    (3, 4),
    (4, None),
    # Index: mid(MCP,PIP), mid(PIP,DIP), mid(DIP,TIP), TIP
    (5, 6),
    (6, 7),
    (7, 8),
    (8, None),
    # Middle
    (9, 10),
    (10, 11),
    (11, 12),
    (12, None),
    # Ring
    (13, 14),
    (14, 15),
    (15, 16),
    (16, None),
    # Pinky
    (17, 18),
    (18, 19),
    (19, 20),
    (20, None),
]
"""20 link-midpoint segments: 4 per finger, no wrist."""


def _build_midpoint_body_pairs(joints_mapping: dict[int, str]) -> list[tuple[str, str | None]]:
    """Build robot body name pairs for link midpoints from joints mapping.

    Args:
        joints_mapping: MediaPipe index -> robot body name mapping.

    Returns:
        List of (parent_body, child_body) pairs; child_body is None for raw TIP.
    """
    pairs = []
    for parent_mp, child_mp in MIDPOINT_SEGMENTS:
        parent_body = joints_mapping[parent_mp]
        child_body = joints_mapping[child_mp] if child_mp is not None else None
        pairs.append((parent_body, child_body))
    return pairs


# YAML section -> field name mapping for automatic deserialization.
# Each entry maps (yaml_section, yaml_key) -> dataclass_field_name.
_YAML_FIELD_MAP: dict[tuple[str, str], str] = {
    ("optimization", "step_size"): "step_size",
    ("optimization", "smooth_weight"): "smooth_weight",
    ("optimization", "n_iter_first"): "n_iter_first",
    ("optimization", "n_iter"): "n_iter",
    ("optimization", "activate_joint_limits"): "activate_joint_limits",
    ("optimization", "floating_base"): "floating_base",
    ("optimization", "object_sample_count"): "object_sample_count",
    ("retarget", "global_scale"): "global_scale",
    ("retarget", "mediapipe_rotation"): "mediapipe_rotation",
    ("", "hand_side"): "hand_side",
    ("angle_warmup", "weight"): "angle_warmup_weight",
    ("angle_warmup", "iters"): "angle_warmup_iters",
    ("angle_warmup", "anchor_weight"): "angle_anchor_weight",
}
"""Maps ``(yaml_section, yaml_key)`` to ``HandRetargetConfig`` field name."""

_YAML_ENABLED_MAP: dict[str, str] = {
    "link_midpoints": "use_link_midpoints",
    "angle_warmup": "use_angle_warmup",
}
"""Maps YAML section name to the boolean field toggled by its ``enabled`` key."""


# ==============================================================================
# Classes
# ==============================================================================


@dataclass
class HandRetargetConfig:
    """Configuration for interaction mesh hand retargeting."""

    # Model
    mjcf_path: str = ""  # URDF (fixed base, Pinocchio) or scene XML (floating, MuJoCo)
    hand_side: str = "left"
    floating_base: bool = False  # True: 6DOF wrist + 20 finger = 26 DOF (object mode)

    # Optimization
    step_size: float = 0.1  # Trust region radius (box constraint)
    n_iter_first: int = 50  # SQP iterations for first frame
    n_iter: int = 10  # SQP iterations for subsequent frames
    activate_joint_limits: bool = True
    activate_non_penetration: bool = False  # fingertip-object non-penetration (linearized SDF)

    # Object interaction
    object_sample_count: int = 100  # surface points sampled from object mesh

    # Delaunay edge threshold: filter long-range edges before Laplacian computation.
    # Removes cross-finger connections that pollute neighborhood averages.
    # Default 0.06m (60mm) keeps bone edges and close cross-finger edges only.
    # Set to None to use all Delaunay edges (original behavior).
    delaunay_edge_threshold: float | None = 0.06

    # Laplacian distance-decay weight: w_ij = exp(-k * ||e_ij||), normalized per vertex.
    # Shorter edges receive higher weight → local bone structure dominates.
    # k=20.0 means an edge at 50mm gets weight ~0.37 relative to a 0mm edge.
    # Set to None to use uniform weights (original behavior).
    laplacian_distance_weight_k: float | None = 20.0

    # Skeleton topology: use hand bone structure instead of Delaunay
    use_skeleton_topology: bool = False

    # Link-midpoint IM: use 20 link midpoints instead of 21 joint origins
    use_link_midpoints: bool = False

    # Two-stage: S1 cosine IK bone direction alignment + S2 Laplacian position refinement
    # Joint cost: anchor_weight * angle_error + laplacian_weight * lap_error + smooth_weight * smooth
    # Default 5:5:1 ratio (anchor=5, laplacian via retargeter.laplacian_weight, smooth=1)
    use_angle_warmup: bool = True
    angle_warmup_weight: float = 5.0
    angle_warmup_iters: int = 3
    angle_anchor_weight: float = 5.0  # S1 angle anchor in joint cost
    smooth_weight: float = 1.0  # Temporal smoothness (5:5:1 ratio with anchor and laplacian)
    exclude_fingers_from_laplacian: list = None  # Finger indices (0-4) excluded from Laplacian gradient

    # Object-frame Laplacian: compute Laplacian in object local coordinates
    use_object_frame: bool = False

    # MediaPipe preprocessing
    global_scale: float = 1.0
    use_mano_rotation: bool = True  # True: SVD+OPERATOR2MANO (manus data), False: wrist-center only (HO-Cap)
    mediapipe_rotation: dict = field(
        default_factory=lambda: {
            "x": 0.0,
            "y": 0.0,
            "z": 15.0,
        }
    )

    @property
    def joints_mapping(self) -> dict[int, str]:
        """MediaPipe index -> robot body name mapping."""
        return JOINTS_MAPPING_LEFT if self.hand_side == "left" else JOINTS_MAPPING_RIGHT

    @classmethod
    def from_yaml(cls, yaml_path: str, mjcf_path: str = "") -> HandRetargetConfig:
        """Load config from YAML file.

        Reads a nested YAML structure and maps values to dataclass fields
        via the module-level ``_YAML_FIELD_MAP``. Adding a new config field
        only requires adding one entry to the mapping table.

        Args:
            yaml_path: Path to the YAML config file.
            mjcf_path: Override for the ``mjcf_path`` field.

        Returns:
            Populated config instance.
        """
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        cfg = cls(mjcf_path=mjcf_path)
        valid_fields = {f.name for f in dataclasses.fields(cls)}

        # Auto-map (section, key) -> field
        for (section, key), field_name in _YAML_FIELD_MAP.items():
            if field_name not in valid_fields:
                continue
            source = data if section == "" else data.get(section, {})
            if key in source:
                setattr(cfg, field_name, source[key])

        # Handle "enabled" toggles
        for section, field_name in _YAML_ENABLED_MAP.items():
            sec_data = data.get(section, {})
            if sec_data.get("enabled", False):
                setattr(cfg, field_name, True)

        return cfg

    def make_stamp(self) -> str:
        """Return a short, deterministic identifier for this config's algorithm variant.

        Human-readable prefix encodes the most-varied hyperparameters; a 6-char MD5
        hash covers everything else so two configs that differ only in unlisted fields
        still get distinct stamps.

        Example: ``thr60_k20_s50_a3f2c8``
        """
        parts: list[str] = []

        thr = self.delaunay_edge_threshold
        parts.append(f"thr{int(thr * 1000)}" if thr is not None else "thrX")

        k = self.laplacian_distance_weight_k
        parts.append(f"k{int(k)}" if k is not None else "kX")

        parts.append(f"s{self.object_sample_count}")

        if self.use_skeleton_topology:
            parts.append("skel")
        if self.use_link_midpoints:
            parts.append("midpt")
        if self.use_angle_warmup:
            parts.append("aw")
            parts.append("er")
        if not self.activate_joint_limits:
            parts.append("nojl")

        # Hash covers full config (minus mjcf_path which is environment-specific)
        d = dataclasses.asdict(self)
        d.pop("mjcf_path", None)
        h = hashlib.md5(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:6]
        parts.append(h)

        return "_".join(parts)
