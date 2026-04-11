"""
Configuration for interaction-mesh-based hand retargeting.
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


# MediaPipe hand landmark indices
# 0=WRIST, 1-4=THUMB(CMC,MCP,IP,TIP), 5-8=INDEX(MCP,PIP,DIP,TIP),
# 9-12=MIDDLE, 13-16=RING, 17-20=PINKY

# Mapping: MediaPipe index -> WujiHand MuJoCo body name
# 21 points: palm + 4 per finger (link1=MCP, link2=PIP, link3=DIP, link4=near-tip)
# Matches baseline AdaptiveOptimizerAnalytical's full link set:
#   palm_link, finger{1-5}_link1, link2(=link3 for base), link3, link4, tip_link
#
# MediaPipe hand landmarks:
#   0=WRIST, 1=THUMB_CMC, 2=THUMB_MCP, 3=THUMB_IP, 4=THUMB_TIP
#   5=INDEX_MCP, 6=INDEX_PIP, 7=INDEX_DIP, 8=INDEX_TIP
#   9=MIDDLE_MCP, 10=MIDDLE_PIP, 11=MIDDLE_DIP, 12=MIDDLE_TIP
#   13=RING_MCP, 14=RING_PIP, 15=RING_DIP, 16=RING_TIP
#   17=PINKY_MCP, 18=PINKY_PIP, 19=PINKY_DIP, 20=PINKY_TIP
JOINTS_MAPPING_LEFT = {
    0:  "left_palm_link",          # WRIST
    1:  "left_finger1_link1",      # THUMB_CMC  -> link1
    2:  "left_finger1_link3",      # THUMB_MCP  -> link3 (skip link2, thumb has 16mm gap so OK)
    3:  "left_finger1_link4",      # THUMB_IP   -> link4
    4:  "left_finger1_tip_link",   # THUMB_TIP  -> tip_link
    5:  "left_finger2_link1",      # INDEX_MCP  -> link1
    6:  "left_finger2_link3",      # INDEX_PIP  -> link3 (skip link2, only 4mm from link1)
    7:  "left_finger2_link4",      # INDEX_DIP  -> link4
    8:  "left_finger2_tip_link",   # INDEX_TIP  -> tip_link
    9:  "left_finger3_link1",      # MIDDLE_MCP -> link1
    10: "left_finger3_link3",      # MIDDLE_PIP -> link3
    11: "left_finger3_link4",      # MIDDLE_DIP -> link4
    12: "left_finger3_tip_link",   # MIDDLE_TIP -> tip_link
    13: "left_finger4_link1",      # RING_MCP   -> link1
    14: "left_finger4_link3",      # RING_PIP   -> link3
    15: "left_finger4_link4",      # RING_DIP   -> link4
    16: "left_finger4_tip_link",   # RING_TIP   -> tip_link
    17: "left_finger5_link1",      # PINKY_MCP  -> link1
    18: "left_finger5_link3",      # PINKY_PIP  -> link3
    19: "left_finger5_link4",      # PINKY_DIP  -> link4
    20: "left_finger5_tip_link",   # PINKY_TIP  -> tip_link
}

JOINTS_MAPPING_RIGHT = {
    0:  "right_palm_link",
    1:  "right_finger1_link1",
    2:  "right_finger1_link3",
    3:  "right_finger1_link4",
    4:  "right_finger1_tip_link",
    5:  "right_finger2_link1",
    6:  "right_finger2_link3",
    7:  "right_finger2_link4",
    8:  "right_finger2_tip_link",
    9:  "right_finger3_link1",
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

# Fingertip body names (for evaluation)
FINGERTIP_LINKS_LEFT = [
    "left_finger1_tip_link",
    "left_finger2_tip_link",
    "left_finger3_tip_link",
    "left_finger4_tip_link",
    "left_finger5_tip_link",
]

FINGERTIP_MP_INDICES = [4, 8, 12, 16, 20]


@dataclass
class HandRetargetConfig:
    """Configuration for interaction mesh hand retargeting."""

    # Model
    mjcf_path: str = ""  # URDF (fixed base, Pinocchio) or scene XML (floating, MuJoCo)
    hand_side: str = "left"
    floating_base: bool = False  # True: 6DOF wrist + 20 finger = 26 DOF (object mode)

    # Optimization
    step_size: float = 0.1        # Trust region radius (SOC constraint)
    smooth_weight: float = 0.2    # Temporal smoothness weight (matches OmniRetarget default)
    n_iter_first: int = 50        # SQP iterations for first frame
    n_iter: int = 10              # SQP iterations for subsequent frames
    activate_joint_limits: bool = True
    activate_self_collision: bool = False  # Phase 1: off

    # Object interaction
    object_sample_count: int = 100  # surface points sampled from object mesh

    # MediaPipe preprocessing
    # Global scale: auto-computed as robot_palm_to_middle_tip / source_palm_to_middle_tip
    # Set to None for auto-compute, or a float to override
    global_scale: float | None = None
    mediapipe_rotation: dict = field(default_factory=lambda: {
        "x": 0.0, "y": 0.0, "z": 15.0,
    })

    @property
    def joints_mapping(self) -> dict[int, str]:
        if self.hand_side == "left":
            return JOINTS_MAPPING_LEFT
        return JOINTS_MAPPING_RIGHT

    @property
    def fingertip_links(self) -> list[str]:
        side = self.hand_side
        return [f"{side}_finger{i}_tip_link" for i in range(1, 6)]

    @classmethod
    def from_yaml(cls, yaml_path: str, mjcf_path: str = "") -> "HandRetargetConfig":
        """Load config from YAML file."""
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        cfg = cls(mjcf_path=mjcf_path)

        opt = data.get("optimization", {})
        for key in ("step_size", "smooth_weight", "n_iter_first", "n_iter",
                     "activate_joint_limits", "activate_self_collision"):
            if key in opt:
                setattr(cfg, key, opt[key])

        if "floating_base" in opt:
            cfg.floating_base = opt["floating_base"]
        if "object_sample_count" in opt:
            cfg.object_sample_count = opt["object_sample_count"]

        retarget = data.get("retarget", {})
        if "global_scale" in retarget:
            cfg.global_scale = retarget["global_scale"]
        if "mediapipe_rotation" in retarget:
            cfg.mediapipe_rotation = retarget["mediapipe_rotation"]
        if "hand_side" in data:
            cfg.hand_side = data["hand_side"]

        return cfg
