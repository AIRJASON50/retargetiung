"""Visualization helpers (MuJoCo overlays, playback controller, qpos caching).

Separate from ``hand_retarget`` core (which has no mujoco viewer dependency)
so library users retargeting in a headless pipeline do not pull in viewer code.
"""
