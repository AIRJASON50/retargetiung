"""Playback controller for precomputed trajectory replay in MuJoCo viewer.

Encapsulates the pause / step / reverse / timing-advance state machine
that is duplicated across all demo scripts.  Works in two flavours:

- **Precomputed mode** (default): random-access into a cached qpos
  array with pause, single-step, and reverse playback.
- **Live mode**: frame-by-frame advance without rewind (used when
  retargeting is computed on-the-fly).

Custom per-demo key bindings (e.g. visibility toggles) are supported
via an optional *custom_key_handler* callback.
"""
from __future__ import annotations

import time
from collections.abc import Callable

from hand_retarget_viz.overlay import KEY_LEFT, KEY_RIGHT, KEY_SPACE


class PlaybackController:
    """Timing-based playback state machine for MuJoCo viewer demos.

    Args:
        total_frames: Number of frames in the trajectory.
        avg_dt: Average inter-frame interval in seconds.
        speed: Playback speed multiplier (1.0 = realtime).
        loop: If True, wrap around at trajectory boundaries;
            otherwise clamp and auto-pause.
        custom_key_handler: Optional callback ``(keycode: int) -> bool``.
            Called **before** the built-in key handling.  Return True to
            indicate the key was consumed (skip built-in logic).
    """

    def __init__(
        self,
        total_frames: int,
        avg_dt: float,
        speed: float = 1.0,
        loop: bool = True,
        custom_key_handler: Callable[[int], bool] | None = None,
    ) -> None:
        self.total_frames = total_frames
        self.avg_dt = avg_dt
        self.speed = speed
        self.loop = loop
        self.custom_key_handler = custom_key_handler

        self.paused: bool = False
        self.direction: int = 1
        self.frame_idx: int = 0

        self._step_request: int = 0
        self._resume_flag: bool = False
        self._last_frame_time: float = time.time()

    # ----------------------------------------------------------
    # Viewer key callback
    # ----------------------------------------------------------

    def key_callback(self, keycode: int) -> None:
        """Pass this as ``key_callback`` to ``mujoco.viewer.launch_passive``.

        Args:
            keycode: GLFW key code received from the viewer.
        """
        # Let the caller handle custom keys first.
        if self.custom_key_handler is not None and self.custom_key_handler(keycode):
            return

        if keycode == KEY_SPACE:
            was_paused = self.paused
            self.paused = not was_paused
            if was_paused:
                self._resume_flag = True
            status = "PAUSED" if self.paused else "PLAYING"
            print(f"  [{status}] frame {self.frame_idx}/{self.total_frames}")

        elif keycode == KEY_LEFT:
            if self.paused:
                self._step_request = -1
            else:
                self.direction *= -1
                dir_str = ">>>" if self.direction == 1 else "<<<"
                print(f"  [{dir_str}]")

        elif keycode == KEY_RIGHT:
            if self.paused:
                self._step_request = 1
            else:
                if self.direction != 1:
                    self.direction = 1
                    print("  [>>>]")

    # ----------------------------------------------------------
    # Frame advance
    # ----------------------------------------------------------

    def advance(self) -> tuple[int, bool]:
        """Compute the current frame index and whether to update the scene.

        Call this once per viewer loop iteration.

        Returns:
            A ``(frame_idx, need_update)`` tuple.  When *need_update* is
            False the caller should ``viewer.sync()`` and ``continue``
            without re-rendering.
        """
        now = time.time()
        idx = self.frame_idx
        need_update = False

        if self.paused:
            step = self._step_request
            if step != 0:
                idx += step
                idx = max(0, min(idx, self.total_frames - 1))
                self._step_request = 0
                self.frame_idx = idx
                need_update = True
            else:
                time.sleep(0.01)
        else:
            if self._resume_flag:
                self._last_frame_time = now
                self._resume_flag = False
            dt = now - self._last_frame_time
            frames_to_advance = int(dt / (self.avg_dt / self.speed))
            if frames_to_advance >= 1:
                idx += self.direction * frames_to_advance
                self._last_frame_time = now

                if self.loop:
                    idx = idx % self.total_frames
                else:
                    if idx >= self.total_frames:
                        idx = self.total_frames - 1
                        self.paused = True
                    elif idx < 0:
                        idx = 0
                        self.paused = True

                self.frame_idx = idx
                need_update = True
            else:
                time.sleep(0.001)

        return self.frame_idx, need_update
