"""NPZ cache helper for precomputed retargeting trajectories.

Provides a single ``load_or_compute`` function that transparently
loads from a ``.npz`` file when the cached frame count matches, or
falls back to running a user-supplied *compute_fn* and saving the
result.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np


def load_or_compute(
    cache_path: str | Path,
    total_frames: int,
    compute_fn: Callable[[], np.ndarray],
    force_recompute: bool = False,
) -> np.ndarray:
    """Load qpos cache from disk or compute and save it.

    The cache file is a ``.npz`` archive with a single ``"qpos"`` key
    containing an ``(T, nq)`` float64 array.

    Args:
        cache_path: File path for the ``.npz`` cache.
        total_frames: Expected number of frames.  A cache with a
            different frame count is treated as stale.
        compute_fn: Zero-argument callable that returns an
            ``(total_frames, nq)`` ndarray when the cache misses.
        force_recompute: Skip cache loading and always recompute.

    Returns:
        The ``(total_frames, nq)`` qpos array (loaded or freshly
        computed).
    """
    cache_path = Path(cache_path)

    if not force_recompute and cache_path.exists():
        cached = np.load(cache_path)
        qpos = cached["qpos"]
        if len(qpos) == total_frames:
            print(f"Loaded cache: {cache_path} ({total_frames} frames)")
            return qpos
        print(f"Cache frame count mismatch ({len(qpos)} vs {total_frames}), re-computing...")

    qpos = compute_fn()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, qpos=qpos)
    print(f"Saved cache: {cache_path}")
    return qpos
