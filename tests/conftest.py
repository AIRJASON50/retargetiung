"""pytest conftest: add WUJI_SDK_PATH to sys.path for all tests.

Pytest imports this automatically before collecting tests.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_DEFAULT_WUJI_SDK = "/home/l/ws/doc/WujiRepo/wuji_retargeting_private/public"
_WUJI_SDK = os.environ.get("WUJI_SDK_PATH", _DEFAULT_WUJI_SDK)

if not Path(_WUJI_SDK).exists():
    raise RuntimeError(
        f"wuji_retargeting SDK not found at {_WUJI_SDK!r}. "
        "Set WUJI_SDK_PATH env var to the SDK root."
    )

if _WUJI_SDK not in sys.path:
    sys.path.insert(0, _WUJI_SDK)
