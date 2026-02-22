"""Shared constants for yaw/quaternion column detection.

This module exists to avoid circular imports:
- `csu_core` contains file-selection heuristics that need these patterns
- `csu_features` contains yaw feature extraction that also needs them

Keeping the regexes and candidate lists here makes both modules importable
without relying on star-imports.
"""

from __future__ import annotations

import re
from typing import Dict, List

# Prefer explicit head/HMD yaw first.
_YAW_CANDIDATES: List[str] = [
    "HMDYaw",
    "HMDYawDeg",
    "HMDEulerYaw",
    "HMD_EulerYaw",
    "HeadYaw",
    "HeadOrientationYaw",
    "HMDRotationYaw",
    "RotationYaw",
    "EulerYaw",
    "YawDeg",
    "yaw_deg",
    "yaw",
    "hmd_yaw",
    "yaw_hmd",
    # Last resort: Unity logs often include a generic 'Yaw' that is NOT head yaw (e.g., vehicle yaw).
    "Yaw",
]

# Quaternion column fallbacks (scalar-first [w, x, y, z]) used when no yaw column exists.
# NOTE: use flags=IGNORECASE (instead of inline (?i)) to avoid DeprecationWarning.
_QUAT_REGEX: Dict[str, re.Pattern] = {
    "w": re.compile(r"(^|[_\s])(w|qw)$|(quat|quaternion|rotation|orientation).*w$", flags=re.IGNORECASE),
    "x": re.compile(r"(^|[_\s])(x|qx)$|(quat|quaternion|rotation|orientation).*x$", flags=re.IGNORECASE),
    "y": re.compile(r"(^|[_\s])(y|qy)$|(quat|quaternion|rotation|orientation).*y$", flags=re.IGNORECASE),
    "z": re.compile(r"(^|[_\s])(z|qz)$|(quat|quaternion|rotation|orientation).*z$", flags=re.IGNORECASE),
}

# Column name pattern suggesting a quaternion or rotation field.
_QUAT_LIST_COL_PAT: re.Pattern = re.compile(r"(?i)(quat|quaternion|rotation|orientation)")

__all__ = ["_YAW_CANDIDATES", "_QUAT_REGEX", "_QUAT_LIST_COL_PAT"]
