"""Read sequence mappings and locate participant responses and trial recordings."""

from __future__ import annotations

from typing import Any
from pathlib import Path
import numpy as np
import pandas as pd
import re
from utils.constants import (
    PARTICIPANT_PATTERN,
    VIDEO_PATTERN,
)
from utils.features import (
    _passage_time_seconds,
)


def _normalise_video_id(value: Any) -> str | None:
    match = VIDEO_PATTERN.search(str(value))
    return match.group(0).lower() if match else None


def _coerce_binary(series: pd.Series, name: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    unique = set(values.dropna().astype(int).unique())
    if not unique.issubset({0, 1}):
        raise ValueError(f"{name} must contain only 0 and 1; found {sorted(unique)}")
    return values.astype("Int64")


def load_mapping(path: Path) -> pd.DataFrame:
    """Read and validate the 40 cell factorial trial mapping."""

    if not path.is_file():
        raise FileNotFoundError(f"Mapping file not found: {path}")
    mapping = pd.read_csv(path)
    aliases = {
        "video": "video_id",
        "videoID": "video_id",
        "videoId": "video_id",
        "ehmi": "eHMIOn",
        "eHMI": "eHMIOn",
        "distance": "distPed",
        "order": "camera",
    }
    mapping = mapping.rename(columns={key: value for key, value in aliases.items() if key in mapping})
    required = {"video_id", "yielding", "eHMIOn", "camera", "distPed"}
    missing = sorted(required.difference(mapping.columns))
    if missing:
        raise ValueError(f"Mapping is missing required columns: {', '.join(missing)}")

    mapping["video_id"] = mapping["video_id"].map(_normalise_video_id)
    mapping = mapping[mapping["video_id"].notna()].copy()
    mapping = mapping.drop_duplicates("video_id", keep="first")
    mapping["yielding"] = _coerce_binary(mapping["yielding"], "yielding")
    mapping["eHMIOn"] = _coerce_binary(mapping["eHMIOn"], "eHMIOn")
    mapping["camera"] = _coerce_binary(mapping["camera"], "camera")
    mapping["distPed"] = pd.to_numeric(mapping["distPed"], errors="raise").astype(int)

    distance_codes = sorted(mapping["distPed"].dropna().unique())
    distance_map = {code: 2.0 * float(code) for code in distance_codes}
    mapping["distPed_m"] = mapping["distPed"].map(distance_map)

    if len(mapping) != 40:
        raise ValueError(f"Expected 40 experimental videos in mapping, found {len(mapping)}")
    factorial_counts = (
        mapping.groupby(["yielding", "eHMIOn", "camera", "distPed_m"], observed=True)
        .size()
        .reset_index(name="n")
    )
    if len(factorial_counts) != 40 or not (factorial_counts["n"] == 1).all():
        raise ValueError("The trial mapping is not a complete 2 x 2 x 2 x 5 factorial design")
    return mapping.reset_index(drop=True)


def _modal_value(values: pd.Series) -> float:
    """Most frequent value; ties resolve to the smallest value."""
    counts = values.round(2).value_counts()
    top = counts[counts == counts.max()].index
    return float(min(top))


def apply_constant_passage_times(mapping: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``mapping`` with constant passage times.

    The AV followed the same scripted trajectory in every trial of a given
    vehicle behaviour, so passage times are constant across trials. The logged
    per-condition times scatter by one to four 20-ms physics steps, so each is
    replaced by the modal logged value: per vehicle behaviour for the first
    roadside position (P2) and per vehicle behaviour and inter-pedestrian
    distance for the second position (P1), which lies d metres downstream.
    The same schedule is used in the multiped analysis
    (``human_analysis/utils/vehicle_events.py``).
    """

    result = mapping.copy()
    groupings = {
        "cross_p2_time_s": ["yielding"],
        "cross_p1_time_s": ["yielding", "distPed"],
    }
    for column, keys in groupings.items():
        if column not in result.columns:
            continue
        result[column] = pd.to_numeric(result[column], errors="coerce")
        for _, index in result.groupby(keys).groups.items():
            values = result.loc[index, column].dropna()
            if not values.empty:
                result.loc[index, column] = _modal_value(values)
    return result


def attach_passage_timing(
    condition_mapping: pd.DataFrame,
    timing_mapping: pd.DataFrame,
) -> pd.DataFrame:
    """Add crossing timestamps without replacing participant condition assignments."""

    timing_columns = [
        column
        for column in timing_mapping.columns
        if "cross" in str(column).lower() or str(column) == "video_length"
    ]
    if not timing_columns:
        raise ValueError(
            "The common timing mapping does not contain crossing timestamp columns"
        )

    result = condition_mapping.copy()
    timing_index = timing_mapping.set_index("video_id", drop=False)
    missing_video_ids = sorted(
        set(result["video_id"].astype(str)).difference(timing_index.index.astype(str))
    )
    if missing_video_ids:
        raise ValueError(
            "The common timing mapping is missing video IDs: "
            + ", ".join(missing_video_ids)
        )

    for column in timing_columns:
        timing_values = result["video_id"].map(timing_index[column])
        if column in result.columns:
            result[column] = result[column].combine_first(timing_values)
        else:
            result[column] = timing_values

    missing_passage: list[str] = []
    for _, row in result.iterrows():
        try:
            _passage_time_seconds(row)
        except (TypeError, ValueError):
            missing_passage.append(str(row["video_id"]))
    if missing_passage:
        raise ValueError(
            "No usable participant passage timestamp after combining mappings for: "
            + ", ".join(missing_passage)
        )
    return result


def _participant_number(path: Path) -> str:
    match = PARTICIPANT_PATTERN.search(path.name)
    if match:
        return match.group(1)
    digits = re.findall(r"\d+", path.name)
    return digits[-1] if digits else path.name


def discover_participants(root: Path) -> list[Path]:
    """Find folders that contain a participant response file and trial files."""

    if not root.is_dir():
        raise FileNotFoundError(f"Participant response root not found: {root}")
    candidates: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_dir():
            continue
        names = [item.name.lower() for item in path.glob("*.csv")]
        if any("participant" in name for name in names) and any("video_" in name for name in names):
            candidates.append(path)
    if not candidates:
        names = [item.name.lower() for item in root.glob("*.csv")]
        if any("participant" in name for name in names) and any("video_" in name for name in names):
            candidates = [root]
    if not candidates:
        raise FileNotFoundError(f"No participant folders were discovered below {root}")
    return candidates


def _response_candidates(participant_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in participant_dir.glob("*.csv")
        if "participant" in path.name.lower() and "video_" not in path.name.lower()
        and "mapping" not in path.name.lower()
    )


def read_participant_responses(participant_dir: Path) -> pd.DataFrame:
    """Read realised video order and Q1 to Q3 from the headerless response file."""

    blocks: list[pd.DataFrame] = []
    for path in _response_candidates(participant_dir):
        frame: pd.DataFrame | None = None
        for separator in [",", ";"]:
            try:
                candidate = pd.read_csv(path, header=None, sep=separator)
            except Exception:
                continue
            if candidate.shape[1] >= 4:
                frame = candidate
                break
        if frame is None or frame.empty:
            continue
        block = pd.DataFrame({"video_id": frame.iloc[:, 0].map(_normalise_video_id)})
        for index, column in enumerate(["Q1", "Q2", "Q3"], start=1):
            block[column] = pd.to_numeric(frame.iloc[:, index], errors="coerce")
        block = block[block["video_id"].notna()].copy()
        if block.empty:
            continue
        block["response_source"] = str(path)
        blocks.append(block)
    if not blocks:
        raise FileNotFoundError(f"No readable participant response file in {participant_dir}")

    response = pd.concat(blocks, ignore_index=True)
    response = response.drop_duplicates("video_id", keep="first").reset_index(drop=True)
    response["trial_number"] = np.arange(1, len(response) + 1)
    return response


def _read_header(path: Path) -> set[str]:
    try:
        return set(pd.read_csv(path, nrows=0).columns.astype(str))
    except Exception:
        return set()


def find_trial_file(participant_dir: Path, video_id: str) -> Path:
    """Choose the trial time series file by ID and required columns."""

    candidates = sorted(
        path
        for path in participant_dir.glob("*.csv")
        if _normalise_video_id(path.stem) == video_id.lower()
    )
    for path in candidates:
        header = _read_header(path)
        if "Timestamp" in header and "TriggerValueRight" in header:
            return path
    for path in sorted(participant_dir.glob("*.csv")):
        header = _read_header(path)
        if "Timestamp" not in header or "TriggerValueRight" not in header:
            continue
        try:
            first = pd.read_csv(path, usecols=["VideoID"], nrows=1)
            value = _normalise_video_id(first.iloc[0, 0]) if not first.empty else None
        except Exception:
            value = None
        if value == video_id:
            return path
    raise FileNotFoundError(f"Time series for {video_id} not found in {participant_dir}")
