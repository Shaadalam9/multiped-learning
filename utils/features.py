"""Extract trigger and head-orientation measurements from a single trial."""

from __future__ import annotations

from typing import Any
from pathlib import Path
import math
import numpy as np
import pandas as pd
from utils.config import (
    AnalysisSettings,
)


def _passage_time_seconds(row: pd.Series) -> float:
    """Return passage of the participant identified by the camera condition."""

    camera = int(row["camera"])
    preferred = ["cross_p1_time_s", "cross_p1", "crossP1", "crossing_p1"] if camera == 0 else [
        "cross_p2_time_s",
        "cross_p2",
        "crossP2",
        "crossing_p2",
    ]
    for column in preferred:
        if column in row.index and pd.notna(row[column]):
            value = float(row[column])
            return value / 1000.0 if value > 100.0 else value
    fallback = [column for column in row.index if "cross" in str(column).lower()]
    if fallback:
        value = float(row[fallback[0]])
        return value / 1000.0 if value > 100.0 else value
    raise ValueError(f"No participant passage timestamp found for {row['video_id']}")


def _timestamp_scale_to_seconds(values: pd.Series, passage_seconds: float) -> float:
    """Infer seconds, milliseconds, or microseconds from timeline coverage and rate.

    A previous rule divided every long recording by 1,000. That is unsafe: a
    valid 456-second recording sampled at 50 Hz is long in seconds, not a
    456-millisecond recording. The selected scale must place the participant
    passage (and preferably the full primary window) inside the time series and
    produce a plausible sampling frequency.
    """

    timestamps = pd.to_numeric(values, errors="coerce")
    finite = np.sort(timestamps[np.isfinite(timestamps)].to_numpy(float))
    if len(finite) < 2:
        return 1.0
    unique = np.unique(finite)
    spacing = np.diff(unique)
    spacing = spacing[spacing > 0]
    primary_start = passage_seconds - 5.0
    candidates: list[tuple[float, float]] = []
    for scale in (1.0, 1e-3, 1e-6):
        start = float(np.quantile(finite, 0.01) * scale)
        end = float(np.quantile(finite, 0.99) * scale)
        covers_passage = start <= passage_seconds <= end
        covers_primary_window = start <= primary_start and end >= passage_seconds
        if len(spacing):
            rate = 1.0 / (float(np.median(spacing)) * scale)
            plausible_rate = 1.0 <= rate <= 1000.0
            rate_penalty = abs(math.log(rate / 120.0)) if plausible_rate else 20.0
        else:
            plausible_rate = False
            rate_penalty = 20.0
        # Coverage is decisive; the rate breaks ties among plausible units.
        score = (
            1000.0 * float(covers_primary_window)
            + 100.0 * float(covers_passage)
            + 10.0 * float(plausible_rate)
            - rate_penalty
        )
        candidates.append((score, scale))
    return max(candidates, key=lambda item: item[0])[1]


def _timestamps_in_seconds(values: pd.Series, passage_seconds: float) -> pd.Series:
    timestamps = pd.to_numeric(values, errors="coerce")
    return timestamps * _timestamp_scale_to_seconds(timestamps, passage_seconds)


def _normalise_trigger(values: pd.Series) -> tuple[pd.Series, str]:
    trigger = pd.to_numeric(values, errors="coerce")
    finite = trigger[np.isfinite(trigger)]
    if finite.empty:
        return trigger, "unknown"
    scale = "0_to_100" if float(finite.quantile(0.99)) > 1.5 else "0_to_1"
    if scale == "0_to_100":
        trigger = trigger / 100.0
    return trigger, scale


def _quaternion_columns(frame: pd.DataFrame) -> tuple[str, str, str, str] | None:
    alternatives = [
        ("HMDRotationW", "HMDRotationX", "HMDRotationY", "HMDRotationZ"),
        ("HMDQuaternionW", "HMDQuaternionX", "HMDQuaternionY", "HMDQuaternionZ"),
        ("HMD_W", "HMD_X", "HMD_Y", "HMD_Z"),
        ("QuaternionW", "QuaternionX", "QuaternionY", "QuaternionZ"),
    ]
    return next((cols for cols in alternatives if set(cols).issubset(frame.columns)), None)


def _markley_average(quaternions: np.ndarray) -> np.ndarray:
    quaternions = np.asarray(quaternions, dtype=float)
    valid = np.isfinite(quaternions).all(axis=1)
    quaternions = quaternions[valid]
    if not len(quaternions):
        return np.full(4, np.nan)
    norms = np.linalg.norm(quaternions, axis=1)
    quaternions = quaternions[norms > np.finfo(float).eps]
    norms = norms[norms > np.finfo(float).eps]
    if not len(quaternions):
        return np.full(4, np.nan)
    quaternions = quaternions / norms[:, None]
    reference = quaternions[0]
    quaternions[np.dot(quaternions, reference) < 0] *= -1
    eigenvalues, eigenvectors = np.linalg.eigh(quaternions.T @ quaternions / len(quaternions))
    average = eigenvectors[:, np.argmax(eigenvalues)]
    return average if average[0] >= 0 else -average


def _heading_from_quaternions(quaternions: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternions, dtype=float)
    norm = np.linalg.norm(q, axis=1)
    valid = np.isfinite(q).all(axis=1) & (norm > 0)
    qn = np.full_like(q, np.nan)
    qn[valid] = q[valid] / norm[valid, None]
    w, x, y, z = qn.T
    forward_x = 2.0 * (x * z + w * y)
    forward_z = 1.0 - 2.0 * (x * x + y * y)
    return np.degrees(np.arctan2(forward_x, forward_z))


def _unity_heading_degrees(frame: pd.DataFrame) -> pd.Series:
    """Unity horizontal heading from a W X Y Z quaternion.

    Unity's vertical axis is y. The local forward vector (0, 0, 1) is rotated,
    projected onto the x z plane, and converted with atan2(x, z).
    """

    columns = _quaternion_columns(frame)
    if columns is None:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    q = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    return pd.Series(_heading_from_quaternions(q), index=frame.index)


def _unity_heading_trajectory(frame: pd.DataFrame, time_seconds: pd.Series) -> pd.DataFrame:
    """Return one Unity horizontal heading per unique raw timestamp."""

    columns = _quaternion_columns(frame)
    if columns is None:
        return pd.DataFrame(columns=["time_seconds", "heading_deg"])
    work = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    work.insert(0, "time_seconds", pd.to_numeric(time_seconds, errors="coerce"))
    work = work.dropna().sort_values("time_seconds")
    if work.empty:
        return pd.DataFrame(columns=["time_seconds", "heading_deg"])
    if work["time_seconds"].duplicated().any():
        records: list[dict[str, float]] = []
        for timestamp, group in work.groupby("time_seconds", sort=True):
            average = _markley_average(group.loc[:, list(columns)].to_numpy(float))
            heading = _heading_from_quaternions(average.reshape(1, 4))[0]
            records.append({"time_seconds": float(timestamp), "heading_deg": float(heading)})
        return pd.DataFrame(records)
    heading = _heading_from_quaternions(work.loc[:, list(columns)].to_numpy(float))
    return pd.DataFrame(
        {"time_seconds": work["time_seconds"].to_numpy(float), "heading_deg": heading}
    )


def _mean_heading_window(
    trajectory: pd.DataFrame,
    start: float,
    end: float,
    minimum_samples: int = 3,
) -> float:
    """Mean baseline-corrected heading in a half-open time window."""

    mask = (
        (trajectory["time_seconds"] >= start)
        & (trajectory["time_seconds"] < end)
        & np.isfinite(trajectory["heading_corrected_deg"])
    )
    values = trajectory.loc[mask, "heading_corrected_deg"]
    return float(values.mean()) if len(values) >= minimum_samples else np.nan


def _smooth_heading_for_feature_extraction(
    time_seconds: np.ndarray,
    heading_degrees: np.ndarray,
    window_seconds: float = 0.22,
) -> np.ndarray:
    """Centred moving mean with an approximately fixed temporal width."""

    time_seconds = np.asarray(time_seconds, dtype=float)
    heading_degrees = np.asarray(heading_degrees, dtype=float)
    finite_times = time_seconds[np.isfinite(time_seconds)]
    differences = np.diff(np.unique(finite_times))
    differences = differences[differences > 0]
    if not len(differences):
        return heading_degrees.copy()
    samples = max(3, int(round(window_seconds / float(np.median(differences)))))
    if samples % 2 == 0:
        samples += 1
    return (
        pd.Series(heading_degrees)
        .rolling(samples, center=True, min_periods=max(2, samples // 3))
        .mean()
        .to_numpy(float)
    )


def _threshold_suffix(threshold: float) -> str:
    return f"t{int(round(threshold * 100)):02d}"


def extract_trial_features(
    trial_file: Path,
    mapping_row: pd.Series,
    settings: AnalysisSettings,
) -> dict[str, Any]:
    """Extract trigger and heading features from one prespecified common window."""

    frame = pd.read_csv(trial_file)
    required = {"Timestamp", "TriggerValueRight"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{trial_file} is missing columns: {', '.join(sorted(missing))}")

    passage = _passage_time_seconds(mapping_row)
    timestamp_scale = _timestamp_scale_to_seconds(frame["Timestamp"], passage)
    time_seconds = pd.to_numeric(frame["Timestamp"], errors="coerce") * timestamp_scale
    trigger, trigger_scale = _normalise_trigger(frame["TriggerValueRight"])
    window_start = passage - settings.window_seconds
    mask = (
        np.isfinite(time_seconds)
        & np.isfinite(trigger)
        & (time_seconds >= window_start)
        & (time_seconds < passage)
    )
    selected = pd.DataFrame(
        {"time_seconds": time_seconds[mask], "trigger": trigger[mask]},
        index=frame.index[mask],
    )
    if selected.empty:
        raise ValueError("No valid trigger samples in the primary common window")

    relative = (selected["time_seconds"] - window_start) / settings.bin_seconds
    selected["bin_index"] = np.floor(relative + 1e-9).astype(int)
    selected = selected[
        (selected["bin_index"] >= 0) & (selected["bin_index"] < settings.expected_bins)
    ]
    grouped = selected.groupby("bin_index", sort=True)["trigger"]
    bin_max = grouped.max()
    valid_bins = int(bin_max.size)

    result: dict[str, Any] = {
        "window_start_s": window_start,
        "window_end_s": passage,
        "timestamp_scale_to_seconds": timestamp_scale,
        "expected_bins": settings.expected_bins,
        "valid_bins": valid_bins,
        "missing_bins": settings.expected_bins - valid_bins,
        "window_complete": valid_bins == settings.expected_bins,
        "trigger_scale": trigger_scale,
        "trigger_samples": int(selected.shape[0]),
        "mean_trigger": float(selected["trigger"].mean()),
        "peak_trigger": float(selected["trigger"].max()),
    }
    for threshold in settings.sensitivity_thresholds:
        unsafe_states = (bin_max > threshold).astype(int)
        unsafe_bins = int(unsafe_states.sum())
        suffix = _threshold_suffix(threshold)
        result[f"unsafe_bins_{suffix}"] = unsafe_bins
        result[f"safe_bins_{suffix}"] = valid_bins - unsafe_bins
        result[f"unsafe_pct_{suffix}"] = 100.0 * unsafe_bins / valid_bins if valid_bins else np.nan
        if len(unsafe_states) > 2 and unsafe_states.iloc[:-1].std() > 0 and unsafe_states.iloc[1:].std() > 0:
            result[f"unsafe_lag1_{suffix}"] = float(
                np.corrcoef(unsafe_states.iloc[:-1], unsafe_states.iloc[1:])[0, 1]
            )
        else:
            result[f"unsafe_lag1_{suffix}"] = np.nan

    primary_suffix = _threshold_suffix(settings.primary_threshold)
    result["unsafe_bins"] = result[f"unsafe_bins_{primary_suffix}"]
    result["safe_bins"] = result[f"safe_bins_{primary_suffix}"]
    result["unsafe_pct"] = result[f"unsafe_pct_{primary_suffix}"]
    # Event-defined trigger outcomes use the same binned state definition as
    # the primary outcome. They are calculated only for complete windows so a
    # missing bin can never be mistaken for a safe state. Latencies are
    # relative to the common-window start. A zero first-active latency is
    # explicitly left-censored: the participant may have begun pressing before
    # the five-second window.
    primary_states = (
        (bin_max.sort_index() > settings.primary_threshold)
        .astype(float)
        .reindex(range(settings.expected_bins))
    )
    if primary_states.notna().all():
        state_values = primary_states.to_numpy(int)
        active_indices = np.flatnonzero(state_values == 1)
        return_indices = np.flatnonzero(np.diff(state_values) == -1) + 1
        result["any_trigger_press"] = int(bool(active_indices.size))
        result["trigger_active_at_window_start"] = int(state_values[0] == 1)
        result["trigger_press_duration_s"] = float(
            state_values.sum() * settings.bin_seconds
        )
        result["trigger_activation_count"] = int(
            state_values[0] + (np.diff(state_values) == 1).sum()
        )
        result["trigger_first_active_latency_s"] = (
            float(active_indices[0] * settings.bin_seconds)
            if active_indices.size
            else np.nan
        )
        if active_indices.size:
            return_after_activation = return_indices[
                return_indices > active_indices[0]
            ]
            result["trigger_return_to_safe"] = int(
                bool(return_after_activation.size)
            )
            result["trigger_first_return_latency_s"] = (
                float(return_after_activation[0] * settings.bin_seconds)
                if return_after_activation.size
                else np.nan
            )
        else:
            result["trigger_return_to_safe"] = np.nan
            result["trigger_first_return_latency_s"] = np.nan
    else:
        result["any_trigger_press"] = np.nan
        result["trigger_active_at_window_start"] = np.nan
        result["trigger_press_duration_s"] = np.nan
        result["trigger_activation_count"] = np.nan
        result["trigger_first_active_latency_s"] = np.nan
        result["trigger_return_to_safe"] = np.nan
        result["trigger_first_return_latency_s"] = np.nan

    trajectory = _unity_heading_trajectory(frame, time_seconds)
    if not trajectory.empty:
        valid = np.isfinite(trajectory["time_seconds"]) & np.isfinite(trajectory["heading_deg"])
        trajectory = trajectory.loc[valid].sort_values("time_seconds").copy()
    if not trajectory.empty:
        trajectory["heading_unwrapped_deg"] = np.degrees(
            np.unwrap(np.radians(trajectory["heading_deg"].to_numpy(float)))
        )
        baseline_mask = (
            (trajectory["time_seconds"] >= 0.02)
            & (trajectory["time_seconds"] < 0.30)
        )
        baseline_values = trajectory.loc[baseline_mask, "heading_unwrapped_deg"]
        baseline = float(baseline_values.mean()) if len(baseline_values) >= 3 else np.nan
        trajectory["heading_corrected_deg"] = trajectory["heading_unwrapped_deg"] - baseline
        passage_mask = (
            (trajectory["time_seconds"] >= passage - 0.10)
            & (trajectory["time_seconds"] < passage + 0.10)
        )
        passage_values = trajectory.loc[passage_mask, "heading_corrected_deg"].dropna()
        common_mask = (
            (trajectory["time_seconds"] >= window_start)
            & (trajectory["time_seconds"] < passage)
        )
        common_values = trajectory.loc[common_mask, "heading_corrected_deg"].dropna()
        result["heading_baseline_deg"] = baseline
        result["heading_baseline_samples"] = int(baseline_values.size)
        result["heading_passage_mean_deg"] = (
            float(passage_values.mean()) if len(passage_values) >= 3 else np.nan
        )
        # Alias the passage-window outcome to the terminology used in the
        # original participant-level head-heading analysis.
        result["heading_at_pass_deg"] = result["heading_passage_mean_deg"]
        result["heading_passage_samples"] = int(passage_values.size)
        result["heading_common_window_mean_deg"] = (
            float(common_values.mean()) if not common_values.empty else np.nan
        )
        result["heading_common_window_mean_abs_deg"] = (
            float(common_values.abs().mean()) if not common_values.empty else np.nan
        )
        result["heading_common_window_sd_deg"] = (
            float(common_values.std(ddof=1)) if len(common_values) > 1 else np.nan
        )
        common_trajectory = trajectory.loc[
            common_mask, ["time_seconds", "heading_corrected_deg"]
        ].dropna()
        if not common_trajectory.empty:
            relative_bin = np.floor(
                (common_trajectory["time_seconds"] - window_start)
                / settings.bin_seconds
                + 1e-9
            ).astype(int)
            common_trajectory = common_trajectory.assign(common_bin=relative_bin)
            common_trajectory = common_trajectory[
                common_trajectory["common_bin"].between(
                    0, settings.expected_bins - 1
                )
            ]
            common_binned = common_trajectory.groupby(
                "common_bin", sort=True
            )["heading_corrected_deg"].mean()
            bin_numbers = common_binned.index.to_numpy(float)
            headings = common_binned.to_numpy(float)
            elapsed = np.diff(bin_numbers) * settings.bin_seconds
            angular_change = np.abs(np.diff(headings))
            valid_velocity = (
                np.isfinite(elapsed)
                & (elapsed > 0)
                & np.isfinite(angular_change)
            )
            result["heading_yaw_activity_deg_s"] = (
                float(np.mean(angular_change[valid_velocity] / elapsed[valid_velocity]))
                if valid_velocity.any()
                else np.nan
            )
            result["heading_yaw_activity_bin_pairs"] = int(valid_velocity.sum())
        else:
            result["heading_yaw_activity_deg_s"] = np.nan
            result["heading_yaw_activity_bin_pairs"] = 0

        far_pre = _mean_heading_window(trajectory, passage - 3.0, passage - 2.0)
        immediate_pre = _mean_heading_window(trajectory, passage - 0.5, passage)
        immediate_post = _mean_heading_window(trajectory, passage, passage + 0.5)
        result["far_pre_heading_deg"] = far_pre
        result["immediate_pre_heading_deg"] = immediate_pre
        result["immediate_post_heading_deg"] = immediate_post
        result["far_to_pre_change_deg"] = (
            immediate_pre - far_pre
            if np.isfinite(immediate_pre) and np.isfinite(far_pre)
            else np.nan
        )
        result["passage_change_deg"] = (
            immediate_post - immediate_pre
            if np.isfinite(immediate_post) and np.isfinite(immediate_pre)
            else np.nan
        )

        smooth_heading = _smooth_heading_for_feature_extraction(
            trajectory["time_seconds"].to_numpy(float),
            trajectory["heading_corrected_deg"].to_numpy(float),
        )
        minimum_mask = (
            (trajectory["time_seconds"].to_numpy(float) >= 0.50)
            & (trajectory["time_seconds"].to_numpy(float) <= passage - 0.20)
            & np.isfinite(smooth_heading)
        )
        if minimum_mask.any():
            candidate_indices = np.flatnonzero(minimum_mask)
            minimum_index = int(candidate_indices[np.argmin(smooth_heading[minimum_mask])])
            minimum_heading = float(smooth_heading[minimum_index])
            result["minimum_heading_deg"] = minimum_heading
            result["minimum_time_rel_pass_s"] = float(
                trajectory["time_seconds"].iloc[minimum_index] - passage
            )
            at_pass = result["heading_at_pass_deg"]
            result["recovered_by_pass_deg"] = (
                float(at_pass - minimum_heading) if np.isfinite(at_pass) else np.nan
            )
        else:
            result["minimum_heading_deg"] = np.nan
            result["minimum_time_rel_pass_s"] = np.nan
            result["recovered_by_pass_deg"] = np.nan
    else:
        result["heading_baseline_deg"] = np.nan
        result["heading_baseline_samples"] = 0
        result["heading_passage_mean_deg"] = np.nan
        result["heading_at_pass_deg"] = np.nan
        result["heading_passage_samples"] = 0
        result["heading_common_window_mean_deg"] = np.nan
        result["heading_common_window_mean_abs_deg"] = np.nan
        result["heading_common_window_sd_deg"] = np.nan
        result["heading_yaw_activity_deg_s"] = np.nan
        result["heading_yaw_activity_bin_pairs"] = 0
        result["far_pre_heading_deg"] = np.nan
        result["immediate_pre_heading_deg"] = np.nan
        result["immediate_post_heading_deg"] = np.nan
        result["far_to_pre_change_deg"] = np.nan
        result["passage_change_deg"] = np.nan
        result["minimum_heading_deg"] = np.nan
        result["minimum_time_rel_pass_s"] = np.nan
        result["recovered_by_pass_deg"] = np.nan
    return result
