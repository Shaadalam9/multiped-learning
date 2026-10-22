"""Describe temporal changes, carryover and sequence-position associations."""

from __future__ import annotations

from typing import Any

from statsmodels.stats.multitest import multipletests
import numpy as np
import pandas as pd
from utils.constants import (
    GROUP_FIXED,
    GROUP_RANDOMISED,
)
from utils.models.helpers import (
    _condition_formula,
    _fit_gee_with_fallback,
)
from utils.models.participants import (
    _hedges_g,
    _welch_difference,
)


def _participant_temporal_metrics(group: pd.DataFrame, outcome: str) -> dict[str, float]:
    frame = group.dropna(subset=[outcome, "trial_number"]).sort_values("trial_number")
    if len(frame) < 4:
        return {"slope_per_trial": np.nan, "early_late_difference": np.nan}
    x = frame["trial_number"].to_numpy(float)
    y = frame[outcome].to_numpy(float)
    slope = float(np.polyfit(x, y, 1)[0])
    early = y[x <= np.quantile(x, 0.25)]
    late = y[x >= np.quantile(x, 0.75)]
    return {
        "slope_per_trial": slope,
        "early_late_difference": float(np.mean(late) - np.mean(early)),
    }


def temporal_descriptives(trials: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Participant slopes and late minus early drift as secondary summaries."""

    records: list[dict[str, Any]] = []
    requested = [
        "unsafe_pct",
        "Q1",
        "Q2",
        "Q3",
        "mean_trigger",
        "peak_trigger",
        "heading_at_pass_deg",
        "minimum_heading_deg",
        "passage_change_deg",
    ]
    outcomes = [outcome for outcome in requested if outcome in trials.columns]
    for (ordering_group, participant_uid), group in trials.groupby(
        ["ordering_group", "participant_uid"], observed=True
    ):
        for outcome in outcomes:
            metrics = _participant_temporal_metrics(group, outcome)
            for metric, value in metrics.items():
                records.append(
                    {
                        "ordering_group": ordering_group,
                        "participant_uid": participant_uid,
                        "outcome": outcome,
                        "metric": metric,
                        "value": value,
                    }
                )
    participant_metrics = pd.DataFrame(records)
    comparisons: list[dict[str, Any]] = []
    for (outcome, metric), group in participant_metrics.groupby(["outcome", "metric"]):
        randomised = group.loc[group["ordering_group"] == GROUP_RANDOMISED, "value"].dropna().to_numpy(float)
        fixed = group.loc[group["ordering_group"] == GROUP_FIXED, "value"].dropna().to_numpy(float)
        if len(randomised) < 2 or len(fixed) < 2:
            continue
        row = _welch_difference(randomised, fixed)
        row.update(
            {
                "outcome": outcome,
                "metric": metric,
                "n_randomised": len(randomised),
                "n_fixed": len(fixed),
                "hedges_g_fixed_minus_randomised": _hedges_g(randomised, fixed),
            }
        )
        comparisons.append(row)
    table = pd.DataFrame(comparisons)
    if not table.empty:
        table["p_value_adjusted_fdr"] = multipletests(table["p_value"], method="fdr_bh")[1]
        table["multiplicity_method"] = "Benjamini Hochberg across drift tests"
    return participant_metrics, table


def _condition_adjusted_residuals(frame: pd.DataFrame, outcome: str) -> pd.Series:
    """Remove current trial factorial condition means within one participant."""

    columns = ["yielding", "eHMIOn", "camera", "distPed_m"]
    work = frame[[outcome, *columns]].copy()
    valid = work.notna().all(axis=1)
    residuals = pd.Series(np.nan, index=frame.index, dtype=float)
    if valid.sum() < 10:
        return residuals
    condition = pd.get_dummies(
        work.loc[valid, columns].astype(
            {"yielding": "int64", "eHMIOn": "int64", "camera": "int64", "distPed_m": "float64"}
        ),
        columns=columns,
        drop_first=True,
        dtype=float,
    )
    design = np.column_stack([np.ones(len(condition)), condition.to_numpy(float)])
    values = work.loc[valid, outcome].to_numpy(float)
    coefficients, _, _, _ = np.linalg.lstsq(design, values, rcond=None)
    residuals.loc[valid] = values - design @ coefficients
    return residuals


def carryover_analysis(trials: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exploratory previous trial effects adjusted for the current condition."""

    requested = [
        "unsafe_pct",
        "Q1",
        "Q2",
        "Q3",
        "mean_trigger",
        "heading_at_pass_deg",
        "passage_change_deg",
    ]
    outcomes = [outcome for outcome in requested if outcome in trials.columns]
    previous_factors = ["yielding", "eHMIOn", "camera", "distPed_m"]
    records: list[dict[str, Any]] = []
    for (ordering_group, participant_uid), participant in trials.groupby(
        ["ordering_group", "participant_uid"], observed=True
    ):
        participant = participant.sort_values("trial_number").copy()
        for factor in previous_factors:
            participant[f"previous_{factor}"] = participant[factor].shift(1)
        for outcome in outcomes:
            participant["adjusted_outcome"] = _condition_adjusted_residuals(participant, outcome)
            for factor in previous_factors:
                previous = f"previous_{factor}"
                frame = participant.dropna(subset=["adjusted_outcome", previous])
                effect = np.nan
                if factor == "distPed_m":
                    if len(frame) >= 10 and frame[previous].nunique() > 1:
                        effect = float(
                            np.polyfit(
                                frame[previous].to_numpy(float),
                                frame["adjusted_outcome"].to_numpy(float),
                                1,
                            )[0]
                        )
                    estimand = "adjusted slope per previous metre"
                else:
                    zero = frame.loc[frame[previous] == 0, "adjusted_outcome"]
                    one = frame.loc[frame[previous] == 1, "adjusted_outcome"]
                    if len(zero) >= 2 and len(one) >= 2:
                        effect = float(one.mean() - zero.mean())
                    estimand = "adjusted previous 1 minus previous 0"
                records.append(
                    {
                        "ordering_group": ordering_group,
                        "participant_uid": participant_uid,
                        "outcome": outcome,
                        "previous_factor": factor,
                        "estimand": estimand,
                        "carryover_effect": effect,
                    }
                )

    participant_effects = pd.DataFrame(records)
    comparisons: list[dict[str, Any]] = []
    for (outcome, factor), frame in participant_effects.groupby(
        ["outcome", "previous_factor"], observed=True
    ):
        randomised = frame.loc[
            frame["ordering_group"] == GROUP_RANDOMISED, "carryover_effect"
        ].dropna().to_numpy(float)
        fixed = frame.loc[
            frame["ordering_group"] == GROUP_FIXED, "carryover_effect"
        ].dropna().to_numpy(float)
        if len(randomised) < 2 or len(fixed) < 2:
            continue
        row = _welch_difference(randomised, fixed)
        row.update(
            {
                "outcome": outcome,
                "previous_factor": factor,
                "n_randomised": len(randomised),
                "n_fixed": len(fixed),
                "hedges_g_fixed_minus_randomised": _hedges_g(randomised, fixed),
            }
        )
        comparisons.append(row)
    table = pd.DataFrame(comparisons)
    if not table.empty:
        table["p_value_adjusted_fdr"] = multipletests(table["p_value"], method="fdr_bh")[1]
        table["multiplicity_method"] = "Benjamini Hochberg across carryover tests"
    return participant_effects, table


def sequence_position_audit(trials: pd.DataFrame) -> pd.DataFrame:
    """Quantify trial position association with each condition within each group."""

    records: list[dict[str, Any]] = []
    for (ordering_group, participant_uid), frame in trials.groupby(
        ["ordering_group", "participant_uid"], observed=True
    ):
        trial = frame["trial_number"].to_numpy(float)
        for factor in ["yielding", "eHMIOn", "camera", "distPed_m"]:
            values = pd.to_numeric(frame[factor], errors="coerce").to_numpy(float)
            valid = np.isfinite(trial) & np.isfinite(values)
            correlation = (
                float(np.corrcoef(trial[valid], values[valid])[0, 1])
                if valid.sum() > 2 and np.std(values[valid]) > 0
                else np.nan
            )
            records.append(
                {
                    "ordering_group": ordering_group,
                    "participant_uid": participant_uid,
                    "factor": factor,
                    "trial_position_correlation": correlation,
                }
            )
    participant = pd.DataFrame(records)
    summary = (
        participant.groupby(["ordering_group", "factor"], observed=True)[
            "trial_position_correlation"
        ]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
    )
    return summary


HEAD_TURN_THRESHOLD_DEG = 45.0


def head_turn_towards_pedestrian(trials: pd.DataFrame) -> pd.DataFrame:
    """Describe turns away from the vehicle, towards the out-of-view pedestrian side.

    With the participant first (camera 1), the virtual pedestrian stood on the side
    away from the approaching vehicle; with the participant second it stood between
    the participant and the vehicle. Head orientation is not eye gaze.
    """

    frame = trials.dropna(subset=["maximum_heading_deg"]).assign(
        turned=lambda d: d["maximum_heading_deg"] > HEAD_TURN_THRESHOLD_DEG
    )
    records: list[dict[str, Any]] = []
    for (ordering_group, camera), cell in frame.groupby(["ordering_group", "camera"], observed=True):
        records.append(_head_turn_record(ordering_group, camera, cell))
    for camera, cell in frame.groupby("camera", observed=True):
        records.append(_head_turn_record("both", camera, cell))
    return pd.DataFrame(records)


def _head_turn_record(ordering_group: str, camera: int, cell: pd.DataFrame) -> dict[str, Any]:
    by_participant = cell.groupby("participant_uid", observed=True)["turned"].any()
    return {
        "ordering_group": ordering_group,
        "pedestrian_position": "Participant first" if camera else "Participant second",
        "virtual_pedestrian_in_view_towards_vehicle": not bool(camera),
        "threshold_deg": HEAD_TURN_THRESHOLD_DEG,
        "trials": int(len(cell)),
        "trials_turned_pct": float(100.0 * cell["turned"].mean()),
        "median_maximum_heading_deg": float(cell["maximum_heading_deg"].median()),
        "median_minimum_heading_deg": float(cell["minimum_heading_deg"].median()),
        "participants": int(by_participant.size),
        "participants_turned_at_least_once": int(by_participant.sum()),
    }


FIRST_TRIAL_POSITIONS = ("trial 1", "trials 2 to 3", "trials 4 to 40")
FIRST_TRIAL_RATINGS = ("Q1", "Q2", "Q3")
# The single Q2 condition that differed between the groups: always trial 1 in the
# fixed order group.
FIRST_FIXED_CONDITION = {"yielding": 1, "eHMIOn": 0, "camera": 0, "distPed_m": 2.0}
FIRST_TRIAL_POSITION_BANDS = ((1, 5), (6, 20), (21, 40))


def _position_band(trial_number: pd.Series) -> pd.Series:
    return pd.Series(
        np.where(
            trial_number == 1,
            FIRST_TRIAL_POSITIONS[0],
            np.where(trial_number <= 3, FIRST_TRIAL_POSITIONS[1], FIRST_TRIAL_POSITIONS[2]),
        ),
        index=trial_number.index,
    )


def first_trial_rating_check(trials: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Exploratory check of first-trial ratings in the randomised order group.

    Every condition could come first in the randomised order group, so ratings at
    trial 1 can be compared with later trials after adjusting for condition. The
    second table follows the one condition that was always first in the fixed
    order group. Unadjusted for multiplicity; exploratory only.
    """

    randomised = trials[trials["ordering_group"] == GROUP_RANDOMISED].copy()
    randomised["position_band"] = _position_band(randomised["trial_number"])
    reference = FIRST_TRIAL_POSITIONS[2]
    term = f"C(position_band, Treatment('{reference}'))"
    rows: list[dict[str, Any]] = []
    for outcome in FIRST_TRIAL_RATINGS:
        frame = randomised.dropna(subset=[outcome])
        result, working = _fit_gee_with_fallback(
            f"{outcome} ~ {term} + {_condition_formula()}", frame, "gaussian"
        )
        intervals = result.conf_int()
        for band in FIRST_TRIAL_POSITIONS[:2]:
            name = f"{term}[T.{band}]"
            rows.append(
                {
                    "ordering_group": GROUP_RANDOMISED,
                    "outcome": outcome,
                    "position": band,
                    "reference": reference,
                    "difference_rating_points": float(result.params[name]),
                    "ci_low": float(intervals.loc[name, 0]),
                    "ci_high": float(intervals.loc[name, 1]),
                    "p_value": float(result.pvalues[name]),
                    "trials": int((frame["position_band"] == band).sum()),
                    "working_correlation": working,
                    "adjustment": "experimental condition; not adjusted for multiplicity",
                }
            )

    mask = np.ones(len(trials), dtype=bool)
    for column, value in FIRST_FIXED_CONDITION.items():
        mask &= trials[column].to_numpy() == value
    condition = trials[mask]
    position_rows: list[dict[str, Any]] = []
    for low, high in FIRST_TRIAL_POSITION_BANDS:
        cell = condition[
            (condition["ordering_group"] == GROUP_RANDOMISED)
            & condition["trial_number"].between(low, high)
        ]
        position_rows.append(
            {
                "ordering_group": GROUP_RANDOMISED,
                "trial_positions": f"{low} to {high}",
                "participants": int(len(cell)),
                "mean_Q2": float(cell["Q2"].mean()),
            }
        )
    fixed = condition[condition["ordering_group"] == GROUP_FIXED]
    position_rows.append(
        {
            "ordering_group": GROUP_FIXED,
            "trial_positions": "1",
            "participants": int(len(fixed)),
            "mean_Q2": float(fixed["Q2"].mean()),
        }
    )
    return {
        "first_trial_rating_check": pd.DataFrame(rows),
        "first_condition_q2_by_position": pd.DataFrame(position_rows),
    }
