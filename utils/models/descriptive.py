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
