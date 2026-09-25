"""Participant-level descriptive statistics and group comparisons."""

from __future__ import annotations

from typing import Any

import math
import numpy as np
import pandas as pd
import scipy.stats as st
from utils.constants import (
    GROUP_FIXED,
    GROUP_LABELS,
    GROUP_RANDOMISED,
    OUTCOME_SPECS,
    PARTICIPANT_OUTCOMES,
)
from utils.models.helpers import (
    _add_global_holm,
    _holm_within,
)


def _hedges_g(first: np.ndarray, second: np.ndarray) -> float:
    """Hedges g where positive means fixed sequence is higher."""

    n1, n2 = len(first), len(second)
    if n1 < 2 or n2 < 2:
        return np.nan
    pooled_var = ((n1 - 1) * np.var(first, ddof=1) + (n2 - 1) * np.var(second, ddof=1)) / (
        n1 + n2 - 2
    )
    if pooled_var <= 0:
        return np.nan
    d = (np.mean(second) - np.mean(first)) / math.sqrt(pooled_var)
    correction = 1.0 - 3.0 / (4.0 * (n1 + n2) - 9.0)
    return correction * d


def _welch_difference(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    """Fixed sequence minus randomised order, with Welch confidence interval."""

    n1, n2 = len(first), len(second)
    mean_difference = float(np.mean(second) - np.mean(first))
    v1, v2 = np.var(first, ddof=1), np.var(second, ddof=1)
    se2 = v1 / n1 + v2 / n2
    se = math.sqrt(se2)
    numerator = se2**2
    denominator = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
    df = numerator / denominator if denominator > 0 else np.nan
    critical = st.t.ppf(0.975, df) if np.isfinite(df) else np.nan
    t_value = mean_difference / se if se > 0 else np.nan
    p_value = 2.0 * st.t.sf(abs(t_value), df) if np.isfinite(t_value) else np.nan
    return {
        "difference_fixed_minus_randomised": mean_difference,
        "standard_error": se,
        "degrees_of_freedom": df,
        "ci_low": mean_difference - critical * se,
        "ci_high": mean_difference + critical * se,
        "t": t_value,
        "p_value": p_value,
    }


def participant_level_analysis(
    trials: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outcomes = [outcome for outcome in PARTICIPANT_OUTCOMES if outcome in trials.columns]
    participant_means = (
        trials.groupby(
            ["ordering_group", "ordering_group_label", "participant_uid"], observed=True
        )[outcomes]
        .mean()
        .reset_index()
    )
    if "any_trigger_press" in participant_means:
        participant_means["any_trigger_press"] *= 100.0
    descriptive_records: list[dict[str, Any]] = []
    comparison_records: list[dict[str, Any]] = []
    for outcome in outcomes:
        spec = OUTCOME_SPECS[outcome]
        for group in [GROUP_RANDOMISED, GROUP_FIXED]:
            values = participant_means.loc[
                participant_means["ordering_group"] == group, outcome
            ].dropna()
            descriptive_records.append(
                {
                    "outcome": outcome,
                    "outcome_label": spec["label"],
                    "outcome_family": spec["family"],
                    "analysis_role": spec["role"],
                    "units": spec["units"],
                    "ordering_group": group,
                    "ordering_group_label": GROUP_LABELS[group],
                    "n_participants": int(values.size),
                    "mean": float(values.mean()) if not values.empty else np.nan,
                    "standard_deviation": float(values.std(ddof=1)) if values.size > 1 else np.nan,
                    "median": float(values.median()) if not values.empty else np.nan,
                    "q1": float(values.quantile(0.25)) if not values.empty else np.nan,
                    "q3": float(values.quantile(0.75)) if not values.empty else np.nan,
                }
            )
        randomised = participant_means.loc[
            participant_means["ordering_group"] == GROUP_RANDOMISED, outcome
        ].dropna().to_numpy(float)
        fixed = participant_means.loc[
            participant_means["ordering_group"] == GROUP_FIXED, outcome
        ].dropna().to_numpy(float)
        if len(randomised) < 2 or len(fixed) < 2:
            continue
        comparison = _welch_difference(randomised, fixed)
        comparison.update(
            {
                "outcome": outcome,
                "outcome_label": spec["label"],
                "outcome_family": spec["family"],
                "analysis_role": spec["role"],
                "units": spec["units"],
                "n_randomised": len(randomised),
                "n_fixed": len(fixed),
                "hedges_g_fixed_minus_randomised": _hedges_g(randomised, fixed),
            }
        )
        comparison_records.append(comparison)

    comparisons = pd.DataFrame(comparison_records)
    if not comparisons.empty:
        comparisons = _holm_within(comparisons, ["outcome_family"])
        comparisons["multiplicity_method"] = (
            "Holm across participant-mean outcomes within outcome family"
        )
        comparisons = _add_global_holm(comparisons)
    return (
        participant_means,
        pd.DataFrame(descriptive_records),
        comparisons,
    )
