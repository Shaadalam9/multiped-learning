"""Condition-adjusted linear changes over trial position."""

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
    LOGGER,
    OUTCOME_SPECS,
    TEMPORAL_MODEL_OUTCOMES,
)
from utils.models.helpers import (
    _add_global_holm,
    _average_prediction,
    _design_formula,
    _factorial_prediction_grid,
    _fit_gee_with_fallback,
    _holm_within,
    _secondary_coefficient_table,
    _wald_test,
)


def _break_segment(trial_number: pd.Series | np.ndarray | float) -> Any:
    """Scheduled break-opportunity segment for trial positions 1 to 40."""

    values = np.asarray(trial_number, dtype=float)
    result = np.where(
        values <= 14,
        "before_break_14",
        np.where(values <= 26, "after_break_14", "after_break_26"),
    )
    return result.item() if result.ndim == 0 else result


def _prepare_progression_frame(trials: pd.DataFrame) -> pd.DataFrame:
    frame = trials.copy()
    # Ten-trial units make the linear term interpretable without changing the
    # model fit.  The midpoint is fixed by design, not estimated from missingness.
    frame["trial_scaled"] = (pd.to_numeric(frame["trial_number"], errors="coerce") - 20.5) / 10.0
    frame["break_segment"] = _break_segment(frame["trial_number"])
    return frame


def _progression_prediction_table(
    result: Any,
    outcome: str,
    spec: dict[str, Any],
    working_correlation: str,
) -> pd.DataFrame:
    """Equal-cell predictions over trial position for adjusted temporal plots."""

    base = _factorial_prediction_grid()
    link = "logit" if spec["model_family"] == "binomial" else "identity"
    scale = float(spec["scale"])
    critical = st.norm.ppf(0.975)
    rows: list[dict[str, Any]] = []
    design_info = result.model.data.design_info
    for group in [GROUP_RANDOMISED, GROUP_FIXED]:
        for trial_number in range(1, 41):
            grid = base.copy()
            grid["ordering_group"] = group
            grid["trial_scaled"] = (trial_number - 20.5) / 10.0
            grid["break_segment"] = _break_segment(float(trial_number))
            estimate, _, variance = _average_prediction(result, design_info, grid, link)
            standard_error = math.sqrt(variance)
            lower = estimate - critical * standard_error
            upper = estimate + critical * standard_error
            if link == "logit":
                lower, upper = max(0.0, lower), min(1.0, upper)
            rows.append(
                {
                    "outcome": outcome,
                    "outcome_label": spec["label"],
                    "outcome_family": spec["family"],
                    "analysis_role": spec["role"],
                    "units": spec["units"],
                    "ordering_group": group,
                    "ordering_group_label": GROUP_LABELS[group],
                    "trial_number": trial_number,
                    "break_segment": _break_segment(float(trial_number)),
                    "adjusted_estimate": scale * estimate,
                    "ci_low": scale * lower,
                    "ci_high": scale * upper,
                    "model_family": spec["model_family"],
                    "working_correlation": working_correlation,
                    "standardisation": "equal weighting of the 2 by 2 by 2 by 5 condition grid",
                }
            )
    return pd.DataFrame(rows)


def run_adjusted_temporal_models(trials: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Condition-adjusted progression models for trigger, ratings, and heading."""

    frame_all = _prepare_progression_frame(trials)
    group_term = "C(ordering_group, Treatment(reference='randomised_order'))"
    rhs = (
        f"{_design_formula(False)} + trial_scaled + I(trial_scaled ** 2) "
        f"+ C(break_segment) + {group_term}:trial_scaled "
        f"+ {group_term}:I(trial_scaled ** 2)"
    )
    coefficient_tables: list[pd.DataFrame] = []
    test_rows: list[dict[str, Any]] = []
    prediction_tables: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []
    critical = st.norm.ppf(0.975)

    for outcome in TEMPORAL_MODEL_OUTCOMES:
        spec = OUTCOME_SPECS[outcome]
        if outcome not in frame_all.columns:
            failures.append({"outcome": outcome, "reason": "Outcome column was not available"})
            continue
        frame = frame_all.dropna(
            subset=[outcome, "participant_uid", "trial_scaled", "break_segment"]
        ).copy()
        if frame.empty or frame["participant_uid"].nunique() < 3 or frame[outcome].nunique() < 2:
            failures.append({"outcome": outcome, "reason": "Insufficient observations or variation"})
            continue
        try:
            result, working = _fit_gee_with_fallback(
                f"{outcome} ~ {rhs}", frame, spec["model_family"]
            )
        except RuntimeError as exc:
            failures.append({"outcome": outcome, "reason": str(exc)})
            LOGGER.warning("Adjusted temporal model omitted for %s: %s", outcome, exc)
            continue

        coefficients = _secondary_coefficient_table(result, outcome, spec, working)
        coefficients.insert(0, "model", "condition_adjusted_trial_position")
        coefficient_tables.append(coefficients)
        prediction_tables.append(
            _progression_prediction_table(result, outcome, spec, working)
        )

        names = list(pd.Series(result.params).index.astype(str))
        linear = [
            index
            for index, name in enumerate(names)
            if "C(ordering_group" in name
            and ":trial_scaled" in name
            and "I(trial_scaled ** 2)" not in name
        ]
        quadratic = [
            index
            for index, name in enumerate(names)
            if "C(ordering_group" in name and "I(trial_scaled ** 2)" in name
        ]
        for test_name, indexes in [
            ("group by linear trial position", linear),
            ("group by quadratic trial position", quadratic),
            ("joint group by trial position", linear + quadratic),
        ]:
            test = _wald_test(result, indexes, test_name)
            estimate = standard_error = ci_low = ci_high = np.nan
            standardised_estimate = standardised_ci_low = standardised_ci_high = np.nan
            estimate_scale = "joint Wald test"
            if len(indexes) == 1:
                estimate = float(result.params.iloc[indexes[0]])
                covariance = np.asarray(result.cov_params(), dtype=float)
                standard_error = math.sqrt(
                    max(0.0, float(covariance[indexes[0], indexes[0]]))
                )
                ci_low = estimate - critical * standard_error
                ci_high = estimate + critical * standard_error
                if "quadratic" in test_name:
                    estimate_scale = (
                        "log odds per squared ten-trial unit"
                        if spec["model_family"] == "binomial"
                        else f"{spec['units']} per squared ten-trial unit"
                    )
                else:
                    estimate_scale = (
                        "log odds per ten trials"
                        if spec["model_family"] == "binomial"
                        else f"{spec['units']} per ten trials"
                    )
                if spec["model_family"] != "binomial":
                    scale = float(spec["scale"])
                    estimate *= scale
                    standard_error *= scale
                    ci_low *= scale
                    ci_high *= scale
                    outcome_sd = float(pd.to_numeric(frame[outcome], errors="coerce").std(ddof=1)) * scale
                    if np.isfinite(outcome_sd) and outcome_sd > 0:
                        standardised_estimate = estimate / outcome_sd
                        standardised_ci_low = ci_low / outcome_sd
                        standardised_ci_high = ci_high / outcome_sd
            test_rows.append(
                {
                    "outcome": outcome,
                    "outcome_label": spec["label"],
                    "outcome_family": spec["family"],
                    "analysis_role": spec["role"],
                    "test": test_name,
                    "df": test["df"],
                    "chi_square": test["chi_square"],
                    "estimate_fixed_minus_randomised": estimate,
                    "standard_error": standard_error,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "standardised_estimate": standardised_estimate,
                    "standardised_ci_low": standardised_ci_low,
                    "standardised_ci_high": standardised_ci_high,
                    "estimate_scale": estimate_scale,
                    "p_value": test["p_value"],
                    "working_correlation": working,
                    "condition_adjustment": (
                        'full factorial condition structure plus scheduled break-opportunity segment'
                    ),
                }
            )

    tests = pd.DataFrame(test_rows)
    if not tests.empty:
        tests = _holm_within(tests, ["outcome_family", "test"])
        tests["multiplicity_method"] = (
            "Holm across outcomes within family, separately for each temporal test"
        )
        tests = _add_global_holm(tests, ["test"])
    return {
        "temporal_adjusted_coefficients": (
            pd.concat(coefficient_tables, ignore_index=True)
            if coefficient_tables
            else pd.DataFrame()
        ),
        "temporal_adjusted_tests": tests,
        "temporal_adjusted_predictions": (
            pd.concat(prediction_tables, ignore_index=True)
            if prediction_tables
            else pd.DataFrame()
        ),
        "temporal_adjusted_failures": pd.DataFrame(
            failures, columns=["outcome", "reason"]
        ),
    }
