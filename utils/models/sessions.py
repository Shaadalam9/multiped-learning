"""Compare responses over the three break-defined session segments."""

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
    SESSION_SEGMENTS,
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
from utils.models.progression import (
    _prepare_progression_frame,
)


def _delta_contrast(
    result: Any,
    estimate: float,
    gradient: np.ndarray,
    scale: float = 1.0,
) -> dict[str, float]:
    """Delta-method contrast on the response scale."""

    covariance = np.asarray(result.cov_params(), dtype=float)
    variance = float(gradient @ covariance @ gradient)
    standard_error = math.sqrt(max(0.0, variance))
    z_value = estimate / standard_error if standard_error > 0 else np.nan
    critical = st.norm.ppf(0.975)
    return {
        "estimate": scale * estimate,
        "standard_error": scale * standard_error,
        "ci_low": scale * (estimate - critical * standard_error),
        "ci_high": scale * (estimate + critical * standard_error),
        "z": z_value,
        "p_value": (
            float(2.0 * st.norm.sf(abs(z_value)))
            if np.isfinite(z_value)
            else np.nan
        ),
    }


def run_session_segment_models(trials: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Categorical early/middle/late robustness models aligned to break opportunities.

    The polynomial models provide a compact trajectory, but their conclusion can
    depend on the assumed curve. This analysis instead treats trials 1-14,
    15-26, and 27-40 as categories. The segments are labelled by scheduled break
    opportunities; they do not imply that a participant actually took a break.
    """

    frame_all = _prepare_progression_frame(trials)
    group_term = "C(ordering_group, Treatment(reference='randomised_order'))"
    rhs = (
        f"{_design_formula(False)} + C(break_segment) "
        f"+ {group_term}:C(break_segment)"
    )
    coefficient_tables: list[pd.DataFrame] = []
    prediction_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for outcome in TEMPORAL_MODEL_OUTCOMES:
        spec = OUTCOME_SPECS[outcome]
        if outcome not in frame_all.columns:
            failures.append({"outcome": outcome, "reason": "Outcome column was not available"})
            continue
        frame = frame_all.dropna(
            subset=[outcome, "participant_uid", "break_segment"]
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
            LOGGER.warning("Session-segment model omitted for %s: %s", outcome, exc)
            continue

        coefficients = _secondary_coefficient_table(result, outcome, spec, working)
        coefficients.insert(0, "model", "categorical_session_segment")
        coefficient_tables.append(coefficients)
        link = "logit" if spec["model_family"] == "binomial" else "identity"
        scale = float(spec["scale"])
        design_info = result.model.data.design_info
        components: dict[tuple[str, str], tuple[float, np.ndarray, float]] = {}
        critical = st.norm.ppf(0.975)
        for ordering_group in [GROUP_RANDOMISED, GROUP_FIXED]:
            for segment, segment_label in SESSION_SEGMENTS:
                grid = _factorial_prediction_grid()
                grid["ordering_group"] = ordering_group
                grid["break_segment"] = segment
                estimate, gradient, variance = _average_prediction(
                    result, design_info, grid, link
                )
                components[(ordering_group, segment)] = (estimate, gradient, variance)
                standard_error = math.sqrt(variance)
                prediction_rows.append(
                    {
                        "outcome": outcome,
                        "outcome_label": spec["label"],
                        "outcome_family": spec["family"],
                        "analysis_role": spec["role"],
                        "units": spec["units"],
                        "ordering_group": ordering_group,
                        "ordering_group_label": GROUP_LABELS[ordering_group],
                        "session_segment": segment,
                        "session_segment_label": segment_label,
                        "adjusted_estimate": scale * estimate,
                        "ci_low": scale * (estimate - critical * standard_error),
                        "ci_high": scale * (estimate + critical * standard_error),
                        "working_correlation": working,
                        "standardisation": "equal weighting of the 2 by 2 by 2 by 5 condition grid",
                    }
                )

        names = list(pd.Series(result.params).index.astype(str))
        interaction_indices = [
            index
            for index, name in enumerate(names)
            if "C(ordering_group" in name and "C(break_segment)" in name
        ]
        joint = _wald_test(
            result, interaction_indices, "joint group by session segment"
        )
        test_rows.append(
            {
                "outcome": outcome,
                "outcome_label": spec["label"],
                "outcome_family": spec["family"],
                "analysis_role": spec["role"],
                "test": joint["test"],
                "df": joint["df"],
                "chi_square": joint["chi_square"],
                "estimate": np.nan,
                "standard_error": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "z": np.nan,
                "p_value": joint["p_value"],
                "estimate_scale": "joint Wald test",
                "working_correlation": working,
            }
        )

        early = SESSION_SEGMENTS[0][0]
        late = SESSION_SEGMENTS[-1][0]
        changes: dict[str, tuple[float, np.ndarray]] = {}
        for ordering_group in [GROUP_RANDOMISED, GROUP_FIXED]:
            late_estimate, late_gradient, _ = components[(ordering_group, late)]
            early_estimate, early_gradient, _ = components[(ordering_group, early)]
            change = late_estimate - early_estimate
            gradient = late_gradient - early_gradient
            changes[ordering_group] = (change, gradient)
            statistics = _delta_contrast(result, change, gradient, scale)
            test_rows.append(
                {
                    "outcome": outcome,
                    "outcome_label": spec["label"],
                    "outcome_family": spec["family"],
                    "analysis_role": spec["role"],
                    "test": f"early-to-late change: {ordering_group}",
                    "df": 1,
                    "chi_square": statistics["z"] ** 2,
                    **statistics,
                    "estimate_scale": spec["units"],
                    "working_correlation": working,
                }
            )

        fixed_change, fixed_gradient = changes[GROUP_FIXED]
        random_change, random_gradient = changes[GROUP_RANDOMISED]
        difference_statistics = _delta_contrast(
            result,
            fixed_change - random_change,
            fixed_gradient - random_gradient,
            scale,
        )
        test_rows.append(
            {
                "outcome": outcome,
                "outcome_label": spec["label"],
                "outcome_family": spec["family"],
                "analysis_role": spec["role"],
                "test": "difference in early-to-late change",
                "df": 1,
                "chi_square": difference_statistics["z"] ** 2,
                **difference_statistics,
                "estimate_scale": spec["units"],
                "working_correlation": working,
            }
        )

    tests = pd.DataFrame(test_rows)
    if not tests.empty:
        tests = _holm_within(tests, ["outcome_family", "test"])
        tests["multiplicity_method"] = (
            "Holm across outcomes within family, separately for each session-segment test"
        )
        tests = _add_global_holm(tests, ["test"])
    return {
        "session_segment_coefficients": (
            pd.concat(coefficient_tables, ignore_index=True)
            if coefficient_tables
            else pd.DataFrame()
        ),
        "session_segment_predictions": pd.DataFrame(prediction_rows),
        "session_segment_tests": tests,
        "session_segment_failures": pd.DataFrame(
            failures, columns=["outcome", "reason"]
        ),
    }
