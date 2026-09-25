"""Model responses in relation to prior exposure to experimental factors."""

from __future__ import annotations

from typing import Any

import math
import numpy as np
import pandas as pd
import scipy.stats as st
from utils.constants import (
    EXPOSURE_FACTORS,
    EXPOSURE_MODEL_OUTCOMES,
    LOGGER,
    OUTCOME_SPECS,
)
from utils.models.helpers import (
    _add_global_holm,
    _design_formula,
    _fit_gee_with_fallback,
    _holm_within,
    _secondary_coefficient_table,
    _wald_test,
)


def _prepare_prior_exposures(trials: pd.DataFrame) -> pd.DataFrame:
    """Count previous trials at the current level of each factorial variable."""

    frame = trials.sort_values(["participant_uid", "trial_number"]).copy()
    for factor, _ in EXPOSURE_FACTORS:
        name = f"prior_same_{factor}_per10"
        frame[name] = (
            frame.groupby(["participant_uid", factor], observed=True).cumcount() / 10.0
        )
    return frame


def run_prior_exposure_models(trials: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Replace trial position with prior same-level factorial exposure counts."""

    frame_all = _prepare_prior_exposures(trials)
    group_term = "C(ordering_group, Treatment(reference='randomised_order'))"
    exposure_columns = [f"prior_same_{factor}_per10" for factor, _ in EXPOSURE_FACTORS]
    rhs = (
        f"{_design_formula(False)} + "
        + " + ".join(exposure_columns)
        + " + "
        + " + ".join(f"{group_term}:{column}" for column in exposure_columns)
    )
    coefficient_tables: list[pd.DataFrame] = []
    test_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    critical = st.norm.ppf(0.975)

    for outcome in EXPOSURE_MODEL_OUTCOMES:
        spec = OUTCOME_SPECS[outcome]
        if outcome not in frame_all.columns:
            failures.append({"outcome": outcome, "reason": "Outcome column was not available"})
            continue
        frame = frame_all.dropna(subset=[outcome, "participant_uid", *exposure_columns]).copy()
        if frame.empty or frame["participant_uid"].nunique() < 3 or frame[outcome].nunique() < 2:
            failures.append({"outcome": outcome, "reason": "Insufficient observations or variation"})
            continue
        try:
            result, working = _fit_gee_with_fallback(
                f"{outcome} ~ {rhs}", frame, spec["model_family"]
            )
        except RuntimeError as exc:
            failures.append({"outcome": outcome, "reason": str(exc)})
            LOGGER.warning("Prior-exposure model omitted for %s: %s", outcome, exc)
            continue

        coefficients = _secondary_coefficient_table(result, outcome, spec, working)
        coefficients.insert(0, "model", "prior_same_level_exposure")
        coefficient_tables.append(coefficients)
        names = list(pd.Series(result.params).index.astype(str))
        interaction_indexes: list[int] = []
        for (factor, factor_label), column in zip(EXPOSURE_FACTORS, exposure_columns):
            indexes = [
                index
                for index, name in enumerate(names)
                if "C(ordering_group" in name and name.endswith(f":{column}")
            ]
            interaction_indexes.extend(indexes)
            test = _wald_test(result, indexes, f"group by prior {factor_label} exposure")
            estimate = standard_error = ci_low = ci_high = np.nan
            standardised_estimate = np.nan
            estimate_scale = "joint Wald test"
            if len(indexes) == 1:
                estimate = float(result.params.iloc[indexes[0]])
                covariance = np.asarray(result.cov_params(), dtype=float)
                standard_error = math.sqrt(
                    max(0.0, float(covariance[indexes[0], indexes[0]]))
                )
                ci_low = estimate - critical * standard_error
                ci_high = estimate + critical * standard_error
                estimate_scale = (
                    "log odds per ten prior same-level exposures"
                    if spec["model_family"] == "binomial"
                    else f"{spec['units']} per ten prior same-level exposures"
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
            test_rows.append(
                {
                    "outcome": outcome,
                    "outcome_label": spec["label"],
                    "outcome_family": spec["family"],
                    "analysis_role": spec["role"],
                    "exposure_factor": factor,
                    "test": test["test"],
                    "df": test["df"],
                    "chi_square": test["chi_square"],
                    "estimate_fixed_minus_randomised": estimate,
                    "standard_error": standard_error,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "standardised_estimate": standardised_estimate,
                    "estimate_scale": estimate_scale,
                    "p_value": test["p_value"],
                    "working_correlation": working,
                }
            )
        joint = _wald_test(result, interaction_indexes, "joint group by prior exposure")
        test_rows.append(
            {
                "outcome": outcome,
                "outcome_label": spec["label"],
                "outcome_family": spec["family"],
                "analysis_role": spec["role"],
                "exposure_factor": "joint",
                "test": joint["test"],
                "df": joint["df"],
                "chi_square": joint["chi_square"],
                "estimate_fixed_minus_randomised": np.nan,
                "standard_error": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "estimate_scale": "joint Wald test",
                "p_value": joint["p_value"],
                "working_correlation": working,
            }
        )

    tests = pd.DataFrame(test_rows)
    if not tests.empty:
        tests = _holm_within(tests, ["outcome_family", "test"])
        tests["multiplicity_method"] = (
            "Holm across outcomes within family, separately for each exposure test"
        )
        tests = _add_global_holm(tests, ["test"])
    return {
        "exposure_adjusted_coefficients": (
            pd.concat(coefficient_tables, ignore_index=True)
            if coefficient_tables
            else pd.DataFrame()
        ),
        "exposure_adjusted_tests": tests,
        "exposure_adjusted_failures": pd.DataFrame(
            failures, columns=["outcome", "reason"]
        ),
    }
