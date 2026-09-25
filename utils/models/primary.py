"""Primary grouped-binomial comparisons and condition-specific contrasts."""

from __future__ import annotations

from typing import Any
import math
from statsmodels.stats.multitest import multipletests
import numpy as np
import pandas as pd
import scipy.stats as st
from utils.config import (
    AnalysisSettings,
)
from utils.constants import (
    GROUP_FIXED,
    GROUP_LABELS,
    GROUP_RANDOMISED,
    LOGGER,
    OUTCOME_SPECS,
)
from utils.models.helpers import (
    _average_prediction,
    _binomial_diagnostics,
    _coefficient_table,
    _condition_cell_metadata,
    _factorial_prediction_grid,
    _fit_grouped_binomial,
    _omnibus_order_tests,
)


def condition_cell_group_contrasts(
    result: Any,
    design_info: Any,
    outcome: str,
    spec: dict[str, Any],
    working_correlation: str,
    model_label: str,
    threshold: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate every factorial cell and its fixed minus randomised contrast.

    Point estimates and robust delta method standard errors are calculated on
    the response scale. The 40 cell comparisons within each outcome receive
    Holm adjusted p values and Bonferroni simultaneous 95 percent confidence
    intervals. Pointwise intervals are retained only for numerical audit.
    """

    base = _factorial_prediction_grid()
    link = "logit" if spec["model_family"] == "binomial" else "identity"
    scale = float(spec["scale"])
    covariance = np.asarray(result.cov_params(), dtype=float)
    pointwise_critical = float(st.norm.ppf(0.975))
    family_size = int(len(base))
    simultaneous_critical = float(
        st.norm.ppf(1.0 - 0.05 / (2.0 * family_size))
    )
    estimate_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []

    for cell_record in base.to_dict(orient="records"):
        cell = {
            "yielding": int(cell_record["yielding"]),
            "eHMIOn": int(cell_record["eHMIOn"]),
            "camera": int(cell_record["camera"]),
            "distPed_m": float(cell_record["distPed_m"]),
        }
        metadata = _condition_cell_metadata(cell)
        group_results: dict[str, tuple[float, np.ndarray, float]] = {}
        for group in [GROUP_RANDOMISED, GROUP_FIXED]:
            prediction_frame = pd.DataFrame([{**cell, "ordering_group": group}])
            estimate, gradient, variance = _average_prediction(
                result,
                design_info,
                prediction_frame,
                link,
            )
            group_results[group] = (estimate, gradient, variance)
            standard_error = math.sqrt(variance)
            lower = estimate - pointwise_critical * standard_error
            upper = estimate + pointwise_critical * standard_error
            if link == "logit":
                lower, upper = max(0.0, lower), min(1.0, upper)
            estimate_rows.append(
                {
                    "outcome": outcome,
                    "outcome_label": spec["label"],
                    "outcome_family": spec["family"],
                    "analysis_role": spec["role"],
                    "units": spec["units"],
                    "model": model_label,
                    "threshold": threshold,
                    **cell,
                    **metadata,
                    "ordering_group": group,
                    "ordering_group_label": GROUP_LABELS[group],
                    "estimate": scale * estimate,
                    "standard_error": scale * standard_error,
                    "pointwise_ci_low": scale * lower,
                    "pointwise_ci_high": scale * upper,
                    "model_family": spec["model_family"],
                    "working_correlation": working_correlation,
                }
            )

        fixed_estimate, fixed_gradient, _ = group_results[GROUP_FIXED]
        random_estimate, random_gradient, _ = group_results[GROUP_RANDOMISED]
        contrast_gradient = fixed_gradient - random_gradient
        contrast_variance = float(
            contrast_gradient @ covariance @ contrast_gradient
        )
        contrast_standard_error = math.sqrt(max(0.0, contrast_variance))
        difference = fixed_estimate - random_estimate
        z_value = (
            difference / contrast_standard_error
            if contrast_standard_error > 0
            else np.nan
        )
        contrast_rows.append(
            {
                "outcome": outcome,
                "outcome_label": spec["label"],
                "outcome_family": spec["family"],
                "analysis_role": spec["role"],
                "units": spec["units"],
                "model": model_label,
                "threshold": threshold,
                **cell,
                **metadata,
                "randomised_estimate": scale * random_estimate,
                "fixed_estimate": scale * fixed_estimate,
                "difference_fixed_minus_randomised": scale * difference,
                "standard_error": scale * contrast_standard_error,
                "pointwise_ci_low": scale
                * (
                    difference
                    - pointwise_critical * contrast_standard_error
                ),
                "pointwise_ci_high": scale
                * (
                    difference
                    + pointwise_critical * contrast_standard_error
                ),
                "simultaneous_ci_low": scale
                * (
                    difference
                    - simultaneous_critical * contrast_standard_error
                ),
                "simultaneous_ci_high": scale
                * (
                    difference
                    + simultaneous_critical * contrast_standard_error
                ),
                "z": z_value,
                "p_value": (
                    2.0 * st.norm.sf(abs(z_value))
                    if np.isfinite(z_value)
                    else np.nan
                ),
                "simultaneous_confidence_level": 0.95,
                "simultaneous_method": (
                    "Bonferroni across 40 condition cells within outcome"
                ),
                "model_family": spec["model_family"],
                "working_correlation": working_correlation,
            }
        )

    contrasts = pd.DataFrame(contrast_rows)
    contrasts["p_value_holm_within_outcome"] = np.nan
    valid = pd.to_numeric(contrasts["p_value"], errors="coerce").notna()
    if valid.any():
        contrasts.loc[valid, "p_value_holm_within_outcome"] = multipletests(
            contrasts.loc[valid, "p_value"],
            method="holm",
        )[1]
    contrasts["multiplicity_method_within_outcome"] = (
        "Holm across 40 condition cells within outcome"
    )
    contrasts["simultaneous_ci_excludes_zero"] = (
        (contrasts["simultaneous_ci_low"] > 0.0)
        | (contrasts["simultaneous_ci_high"] < 0.0)
    )
    return pd.DataFrame(estimate_rows), contrasts


def marginal_ordering_contrast(
    result: Any,
    design_info: Any,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Equal cell weighted predicted group means and fixed minus randomised contrast."""

    base = _factorial_prediction_grid()
    estimates: dict[str, tuple[float, np.ndarray, float]] = {}
    rows: list[dict[str, Any]] = []
    critical = st.norm.ppf(0.975)
    for group in [GROUP_RANDOMISED, GROUP_FIXED]:
        grid = base.copy()
        grid["ordering_group"] = group
        estimate, gradient, variance = _average_prediction(result, design_info, grid, "logit")
        estimates[group] = (estimate, gradient, variance)
        se = math.sqrt(variance)
        rows.append(
            {
                "threshold": threshold,
                "ordering_group": group,
                "ordering_group_label": GROUP_LABELS[group],
                "predicted_unsafe_pct": 100.0 * estimate,
                "ci_low": 100.0 * max(0.0, estimate - critical * se),
                "ci_high": 100.0 * min(1.0, estimate + critical * se),
            }
        )

    fixed_estimate, fixed_gradient, _ = estimates[GROUP_FIXED]
    random_estimate, random_gradient, _ = estimates[GROUP_RANDOMISED]
    gradient_difference = fixed_gradient - random_gradient
    variance_difference = float(
        gradient_difference @ np.asarray(result.cov_params()) @ gradient_difference
    )
    se_difference = math.sqrt(max(0.0, variance_difference))
    difference = fixed_estimate - random_estimate
    z_value = difference / se_difference if se_difference > 0 else np.nan
    contrast = pd.DataFrame(
        [
            {
                "threshold": threshold,
                "contrast": "fixed sequence minus randomised order",
                "difference_percentage_points": 100.0 * difference,
                "standard_error_percentage_points": 100.0 * se_difference,
                "ci_low": 100.0 * (difference - critical * se_difference),
                "ci_high": 100.0 * (difference + critical * se_difference),
                "z": z_value,
                "p_value": 2.0 * st.norm.sf(abs(z_value)) if np.isfinite(z_value) else np.nan,
            }
        ]
    )
    return pd.DataFrame(rows), contrast


def _factor_effect_prediction(
    result: Any,
    design_info: Any,
    ordering_group: str,
    focal_factor: str,
    conditioning_values: dict[str, int],
) -> tuple[float, np.ndarray, float]:
    """Response scale effect of changing one binary factor from 0 to 1."""

    template = pd.DataFrame({"distPed_m": [2.0, 4.0, 6.0, 8.0, 10.0]})
    template["ordering_group"] = ordering_group
    for factor, value in conditioning_values.items():
        template[factor] = value
    low = template.copy()
    high = template.copy()
    low[focal_factor] = 0
    high[focal_factor] = 1
    high_estimate, high_gradient, _ = _average_prediction(result, design_info, high, "logit")
    low_estimate, low_gradient, _ = _average_prediction(result, design_info, low, "logit")
    gradient = high_gradient - low_gradient
    variance = float(gradient @ np.asarray(result.cov_params()) @ gradient)
    return high_estimate - low_estimate, gradient, max(0.0, variance)


def condition_effect_contrasts(
    result: Any,
    design_info: Any,
    threshold: float,
) -> pd.DataFrame:
    """Compare the 12 final-analysis condition contrasts between ordering groups."""

    specifications: list[tuple[str, str, str, dict[str, int]]] = []
    for yielding in [0, 1]:
        for camera in [0, 1]:
            specifications.append(
                (
                    "conditional eHMI",
                    f"eHMI on minus off | yielding={yielding}, relative_order={camera}",
                    "eHMIOn",
                    {"yielding": yielding, "camera": camera},
                )
            )
    for yielding in [0, 1]:
        for ehmi in [0, 1]:
            specifications.append(
                (
                    "relative pedestrian order",
                    f"participant first minus participant second | yielding={yielding}, eHMI={ehmi}",
                    "camera",
                    {"yielding": yielding, "eHMIOn": ehmi},
                )
            )
    for ehmi in [0, 1]:
        for camera in [0, 1]:
            specifications.append(
                (
                    "AV behaviour",
                    f"yielding minus non-yielding | eHMI={ehmi}, relative_order={camera}",
                    "yielding",
                    {"eHMIOn": ehmi, "camera": camera},
                )
            )

    covariance = np.asarray(result.cov_params())
    critical = st.norm.ppf(0.975)
    rows: list[dict[str, Any]] = []
    for family, label, focal_factor, conditioning in specifications:
        effects: dict[str, tuple[float, np.ndarray, float]] = {}
        for group in [GROUP_RANDOMISED, GROUP_FIXED]:
            effects[group] = _factor_effect_prediction(
                result,
                design_info,
                group,
                focal_factor,
                conditioning,
            )
        random_effect, random_gradient, random_variance = effects[GROUP_RANDOMISED]
        fixed_effect, fixed_gradient, fixed_variance = effects[GROUP_FIXED]
        interaction_gradient = fixed_gradient - random_gradient
        interaction_variance = float(interaction_gradient @ covariance @ interaction_gradient)
        interaction_se = math.sqrt(max(0.0, interaction_variance))
        interaction = fixed_effect - random_effect
        z_value = interaction / interaction_se if interaction_se > 0 else np.nan
        rows.append(
            {
                "threshold": threshold,
                "contrast_family": family,
                "contrast": label,
                "randomised_effect_percentage_points": 100.0 * random_effect,
                "randomised_ci_low": 100.0
                * (random_effect - critical * math.sqrt(random_variance)),
                "randomised_ci_high": 100.0
                * (random_effect + critical * math.sqrt(random_variance)),
                "fixed_effect_percentage_points": 100.0 * fixed_effect,
                "fixed_ci_low": 100.0 * (fixed_effect - critical * math.sqrt(fixed_variance)),
                "fixed_ci_high": 100.0 * (fixed_effect + critical * math.sqrt(fixed_variance)),
                "difference_of_effects_percentage_points": 100.0 * interaction,
                "difference_ci_low": 100.0 * (interaction - critical * interaction_se),
                "difference_ci_high": 100.0 * (interaction + critical * interaction_se),
                "z": z_value,
                "p_value": 2.0 * st.norm.sf(abs(z_value)) if np.isfinite(z_value) else np.nan,
            }
        )
    table = pd.DataFrame(rows)
    table["p_value_holm"] = np.nan
    for _, indexes in table.groupby("contrast_family", observed=True).groups.items():
        table.loc[list(indexes), "p_value_holm"] = multipletests(
            table.loc[list(indexes), "p_value"], method="holm"
        )[1]
    table["multiplicity_method"] = "Holm within each four contrast family"
    return table


def run_primary_models(
    trials: pd.DataFrame,
    settings: AnalysisSettings,
) -> dict[str, pd.DataFrame]:
    """Fit the primary and threshold sensitivity binomial models."""

    coefficient_tables: list[pd.DataFrame] = []
    omnibus_tables: list[pd.DataFrame] = []
    marginal_tables: list[pd.DataFrame] = []
    contrast_tables: list[pd.DataFrame] = []
    temporal_coefficients: list[pd.DataFrame] = []
    temporal_omnibus: list[pd.DataFrame] = []
    condition_contrasts: list[pd.DataFrame] = []
    condition_cell_estimates: list[pd.DataFrame] = []
    condition_cell_contrasts: list[pd.DataFrame] = []
    diagnostic_tables: list[pd.DataFrame] = []
    model_failures: list[dict[str, Any]] = []

    for threshold in settings.sensitivity_thresholds:
        try:
            result, design_info, model_frame, _ = _fit_grouped_binomial(trials, threshold, False)
        except RuntimeError as exc:
            if np.isclose(threshold, settings.primary_threshold):
                raise
            model_failures.append(
                {
                    "model": "mean_and_condition",
                    "threshold": threshold,
                    "reason": str(exc),
                }
            )
            LOGGER.warning("Sensitivity model omitted: %s", exc)
            continue
        coefficient_tables.append(_coefficient_table(result, "mean_and_condition", threshold))
        diagnostic_tables.append(
            _binomial_diagnostics(result, model_frame, "mean_and_condition", threshold)
        )
        omnibus_tables.append(_omnibus_order_tests(result, False, threshold))
        marginals, contrast = marginal_ordering_contrast(result, design_info, threshold)
        marginal_tables.append(marginals)
        contrast_tables.append(contrast)
        if np.isclose(threshold, settings.primary_threshold):
            condition_contrasts.append(
                condition_effect_contrasts(result, design_info, threshold)
            )
            primary_cell_spec = {
                **OUTCOME_SPECS["unsafe_pct"],
                "model_family": "binomial",
                "scale": 100.0,
            }
            cell_estimates, cell_contrasts = condition_cell_group_contrasts(
                result,
                design_info,
                "unsafe_pct",
                primary_cell_spec,
                "participant clustered sandwich GLM",
                "primary grouped binomial",
                threshold,
            )
            condition_cell_estimates.append(cell_estimates)
            condition_cell_contrasts.append(cell_contrasts)

    temporal_result, _, temporal_frame, _ = _fit_grouped_binomial(
        trials, settings.primary_threshold, True
    )
    temporal_coefficients.append(
        _coefficient_table(temporal_result, "temporal", settings.primary_threshold)
    )
    temporal_omnibus.append(
        _omnibus_order_tests(temporal_result, True, settings.primary_threshold)
    )
    diagnostic_tables.append(
        _binomial_diagnostics(
            temporal_result,
            temporal_frame,
            "temporal",
            settings.primary_threshold,
        )
    )
    return {
        "primary_binomial_coefficients": pd.concat(coefficient_tables, ignore_index=True),
        "primary_omnibus_tests": pd.concat(omnibus_tables, ignore_index=True),
        "primary_marginal_estimates": pd.concat(marginal_tables, ignore_index=True),
        "primary_marginal_contrasts": pd.concat(contrast_tables, ignore_index=True),
        "temporal_binomial_coefficients": pd.concat(temporal_coefficients, ignore_index=True),
        "temporal_omnibus_tests": pd.concat(temporal_omnibus, ignore_index=True),
        "primary_condition_effect_contrasts": pd.concat(
            condition_contrasts, ignore_index=True
        ),
        "primary_condition_cell_estimates": pd.concat(
            condition_cell_estimates, ignore_index=True
        ),
        "primary_condition_cell_contrasts": pd.concat(
            condition_cell_contrasts, ignore_index=True
        ),
        "binomial_model_diagnostics": pd.concat(diagnostic_tables, ignore_index=True),
        "model_failures": pd.DataFrame(
            model_failures,
            columns=["model", "threshold", "reason"],
        ),
    }
