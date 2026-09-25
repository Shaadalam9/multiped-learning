"""Shared model fitting, prediction, contrast and multiplicity utilities."""

from __future__ import annotations

from typing import Iterable

from typing import Any
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.cov_struct import Independence
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning
from typing import Sequence
from patsy import build_design_matrices
from patsy import dmatrix
from scipy.special import expit
import math
from statsmodels.stats.multitest import multipletests
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as st
import warnings
from utils.constants import (
    GROUP_FIXED,
    GROUP_LABELS,
    GROUP_RANDOMISED,
)
from utils.features import (
    _threshold_suffix,
)


def _condition_formula() -> str:
    """Factor structure used in the final randomised order study analysis."""

    return (
        "C(yielding) * C(eHMIOn) * C(camera) + "
        "C(distPed_m) * (C(yielding) + C(eHMIOn) + C(camera))"
    )


def _design_formula(include_temporal: bool = False) -> str:
    group = "C(ordering_group, Treatment(reference='randomised_order'))"
    condition = _condition_formula()
    formula = f"{group} * ({condition})"
    if include_temporal:
        formula += (
            " + trial_centered * C(yielding) * C(eHMIOn) "
            "+ I(trial_centered ** 2) "
            f"+ {group}:trial_centered "
            f"+ {group}:trial_centered:C(yielding) "
            f"+ {group}:trial_centered:C(eHMIOn) "
            f"+ {group}:trial_centered:C(yielding):C(eHMIOn) "
            f"+ {group}:I(trial_centered ** 2)"
        )
    return formula


def _fit_grouped_binomial(
    frame: pd.DataFrame,
    threshold: float,
    include_temporal: bool,
    additional_rhs: str = "",
) -> tuple[Any, Any, pd.DataFrame, str]:
    suffix = _threshold_suffix(threshold)
    unsafe_col = f"unsafe_bins_{suffix}"
    safe_col = f"safe_bins_{suffix}"
    required = [unsafe_col, safe_col, "participant_uid", "ordering_group"]
    model_frame = frame.dropna(subset=required).copy()
    if include_temporal:
        model_frame["trial_centered"] = model_frame["trial_number"] - model_frame["trial_number"].mean()
    formula = _design_formula(include_temporal)
    if additional_rhs.strip():
        formula = f"{formula} + {additional_rhs}"
    design = dmatrix(formula, model_frame, return_type="dataframe")
    endog = model_frame[[unsafe_col, safe_col]].to_numpy(float)
    if endog[:, 0].sum() <= 0 or endog[:, 1].sum() <= 0:
        raise RuntimeError(
            f"Grouped binomial outcome is constant at threshold {threshold}; the model is not identifiable"
        )
    model = sm.GLM(endog, design, family=sm.families.Binomial())
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", PerfectSeparationWarning)
            result = model.fit(
                cov_type="cluster",
                cov_kwds={"groups": model_frame["participant_uid"].to_numpy()},
            )
        if any(issubclass(item.category, PerfectSeparationWarning) for item in caught):
            raise RuntimeError(
                f"Perfect separation was detected at threshold {threshold}; estimates are not identifiable"
            )
    except Exception as exc:
        raise RuntimeError(f"Grouped binomial model failed at threshold {threshold}: {exc}") from exc
    return result, design.design_info, model_frame, formula


def _coefficient_table(result: Any, model_name: str, threshold: float) -> pd.DataFrame:
    params = pd.Series(result.params)
    covariance = np.asarray(result.cov_params())
    se = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    z_values = np.divide(
        params.to_numpy(float),
        se,
        out=np.full(len(params), np.nan, dtype=float),
        where=se > 0,
    )
    p_values = 2.0 * st.norm.sf(np.abs(z_values))
    critical = st.norm.ppf(0.975)
    lower = params.to_numpy(float) - critical * se
    upper = params.to_numpy(float) + critical * se
    return pd.DataFrame(
        {
            "model": model_name,
            "threshold": threshold,
            "term": params.index.astype(str),
            "log_odds": params.to_numpy(float),
            "standard_error": se,
            "z": z_values,
            "p_value": p_values,
            "odds_ratio": np.exp(np.clip(params.to_numpy(float), -50.0, 50.0)),
            "odds_ratio_ci_low": np.exp(np.clip(lower, -50.0, 50.0)),
            "odds_ratio_ci_high": np.exp(np.clip(upper, -50.0, 50.0)),
        }
    )


def _binomial_diagnostics(
    result: Any,
    model_frame: pd.DataFrame,
    model_name: str,
    threshold: float,
) -> pd.DataFrame:
    lag_column = f"unsafe_lag1_{_threshold_suffix(threshold)}"
    return pd.DataFrame(
        [
            {
                "model": model_name,
                "threshold": threshold,
                "trial_observations": len(model_frame),
                "participant_clusters": model_frame["participant_uid"].nunique(),
                "pearson_dispersion": float(result.pearson_chi2 / result.df_resid)
                if result.df_resid > 0
                else np.nan,
                "mean_trial_lag1_binary_autocorrelation": float(model_frame[lag_column].mean())
                if lag_column in model_frame.columns
                else np.nan,
                "median_trial_lag1_binary_autocorrelation": float(model_frame[lag_column].median())
                if lag_column in model_frame.columns
                else np.nan,
            }
        ]
    )


def _wald_test(result: Any, indices: Sequence[int], label: str) -> dict[str, Any]:
    if not indices:
        return {
            "test": label,
            "df": 0,
            "requested_constraints": 0,
            "chi_square": np.nan,
            "p_value": np.nan,
            "terms": "",
        }
    names = list(pd.Series(result.params).index)
    parameters = np.asarray(result.params, dtype=float)[list(indices)]
    covariance = np.asarray(result.cov_params(), dtype=float)[np.ix_(indices, indices)]
    if not np.isfinite(parameters).all() or not np.isfinite(covariance).all():
        return {
            "test": label,
            "df": 0,
            "requested_constraints": len(indices),
            "chi_square": np.nan,
            "p_value": np.nan,
            "terms": " | ".join(names[index] for index in indices),
        }
    try:
        rank = int(np.linalg.matrix_rank(covariance))
    except np.linalg.LinAlgError:
        rank = 0
    if rank > 0:
        statistic = float(parameters @ np.linalg.pinv(covariance) @ parameters)
        p_value = float(st.chi2.sf(statistic, rank))
    else:
        statistic = np.nan
        p_value = np.nan
    return {
        "test": label,
        "df": rank,
        "requested_constraints": len(indices),
        "chi_square": statistic,
        "p_value": p_value,
        "terms": " | ".join(names[index] for index in indices),
    }


def _omnibus_order_tests(result: Any, temporal: bool, threshold: float) -> pd.DataFrame:
    names = list(pd.Series(result.params).index.astype(str))
    group_token = "C(ordering_group"
    group_indices = [index for index, name in enumerate(names) if group_token in name]
    interaction_indices = [
        index for index in group_indices if ":" in names[index] and "trial_centered" not in names[index]
    ]
    rows = [
        _wald_test(result, group_indices, "all ordering group terms"),
        _wald_test(result, interaction_indices, "ordering group by condition interactions"),
    ]
    for factor, label in [
        ("C(yielding)", "ordering group by yielding"),
        ("C(eHMIOn)", "ordering group by eHMI"),
        ("C(camera)", "ordering group by relative order"),
        ("C(distPed_m)", "ordering group by pedestrian distance"),
    ]:
        indices = [index for index in group_indices if factor in names[index]]
        rows.append(_wald_test(result, indices, label))
    if temporal:
        linear = [
            index
            for index in group_indices
            if "trial_centered" in names[index] and "I(trial_centered" not in names[index]
        ]
        quadratic = [
            index for index in group_indices if "I(trial_centered" in names[index]
        ]
        rows.extend(
            [
                _wald_test(result, linear, "ordering group by linear trial position"),
                _wald_test(result, quadratic, "ordering group by quadratic trial position"),
                _wald_test(result, linear + quadratic, "joint ordering group by trial position"),
            ]
        )
    table = pd.DataFrame(rows)
    table.insert(0, "threshold", threshold)
    table.insert(0, "model", "temporal" if temporal else "mean_and_condition")
    return table


def _factorial_prediction_grid() -> pd.DataFrame:
    return pd.MultiIndex.from_product(
        [[0, 1], [0, 1], [0, 1], [2.0, 4.0, 6.0, 8.0, 10.0]],
        names=["yielding", "eHMIOn", "camera", "distPed_m"],
    ).to_frame(index=False)


def _condition_cell_metadata(cell: dict[str, Any]) -> dict[str, Any]:
    """Return stable machine and manuscript labels for one factorial cell."""

    yielding = int(cell["yielding"])
    ehmi = int(cell["eHMIOn"])
    camera = int(cell["camera"])
    distance = float(cell["distPed_m"])
    yielding_label = "Yielding" if yielding else "Non yielding"
    ehmi_label = "eHMI on" if ehmi else "eHMI off"
    relative_order_label = "Participant first" if camera else "Participant second"
    distance_label = f"{distance:g} m"
    return {
        "condition_id": f"Y{yielding}_E{ehmi}_R{camera}_D{distance:g}",
        "condition_label": (
            f"{yielding_label}; {ehmi_label}; "
            f"{relative_order_label}; {distance_label}"
        ),
        "yielding_label": yielding_label,
        "ehmi_label": ehmi_label,
        "relative_order_label": relative_order_label,
        "distance_label": distance_label,
    }


def _holm_within(table: pd.DataFrame, group_columns: Sequence[str]) -> pd.DataFrame:
    table = table.copy()
    table["p_value_adjusted"] = np.nan
    table["multiplicity_method"] = "Holm"
    if table.empty:
        return table
    grouped = table.groupby(list(group_columns), dropna=False, sort=False)
    for _, index in grouped.groups.items():
        p = pd.to_numeric(table.loc[index, "p_value"], errors="coerce")
        valid = p.notna()
        if valid.any():
            adjusted = multipletests(p[valid], method="holm")[1]
            table.loc[p[valid].index, "p_value_adjusted"] = adjusted
    return table


def _add_global_holm(
    table: pd.DataFrame,
    group_columns: Sequence[str] = (),
    output_column: str = "p_value_adjusted_global",
) -> pd.DataFrame:
    """Add a conservative Holm correction across all rows or stated strata."""

    table = table.copy()
    table[output_column] = np.nan
    if table.empty:
        return table
    groups: Iterable[Any]
    if group_columns:
        groups = table.groupby(list(group_columns), dropna=False, sort=False).groups.values()
    else:
        groups = [table.index]
    for index in groups:
        index = list(index)
        p = pd.to_numeric(table.loc[index, "p_value"], errors="coerce")
        valid = p.notna()
        if valid.any():
            table.loc[p[valid].index, output_column] = multipletests(
                p[valid], method="holm"
            )[1]
    return table


def _fit_gee_with_fallback(
    formula: str,
    frame: pd.DataFrame,
    model_family: str,
    require_convergence: bool = False,
) -> tuple[Any, str]:
    """Fit a participant-clustered GEE, retaining a documented fallback."""

    failures: list[str] = []
    for structure_name, structure in [
        ("exchangeable", Exchangeable()),
        ("independence fallback", Independence()),
    ]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", PerfectSeparationWarning)
                warnings.simplefilter("ignore", RuntimeWarning)
                result = smf.gee(
                    formula,
                    groups="participant_uid",
                    data=frame,
                    family=(
                        sm.families.Binomial()
                        if model_family == "binomial"
                        else sm.families.Gaussian()
                    ),
                    cov_struct=structure,
                ).fit()
            if require_convergence and not bool(
                getattr(result, "converged", True)
            ):
                raise RuntimeError("model did not converge")
            if not np.isfinite(np.asarray(result.params, dtype=float)).all():
                raise RuntimeError("nonfinite coefficient estimate")
            if not np.isfinite(np.asarray(result.cov_params(), dtype=float)).all():
                raise RuntimeError("nonfinite robust covariance")
            return result, structure_name
        except Exception as exc:
            failures.append(f"{structure_name}: {exc}")
    raise RuntimeError(" | ".join(failures))


def _secondary_coefficient_table(
    result: Any,
    outcome: str,
    spec: dict[str, Any],
    working_correlation: str,
) -> pd.DataFrame:
    params = pd.Series(result.params)
    covariance = np.asarray(result.cov_params(), dtype=float)
    standard_error = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    z_values = np.divide(
        params.to_numpy(float),
        standard_error,
        out=np.full(len(params), np.nan, dtype=float),
        where=standard_error > 0,
    )
    critical = st.norm.ppf(0.975)
    return pd.DataFrame(
        {
            "outcome": outcome,
            "outcome_label": spec["label"],
            "outcome_family": spec["family"],
            "analysis_role": spec["role"],
            "model_family": spec["model_family"],
            "working_correlation": working_correlation,
            "term": params.index.astype(str),
            "estimate_on_link_scale": params.to_numpy(float),
            "standard_error": standard_error,
            "ci_low": params.to_numpy(float) - critical * standard_error,
            "ci_high": params.to_numpy(float) + critical * standard_error,
            "z": z_values,
            "p_value": 2.0 * st.norm.sf(np.abs(z_values)),
        }
    )


def _secondary_marginal_ordering_contrast(
    result: Any,
    design_info: Any,
    outcome: str,
    spec: dict[str, Any],
    working_correlation: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Equal-cell marginal group estimates for a secondary GEE outcome."""

    base = _factorial_prediction_grid()
    link = "logit" if spec["model_family"] == "binomial" else "identity"
    scale = float(spec["scale"])
    estimates: dict[str, tuple[float, np.ndarray, float]] = {}
    estimate_rows: list[dict[str, Any]] = []
    critical = st.norm.ppf(0.975)
    for group in [GROUP_RANDOMISED, GROUP_FIXED]:
        grid = base.copy()
        grid["ordering_group"] = group
        estimate, gradient, variance = _average_prediction(
            result, design_info, grid, link
        )
        estimates[group] = (estimate, gradient, variance)
        standard_error = math.sqrt(variance)
        lower = estimate - critical * standard_error
        upper = estimate + critical * standard_error
        if link == "logit":
            lower, upper = max(0.0, lower), min(1.0, upper)
        estimate_rows.append(
            {
                "outcome": outcome,
                "outcome_label": spec["label"],
                "outcome_family": spec["family"],
                "analysis_role": spec["role"],
                "units": spec["units"],
                "ordering_group": group,
                "ordering_group_label": GROUP_LABELS[group],
                "marginal_estimate": scale * estimate,
                "ci_low": scale * lower,
                "ci_high": scale * upper,
                "standardisation": "equal weighting of the 2 by 2 by 2 by 5 condition grid",
                "model_family": spec["model_family"],
                "working_correlation": working_correlation,
            }
        )

    fixed_estimate, fixed_gradient, _ = estimates[GROUP_FIXED]
    random_estimate, random_gradient, _ = estimates[GROUP_RANDOMISED]
    gradient = fixed_gradient - random_gradient
    variance = float(gradient @ np.asarray(result.cov_params()) @ gradient)
    standard_error = math.sqrt(max(0.0, variance))
    difference = fixed_estimate - random_estimate
    z_value = difference / standard_error if standard_error > 0 else np.nan
    contrast = pd.DataFrame(
        [
            {
                "outcome": outcome,
                "outcome_label": spec["label"],
                "outcome_family": spec["family"],
                "analysis_role": spec["role"],
                "units": spec["units"],
                "contrast": "fixed sequence minus randomised order",
                "difference": scale * difference,
                "standard_error": scale * standard_error,
                "ci_low": scale * (difference - critical * standard_error),
                "ci_high": scale * (difference + critical * standard_error),
                "z": z_value,
                "p_value": (
                    2.0 * st.norm.sf(abs(z_value)) if np.isfinite(z_value) else np.nan
                ),
                "standardisation": "equal weighting of the 2 by 2 by 2 by 5 condition grid",
                "model_family": spec["model_family"],
                "working_correlation": working_correlation,
            }
        ]
    )
    return pd.DataFrame(estimate_rows), contrast


def _average_prediction(
    result: Any,
    design_info: Any,
    prediction_frame: pd.DataFrame,
    link: str,
) -> tuple[float, np.ndarray, float]:
    design = np.asarray(
        build_design_matrices([design_info], prediction_frame, return_type="dataframe")[0]
    )
    beta = np.asarray(result.params, dtype=float)
    linear = design @ beta
    if link == "logit":
        predicted = expit(linear)
        gradients = predicted[:, None] * (1.0 - predicted[:, None]) * design
    else:
        predicted = linear
        gradients = design
    gradient = gradients.mean(axis=0)
    estimate = float(predicted.mean())
    variance = float(gradient @ np.asarray(result.cov_params()) @ gradient)
    return estimate, gradient, max(0.0, variance)
