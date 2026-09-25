"""Exploratory nonlinear changes over trial position."""

from __future__ import annotations

from typing import Any
from typing import Sequence
from patsy import build_design_matrices
from itertools import combinations
from patsy import dmatrix
import math
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
    TEMPORAL_MODEL_OUTCOMES,
)
from utils.models.helpers import (
    _add_global_holm,
    _average_prediction,
    _condition_formula,
    _design_formula,
    _factorial_prediction_grid,
    _fit_gee_with_fallback,
    _holm_within,
    _secondary_coefficient_table,
    _wald_test,
)
from utils.models.progression import (
    _break_segment,
    _prepare_progression_frame,
)


def _prepare_residualised_spline_basis(
    trials: pd.DataFrame,
    settings: AnalysisSettings,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Create a stable spline basis orthogonal to condition within each cohort."""

    frame = _prepare_progression_frame(trials)
    degrees = settings.spline_degrees_of_freedom
    prepared_groups: dict[str, dict[str, Any]] = {}
    prediction_lookup_rows: list[dict[str, Any]] = []

    for ordering_group in [GROUP_RANDOMISED, GROUP_FIXED]:
        group_index = frame.index[
            frame["ordering_group"] == ordering_group
        ]
        group_frame = frame.loc[group_index].copy()
        raw_design = dmatrix(
            f"cr(trial_scaled, df={degrees}) - 1",
            group_frame,
            return_type="dataframe",
        )
        raw_basis = np.asarray(raw_design, dtype=float)
        nuisance_design = dmatrix(
            f"1 + {_condition_formula()} + C(break_segment)",
            group_frame,
            return_type="dataframe",
        )
        nuisance_matrix = np.asarray(nuisance_design, dtype=float)
        projection = np.linalg.pinv(nuisance_matrix) @ raw_basis
        residual_basis = raw_basis - nuisance_matrix @ projection
        singular_values = np.linalg.svd(
            residual_basis,
            compute_uv=False,
        )
        tolerance = (
            singular_values[0] * max(residual_basis.shape) * np.finfo(float).eps
            if len(singular_values) and singular_values[0] > 0
            else 0.0
        )
        rank = int(np.sum(singular_values > tolerance))
        prepared_groups[ordering_group] = {
            "group_index": group_index,
            "raw_design": raw_design,
            "nuisance_design": nuisance_design,
            "projection": projection,
            "residual_basis": residual_basis,
            "rank": rank,
        }

    common_rank = min(
        int(prepared_groups[group]["rank"])
        for group in [GROUP_RANDOMISED, GROUP_FIXED]
    )
    if common_rank < 2:
        raise RuntimeError(
            "Fewer than two spline components remained after condition "
            "residualisation"
        )

    selected_columns: tuple[int, ...] | None = None
    best_score = -np.inf
    for candidate in combinations(range(degrees), common_rank):
        score_components: list[float] = []
        valid = True
        for ordering_group in [GROUP_RANDOMISED, GROUP_FIXED]:
            candidate_basis = prepared_groups[ordering_group][
                "residual_basis"
            ][:, candidate]
            singular_values = np.linalg.svd(
                candidate_basis,
                compute_uv=False,
            )
            if (
                len(singular_values) != common_rank
                or singular_values[-1] <= 1e-10
            ):
                valid = False
                break
            score_components.append(
                float(singular_values[-1] / singular_values[0])
            )
        if valid:
            score = min(score_components)
            if score > best_score:
                best_score = score
                selected_columns = tuple(candidate)
    if selected_columns is None:
        raise RuntimeError(
            "No common full-rank spline basis remained in both cohorts"
        )

    basis_columns = [
        f"spline_residual_{index + 1}"
        for index in range(common_rank)
    ]
    for ordering_group in [GROUP_RANDOMISED, GROUP_FIXED]:
        prepared = prepared_groups[ordering_group]
        group_index = prepared["group_index"]
        raw_design = prepared["raw_design"]
        nuisance_design = prepared["nuisance_design"]
        projection = prepared["projection"][:, selected_columns]
        residual_basis = prepared["residual_basis"][:, selected_columns]
        root_mean_square = np.sqrt(np.mean(residual_basis**2, axis=0))
        if (
            len(root_mean_square) != len(basis_columns)
            or not np.isfinite(root_mean_square).all()
            or np.any(root_mean_square <= 1e-10)
            or np.linalg.matrix_rank(residual_basis) != len(basis_columns)
        ):
            raise RuntimeError(
                f"Residualised spline basis was rank deficient for {ordering_group}"
            )
        residual_basis /= root_mean_square
        for column_index, column in enumerate(basis_columns):
            frame.loc[group_index, column] = residual_basis[:, column_index]

        for trial_number in range(1, 41):
            grid = _factorial_prediction_grid()
            grid["trial_scaled"] = (trial_number - 20.5) / 10.0
            grid["break_segment"] = _break_segment(float(trial_number))
            raw_grid = np.asarray(
                build_design_matrices(
                    [raw_design.design_info],
                    grid,
                )[0],
                dtype=float,
            )[:, selected_columns]
            nuisance_grid = np.asarray(
                build_design_matrices(
                    [nuisance_design.design_info],
                    grid,
                )[0],
                dtype=float,
            )
            residual_grid = (
                raw_grid - nuisance_grid @ projection
            ) / root_mean_square
            record: dict[str, Any] = {
                "ordering_group": ordering_group,
                "trial_number": trial_number,
            }
            for column_index, column in enumerate(basis_columns):
                record[column] = float(
                    np.mean(residual_grid[:, column_index])
                )
            prediction_lookup_rows.append(record)

    return frame, basis_columns, pd.DataFrame(prediction_lookup_rows)


def _spline_progression_prediction_table(
    result: Any,
    outcome: str,
    spec: dict[str, Any],
    working_correlation: str,
    basis_columns: Sequence[str],
    prediction_lookup: pd.DataFrame,
) -> pd.DataFrame:
    """Equal-cell spline predictions using the residualised basis lookup."""

    link = "logit" if spec["model_family"] == "binomial" else "identity"
    scale = float(spec["scale"])
    critical = st.norm.ppf(0.975)
    design_info = result.model.data.design_info
    rows: list[dict[str, Any]] = []
    for ordering_group in [GROUP_RANDOMISED, GROUP_FIXED]:
        for trial_number in range(1, 41):
            lookup = prediction_lookup[
                (prediction_lookup["ordering_group"] == ordering_group)
                & (prediction_lookup["trial_number"] == trial_number)
            ]
            if len(lookup) != 1:
                raise RuntimeError(
                    "Spline prediction lookup did not contain exactly one row "
                    f"for {ordering_group}, trial {trial_number}"
                )
            grid = _factorial_prediction_grid()
            grid["ordering_group"] = ordering_group
            grid["break_segment"] = _break_segment(float(trial_number))
            for column in basis_columns:
                grid[column] = float(lookup.iloc[0][column])
            estimate, _, variance = _average_prediction(
                result,
                design_info,
                grid,
                link,
            )
            standard_error = math.sqrt(max(0.0, variance))
            lower = estimate - critical * standard_error
            upper = estimate + critical * standard_error
            if link == "logit":
                lower, upper = max(0.0, lower), min(1.0, upper)
            rows.append(
                {
                    "outcome": outcome,
                    "outcome_label": spec["label"],
                    "outcome_family": spec["family"],
                    "analysis_role": "post hoc exploratory spline sensitivity",
                    "units": spec["units"],
                    "ordering_group": ordering_group,
                    "ordering_group_label": GROUP_LABELS[ordering_group],
                    "trial_number": trial_number,
                    "break_segment": _break_segment(float(trial_number)),
                    "adjusted_estimate": scale * estimate,
                    "ci_low": scale * lower,
                    "ci_high": scale * upper,
                    "model_family": spec["model_family"],
                    "working_correlation": working_correlation,
                    "standardisation": (
                        "equal weighting of the 2 by 2 by 2 by 5 condition grid"
                    ),
                }
            )
    return pd.DataFrame(rows)


def run_spline_temporal_models(
    trials: pd.DataFrame,
    settings: AnalysisSettings,
) -> dict[str, pd.DataFrame]:
    """Exploratory nonlinear progression sensitivity using natural cubic splines."""

    frame_all, basis_columns, prediction_lookup = (
        _prepare_residualised_spline_basis(trials, settings)
    )
    group_term = "C(ordering_group, Treatment(reference='randomised_order'))"
    basis_rhs = " + ".join(basis_columns)
    interaction_rhs = " + ".join(
        f"{group_term}:{column}" for column in basis_columns
    )
    rhs = (
        f"{_design_formula(False)} + C(break_segment) + {basis_rhs} "
        f"+ {interaction_rhs}"
    )
    coefficient_tables: list[pd.DataFrame] = []
    prediction_tables: list[pd.DataFrame] = []
    test_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for outcome in TEMPORAL_MODEL_OUTCOMES:
        spec = OUTCOME_SPECS[outcome]
        if outcome not in frame_all.columns:
            failures.append(
                {"outcome": outcome, "reason": "Outcome column was not available"}
            )
            continue
        frame = frame_all.dropna(
            subset=[outcome, "participant_uid", "break_segment", *basis_columns]
        ).copy()
        if (
            frame.empty
            or frame["participant_uid"].nunique() < 3
            or frame[outcome].nunique() < 2
        ):
            failures.append(
                {"outcome": outcome, "reason": "Insufficient observations or variation"}
            )
            continue
        try:
            result, working = _fit_gee_with_fallback(
                f"{outcome} ~ {rhs}",
                frame,
                spec["model_family"],
                require_convergence=True,
            )
            names = list(pd.Series(result.params).index.astype(str))
            interaction_indices = [
                index
                for index, name in enumerate(names)
                if "C(ordering_group" in name
                and "spline_residual_" in name
            ]
            interaction_covariance = np.asarray(
                result.cov_params(),
                dtype=float,
            )[np.ix_(interaction_indices, interaction_indices)]
            symmetric_covariance = (
                interaction_covariance + interaction_covariance.T
            ) / 2.0
            eigenvalues = np.linalg.eigvalsh(symmetric_covariance)
            tolerance = 1e-8 * max(
                1.0,
                float(np.max(np.abs(eigenvalues))),
            )
            if (
                len(interaction_indices) != len(basis_columns)
                or not np.isfinite(eigenvalues).all()
                or float(eigenvalues.min()) < -tolerance
            ):
                raise RuntimeError(
                    "spline interaction covariance was not positive semidefinite"
                )
        except (RuntimeError, np.linalg.LinAlgError) as exc:
            failures.append({"outcome": outcome, "reason": str(exc)})
            LOGGER.warning("Spline temporal model omitted for %s: %s", outcome, exc)
            continue

        coefficients = _secondary_coefficient_table(
            result,
            outcome,
            spec,
            working,
        )
        coefficients.insert(0, "model", "condition_residualised_natural_cubic_spline")
        coefficients.insert(
            1,
            "spline_degrees_of_freedom",
            settings.spline_degrees_of_freedom,
        )
        coefficient_tables.append(coefficients)
        prediction_tables.append(
            _spline_progression_prediction_table(
                result,
                outcome,
                spec,
                working,
                basis_columns,
                prediction_lookup,
            )
        )

        test = _wald_test(
            result,
            interaction_indices,
            "joint group by residualised spline trial trajectory",
        )
        test_rows.append(
            {
                "outcome": outcome,
                "outcome_label": spec["label"],
                "outcome_family": spec["family"],
                "analysis_role": "post hoc exploratory spline sensitivity",
                "test": test["test"],
                "df": test["df"],
                "requested_constraints": test["requested_constraints"],
                "chi_square": test["chi_square"],
                "p_value": test["p_value"],
                "working_correlation": working,
                "spline_degrees_of_freedom": settings.spline_degrees_of_freedom,
                "independent_spline_components": len(basis_columns),
                "condition_adjustment": (
                    "spline basis residualised within cohort against factorial "
                    "condition and scheduled break segment; outcome model retains "
                    "the full group by condition structure"
                ),
                "terms": test["terms"],
            }
        )

    tests = pd.DataFrame(test_rows)
    if not tests.empty:
        tests = _holm_within(tests, ["outcome_family", "test"])
        tests["multiplicity_method"] = (
            "Holm across outcomes within family for the spline trajectory test"
        )
        tests = _add_global_holm(tests)
    return {
        "spline_temporal_coefficients": (
            pd.concat(coefficient_tables, ignore_index=True)
            if coefficient_tables
            else pd.DataFrame()
        ),
        "spline_temporal_predictions": (
            pd.concat(prediction_tables, ignore_index=True)
            if prediction_tables
            else pd.DataFrame()
        ),
        "spline_temporal_tests": tests,
        "spline_temporal_failures": pd.DataFrame(
            failures,
            columns=["outcome", "reason"],
        ),
    }
