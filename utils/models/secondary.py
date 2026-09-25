"""Secondary outcome models and combined individual-condition follow-up tables."""

from __future__ import annotations

from typing import Any

from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.cov_struct import Independence
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
from utils.constants import (
    LOGGER,
    OUTCOME_SPECS,
    SECONDARY_MODEL_OUTCOMES,
)
from utils.models.helpers import (
    _add_global_holm,
    _design_formula,
    _holm_within,
    _secondary_coefficient_table,
    _secondary_marginal_ordering_contrast,
    _wald_test,
)
from utils.models.primary import (
    condition_cell_group_contrasts,
)


def run_secondary_gee(trials: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Fit participant-clustered GEE models and paper-ready marginal contrasts."""

    omnibus_records: list[dict[str, Any]] = []
    coefficient_tables: list[pd.DataFrame] = []
    marginal_estimate_tables: list[pd.DataFrame] = []
    marginal_contrast_tables: list[pd.DataFrame] = []
    condition_cell_estimate_tables: list[pd.DataFrame] = []
    condition_cell_contrast_tables: list[pd.DataFrame] = []
    failure_records: list[dict[str, Any]] = []
    rhs = _design_formula(False)
    for outcome in SECONDARY_MODEL_OUTCOMES:
        spec = OUTCOME_SPECS[outcome]
        if outcome not in trials.columns:
            failure_records.append(
                {"outcome": outcome, "reason": "Outcome column was not available"}
            )
            continue
        frame = trials.dropna(subset=[outcome, "participant_uid"]).copy()
        if (
            frame.empty
            or frame["participant_uid"].nunique() < 3
            or frame[outcome].nunique() < 2
        ):
            LOGGER.warning("Skipping secondary GEE for %s because variation is insufficient", outcome)
            failure_records.append(
                {"outcome": outcome, "reason": "Insufficient observations or variation"}
            )
            continue
        result = None
        failure_messages: list[str] = []
        working_structure = "exchangeable"
        for structure_name, structure in [
            ("exchangeable", Exchangeable()),
            ("independence fallback", Independence()),
        ]:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", PerfectSeparationWarning)
                    warnings.simplefilter("ignore", RuntimeWarning)
                    model = smf.gee(
                        f"{outcome} ~ {rhs}",
                        groups="participant_uid",
                        data=frame,
                        family=(
                            sm.families.Binomial()
                            if spec["model_family"] == "binomial"
                            else sm.families.Gaussian()
                        ),
                        cov_struct=structure,
                    )
                    candidate = model.fit()
                if not np.isfinite(np.asarray(candidate.params, dtype=float)).all():
                    raise RuntimeError("nonfinite coefficient estimate")
                if not np.isfinite(np.asarray(candidate.cov_params(), dtype=float)).all():
                    raise RuntimeError("nonfinite robust covariance")
                result = candidate
                working_structure = structure_name
                break
            except Exception as exc:
                failure_messages.append(f"{structure_name}: {exc}")
        if result is None:
            failure_records.append(
                {"outcome": outcome, "reason": " | ".join(failure_messages)}
            )
            LOGGER.warning("Secondary GEE omitted for %s: %s", outcome, failure_messages)
            continue

        design_info = result.model.data.design_info
        coefficient_tables.append(
            _secondary_coefficient_table(result, outcome, spec, working_structure)
        )
        marginal_estimates, marginal_contrast = _secondary_marginal_ordering_contrast(
            result,
            design_info,
            outcome,
            spec,
            working_structure,
        )
        marginal_estimate_tables.append(marginal_estimates)
        marginal_contrast_tables.append(marginal_contrast)
        cell_estimates, cell_contrasts = condition_cell_group_contrasts(
            result,
            design_info,
            outcome,
            spec,
            working_structure,
            "secondary participant clustered GEE",
        )
        condition_cell_estimate_tables.append(cell_estimates)
        condition_cell_contrast_tables.append(cell_contrasts)

        names = list(pd.Series(result.params).index.astype(str))
        group_indices = [index for index, name in enumerate(names) if "C(ordering_group" in name]
        tests = [
            _wald_test(result, group_indices, "all ordering group terms"),
            _wald_test(
                result,
                [index for index in group_indices if ":" in names[index]],
                "ordering group by condition interactions",
            ),
        ]
        for row in tests:
            row["outcome"] = outcome
            row["outcome_label"] = spec["label"]
            row["outcome_family"] = spec["family"]
            row["analysis_role"] = spec["role"]
            row["model_family"] = spec["model_family"]
            row["working_correlation"] = working_structure
            row["failure_reason"] = ""
            omnibus_records.append(row)

    omnibus = _holm_within(
        pd.DataFrame(omnibus_records), ["outcome_family", "test"]
    )
    if not omnibus.empty:
        omnibus["multiplicity_method"] = (
            "Holm across outcomes within family, separately for each omnibus test"
        )
        omnibus = _add_global_holm(omnibus, ["test"])
    marginal_contrasts = _holm_within(
        pd.concat(marginal_contrast_tables, ignore_index=True)
        if marginal_contrast_tables
        else pd.DataFrame(),
        ["outcome_family"],
    )
    if not marginal_contrasts.empty:
        marginal_contrasts["multiplicity_method"] = (
            "Holm across marginal group contrasts within outcome family"
        )
        marginal_contrasts = _add_global_holm(marginal_contrasts)
    return {
        "secondary_gee_omnibus": omnibus,
        "secondary_gee_coefficients": (
            pd.concat(coefficient_tables, ignore_index=True)
            if coefficient_tables
            else pd.DataFrame()
        ),
        "secondary_marginal_estimates": (
            pd.concat(marginal_estimate_tables, ignore_index=True)
            if marginal_estimate_tables
            else pd.DataFrame()
        ),
        "secondary_marginal_contrasts": marginal_contrasts,
        "secondary_condition_cell_estimates": (
            pd.concat(condition_cell_estimate_tables, ignore_index=True)
            if condition_cell_estimate_tables
            else pd.DataFrame()
        ),
        "secondary_condition_cell_contrasts": (
            pd.concat(condition_cell_contrast_tables, ignore_index=True)
            if condition_cell_contrast_tables
            else pd.DataFrame()
        ),
        "secondary_model_failures": pd.DataFrame(
            failure_records, columns=["outcome", "reason"]
        ),
    }


def assemble_condition_cell_followups(
    primary: dict[str, pd.DataFrame],
    secondary: dict[str, pd.DataFrame],
    primary_threshold: float,
) -> dict[str, pd.DataFrame]:
    """Combine condition cell estimates and create an interpretable summary."""

    estimates = pd.concat(
        [
            primary["primary_condition_cell_estimates"],
            secondary["secondary_condition_cell_estimates"],
        ],
        ignore_index=True,
    )
    contrasts = pd.concat(
        [
            primary["primary_condition_cell_contrasts"],
            secondary["secondary_condition_cell_contrasts"],
        ],
        ignore_index=True,
    )
    contrasts = _add_global_holm(
        contrasts,
        output_column="p_value_holm_across_all_cells_and_outcomes",
    )
    contrasts["multiplicity_method_global"] = (
        "Holm across every condition cell follow up comparison and outcome"
    )

    primary_omnibus = primary["primary_omnibus_tests"]
    primary_omnibus = primary_omnibus[
        np.isclose(primary_omnibus["threshold"], primary_threshold)
        & (
            primary_omnibus["test"]
            == "ordering group by condition interactions"
        )
    ]
    secondary_omnibus = secondary["secondary_gee_omnibus"]
    secondary_omnibus = secondary_omnibus[
        secondary_omnibus["test"]
        == "ordering group by condition interactions"
    ]

    summary_rows: list[dict[str, Any]] = []
    for outcome, outcome_cells in contrasts.groupby(
        "outcome",
        observed=True,
        sort=False,
    ):
        if outcome == "unsafe_pct":
            omnibus = primary_omnibus
            omnibus_adjusted = np.nan
            omnibus_global = np.nan
            omnibus_scope = (
                "Primary outcome at the prespecified 0.10 threshold"
            )
        else:
            omnibus = secondary_omnibus[
                secondary_omnibus["outcome"] == outcome
            ]
            omnibus_adjusted = (
                float(omnibus["p_value_adjusted"].iloc[0])
                if not omnibus.empty
                else np.nan
            )
            omnibus_global = (
                float(omnibus["p_value_adjusted_global"].iloc[0])
                if not omnibus.empty
                else np.nan
            )
            omnibus_scope = (
                "Secondary or exploratory outcome with family and global Holm correction"
            )
        omnibus_p_value = (
            float(omnibus["p_value"].iloc[0])
            if not omnibus.empty
            else np.nan
        )
        omnibus_selected = bool(
            np.isfinite(omnibus_p_value)
            and (
                (
                    outcome == "unsafe_pct"
                    and omnibus_p_value < 0.05
                )
                or (
                    outcome != "unsafe_pct"
                    and np.isfinite(omnibus_global)
                    and omnibus_global < 0.05
                )
            )
        )

        absolute_difference = pd.to_numeric(
            outcome_cells["difference_fixed_minus_randomised"],
            errors="coerce",
        ).abs()
        largest = outcome_cells.loc[absolute_difference.idxmax()]
        within_significant = (
            pd.to_numeric(
                outcome_cells["p_value_holm_within_outcome"],
                errors="coerce",
            )
            < 0.05
        )
        global_significant = (
            pd.to_numeric(
                outcome_cells[
                    "p_value_holm_across_all_cells_and_outcomes"
                ],
                errors="coerce",
            )
            < 0.05
        )
        simultaneous_exclusion = outcome_cells[
            "simultaneous_ci_excludes_zero"
        ].fillna(False)
        summary_rows.append(
            {
                "outcome": outcome,
                "outcome_label": largest["outcome_label"],
                "outcome_family": largest["outcome_family"],
                "analysis_role": largest["analysis_role"],
                "units": largest["units"],
                "omnibus_interaction_p_value": omnibus_p_value,
                "omnibus_interaction_p_value_adjusted_within_family": (
                    omnibus_adjusted
                ),
                "omnibus_interaction_p_value_adjusted_global": omnibus_global,
                "omnibus_multiplicity_scope": omnibus_scope,
                "omnibus_selected_for_condition_followup_figure": (
                    omnibus_selected
                ),
                "condition_cells": int(len(outcome_cells)),
                "cells_holm_significant_within_outcome": int(
                    within_significant.sum()
                ),
                "cells_holm_significant_across_all_cells_and_outcomes": int(
                    global_significant.sum()
                ),
                "cells_with_simultaneous_ci_excluding_zero": int(
                    simultaneous_exclusion.sum()
                ),
                "largest_absolute_difference_condition_id": largest[
                    "condition_id"
                ],
                "largest_absolute_difference_condition": largest[
                    "condition_label"
                ],
                "largest_difference_fixed_minus_randomised": largest[
                    "difference_fixed_minus_randomised"
                ],
                "largest_difference_simultaneous_ci_low": largest[
                    "simultaneous_ci_low"
                ],
                "largest_difference_simultaneous_ci_high": largest[
                    "simultaneous_ci_high"
                ],
                "largest_difference_p_value_holm_within_outcome": largest[
                    "p_value_holm_within_outcome"
                ],
                "largest_difference_p_value_holm_global": largest[
                    "p_value_holm_across_all_cells_and_outcomes"
                ],
                "interpretation": (
                    "Omnibus interaction retained for structured follow up; "
                    "individual cells require simultaneous interval or adjusted p value"
                    if omnibus_selected
                    else
                    "No multiplicity controlled omnibus interaction; cell estimates are descriptive"
                ),
            }
        )

    return {
        "condition_cell_estimates": estimates,
        "condition_cell_contrasts": contrasts,
        "condition_cell_followup_summary": pd.DataFrame(summary_rows),
    }
