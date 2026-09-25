"""Estimate how the eHMI effect changes with participant experience."""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from utils.config import (
    AnalysisSettings,
)
from utils.constants import (
    EHMI_LEARNING_OUTCOMES,
    GROUP_FIXED,
    GROUP_LABELS,
    GROUP_RANDOMISED,
    LOGGER,
    OUTCOME_SPECS,
)
from utils.models.helpers import (
    _add_global_holm,
    _average_prediction,
    _coefficient_table,
    _design_formula,
    _fit_gee_with_fallback,
    _fit_grouped_binomial,
    _holm_within,
    _secondary_coefficient_table,
)
from utils.models.progression import (
    _break_segment,
    _prepare_progression_frame,
)
from utils.models.sessions import (
    _delta_contrast,
)


def _ehmi_effect_components(
    result: Any,
    design_info: Any,
    ordering_group: str,
    prediction_values: dict[str, Any],
    link: str,
) -> tuple[float, np.ndarray]:
    """Yielding-trial eHMI-on minus eHMI-off effect at one progression value."""

    base = pd.MultiIndex.from_product(
        [[0, 1], [2.0, 4.0, 6.0, 8.0, 10.0]],
        names=["camera", "distPed_m"],
    ).to_frame(index=False)
    estimates: dict[int, tuple[float, np.ndarray]] = {}
    for ehmi in [0, 1]:
        grid = base.copy()
        grid["yielding"] = 1
        grid["eHMIOn"] = ehmi
        grid["ordering_group"] = ordering_group
        for column, value in prediction_values.items():
            grid[column] = value
        estimate, gradient, _ = _average_prediction(result, design_info, grid, link)
        estimates[ehmi] = (estimate, gradient)
    off_estimate, off_gradient = estimates[0]
    on_estimate, on_gradient = estimates[1]
    return on_estimate - off_estimate, on_gradient - off_gradient


def run_ehmi_learning_models(
    trials: pd.DataFrame, settings: AnalysisSettings
) -> dict[str, pd.DataFrame]:
    """Focused cue-learning check for primary unsafety and Q3 only.

    The first estimand is the change from trial 1 to trial 40 in the
    yielding-trial eHMI-on minus eHMI-off contrast. The sensitivity estimand
    replaces trial position with zero versus nine prior encounters with the
    yielding-plus-eHMI cue. Together they address acquisition of this particular
    conditional cue more directly than a generic trial-position slope.
    """

    frame_all = trials.sort_values(["participant_uid", "trial_number"]).copy()
    yielding_ehmi = (
        (pd.to_numeric(frame_all["yielding"], errors="coerce") == 1)
        & (pd.to_numeric(frame_all["eHMIOn"], errors="coerce") == 1)
    ).astype(int)
    frame_all["prior_yielding_ehmi_per5"] = (
        yielding_ehmi.groupby(frame_all["participant_uid"]).cumsum()
        - yielding_ehmi
    ) / 5.0
    frame_all = _prepare_progression_frame(frame_all)
    cue_trials = frame_all[
        (pd.to_numeric(frame_all["yielding"], errors="coerce") == 1)
        & (pd.to_numeric(frame_all["eHMIOn"], errors="coerce") == 1)
    ]
    # Use the largest exposure count observed on a current cue-present trial.
    # Counts of ten can occur only on later cue-absent trials and would require
    # extrapolating the eHMI-on contrast outside its observed support.
    maximum_prior_exposures = int(
        round(float(cue_trials["prior_yielding_ehmi_per5"].max()) * 5.0)
    )
    group_term = "C(ordering_group, Treatment(reference='randomised_order'))"
    trial_rhs = (
        "trial_scaled * C(yielding) * C(eHMIOn) + I(trial_scaled ** 2) "
        f"+ C(break_segment) + {group_term}:trial_scaled "
        f"+ {group_term}:trial_scaled:C(yielding) "
        f"+ {group_term}:trial_scaled:C(eHMIOn) "
        f"+ {group_term}:trial_scaled:C(yielding):C(eHMIOn) "
        f"+ {group_term}:I(trial_scaled ** 2)"
    )
    exposure_rhs = (
        "prior_yielding_ehmi_per5 * C(yielding) * C(eHMIOn) "
        f"+ {group_term}:prior_yielding_ehmi_per5 "
        f"+ {group_term}:prior_yielding_ehmi_per5:C(yielding) "
        f"+ {group_term}:prior_yielding_ehmi_per5:C(eHMIOn) "
        f"+ {group_term}:prior_yielding_ehmi_per5:C(yielding):C(eHMIOn)"
    )
    progression_specs = [
        {
            "name": "trial_position",
            "model_name": "ehmi_cue_learning_by_trial_position",
            "rhs": trial_rhs,
            "required": ["trial_scaled", "break_segment"],
            "early_key": "trial_1",
            "late_key": "trial_40",
            "early_label": "Trial 1",
            "late_label": "Trial 40",
            "early_values": {
                "trial_scaled": (1.0 - 20.5) / 10.0,
                "break_segment": _break_segment(1.0),
            },
            "late_values": {
                "trial_scaled": (40.0 - 20.5) / 10.0,
                "break_segment": _break_segment(40.0),
            },
        },
        {
            "name": "prior_yielding_ehmi_exposure",
            "model_name": "ehmi_cue_learning_by_prior_exposure",
            "rhs": exposure_rhs,
            "required": ["prior_yielding_ehmi_per5"],
            "early_key": "zero_prior_exposures",
            "late_key": "maximum_prior_exposures",
            "early_label": "0 prior yielding-eHMI encounters",
            "late_label": f"{maximum_prior_exposures} prior yielding-eHMI encounters",
            "early_values": {"prior_yielding_ehmi_per5": 0.0},
            "late_values": {
                "prior_yielding_ehmi_per5": maximum_prior_exposures / 5.0
            },
        },
    ]
    coefficient_tables: list[pd.DataFrame] = []
    effect_rows: list[dict[str, Any]] = []
    change_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for progression in progression_specs:
        for outcome in EHMI_LEARNING_OUTCOMES:
            spec = OUTCOME_SPECS[outcome]
            frame = frame_all.dropna(
                subset=[outcome, "participant_uid", *progression["required"]]
            ).copy()
            try:
                if outcome == "unsafe_pct":
                    result, design_info, frame, _ = _fit_grouped_binomial(
                        frame,
                        settings.primary_threshold,
                        False,
                        additional_rhs=progression["rhs"],
                    )
                    coefficients = _coefficient_table(
                        result, progression["model_name"], settings.primary_threshold
                    )
                    coefficients.insert(0, "outcome", outcome)
                    working = "participant-clustered sandwich GLM"
                    link = "logit"
                else:
                    result, working = _fit_gee_with_fallback(
                        f"{outcome} ~ {_design_formula(False)} + {progression['rhs']}",
                        frame,
                        spec["model_family"],
                    )
                    design_info = result.model.data.design_info
                    coefficients = _secondary_coefficient_table(
                        result, outcome, spec, working
                    )
                    coefficients.insert(0, "model", progression["model_name"])
                    link = "identity"
                coefficients.insert(0, "progression_metric", progression["name"])
                coefficient_tables.append(coefficients)
            except Exception as exc:
                failures.append(
                    {
                        "progression_metric": progression["name"],
                        "outcome": outcome,
                        "reason": str(exc),
                    }
                )
                LOGGER.warning(
                    "eHMI cue-learning model omitted for %s (%s): %s",
                    outcome,
                    progression["name"],
                    exc,
                )
                continue

            scale = 100.0 if outcome == "unsafe_pct" else float(spec["scale"])
            components: dict[tuple[str, str], tuple[float, np.ndarray]] = {}
            for ordering_group in [GROUP_RANDOMISED, GROUP_FIXED]:
                for endpoint in ["early", "late"]:
                    key = progression[f"{endpoint}_key"]
                    effect, gradient = _ehmi_effect_components(
                        result,
                        design_info,
                        ordering_group,
                        progression[f"{endpoint}_values"],
                        link,
                    )
                    components[(ordering_group, key)] = (effect, gradient)
                    statistics = _delta_contrast(result, effect, gradient, scale)
                    effect_rows.append(
                        {
                            "progression_metric": progression["name"],
                            "progression_value": key,
                            "progression_value_label": progression[f"{endpoint}_label"],
                            "outcome": outcome,
                            "outcome_label": spec["label"],
                            "outcome_family": spec["family"],
                            "analysis_role": "focused exploratory cue-learning check",
                            "ordering_group": ordering_group,
                            "ordering_group_label": GROUP_LABELS[ordering_group],
                            "contrast": "eHMI on minus off within yielding trials",
                            **statistics,
                            "units": spec["units"],
                            "working_correlation": working,
                            "standardisation": "equal weighting over relative order and distance",
                        }
                    )

            changes: dict[str, tuple[float, np.ndarray]] = {}
            for ordering_group in [GROUP_RANDOMISED, GROUP_FIXED]:
                late_effect, late_gradient = components[
                    (ordering_group, progression["late_key"])
                ]
                early_effect, early_gradient = components[
                    (ordering_group, progression["early_key"])
                ]
                change = late_effect - early_effect
                gradient = late_gradient - early_gradient
                changes[ordering_group] = (change, gradient)
                statistics = _delta_contrast(result, change, gradient, scale)
                change_rows.append(
                    {
                        "progression_metric": progression["name"],
                        "comparison_scope": f"within_{ordering_group}",
                        "comparison": (
                            f"{progression['late_label']} minus {progression['early_label']} "
                            f"eHMI effect: {ordering_group}"
                        ),
                        "outcome": outcome,
                        "outcome_label": spec["label"],
                        "outcome_family": spec["family"],
                        "analysis_role": "focused exploratory cue-learning check",
                        "ordering_group": ordering_group,
                        "ordering_group_label": GROUP_LABELS[ordering_group],
                        **statistics,
                        "units": spec["units"],
                        "working_correlation": working,
                    }
                )

            fixed_change, fixed_gradient = changes[GROUP_FIXED]
            random_change, random_gradient = changes[GROUP_RANDOMISED]
            statistics = _delta_contrast(
                result,
                fixed_change - random_change,
                fixed_gradient - random_gradient,
                scale,
            )
            change_rows.append(
                {
                    "progression_metric": progression["name"],
                    "comparison_scope": "between_groups",
                    "comparison": "fixed minus randomised difference in cue-learning change",
                    "outcome": outcome,
                    "outcome_label": spec["label"],
                    "outcome_family": spec["family"],
                    "analysis_role": "focused exploratory cue-learning check",
                    "ordering_group": "between_groups",
                    "ordering_group_label": "Fixed minus randomised",
                    **statistics,
                    "units": spec["units"],
                    "working_correlation": working,
                }
            )

    changes = pd.DataFrame(change_rows)
    if not changes.empty:
        changes = _holm_within(
            changes, ["progression_metric", "comparison_scope"]
        )
        changes["multiplicity_method"] = (
            "Holm across the two focused outcomes separately for each cue-learning comparison"
        )
        changes = _add_global_holm(changes)
    return {
        "ehmi_learning_coefficients": (
            pd.concat(coefficient_tables, ignore_index=True, sort=False)
            if coefficient_tables
            else pd.DataFrame()
        ),
        "ehmi_learning_effects": pd.DataFrame(effect_rows),
        "ehmi_learning_change_tests": changes,
        "ehmi_learning_failures": pd.DataFrame(
            failures, columns=["progression_metric", "outcome", "reason"]
        ),
    }
