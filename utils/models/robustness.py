"""Bootstrap, exclusion and precision analyses for the primary outcome."""

from __future__ import annotations

from typing import Any
import math
import numpy as np
import pandas as pd
import scipy.stats as st
from utils.config import (
    AnalysisSettings,
)
from utils.constants import (
    GROUP_FIXED,
    GROUP_RANDOMISED,
)
from utils.models.participants import (
    _welch_difference,
)


def _primary_participant_values(
    trials: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Return one primary outcome mean per independent participant."""

    means = (
        trials.groupby(
            ["ordering_group", "participant_uid"],
            observed=True,
        )["unsafe_pct"]
        .mean()
        .reset_index()
    )
    randomised = means.loc[
        means["ordering_group"] == GROUP_RANDOMISED,
        "unsafe_pct",
    ].dropna().to_numpy(float)
    fixed = means.loc[
        means["ordering_group"] == GROUP_FIXED,
        "unsafe_pct",
    ].dropna().to_numpy(float)
    if len(randomised) < 3 or len(fixed) < 3:
        raise RuntimeError(
            "At least three participants per cohort are required for primary "
            "participant-cluster robustness analyses"
        )
    return means, randomised, fixed


def _difference_after_exclusions(
    participant_means: pd.DataFrame,
    excluded_participants: set[str],
    scenario: str,
    exclusion_reason: str,
) -> dict[str, Any]:
    retained = participant_means[
        ~participant_means["participant_uid"].astype(str).isin(excluded_participants)
    ]
    randomised = retained.loc[
        retained["ordering_group"] == GROUP_RANDOMISED,
        "unsafe_pct",
    ].dropna().to_numpy(float)
    fixed = retained.loc[
        retained["ordering_group"] == GROUP_FIXED,
        "unsafe_pct",
    ].dropna().to_numpy(float)
    if len(randomised) < 2 or len(fixed) < 2:
        return {
            "scenario": scenario,
            "exclusion_reason": exclusion_reason,
            "excluded_participants": " | ".join(sorted(excluded_participants)),
            "n_randomised": len(randomised),
            "n_fixed": len(fixed),
            "difference_fixed_minus_randomised": np.nan,
            "standard_error": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_value": np.nan,
        }
    result = _welch_difference(randomised, fixed)
    return {
        "scenario": scenario,
        "exclusion_reason": exclusion_reason,
        "excluded_participants": " | ".join(sorted(excluded_participants)),
        "n_randomised": len(randomised),
        "n_fixed": len(fixed),
        **result,
    }


def run_primary_robustness_analyses(
    trials: pd.DataFrame,
    sequences: pd.DataFrame,
    settings: AnalysisSettings,
) -> dict[str, pd.DataFrame]:
    """Participant-cluster bootstrap, influence, Bayesian, and precision checks."""

    participant_means, randomised, fixed = _primary_participant_values(trials)
    observed = float(np.mean(fixed) - np.mean(randomised))
    rng = np.random.default_rng(settings.bootstrap_seed)

    random_indices = rng.integers(
        0,
        len(randomised),
        size=(settings.bootstrap_replicates, len(randomised)),
    )
    fixed_indices = rng.integers(
        0,
        len(fixed),
        size=(settings.bootstrap_replicates, len(fixed)),
    )
    bootstrap_differences = (
        fixed[fixed_indices].mean(axis=1)
        - randomised[random_indices].mean(axis=1)
    )
    bootstrap_low, bootstrap_high = np.quantile(
        bootstrap_differences,
        [0.025, 0.975],
    )
    bootstrap_replicates = pd.DataFrame(
        {
            "replicate": np.arange(
                1,
                settings.bootstrap_replicates + 1,
                dtype=int,
            ),
            "difference_percentage_points": bootstrap_differences,
        }
    )
    bootstrap_summary = pd.DataFrame(
        [
            {
                "outcome": "unsafe_pct",
                "contrast": "fixed sequence minus randomised order",
                "observed_difference_percentage_points": observed,
                "bootstrap_standard_error_percentage_points": float(
                    np.std(bootstrap_differences, ddof=1)
                ),
                "percentile_ci_low": float(bootstrap_low),
                "percentile_ci_high": float(bootstrap_high),
                "participants_randomised": len(randomised),
                "participants_fixed": len(fixed),
                "bootstrap_replicates": settings.bootstrap_replicates,
                "seed": settings.bootstrap_seed,
                "resampling_unit": (
                    "participant, sampled independently within ordering cohort"
                ),
                "interpretation": (
                    "robustness confidence interval; not an additional "
                    "significance search"
                ),
            }
        ]
    )

    leave_one_out_rows: list[dict[str, Any]] = []
    for _, participant in participant_means.iterrows():
        participant_uid = str(participant["participant_uid"])
        retained = participant_means[
            participant_means["participant_uid"].astype(str) != participant_uid
        ]
        random_values = retained.loc[
            retained["ordering_group"] == GROUP_RANDOMISED,
            "unsafe_pct",
        ].dropna().to_numpy(float)
        fixed_values = retained.loc[
            retained["ordering_group"] == GROUP_FIXED,
            "unsafe_pct",
        ].dropna().to_numpy(float)
        difference = float(np.mean(fixed_values) - np.mean(random_values))
        leave_one_out_rows.append(
            {
                "omitted_participant_uid": participant_uid,
                "omitted_ordering_group": participant["ordering_group"],
                "difference_fixed_minus_randomised_percentage_points": difference,
                "change_from_full_sample_percentage_points": difference - observed,
                "sign_changed_from_full_sample": bool(
                    np.sign(difference) != np.sign(observed)
                ),
                "n_randomised": len(random_values),
                "n_fixed": len(fixed_values),
            }
        )
    leave_one_out = pd.DataFrame(leave_one_out_rows)
    leave_one_out_summary = pd.DataFrame(
        [
            {
                "outcome": "unsafe_pct",
                "full_sample_difference_percentage_points": observed,
                "minimum_leave_one_out_difference_percentage_points": float(
                    leave_one_out[
                        "difference_fixed_minus_randomised_percentage_points"
                    ].min()
                ),
                "maximum_leave_one_out_difference_percentage_points": float(
                    leave_one_out[
                        "difference_fixed_minus_randomised_percentage_points"
                    ].max()
                ),
                "maximum_absolute_change_percentage_points": float(
                    leave_one_out[
                        "change_from_full_sample_percentage_points"
                    ].abs().max()
                ),
                "number_of_sign_changes": int(
                    leave_one_out["sign_changed_from_full_sample"].sum()
                ),
                "participants_checked": len(leave_one_out),
            }
        ]
    )

    randomised_sequences = sequences[
        sequences["ordering_group"] == GROUP_RANDOMISED
    ].copy()
    duplicated_hashes = set(
        randomised_sequences.loc[
            randomised_sequences.duplicated("sequence_hash", keep=False),
            "sequence_hash",
        ].astype(str)
    )
    duplicate_participants = set(
        randomised_sequences.loc[
            randomised_sequences["sequence_hash"].astype(str).isin(
                duplicated_hashes
            ),
            "participant_uid",
        ].astype(str)
    )
    mismatch_participants = set(
        sequences.loc[
            ~sequences["mapping_order_matches_response"].fillna(False),
            "participant_uid",
        ].astype(str)
    )
    design_sensitivity = pd.DataFrame(
        [
            _difference_after_exclusions(
                participant_means,
                set(),
                "full sample",
                "none",
            ),
            _difference_after_exclusions(
                participant_means,
                duplicate_participants,
                "exclude randomised participants sharing a realised sequence",
                "duplicate randomised sequence",
            ),
            _difference_after_exclusions(
                participant_means,
                mismatch_participants,
                "exclude participants with mapping and response order mismatch",
                "mapping order mismatch",
            ),
            _difference_after_exclusions(
                participant_means,
                duplicate_participants | mismatch_participants,
                "exclude all sequence audit flags",
                "duplicate randomised sequence or mapping order mismatch",
            ),
        ]
    )

    bayesian_rng = np.random.default_rng(settings.bootstrap_seed + 1)
    random_weights = bayesian_rng.exponential(
        1.0,
        size=(settings.bayesian_bootstrap_draws, len(randomised)),
    )
    random_weights /= random_weights.sum(axis=1, keepdims=True)
    fixed_weights = bayesian_rng.exponential(
        1.0,
        size=(settings.bayesian_bootstrap_draws, len(fixed)),
    )
    fixed_weights /= fixed_weights.sum(axis=1, keepdims=True)
    bayesian_differences = (
        fixed_weights @ fixed
        - random_weights @ randomised
    )
    bayesian_low, bayesian_high = np.quantile(
        bayesian_differences,
        [0.025, 0.975],
    )
    bayesian_draws = pd.DataFrame(
        {
            "draw": np.arange(
                1,
                settings.bayesian_bootstrap_draws + 1,
                dtype=int,
            ),
            "difference_percentage_points": bayesian_differences,
        }
    )
    bayesian_summary = pd.DataFrame(
        [
            {
                "outcome": "unsafe_pct",
                "contrast": "fixed sequence minus randomised order",
                "posterior_mean_difference_percentage_points": float(
                    np.mean(bayesian_differences)
                ),
                "posterior_median_difference_percentage_points": float(
                    np.median(bayesian_differences)
                ),
                "credible_interval_low": float(bayesian_low),
                "credible_interval_high": float(bayesian_high),
                "posterior_probability_fixed_greater_than_randomised": float(
                    np.mean(bayesian_differences > 0)
                ),
                "posterior_probability_fixed_less_than_randomised": float(
                    np.mean(bayesian_differences < 0)
                ),
                "draws": settings.bayesian_bootstrap_draws,
                "seed": settings.bootstrap_seed + 1,
                "method": (
                    "participant-level Bayesian bootstrap with independent "
                    "Dirichlet weights within each cohort"
                ),
                "practical_effect_threshold": (
                    "not evaluated because no independently justified threshold "
                    "was supplied"
                ),
            }
        ]
    )

    welch = _welch_difference(randomised, fixed)
    current_standard_error = float(welch["standard_error"])
    current_total = len(randomised) + len(fixed)
    planning_rows: list[dict[str, Any]] = []
    alpha = 0.05
    z_alpha = float(st.norm.ppf(1.0 - alpha / 2.0))
    for power in settings.planning_power:
        z_power = float(st.norm.ppf(power))
        current_mde = (z_alpha + z_power) * current_standard_error
        planning_rows.append(
            {
                "scenario": "current design minimum detectable difference",
                "alpha_two_sided": alpha,
                "power": power,
                "target_difference_percentage_points": np.nan,
                "minimum_detectable_difference_percentage_points": current_mde,
                "current_total_participants": current_total,
                "required_total_participants": current_total,
                "required_participants_per_cohort": max(
                    len(randomised),
                    len(fixed),
                ),
                "assumptions": (
                    "normal approximation, participant is the independent unit, "
                    "equal cohort allocation, current participant-level variance"
                ),
            }
        )
        for target in settings.planning_effect_sizes_percentage_points:
            multiplier = (current_mde / target) ** 2
            required_total = max(
                current_total,
                int(math.ceil(current_total * multiplier)),
            )
            if required_total % 2:
                required_total += 1
            planning_rows.append(
                {
                    "scenario": "prospective sample required for target difference",
                    "alpha_two_sided": alpha,
                    "power": power,
                    "target_difference_percentage_points": target,
                    "minimum_detectable_difference_percentage_points": target,
                    "current_total_participants": current_total,
                    "required_total_participants": required_total,
                    "required_participants_per_cohort": required_total // 2,
                    "assumptions": (
                        "normal approximation, participant is the independent "
                        "unit, equal cohort allocation, current participant-level "
                        "variance; planning scenario is not an equivalence margin"
                    ),
                }
            )

    return {
        "primary_cluster_bootstrap_summary": bootstrap_summary,
        "primary_cluster_bootstrap_replicates": bootstrap_replicates,
        "primary_leave_one_participant_out": leave_one_out,
        "primary_leave_one_participant_out_summary": leave_one_out_summary,
        "primary_design_audit_sensitivity": design_sensitivity,
        "primary_bayesian_bootstrap_summary": bayesian_summary,
        "primary_bayesian_bootstrap_draws": bayesian_draws,
        "primary_precision_planning": pd.DataFrame(planning_rows),
    }
