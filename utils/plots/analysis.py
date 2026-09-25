"""Generate all analysis plots through small, purpose-specific functions."""

from __future__ import annotations

import pandas as pd
from utils.config import StudyConfig
from utils.plots.outcomes import plot_primary_marginal_estimates
from utils.plots.outcomes import plot_participant_distributions
from utils.plots.sequence import plot_trial_position_profiles
from utils.plots.contrasts import plot_condition_effect_differences
from utils.plots.sequence import plot_sequence_composition
from utils.plots.outcomes import plot_questionnaire_ratings
from utils.plots.outcomes import plot_head_heading_at_passage
from utils.plots.outcomes import plot_trigger_activation_and_magnitude
from utils.plots.outcomes import plot_head_movement_outcomes
from utils.plots.temporal import plot_condition_adjusted_trial_profiles
from utils.plots.temporal import plot_adjusted_temporal_effects
from utils.plots.temporal import plot_prior_exposure_interactions
from utils.plots.outcomes import plot_event_defined_trigger_outcomes
from utils.plots.temporal import plot_adjusted_session_segments
from utils.plots.temporal import plot_ehmi_cue_learning_contrasts
from utils.plots.temporal import plot_spline_trial_trajectories
from utils.plots.robustness import plot_primary_robustness
from utils.plots.contrasts import plot_condition_cell_followup_heatmap


def create_figures(
    trials: pd.DataFrame,
    participant_means: pd.DataFrame,
    marginal_estimates: pd.DataFrame,
    condition_effects: pd.DataFrame,
    condition_cell_contrasts: pd.DataFrame,
    condition_cell_summary: pd.DataFrame,
    temporal_predictions: pd.DataFrame,
    temporal_tests: pd.DataFrame,
    secondary_marginal_estimates: pd.DataFrame,
    session_segment_predictions: pd.DataFrame,
    session_segment_tests: pd.DataFrame,
    ehmi_learning_effects: pd.DataFrame,
    ehmi_learning_change_tests: pd.DataFrame,
    exposure_tests: pd.DataFrame,
    config: StudyConfig,
    spline_predictions: pd.DataFrame | None = None,
    cluster_bootstrap_replicates: pd.DataFrame | None = None,
    leave_one_out: pd.DataFrame | None = None,
    bayesian_bootstrap_draws: pd.DataFrame | None = None,
) -> None:
    """Export the analysis figures; optional models produce plots when available."""
    config.figures.mkdir(parents=True, exist_ok=True)
    plot_primary_marginal_estimates(
        marginal_estimates=marginal_estimates,
        config=config,
    )
    plot_participant_distributions(
        participant_means=participant_means,
        config=config,
    )
    plot_trial_position_profiles(
        trials=trials,
        config=config,
    )
    plot_condition_effect_differences(
        condition_effects=condition_effects,
        config=config,
    )
    plot_sequence_composition(
        trials=trials,
        config=config,
    )
    plot_questionnaire_ratings(
        participant_means=participant_means,
        config=config,
    )
    plot_head_heading_at_passage(
        participant_means=participant_means,
        config=config,
    )
    plot_trigger_activation_and_magnitude(
        participant_means=participant_means,
        config=config,
    )
    plot_head_movement_outcomes(
        participant_means=participant_means,
        config=config,
    )
    plot_condition_adjusted_trial_profiles(
        temporal_predictions=temporal_predictions,
        config=config,
    )
    plot_adjusted_temporal_effects(
        temporal_tests=temporal_tests,
        config=config,
    )
    plot_prior_exposure_interactions(
        exposure_tests=exposure_tests,
        config=config,
    )
    plot_event_defined_trigger_outcomes(
        secondary_marginal_estimates=secondary_marginal_estimates,
        config=config,
    )
    plot_adjusted_session_segments(
        session_segment_predictions=session_segment_predictions,
        config=config,
    )
    plot_ehmi_cue_learning_contrasts(
        ehmi_learning_change_tests=ehmi_learning_change_tests,
        config=config,
    )
    plot_spline_trial_trajectories(
        config=config,
        spline_predictions=spline_predictions,
    )
    plot_primary_robustness(
        config=config,
        cluster_bootstrap_replicates=cluster_bootstrap_replicates,
        leave_one_out=leave_one_out,
        bayesian_bootstrap_draws=bayesian_bootstrap_draws,
    )
    plot_condition_cell_followup_heatmap(
        condition_cell_contrasts=condition_cell_contrasts,
        condition_cell_summary=condition_cell_summary,
        config=config,
    )
