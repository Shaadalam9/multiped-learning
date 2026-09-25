"""Shared outcome definitions, group labels and project paths."""

from __future__ import annotations

from typing import Any
from pathlib import Path
from custom_logger import logger
import re


SCRIPT_VERSION = "2026-09-24.3"


LOGGER = logger


DEFAULT_CONFIG_FILENAMES = ("config", "config.comparison.json")


VIDEO_PATTERN = re.compile(r"video_\d+", re.IGNORECASE)


PARTICIPANT_PATTERN = re.compile(r"participant[_-]?(\d+)", re.IGNORECASE)


COMMON_CONFIG_KEYS = (
    "unshuffled_mapping",
    "timing_mapping",
    "mapping",
    "shuffled_mapping_filename",
    "plotly_template",
    "output",
    "figures",
    "save_final",
    "auto_open",
    "shuffled_data",
    "shuffled_intake_questionnaire",
    "shuffled_post_experiment_questionnaire",
    "unshuffled_data",
    "unshuffled_intake_questionnaire",
    "unshuffled_post_experiment_questionnaire",
    "always_analyse",
    "logger_level",
    "kp_resolution",
    "yaw_resolution",
    "smoothen_signal",
    "freq",
    "mincutoff",
    "beta",
    "font_family",
    "font_size",
    "p_value",
    "window_seconds",
    "bin_seconds",
    "primary_threshold",
    "sensitivity_thresholds",
    "require_complete_window",
    "minimum_valid_trials_per_participant",
    "bootstrap_replicates",
    "bootstrap_seed",
    "bayesian_bootstrap_draws",
    "spline_degrees_of_freedom",
    "planning_effect_sizes_percentage_points",
    "planning_power",
)


GROUP_RANDOMISED = "randomised_order"


GROUP_FIXED = "fixed_sequence"


GROUP_LABELS = {
    GROUP_RANDOMISED: "Randomised order",
    GROUP_FIXED: "Fixed sequence",
}


OUTCOME_SPECS: dict[str, dict[str, Any]] = {
    "unsafe_pct": {
        "label": "Trigger-active bins",
        "units": "percentage points",
        "family": "primary_trigger",
        "role": "primary",
        "model_family": "gaussian",
        "scale": 1.0,
    },
    "Q1": {
        "label": "Rating of the other pedestrian's behaviour (Q1)",
        "units": "rating points",
        "family": "ratings",
        "role": "secondary",
        "model_family": "gaussian",
        "scale": 1.0,
    },
    "Q2": {
        "label": "Rating of the distance between pedestrians (Q2)",
        "units": "rating points",
        "family": "ratings",
        "role": "secondary",
        "model_family": "gaussian",
        "scale": 1.0,
    },
    "Q3": {
        "label": "Rating of the vehicle's intention (Q3)",
        "units": "rating points",
        "family": "ratings",
        "role": "secondary",
        "model_family": "gaussian",
        "scale": 1.0,
    },
    "mean_trigger": {
        "label": "Mean trigger value",
        "units": "trigger units (0 to 1)",
        "family": "trigger_secondary",
        "role": "secondary",
        "model_family": "gaussian",
        "scale": 1.0,
    },
    "peak_trigger": {
        "label": "Peak trigger value",
        "units": "trigger units (0 to 1)",
        "family": "trigger_secondary",
        "role": "exploratory",
        "model_family": "gaussian",
        "scale": 1.0,
    },
    "any_trigger_press": {
        "label": "Trials with any trigger activation",
        "units": "percentage points",
        "family": "trigger_secondary",
        "role": "secondary",
        "model_family": "binomial",
        "scale": 100.0,
    },
    "trigger_first_active_latency_s": {
        "label": "Time to first trigger-active bin",
        "units": "seconds from common-window start",
        "family": "trigger_event",
        "role": "exploratory",
        "model_family": "gaussian",
        "scale": 1.0,
    },
    "trigger_return_to_safe": {
        "label": "Return to safe after trigger activation",
        "units": "percentage points",
        "family": "trigger_event",
        "role": "exploratory",
        "model_family": "binomial",
        "scale": 100.0,
    },
    "trigger_first_return_latency_s": {
        "label": "Time to first return to safe",
        "units": "seconds from common-window start",
        "family": "trigger_event",
        "role": "exploratory",
        "model_family": "gaussian",
        "scale": 1.0,
    },
    "heading_at_pass_deg": {
        "label": "Horizontal head heading at passage",
        "units": "degrees",
        "family": "head_movement",
        "role": "secondary",
        "model_family": "gaussian",
        "scale": 1.0,
    },
    "minimum_heading_deg": {
        "label": "Minimum pre-passage head heading",
        "units": "degrees",
        "family": "head_movement",
        "role": "exploratory",
        "model_family": "gaussian",
        "scale": 1.0,
    },
    "passage_change_deg": {
        "label": "Head-heading change across passage",
        "units": "degrees",
        "family": "head_movement",
        "role": "exploratory",
        "model_family": "gaussian",
        "scale": 1.0,
    },
    "heading_common_window_sd_deg": {
        "label": "Pre-passage head-heading variability",
        "units": "degrees",
        "family": "head_movement",
        "role": "exploratory",
        "model_family": "gaussian",
        "scale": 1.0,
    },
    "heading_yaw_activity_deg_s": {
        "label": "Pre-passage head-yaw activity",
        "units": "degrees per second",
        "family": "head_movement",
        "role": "exploratory",
        "model_family": "gaussian",
        "scale": 1.0,
    },
    "far_to_pre_change_deg": {
        "label": "Far-to-immediate-pre head-heading change",
        "units": "degrees",
        "family": "head_movement_descriptive",
        "role": "exploratory",
        "model_family": "gaussian",
        "scale": 1.0,
    },
    "recovered_by_pass_deg": {
        "label": "Head-heading recovery by passage",
        "units": "degrees",
        "family": "head_movement_descriptive",
        "role": "exploratory",
        "model_family": "gaussian",
        "scale": 1.0,
    },
}


SECONDARY_MODEL_OUTCOMES = (
    "Q1",
    "Q2",
    "Q3",
    "mean_trigger",
    "peak_trigger",
    "any_trigger_press",
    "trigger_first_active_latency_s",
    "trigger_return_to_safe",
    "trigger_first_return_latency_s",
    "heading_at_pass_deg",
    "minimum_heading_deg",
    "passage_change_deg",
    "heading_common_window_sd_deg",
    "heading_yaw_activity_deg_s",
)


PARTICIPANT_OUTCOMES = (
    "unsafe_pct",
    *SECONDARY_MODEL_OUTCOMES,
    "far_to_pre_change_deg",
    "recovered_by_pass_deg",
)


TEMPORAL_MODEL_OUTCOMES = (
    "unsafe_pct",
    "mean_trigger",
    "any_trigger_press",
    "trigger_first_active_latency_s",
    "trigger_return_to_safe",
    "trigger_first_return_latency_s",
    "Q1",
    "Q2",
    "Q3",
    "heading_at_pass_deg",
    "minimum_heading_deg",
    "passage_change_deg",
    "heading_common_window_sd_deg",
    "heading_yaw_activity_deg_s",
)


EXPOSURE_MODEL_OUTCOMES = tuple(
    outcome
    for outcome in TEMPORAL_MODEL_OUTCOMES
    if OUTCOME_SPECS[outcome]["family"] != "trigger_event"
)


EHMI_LEARNING_OUTCOMES = ("unsafe_pct", "Q3")


EXPOSURE_FACTORS = (
    ("yielding", "AV behaviour"),
    ("eHMIOn", "eHMI status"),
    ("camera", "relative pedestrian order"),
    ("distPed_m", "pedestrian distance level"),
)


INTAKE_AGE = "What is your age (in years)?"


INTAKE_GENDER = "What is your gender?"


INTAKE_NATIONALITY = "What is your nationality?"


INTAKE_VR = "How often in the last month have you experienced virtual reality?"


SESSION_SEGMENTS = (
    ("before_break_14", "Trials 1-14"),
    ("after_break_14", "Trials 15-26"),
    ("after_break_26", "Trials 27-40"),
)


RESULTS_LOG_TABLES = (
    "sample_flow",
    "mapping_audit",
    "trigger_event_flow",
    "baseline_demographic_descriptives",
    "baseline_demographic_comparisons",
    "outcome_quality_audit",
    "primary_marginal_estimates",
    "primary_marginal_contrasts",
    "primary_omnibus_tests",
    "primary_condition_effect_contrasts",
    "condition_cell_followup_summary",
    "condition_cell_contrasts",
    "condition_cell_estimates",
    "primary_binomial_coefficients",
    "binomial_model_diagnostics",
    "secondary_marginal_estimates",
    "secondary_marginal_contrasts",
    "secondary_gee_omnibus",
    "secondary_gee_coefficients",
    "participant_level_descriptives",
    "participant_level_comparisons",
    "primary_cluster_bootstrap_summary",
    "primary_leave_one_participant_out_summary",
    "primary_design_audit_sensitivity",
    "primary_bayesian_bootstrap_summary",
    "primary_precision_planning",
    "temporal_omnibus_tests",
    "temporal_binomial_coefficients",
    "temporal_descriptive_comparisons",
    "temporal_adjusted_tests",
    "temporal_adjusted_coefficients",
    "spline_temporal_tests",
    "spline_temporal_coefficients",
    "session_segment_predictions",
    "session_segment_tests",
    "session_segment_coefficients",
    "ehmi_learning_effects",
    "ehmi_learning_change_tests",
    "ehmi_learning_coefficients",
    "exposure_adjusted_tests",
    "exposure_adjusted_coefficients",
    "carryover_comparisons",
    "demographic_adjusted_primary_contrast",
    "demographic_adjusted_coefficients",
    "demographic_adjusted_participant_outcomes",
    "model_failures",
    "secondary_model_failures",
    "temporal_adjusted_failures",
    "spline_temporal_failures",
    "session_segment_failures",
    "ehmi_learning_failures",
    "exposure_adjusted_failures",
)


CACHE_SCHEMA = 1


PROJECT_ROOT = Path(__file__).resolve().parent.parent
