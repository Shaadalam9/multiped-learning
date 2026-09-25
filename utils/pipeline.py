"""Coordinate raw-data extraction, statistical models and result exports."""

from __future__ import annotations

import pandas as pd
from utils.config import (
    StudyConfig,
)
from utils.constants import (
    GROUP_FIXED,
    GROUP_RANDOMISED,
    LOGGER,
)
from utils.data_io import (
    apply_constant_passage_times,
    attach_passage_timing,
    load_mapping,
)
from utils.extraction import (
    duplicate_sequence_audit,
    extract_ordering_group,
    mapping_audit_table,
    sample_flow_table,
    validate_sample,
)
from utils.models.demographics import (
    baseline_demographic_tables,
    demographic_adjusted_participant_outcomes,
    demographic_adjusted_sensitivity,
    load_demographics,
)
from utils.models.descriptive import (
    carryover_analysis,
    sequence_position_audit,
    temporal_descriptives,
)
from utils.models.exposure import (
    run_prior_exposure_models,
)
from utils.models.learning import (
    run_ehmi_learning_models,
)
from utils.models.participants import (
    participant_level_analysis,
)
from utils.models.primary import (
    run_primary_models,
)
from utils.models.progression import (
    run_adjusted_temporal_models,
)
from utils.models.robustness import (
    run_primary_robustness_analyses,
)
from utils.models.secondary import (
    assemble_condition_cell_followups,
    run_secondary_gee,
)
from utils.models.sessions import (
    run_session_segment_models,
)
from utils.models.splines import (
    run_spline_temporal_models,
)
from utils.plots import (
    regenerate_all_figures,
)
from utils.reporting import (
    outcome_dictionary,
    outcome_quality_audit,
    questionnaire_descriptives,
    trigger_event_flow,
    write_and_emit_results_log,
    write_extraction_diagnostics,
    write_method_note,
    write_tables,
)
from utils.snapshot import (
    save_result_snapshot,
)


def run_analysis(config: StudyConfig) -> dict[str, pd.DataFrame]:
    """Run the full prespecified comparison and return every output table."""

    config.output.mkdir(parents=True, exist_ok=True)
    config.figures.mkdir(parents=True, exist_ok=True)
    fixed_mapping = load_mapping(config.mapping)
    timing_mapping_path = config.timing_mapping or config.mapping
    # Constant passage times per vehicle behaviour (and distance), not the
    # scattered per-condition simulator log.
    timing_mapping = apply_constant_passage_times(load_mapping(timing_mapping_path))
    try:
        attach_passage_timing(timing_mapping, timing_mapping)
    except ValueError as exc:
        raise ValueError(
            f"Invalid common passage timing mapping {timing_mapping_path}: {exc}. "
            "Use the repository mapping.csv containing cross_p1_time_s and "
            "cross_p2_time_s, not the condition-only Data_unshuffled/mapping.csv."
        ) from exc
    LOGGER.info(
        "Using common passage timing mapping %s",
        timing_mapping_path,
    )
    LOGGER.info("Extracting randomised order group from %s", config.randomised_data)
    randomised, audit_randomised, sequences_randomised = extract_ordering_group(
        config.randomised_data,
        GROUP_RANDOMISED,
        config.settings,
        participant_mapping_filename=config.randomised_mapping_filename,
        timing_mapping=timing_mapping,
        timing_mapping_path=timing_mapping_path,
    )
    LOGGER.info(
        "Extracting fixed sequence group from %s using shared mapping %s",
        config.fixed_data,
        config.mapping,
    )
    fixed, audit_fixed, sequences_fixed = extract_ordering_group(
        config.fixed_data,
        GROUP_FIXED,
        config.settings,
        shared_mapping=fixed_mapping,
        shared_mapping_path=config.mapping,
        timing_mapping=timing_mapping,
        timing_mapping_path=timing_mapping_path,
    )
    trials = pd.concat([randomised, fixed], ignore_index=True)
    audit = pd.concat([audit_randomised, audit_fixed], ignore_index=True)
    sequences = pd.concat([sequences_randomised, sequences_fixed], ignore_index=True)
    write_extraction_diagnostics(audit, sequences, config.output)
    trials, warnings_out = validate_sample(trials, audit, sequences, config.settings)

    primary = run_primary_models(trials, config.settings)
    dispersion = pd.to_numeric(
        primary["binomial_model_diagnostics"]["pearson_dispersion"], errors="coerce"
    )
    if dispersion.notna().any() and float(dispersion.max()) > 2.0:
        warnings_out.append(
            "The grouped-binomial Pearson dispersion exceeded 2. Cluster-robust "
            "inference and independent-participant sensitivity estimates must be "
            "reported; model-based binomial standard errors must not be used."
        )
    secondary = run_secondary_gee(trials)
    condition_cells = assemble_condition_cell_followups(
        primary,
        secondary,
        config.settings.primary_threshold,
    )
    for internal_name in [
        "primary_condition_cell_estimates",
        "primary_condition_cell_contrasts",
    ]:
        primary.pop(internal_name, None)
    for internal_name in [
        "secondary_condition_cell_estimates",
        "secondary_condition_cell_contrasts",
    ]:
        secondary.pop(internal_name, None)
    temporal_adjusted = run_adjusted_temporal_models(trials)
    spline_temporal = run_spline_temporal_models(trials, config.settings)
    session_segments = run_session_segment_models(trials)
    ehmi_learning = run_ehmi_learning_models(trials, config.settings)
    exposure_adjusted = run_prior_exposure_models(trials)
    participant_means, participant_desc, participant_tests = participant_level_analysis(trials)
    primary_robustness = run_primary_robustness_analyses(
        trials,
        sequences,
        config.settings,
    )
    drift_participant, drift_tests = temporal_descriptives(trials)
    carryover_participant, carryover_tests = carryover_analysis(trials)
    demographics, demographic_linkage = load_demographics(config, trials)
    demographic_descriptives, demographic_comparisons = baseline_demographic_tables(
        demographics, trials
    )
    demographic_sensitivity = demographic_adjusted_sensitivity(
        trials, demographics, config.settings
    )
    adjusted_participant_outcomes = demographic_adjusted_participant_outcomes(
        participant_means, demographics
    )

    tables: dict[str, pd.DataFrame] = {
        "trial_level_common_window": trials,
        "exclusion_audit": audit,
        "sequence_audit": sequences,
        "mapping_audit": mapping_audit_table(sequences),
        "duplicate_sequence_audit": duplicate_sequence_audit(sequences),
        "sample_flow": sample_flow_table(trials, audit, sequences),
        "trigger_event_flow": trigger_event_flow(trials),
        "questionnaire_file_audit": questionnaire_descriptives(config),
        "demographic_linkage_audit": demographic_linkage,
        "baseline_demographic_descriptives": demographic_descriptives,
        "baseline_demographic_comparisons": demographic_comparisons,
        "outcome_dictionary": outcome_dictionary(),
        "outcome_quality_audit": outcome_quality_audit(trials),
        "participant_level_means": participant_means,
        "participant_level_descriptives": participant_desc,
        "participant_level_comparisons": participant_tests,
        "participant_temporal_metrics": drift_participant,
        "temporal_descriptive_comparisons": drift_tests,
        "participant_carryover_metrics": carryover_participant,
        "carryover_comparisons": carryover_tests,
        "sequence_position_associations": sequence_position_audit(trials),
        "demographic_adjusted_participant_outcomes": adjusted_participant_outcomes,
        **primary,
        **secondary,
        **condition_cells,
        **temporal_adjusted,
        **spline_temporal,
        **session_segments,
        **ehmi_learning,
        **exposure_adjusted,
        **demographic_sensitivity,
        **primary_robustness,
    }
    write_tables(tables, config.output)
    save_result_snapshot(tables, config, "raw-data analysis")
    regenerate_all_figures(tables, config)
    write_method_note(config, warnings_out)
    write_and_emit_results_log(tables, config.output, warnings_out)
    LOGGER.info("Analysis complete. Outputs written to %s", config.output)
    return tables
