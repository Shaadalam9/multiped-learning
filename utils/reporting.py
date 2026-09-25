"""Build descriptive audit tables and export numerical reports and analysis notes."""

from __future__ import annotations

from typing import Any

from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd
from utils.config import (
    StudyConfig,
)
from utils.constants import (
    GROUP_FIXED,
    GROUP_LABELS,
    GROUP_RANDOMISED,
    LOGGER,
    OUTCOME_SPECS,
    RESULTS_LOG_TABLES,
    SCRIPT_VERSION,
)
from utils.extraction import (
    mapping_audit_table,
)


def outcome_dictionary() -> pd.DataFrame:
    definitions = {
        "unsafe_pct": (
            'Percentage of valid 100 ms bins above the primary trigger threshold in the '
            'five-second pre-passage window.'
        ),
        "Q1": "Trial-level 0 to 100 rating of the behaviour of the other pedestrian.",
        "Q2": "Trial-level 0 to 100 rating of the distance between the pedestrians.",
        "Q3": "Trial-level 0 to 100 rating of the intention of the vehicle.",
        "mean_trigger": "Mean normalised analogue trigger value in the five-second pre-passage window.",
        "peak_trigger": "Maximum normalised analogue trigger value in the five-second pre-passage window.",
        "any_trigger_press": "Indicator that at least one 100 ms bin exceeded the primary trigger threshold.",
        "trigger_first_active_latency_s": (
            'Time from the start of the five-second common window to the first '
            'trigger-active 100 ms bin, defined only for trials with activation; zero is '
            'left-censored because a press may have begun before the window.'
        ),
        "trigger_return_to_safe": (
            'Among trials with trigger activation, indicator that a subsequent 100 ms '
            'bin returned below or equal to the primary threshold before passage.'
        ),
        "trigger_first_return_latency_s": (
            'Time from the common-window start to the first return-to-safe bin, defined '
            'only when activation was followed by a return before passage.'
        ),
        "heading_at_pass_deg": (
            'Mean baseline-corrected Unity horizontal head heading from 100 ms before to '
            '100 ms after participant passage.'
        ),
        "minimum_heading_deg": (
            'Minimum smoothed baseline-corrected horizontal head heading from 0.5 s '
            'after trial onset to 0.2 s before passage.'
        ),
        "passage_change_deg": "Mean heading in the first 0.5 s after passage minus the final 0.5 s before passage.",
        "heading_common_window_sd_deg": (
            'Within-trial standard deviation of baseline-corrected heading in the '
            'five-second pre-passage window.'
        ),
        "heading_yaw_activity_deg_s": (
            'Mean absolute rate of change between successive available 100 ms means of '
            'unwrapped, baseline-corrected Unity horizontal HMD heading in the '
            'five-second pre-passage window.'
        ),
        "far_to_pre_change_deg": "Final 0.5 s pre-passage mean minus the mean from 3 to 2 s before passage.",
        "recovered_by_pass_deg": "Heading at passage minus the minimum pre-passage heading.",
    }
    return pd.DataFrame(
        [
            {
                "outcome": outcome,
                "outcome_label": spec["label"],
                "definition": definitions[outcome],
                "units": spec["units"],
                "outcome_family": spec["family"],
                "analysis_role": spec["role"],
                "model_family": spec["model_family"],
            }
            for outcome, spec in OUTCOME_SPECS.items()
        ]
    )


def outcome_quality_audit(trials: pd.DataFrame) -> pd.DataFrame:
    """Group-specific availability and boundary checks for every paper outcome."""

    rows: list[dict[str, Any]] = []
    upper_bounds = {
        "unsafe_pct": 100.0,
        "Q1": 100.0,
        "Q2": 100.0,
        "Q3": 100.0,
        "mean_trigger": 1.0,
        "peak_trigger": 1.0,
        "any_trigger_press": 1.0,
        "trigger_return_to_safe": 1.0,
    }
    for outcome, spec in OUTCOME_SPECS.items():
        if outcome not in trials.columns:
            continue
        for group in [GROUP_RANDOMISED, GROUP_FIXED]:
            group_frame = trials[trials["ordering_group"] == group]
            values = pd.to_numeric(group_frame[outcome], errors="coerce")
            observed = values.dropna()
            reported = observed * float(spec["scale"])
            rows.append(
                {
                    "outcome": outcome,
                    "outcome_label": spec["label"],
                    "outcome_family": spec["family"],
                    "analysis_role": spec["role"],
                    "ordering_group": group,
                    "trials_total": int(len(group_frame)),
                    "trials_observed": int(observed.size),
                    "trials_missing": int(values.isna().sum()),
                    "participants_observed": int(
                        group_frame.loc[values.notna(), "participant_uid"].nunique()
                    ),
                    "minimum": float(reported.min()) if not reported.empty else np.nan,
                    "maximum": float(reported.max()) if not reported.empty else np.nan,
                    "mean": float(reported.mean()) if not reported.empty else np.nan,
                    "standard_deviation": (
                        float(reported.std(ddof=1)) if len(reported) > 1 else np.nan
                    ),
                    "zero_percentage": (
                        float(100.0 * np.mean(observed == 0))
                        if not observed.empty
                        else np.nan
                    ),
                    "ceiling_percentage": (
                        float(100.0 * np.mean(np.isclose(observed, upper_bounds[outcome])))
                        if not observed.empty and outcome in upper_bounds
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def trigger_event_flow(trials: pd.DataFrame) -> pd.DataFrame:
    """Describe the nested denominators of event-defined trigger outcomes."""

    rows: list[dict[str, Any]] = []
    for ordering_group in [GROUP_RANDOMISED, GROUP_FIXED]:
        frame = trials[trials["ordering_group"] == ordering_group]
        activated = pd.to_numeric(
            frame.get("any_trigger_press", pd.Series(index=frame.index, dtype=float)),
            errors="coerce",
        )
        active_start = pd.to_numeric(
            frame.get(
                "trigger_active_at_window_start",
                pd.Series(index=frame.index, dtype=float),
            ),
            errors="coerce",
        )
        returned = pd.to_numeric(
            frame.get("trigger_return_to_safe", pd.Series(index=frame.index, dtype=float)),
            errors="coerce",
        )
        activations = pd.to_numeric(
            frame.get("trigger_activation_count", pd.Series(index=frame.index, dtype=float)),
            errors="coerce",
        )
        active_trials = activated.eq(1)
        returned_observed = returned[active_trials].dropna()
        rows.append(
            {
                "ordering_group": ordering_group,
                "ordering_group_label": GROUP_LABELS[ordering_group],
                "analysed_trials": int(len(frame)),
                "trials_with_any_activation": int(active_trials.sum()),
                "activation_percentage": (
                    float(100.0 * active_trials.mean()) if len(frame) else np.nan
                ),
                "trials_active_at_window_start": int(active_start.eq(1).sum()),
                "active_at_window_start_percentage": (
                    float(100.0 * active_start.eq(1).mean())
                    if active_start.notna().any()
                    else np.nan
                ),
                "activated_trials_with_return_observed": int(
                    returned_observed.eq(1).sum()
                ),
                "return_to_safe_percentage_among_activated": (
                    float(100.0 * returned_observed.mean())
                    if not returned_observed.empty
                    else np.nan
                ),
                "trials_with_multiple_activations": int(activations.gt(1).sum()),
                "first_active_latency_observed": int(
                    frame.get(
                        "trigger_first_active_latency_s",
                        pd.Series(index=frame.index, dtype=float),
                    ).notna().sum()
                ),
                "first_return_latency_observed": int(
                    frame.get(
                        "trigger_first_return_latency_s",
                        pd.Series(index=frame.index, dtype=float),
                    ).notna().sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def questionnaire_descriptives(config: StudyConfig) -> pd.DataFrame:
    """Record questionnaire file availability and row counts without guessing schemas."""

    rows: list[dict[str, Any]] = []
    files = [
        (GROUP_RANDOMISED, "intake", config.randomised_intake),
        (GROUP_FIXED, "intake", config.fixed_intake),
        (GROUP_RANDOMISED, "post_experiment", config.randomised_post),
        (GROUP_FIXED, "post_experiment", config.fixed_post),
    ]
    for group, questionnaire, path in files:
        record: dict[str, Any] = {
            "ordering_group": group,
            "questionnaire": questionnaire,
            "path": str(path) if path else "",
            "available": bool(path and path.is_file()),
            "rows": np.nan,
            "columns": np.nan,
        }
        if path and path.is_file():
            try:
                frame = pd.read_csv(path)
                record["rows"] = len(frame)
                record["columns"] = len(frame.columns)
            except Exception as exc:
                record["read_error"] = str(exc)
        rows.append(record)
    return pd.DataFrame(rows)


def write_method_note(config: StudyConfig, warnings_out: Sequence[str]) -> None:
    settings = config.settings
    warning_text = "\n".join(f"* {item}" for item in warnings_out) or "* No design warnings were generated."
    text = f"""# Analysis decisions and reporting note

## Study contrast

The analysis compares two independent study runs: a randomised order group and a
fixed sequence group. The fixed sequence cohort was collected first and the
randomised order cohort later. The laboratory, apparatus, software, virtual
environment, procedure, trial structure, conditions, response instructions, and
dependent measures were held constant according to the study records. Ordering
was therefore the only planned procedural difference, but it was not randomly
allocated concurrently between participants. The group term is consequently a
quasi experimental between cohort contrast and should not be described as an
unconfounded causal effect of shuffling.

For the randomised cohort, condition assignments are loaded from the
Participant_{{i}}_mapping.csv file inside each participant folder. For the fixed
sequence cohort, every participant uses the shared condition mapping at
{config.mapping}. Participant passage timestamps are added by video ID from the
common timing mapping at {config.timing_mapping or config.mapping}. Actual trial
position is reconstructed from the recorded participant response order. The row
order of each condition mapping is compared with that recorded order and
exported in mapping_audit.csv together with mapping source and hash information.

## Primary outcome

The primary analysis uses the {settings.window_seconds:.2f} seconds immediately
before participant passage. The interval is half open: start is included and
passage is excluded. It contains {settings.expected_bins} bins of
{settings.bin_seconds:.2f} seconds. A valid bin is unsafe if any raw trigger
sample exceeds {settings.primary_threshold:.2f}. Empty bins are missing and are
not classified as safe. Complete windows are
{str(settings.require_complete_window).lower()} for the primary analysis.

The primary inferential model is a grouped binomial generalised linear model.
The numerator is unsafe bins and the denominator is valid bins. Sandwich
standard errors are clustered by participant. It includes ordering group,
the full yielding by conditional eHMI by relative order interaction, categorical
distance interactions with yielding, eHMI, and relative order, and ordering
group interactions with those condition terms. The reported marginal ordering
contrast equally weights the complete 2 by 2 by 2 by 5 condition grid. Twelve
response scale contrasts compare the conditional eHMI, relative order, and AV
behaviour effects between ordering groups, with Holm correction within each
four contrast family.

The omnibus ordering-group by condition interaction is followed by a structured
response-scale decomposition of all 40 factorial cells. These cell estimates
are not treated as 40 independent discovery tests. Each outcome receives Holm
adjusted p values and Bonferroni simultaneous 95% confidence intervals across
its 40 cells. A further Holm value across every cell and outcome is exported as
a conservative global safeguard. The condition-cell figure includes an outcome
only when the primary omnibus interaction is below 0.05 or the corresponding
secondary interaction survives global correction. Pointwise intervals and raw
p values remain in the CSV solely for numerical audit and must not be used to
claim isolated condition effects.

Thresholds {', '.join(f'{x:.2f}' for x in settings.sensitivity_thresholds)} are
reported as sensitivity analyses. The temporal model additionally tests linear
and quadratic trial position interactions. In the fixed sequence group, trial
position is structurally tied to condition, so this temporal comparison must be
described as adjusted and potentially condition confounded.

Age, gender, and recent VR experience are described by cohort. A sensitivity
model standardises the primary contrast to the pooled measured demographic
distribution. This can assess measured composition differences but cannot remove
unmeasured run level confounding.

## Secondary analyses

Q1, Q2, Q3, analogue trigger magnitude, and head-movement summaries use GEE
with participant clustering. The any-trigger-activation outcome uses binomial
GEE; continuous outcomes use Gaussian GEE with robust covariance. Equal-cell
marginal group estimates, fixed-minus-randomised contrasts, and 95% confidence
intervals are reported. Holm correction is applied across outcomes within the
ratings, trigger, and head-movement families; a global Holm value across all
secondary outcomes is also exported as a conservative post-audit safeguard.
Independent-participant Welch
comparisons are robustness summaries, and HC3 participant-level regressions
adjust for age, age missingness, gender, and recent VR experience.

Head heading is rotation around Unity's vertical y axis, unwrapped and centred
on 0.02 to 0.30 seconds after trial onset. The planned secondary summary is the
200 ms window centred on participant passage. Minimum pre-passage heading,
change from the final 500 ms before to the first 500 ms after passage, and
pre-passage heading variability are explicitly labelled exploratory movement
outcomes. A direct pre-passage HMD yaw-activity measure is also reported as the
mean absolute rate of change between successive available 100 ms heading means.
It measures headset rotation, not gaze or visual attention. Trigger outcomes
distinguish duration above threshold, mean and peak analogue magnitude, whether
any activation occurred, time to the first active bin, whether an activated
trial subsequently returned to safe, and time to that return. Event-defined
latencies use nested observed-event denominators. A first-active latency of zero
is left-censored because the press may have begun before the common window.

Condition-adjusted temporal GEE models are fitted for the primary trigger
summary, continuous and binary trigger outcomes, Q1 to Q3, and the specified
head-heading outcomes. Trial position is expressed in ten-trial units and
entered as linear and quadratic terms. Ordering-group interactions test whether
the temporal shape differs between cohorts. The current factorial condition is
adjusted using the same condition structure as the overall models. Indicators
for the scheduled break opportunities after trials 14 and 26 are included as
session-segment covariates; these indicators do not establish whether a
participant actually took a break.

A post hoc exploratory sensitivity model replaces the polynomial trial terms
with a natural cubic spline using {settings.spline_degrees_of_freedom} degrees
of freedom. Before model fitting, the spline basis is residualised separately
within each cohort against the factorial condition structure and scheduled
break segment. This prevents deterministic condition composition from making
the flexible spline design numerically singular. The outcome model retains the
full group-by-condition structure. The joint ordering-group-by-spline
interaction tests whether the remaining trajectory differs between cohorts
without requiring a quadratic shape. Holm correction is applied within outcome
family and globally. This flexible analysis cannot remove unmeasured
study-run confounding and must not be selected in place of the polynomial model
solely because it produces a smaller p value.

A categorical robustness model separately estimates trials 1-14, 15-26, and
27-40. It reports adjusted segment means, within-cohort late-minus-early
changes, the fixed-minus-randomised difference in those changes, and a joint
group-by-segment test. This avoids requiring the trajectory to be linear or
quadratic, but it remains subject to the fixed-sequence condition-by-position
limitation.

A focused exploratory cue-learning model is limited to the primary unsafe-bin
outcome and Q3. Within yielding trials, it estimates the conditional-eHMI
contrast at trials 1 and 40 and tests the change in that contrast over the
session within each cohort and between cohorts. A sensitivity model replaces
trial position with cumulative prior encounters with the yielding-plus-eHMI
condition and compares zero with the maximum prior exposure. These analyses
address learning of the particular yielding eHMI cue more directly than a
generic time-on-task slope. They do not prove individual learning and cannot
eliminate sequence or study-run confounding.

Prior-exposure sensitivity models replace trial position with the number of
preceding trials at the current yielding, eHMI, relative pedestrian order, and
distance levels. Counts are expressed per ten prior exposures. These models can
show whether a temporal pattern is attenuated when expressed as accumulated
factor-level exposure, but they cannot remove the structural
condition-by-position confounding in the fixed sequence.

## Additional primary robustness analyses

The independent unit for all additional primary robustness checks is the
participant, not the trial. A participant-cluster percentile bootstrap resamples
participants separately within cohort for {settings.bootstrap_replicates}
replicates using seed {settings.bootstrap_seed}. A leave-one-participant-out
analysis reports the complete influence range. Separate audit sensitivities
exclude randomised participants sharing a realised sequence, participants whose
mapping row order differs from recorded response order, and both sets together.
These are stability checks and do not create additional confirmatory
hypotheses.

A participant-level Bayesian bootstrap uses
{settings.bayesian_bootstrap_draws} independent Dirichlet-weight draws within
each cohort. It reports the posterior interval and the probabilities of each
effect direction. It does not report the probability of practical equivalence
because no scientifically justified practical-effect threshold was provided.
This nonparametric sensitivity was chosen instead of inventing a subjective
parametric prior after inspecting the results.

Prospective precision calculations treat the participant as the independent
unit, assume equal cohort allocation and stable participant-level variance, and
use a two-sided alpha of 0.05. They report the current minimum detectable
difference and approximate sample sizes for the illustrative planning values
{', '.join(f'{value:.2f}' for value in settings.planning_effect_sizes_percentage_points)}
percentage points at powers
{', '.join(f'{value:.0%}' for value in settings.planning_power)}. These values
are sample-size scenarios rather than equivalence margins, and the calculation
must not be used to extend data collection until significance is achieved.

No equivalence test is performed because no independently justified smallest
effect size of interest was available. Confidence intervals describe the
precision of differences without converting a nonsignificant result into
evidence of equivalence. Drift and carryover summaries are exploratory and use
Benjamini Hochberg false discovery rate adjustment.

Analysis figures are exported as HTML, high-resolution PNG, vector PDF, editable pickle, and optional EPS. \
Manuscript figures are generated in the same command. A result-table pickle snapshot supports regeneration \
without refitting. The
complete numerical results log contains all inferential tables, while raw
trial-level values and model prediction grids remain in their dedicated CSV
files to keep the log auditable and readable.

## Design and extraction warnings

{warning_text}
"""
    (config.output / "analysis_decisions.md").write_text(text, encoding="utf-8")


def _round_for_export(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in frame.select_dtypes(include=["float"]).columns:
        frame[column] = frame[column].round(8)
    return frame


def write_tables(tables: dict[str, pd.DataFrame], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        _round_for_export(frame).to_csv(output / f"{name}.csv", index=False)


def write_and_emit_results_log(
    tables: dict[str, pd.DataFrame],
    output: Path,
    warnings_out: Sequence[str],
) -> Path:
    """Write and emit every inferential value needed to audit the paper."""

    sections = [
        f"ORDERING COMPARISON NUMERICAL RESULTS - VERSION {SCRIPT_VERSION}",
        "Contrast direction is fixed sequence minus randomised order unless stated otherwise.",
        "Raw trial rows and 40-position prediction grids are retained in CSV files and are not duplicated here.",
        "",
        "DESIGN AND DATA WARNINGS",
        *(f"- {warning}" for warning in warnings_out),
    ]
    for name in RESULTS_LOG_TABLES:
        frame = tables.get(name)
        sections.extend(["", "=" * 96, name.upper(), "=" * 96])
        if frame is None:
            sections.append("Table was not created.")
        elif frame.empty:
            sections.append("No rows.")
        else:
            rounded = _round_for_export(frame)
            with pd.option_context(
                "display.max_rows",
                None,
                "display.max_columns",
                None,
                "display.width",
                260,
                "display.max_colwidth",
                120,
            ):
                sections.append(rounded.to_string(index=False, na_rep="NA"))
    report = "\n".join(sections) + "\n"
    path = output / "analysis_results.log"
    path.write_text(report, encoding="utf-8")
    LOGGER.info("\n%s", report.rstrip())
    LOGGER.info("Complete numerical results log written to %s", path)
    return path


def write_extraction_diagnostics(
    audit: pd.DataFrame,
    sequences: pd.DataFrame,
    output: Path,
) -> None:
    """Persist extraction evidence before sample validation can stop the run."""

    output.mkdir(parents=True, exist_ok=True)
    _round_for_export(audit).to_csv(output / "exclusion_audit.csv", index=False)
    _round_for_export(sequences).to_csv(output / "sequence_audit.csv", index=False)
    _round_for_export(mapping_audit_table(sequences)).to_csv(
        output / "mapping_audit.csv",
        index=False,
    )
    if audit.empty:
        LOGGER.error("No extraction audit rows were produced")
        return

    failed = audit[~audit["included"].fillna(False)].copy()
    if failed.empty:
        return
    counts = (
        failed.groupby(["ordering_group", "reason"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["count", "ordering_group"], ascending=[False, True])
    )
    LOGGER.warning("Extraction failure counts:\n%s", counts.to_string(index=False))
    detail_columns = [
        column
        for column in [
            "ordering_group",
            "participant_id",
            "video_id",
            "reason",
            "detail",
            "mapping_source",
            "timing_mapping_source",
        ]
        if column in failed.columns
    ]
    LOGGER.warning(
        "First extraction failures:\n%s",
        failed[detail_columns].head(12).to_string(index=False),
    )
