#!/usr/bin/env python3
"""Paper focused comparison of randomised trial order and a fixed sequence.

The script reads the two independent ordering groups directly from the raw
participant folders, applies participant specific mappings to the randomised
cohort and the shared mapping to the fixed cohort, reconstructs trial order from
each participant response file, extracts the prespecified common window, and
writes analysis ready tables, model summaries, diagnostics, and figures.

Primary outcome
---------------
For every trial, the five seconds immediately before participant passage are
split into half open 100 ms bins. A valid bin is unsafe when any raw trigger
sample in the bin exceeds 0.10. Missing bins are never counted as safe. The
primary model is a grouped binomial generalised linear model with participant
clustered sandwich standard errors.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import sys
import warnings
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Sequence

SCRIPT_VERSION = "2026-07-24.3"

# The version check intentionally runs before optional scientific dependencies
# are imported. This allows the installer to verify the exact file even when
# the project environment has not yet been activated.
if __name__ == "__main__" and sys.argv[1:] == ["--version"]:
    print(f"ordering_comparison.py {SCRIPT_VERSION}")
    raise SystemExit(0)

import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.special import expit
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import build_design_matrices, dmatrix
from statsmodels.genmod.cov_struct import Exchangeable, Independence
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning

try:
    import common
except ImportError:
    common = None


LOGGER = logging.getLogger("ordering_comparison")
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
    "comparison_analysis",
)

GROUP_RANDOMISED = "randomised_order"
GROUP_FIXED = "fixed_sequence"
GROUP_LABELS = {
    GROUP_RANDOMISED: "Randomised order",
    GROUP_FIXED: "Fixed sequence",
}

# Outcome families are declared once so that model fitting, multiplicity
# correction, tables, figures, and the manuscript note all use the same scope.
# "secondary" outcomes are suitable for planned paper-level interpretation;
# "exploratory" outcomes describe head-movement dynamics without expanding the
# confirmatory primary question.
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

# Outcomes for which session progression is modelled after adjustment for the
# current factorial condition.  The model uses a participant-clustered robust
# covariance, trial position in ten-trial units, a quadratic term, and fixed
# indicators for the two scheduled break opportunities after trials 14 and 26.
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

# The cue-learning analysis is deliberately narrow. Q3 directly asks about
# vehicle-intention understanding, while the grouped-binomial unsafe outcome is
# the behavioural primary outcome. Adding every available outcome to this
# analysis would turn a theory-led check into significance searching.
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


@dataclass(frozen=True)
class AnalysisSettings:
    """Prespecified analysis choices."""

    window_seconds: float = 5.0
    bin_seconds: float = 0.1
    primary_threshold: float = 0.10
    sensitivity_thresholds: tuple[float, ...] = (0.05, 0.10, 0.50)
    require_complete_window: bool = True
    minimum_valid_trials_per_participant: int = 32
    bootstrap_replicates: int = 5000
    bootstrap_seed: int = 2901
    bayesian_bootstrap_draws: int = 20000
    spline_degrees_of_freedom: int = 4
    planning_effect_sizes_percentage_points: tuple[float, ...] = (
        2.5,
        5.0,
        7.5,
        10.0,
    )
    planning_power: tuple[float, ...] = (0.80, 0.90)

    @property
    def expected_bins(self) -> int:
        return int(round(self.window_seconds / self.bin_seconds))


@dataclass(frozen=True)
class StudyConfig:
    """Resolved input and output locations."""

    config_path: Path
    mapping: Path
    output: Path
    figures: Path
    randomised_data: Path
    fixed_data: Path
    randomised_intake: Path | None
    fixed_intake: Path | None
    randomised_post: Path | None
    fixed_post: Path | None
    settings: AnalysisSettings = field(default_factory=AnalysisSettings)
    randomised_mapping_filename: str = "Participant_{participant_id}_mapping.csv"
    timing_mapping: Path | None = None


def _resolve_path(value: str | None, base: Path) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _load_raw_config(config_path: Path) -> dict[str, Any]:
    """Load project settings through common.get_configs when it is available."""

    if common is not None:
        common_root = Path(common.root_dir).expanduser().resolve()
        if config_path.parent == common_root:
            raw: dict[str, Any] = {}
            try:
                for key in COMMON_CONFIG_KEYS:
                    try:
                        raw[key] = common.get_configs(
                            key,
                            config_file_name=config_path.name,
                        )
                    except KeyError:
                        continue
            except SystemExit as exc:
                raise RuntimeError(
                    "common.py could not validate the project config. Check config "
                    "and default.config in the repository root."
                ) from exc
            LOGGER.info(
                "Loaded %d configuration values through common.get_configs from %s",
                len(raw),
                config_path,
            )
            return raw

    LOGGER.warning(
        "common.py was unavailable for %s; falling back to direct JSON loading",
        config_path,
    )
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_config(config_path: Path) -> StudyConfig:
    """Load the attached comparison configuration and optional analysis block."""

    config_path = config_path.expanduser().resolve()
    raw = _load_raw_config(config_path)
    base = config_path.parent

    analysis_raw = raw.get("comparison_analysis", {})
    primary_threshold = float(analysis_raw.get("primary_threshold", 0.10))
    sensitivity_thresholds = [
        float(x) for x in analysis_raw.get("sensitivity_thresholds", [0.05, 0.10, 0.50])
    ]
    if not any(np.isclose(primary_threshold, value) for value in sensitivity_thresholds):
        sensitivity_thresholds.append(primary_threshold)
    sensitivity_thresholds = sorted(set(sensitivity_thresholds))
    settings = AnalysisSettings(
        window_seconds=float(analysis_raw.get("window_seconds", 5.0)),
        bin_seconds=float(analysis_raw.get("bin_seconds", 0.1)),
        primary_threshold=primary_threshold,
        sensitivity_thresholds=tuple(sensitivity_thresholds),
        require_complete_window=bool(analysis_raw.get("require_complete_window", True)),
        minimum_valid_trials_per_participant=int(
            analysis_raw.get("minimum_valid_trials_per_participant", 32)
        ),
        bootstrap_replicates=int(
            analysis_raw.get("bootstrap_replicates", 5000)
        ),
        bootstrap_seed=int(analysis_raw.get("bootstrap_seed", 2901)),
        bayesian_bootstrap_draws=int(
            analysis_raw.get("bayesian_bootstrap_draws", 20000)
        ),
        spline_degrees_of_freedom=int(
            analysis_raw.get("spline_degrees_of_freedom", 4)
        ),
        planning_effect_sizes_percentage_points=tuple(
            float(value)
            for value in analysis_raw.get(
                "planning_effect_sizes_percentage_points",
                [2.5, 5.0, 7.5, 10.0],
            )
        ),
        planning_power=tuple(
            float(value)
            for value in analysis_raw.get("planning_power", [0.80, 0.90])
        ),
    )
    if settings.window_seconds <= 0 or settings.bin_seconds <= 0:
        raise ValueError("window_seconds and bin_seconds must be positive")
    bins_exact = settings.window_seconds / settings.bin_seconds
    if not np.isclose(bins_exact, round(bins_exact)):
        raise ValueError("window_seconds must be an integer multiple of bin_seconds")
    if not all(0.0 <= value <= 1.0 for value in settings.sensitivity_thresholds):
        raise ValueError("Trigger thresholds must lie between 0 and 1")
    if settings.bootstrap_replicates < 1000:
        raise ValueError("bootstrap_replicates must be at least 1000")
    if settings.bayesian_bootstrap_draws < 5000:
        raise ValueError("bayesian_bootstrap_draws must be at least 5000")
    if not 3 <= settings.spline_degrees_of_freedom <= 8:
        raise ValueError("spline_degrees_of_freedom must lie between 3 and 8")
    if not settings.planning_effect_sizes_percentage_points or not all(
        value > 0 for value in settings.planning_effect_sizes_percentage_points
    ):
        raise ValueError(
            "planning_effect_sizes_percentage_points must contain positive values"
        )
    if not settings.planning_power or not all(
        0.50 < value < 1.0 for value in settings.planning_power
    ):
        raise ValueError("planning_power values must lie between 0.50 and 1")

    output = _resolve_path(raw.get("output", "_comparison_output"), base)
    assert output is not None
    figures_raw = raw.get("figures")
    if figures_raw:
        figures_setting = Path(str(figures_raw)).expanduser()
        if figures_setting.is_absolute():
            figures = figures_setting
        elif len(figures_setting.parts) == 1:
            # The legacy config uses "figures". Keep comparison figures inside
            # the comparison output so archiving `_output` includes the plots.
            figures = (output / figures_setting).resolve()
        else:
            figures = (base / figures_setting).resolve()
    else:
        figures = output / "figures"
    assert figures is not None

    repository_mapping_value = raw.get("timing_mapping", raw.get("mapping"))
    if raw.get("unshuffled_mapping"):
        fixed_mapping_value = raw["unshuffled_mapping"]
        timing_mapping_value = repository_mapping_value
        if not timing_mapping_value:
            repository_candidate = (base / "mapping.csv").resolve()
            if not repository_candidate.is_file():
                raise ValueError(
                    "Missing common passage timing mapping. Add "
                    '"timing_mapping": "mapping.csv" (or "mapping": '
                    '"mapping.csv") to config and place the full mapping.csv '
                    "beside ordering_comparison.py."
                )
            timing_mapping_value = str(repository_candidate)
            LOGGER.warning(
                "Config has no timing_mapping or mapping entry; using repository "
                "mapping.csv at %s",
                repository_candidate,
            )
    else:
        # Backwards compatibility for repositories where one complete mapping
        # contains both condition assignments and crossing timestamps.
        fixed_mapping_value = repository_mapping_value
        timing_mapping_value = repository_mapping_value

    required = {
        "unshuffled_mapping": fixed_mapping_value,
        "timing_mapping": timing_mapping_value,
        "shuffled_data": raw.get("shuffled_data"),
        "unshuffled_data": raw.get("unshuffled_data"),
    }
    missing = [key for key, value in required.items() if not value]
    if missing:
        raise ValueError(f"Missing required configuration keys: {', '.join(missing)}")

    mapping = _resolve_path(str(fixed_mapping_value), base)
    timing_mapping = _resolve_path(str(timing_mapping_value), base)
    randomised_data = _resolve_path(str(raw["shuffled_data"]), base)
    fixed_data = _resolve_path(str(raw["unshuffled_data"]), base)
    assert (
        mapping is not None
        and timing_mapping is not None
        and randomised_data is not None
        and fixed_data is not None
    )
    randomised_mapping_filename = str(
        raw.get(
            "shuffled_mapping_filename",
            "Participant_{participant_id}_mapping.csv",
        )
    ).strip()
    if not randomised_mapping_filename:
        raise ValueError("shuffled_mapping_filename cannot be empty")
    if (
        "{participant_id}" not in randomised_mapping_filename
        and "{participant_folder}" not in randomised_mapping_filename
    ):
        raise ValueError(
            "shuffled_mapping_filename must contain {participant_id} or "
            "{participant_folder}"
        )
    try:
        rendered_mapping_name = randomised_mapping_filename.format(
            participant_id="1",
            participant_folder="Participant_1",
        )
    except (KeyError, ValueError) as exc:
        raise ValueError(
            "shuffled_mapping_filename may use only {participant_id} and "
            "{participant_folder}"
        ) from exc
    rendered_mapping_path = Path(rendered_mapping_name)
    if rendered_mapping_path.is_absolute() or ".." in rendered_mapping_path.parts:
        raise ValueError(
            "shuffled_mapping_filename must resolve inside each participant folder"
        )

    return StudyConfig(
        config_path=config_path,
        mapping=mapping,
        output=output,
        figures=figures,
        randomised_data=randomised_data,
        fixed_data=fixed_data,
        randomised_intake=_resolve_path(raw.get("shuffled_intake_questionnaire"), base),
        fixed_intake=_resolve_path(raw.get("unshuffled_intake_questionnaire"), base),
        randomised_post=_resolve_path(
            raw.get("shuffled_post_experiment_questionnaire"), base
        ),
        fixed_post=_resolve_path(
            raw.get("unshuffled_post_experiment_questionnaire"), base
        ),
        settings=settings,
        randomised_mapping_filename=randomised_mapping_filename,
        timing_mapping=timing_mapping,
    )


def _normalise_video_id(value: Any) -> str | None:
    match = VIDEO_PATTERN.search(str(value))
    return match.group(0).lower() if match else None


def _coerce_binary(series: pd.Series, name: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    unique = set(values.dropna().astype(int).unique())
    if not unique.issubset({0, 1}):
        raise ValueError(f"{name} must contain only 0 and 1; found {sorted(unique)}")
    return values.astype("Int64")


def load_mapping(path: Path) -> pd.DataFrame:
    """Read and validate the 40 cell factorial trial mapping."""

    if not path.is_file():
        raise FileNotFoundError(f"Mapping file not found: {path}")
    mapping = pd.read_csv(path)
    aliases = {
        "video": "video_id",
        "videoID": "video_id",
        "videoId": "video_id",
        "ehmi": "eHMIOn",
        "eHMI": "eHMIOn",
        "distance": "distPed",
        "order": "camera",
    }
    mapping = mapping.rename(columns={key: value for key, value in aliases.items() if key in mapping})
    required = {"video_id", "yielding", "eHMIOn", "camera", "distPed"}
    missing = sorted(required.difference(mapping.columns))
    if missing:
        raise ValueError(f"Mapping is missing required columns: {', '.join(missing)}")

    mapping["video_id"] = mapping["video_id"].map(_normalise_video_id)
    mapping = mapping[mapping["video_id"].notna()].copy()
    mapping = mapping.drop_duplicates("video_id", keep="first")
    mapping["yielding"] = _coerce_binary(mapping["yielding"], "yielding")
    mapping["eHMIOn"] = _coerce_binary(mapping["eHMIOn"], "eHMIOn")
    mapping["camera"] = _coerce_binary(mapping["camera"], "camera")
    mapping["distPed"] = pd.to_numeric(mapping["distPed"], errors="raise").astype(int)

    distance_codes = sorted(mapping["distPed"].dropna().unique())
    distance_map = {code: 2.0 * float(code) for code in distance_codes}
    mapping["distPed_m"] = mapping["distPed"].map(distance_map)

    if len(mapping) != 40:
        raise ValueError(f"Expected 40 experimental videos in mapping, found {len(mapping)}")
    factorial_counts = (
        mapping.groupby(["yielding", "eHMIOn", "camera", "distPed_m"], observed=True)
        .size()
        .reset_index(name="n")
    )
    if len(factorial_counts) != 40 or not (factorial_counts["n"] == 1).all():
        raise ValueError("The trial mapping is not a complete 2 x 2 x 2 x 5 factorial design")
    return mapping.reset_index(drop=True)


def attach_passage_timing(
    condition_mapping: pd.DataFrame,
    timing_mapping: pd.DataFrame,
) -> pd.DataFrame:
    """Add crossing timestamps without replacing participant condition assignments."""

    timing_columns = [
        column
        for column in timing_mapping.columns
        if "cross" in str(column).lower() or str(column) == "video_length"
    ]
    if not timing_columns:
        raise ValueError(
            "The common timing mapping does not contain crossing timestamp columns"
        )

    result = condition_mapping.copy()
    timing_index = timing_mapping.set_index("video_id", drop=False)
    missing_video_ids = sorted(
        set(result["video_id"].astype(str)).difference(timing_index.index.astype(str))
    )
    if missing_video_ids:
        raise ValueError(
            "The common timing mapping is missing video IDs: "
            + ", ".join(missing_video_ids)
        )

    for column in timing_columns:
        timing_values = result["video_id"].map(timing_index[column])
        if column in result.columns:
            result[column] = result[column].combine_first(timing_values)
        else:
            result[column] = timing_values

    missing_passage: list[str] = []
    for _, row in result.iterrows():
        try:
            _passage_time_seconds(row)
        except (TypeError, ValueError):
            missing_passage.append(str(row["video_id"]))
    if missing_passage:
        raise ValueError(
            "No usable participant passage timestamp after combining mappings for: "
            + ", ".join(missing_passage)
        )
    return result


def _participant_number(path: Path) -> str:
    match = PARTICIPANT_PATTERN.search(path.name)
    if match:
        return match.group(1)
    digits = re.findall(r"\d+", path.name)
    return digits[-1] if digits else path.name


def discover_participants(root: Path) -> list[Path]:
    """Find folders that contain a participant response file and trial files."""

    if not root.is_dir():
        raise FileNotFoundError(f"Participant response root not found: {root}")
    candidates: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_dir():
            continue
        names = [item.name.lower() for item in path.glob("*.csv")]
        if any("participant" in name for name in names) and any("video_" in name for name in names):
            candidates.append(path)
    if not candidates:
        names = [item.name.lower() for item in root.glob("*.csv")]
        if any("participant" in name for name in names) and any("video_" in name for name in names):
            candidates = [root]
    if not candidates:
        raise FileNotFoundError(f"No participant folders were discovered below {root}")
    return candidates


def _response_candidates(participant_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in participant_dir.glob("*.csv")
        if "participant" in path.name.lower() and "video_" not in path.name.lower()
        and "mapping" not in path.name.lower()
    )


def read_participant_responses(participant_dir: Path) -> pd.DataFrame:
    """Read realised video order and Q1 to Q3 from the headerless response file."""

    blocks: list[pd.DataFrame] = []
    for path in _response_candidates(participant_dir):
        frame: pd.DataFrame | None = None
        for separator in [",", ";"]:
            try:
                candidate = pd.read_csv(path, header=None, sep=separator)
            except Exception:
                continue
            if candidate.shape[1] >= 4:
                frame = candidate
                break
        if frame is None or frame.empty:
            continue
        block = pd.DataFrame({"video_id": frame.iloc[:, 0].map(_normalise_video_id)})
        for index, column in enumerate(["Q1", "Q2", "Q3"], start=1):
            block[column] = pd.to_numeric(frame.iloc[:, index], errors="coerce")
        block = block[block["video_id"].notna()].copy()
        if block.empty:
            continue
        block["response_source"] = str(path)
        blocks.append(block)
    if not blocks:
        raise FileNotFoundError(f"No readable participant response file in {participant_dir}")

    response = pd.concat(blocks, ignore_index=True)
    response = response.drop_duplicates("video_id", keep="first").reset_index(drop=True)
    response["trial_number"] = np.arange(1, len(response) + 1)
    return response


def _read_header(path: Path) -> set[str]:
    try:
        return set(pd.read_csv(path, nrows=0).columns.astype(str))
    except Exception:
        return set()


def find_trial_file(participant_dir: Path, video_id: str) -> Path:
    """Choose the trial time series file by ID and required columns."""

    candidates = sorted(
        path
        for path in participant_dir.glob("*.csv")
        if _normalise_video_id(path.stem) == video_id.lower()
    )
    for path in candidates:
        header = _read_header(path)
        if "Timestamp" in header and "TriggerValueRight" in header:
            return path
    for path in sorted(participant_dir.glob("*.csv")):
        header = _read_header(path)
        if "Timestamp" not in header or "TriggerValueRight" not in header:
            continue
        try:
            first = pd.read_csv(path, usecols=["VideoID"], nrows=1)
            value = _normalise_video_id(first.iloc[0, 0]) if not first.empty else None
        except Exception:
            value = None
        if value == video_id:
            return path
    raise FileNotFoundError(f"Time series for {video_id} not found in {participant_dir}")


def _passage_time_seconds(row: pd.Series) -> float:
    """Return passage of the participant identified by the camera condition."""

    camera = int(row["camera"])
    preferred = ["cross_p1_time_s", "cross_p1", "crossP1", "crossing_p1"] if camera == 0 else [
        "cross_p2_time_s",
        "cross_p2",
        "crossP2",
        "crossing_p2",
    ]
    for column in preferred:
        if column in row.index and pd.notna(row[column]):
            value = float(row[column])
            return value / 1000.0 if value > 100.0 else value
    fallback = [column for column in row.index if "cross" in str(column).lower()]
    if fallback:
        value = float(row[fallback[0]])
        return value / 1000.0 if value > 100.0 else value
    raise ValueError(f"No participant passage timestamp found for {row['video_id']}")


def _timestamp_scale_to_seconds(values: pd.Series, passage_seconds: float) -> float:
    """Infer seconds, milliseconds, or microseconds from timeline coverage and rate.

    A previous rule divided every long recording by 1,000. That is unsafe: a
    valid 456-second recording sampled at 50 Hz is long in seconds, not a
    456-millisecond recording. The selected scale must place the participant
    passage (and preferably the full primary window) inside the time series and
    produce a plausible sampling frequency.
    """

    timestamps = pd.to_numeric(values, errors="coerce")
    finite = np.sort(timestamps[np.isfinite(timestamps)].to_numpy(float))
    if len(finite) < 2:
        return 1.0
    unique = np.unique(finite)
    spacing = np.diff(unique)
    spacing = spacing[spacing > 0]
    primary_start = passage_seconds - 5.0
    candidates: list[tuple[float, float]] = []
    for scale in (1.0, 1e-3, 1e-6):
        start = float(np.quantile(finite, 0.01) * scale)
        end = float(np.quantile(finite, 0.99) * scale)
        covers_passage = start <= passage_seconds <= end
        covers_primary_window = start <= primary_start and end >= passage_seconds
        if len(spacing):
            rate = 1.0 / (float(np.median(spacing)) * scale)
            plausible_rate = 1.0 <= rate <= 1000.0
            rate_penalty = abs(math.log(rate / 120.0)) if plausible_rate else 20.0
        else:
            plausible_rate = False
            rate_penalty = 20.0
        # Coverage is decisive; the rate breaks ties among plausible units.
        score = (
            1000.0 * float(covers_primary_window)
            + 100.0 * float(covers_passage)
            + 10.0 * float(plausible_rate)
            - rate_penalty
        )
        candidates.append((score, scale))
    return max(candidates, key=lambda item: item[0])[1]


def _timestamps_in_seconds(values: pd.Series, passage_seconds: float) -> pd.Series:
    timestamps = pd.to_numeric(values, errors="coerce")
    return timestamps * _timestamp_scale_to_seconds(timestamps, passage_seconds)


def _normalise_trigger(values: pd.Series) -> tuple[pd.Series, str]:
    trigger = pd.to_numeric(values, errors="coerce")
    finite = trigger[np.isfinite(trigger)]
    if finite.empty:
        return trigger, "unknown"
    scale = "0_to_100" if float(finite.quantile(0.99)) > 1.5 else "0_to_1"
    if scale == "0_to_100":
        trigger = trigger / 100.0
    return trigger, scale


def _quaternion_columns(frame: pd.DataFrame) -> tuple[str, str, str, str] | None:
    alternatives = [
        ("HMDRotationW", "HMDRotationX", "HMDRotationY", "HMDRotationZ"),
        ("HMDQuaternionW", "HMDQuaternionX", "HMDQuaternionY", "HMDQuaternionZ"),
        ("HMD_W", "HMD_X", "HMD_Y", "HMD_Z"),
        ("QuaternionW", "QuaternionX", "QuaternionY", "QuaternionZ"),
    ]
    return next((cols for cols in alternatives if set(cols).issubset(frame.columns)), None)


def _markley_average(quaternions: np.ndarray) -> np.ndarray:
    quaternions = np.asarray(quaternions, dtype=float)
    valid = np.isfinite(quaternions).all(axis=1)
    quaternions = quaternions[valid]
    if not len(quaternions):
        return np.full(4, np.nan)
    norms = np.linalg.norm(quaternions, axis=1)
    quaternions = quaternions[norms > np.finfo(float).eps]
    norms = norms[norms > np.finfo(float).eps]
    if not len(quaternions):
        return np.full(4, np.nan)
    quaternions = quaternions / norms[:, None]
    reference = quaternions[0]
    quaternions[np.dot(quaternions, reference) < 0] *= -1
    eigenvalues, eigenvectors = np.linalg.eigh(quaternions.T @ quaternions / len(quaternions))
    average = eigenvectors[:, np.argmax(eigenvalues)]
    return average if average[0] >= 0 else -average


def _heading_from_quaternions(quaternions: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternions, dtype=float)
    norm = np.linalg.norm(q, axis=1)
    valid = np.isfinite(q).all(axis=1) & (norm > 0)
    qn = np.full_like(q, np.nan)
    qn[valid] = q[valid] / norm[valid, None]
    w, x, y, z = qn.T
    forward_x = 2.0 * (x * z + w * y)
    forward_z = 1.0 - 2.0 * (x * x + y * y)
    return np.degrees(np.arctan2(forward_x, forward_z))


def _unity_heading_degrees(frame: pd.DataFrame) -> pd.Series:
    """Unity horizontal heading from a W X Y Z quaternion.

    Unity's vertical axis is y. The local forward vector (0, 0, 1) is rotated,
    projected onto the x z plane, and converted with atan2(x, z).
    """

    columns = _quaternion_columns(frame)
    if columns is None:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    q = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    return pd.Series(_heading_from_quaternions(q), index=frame.index)


def _unity_heading_trajectory(frame: pd.DataFrame, time_seconds: pd.Series) -> pd.DataFrame:
    """Return one Unity horizontal heading per unique raw timestamp."""

    columns = _quaternion_columns(frame)
    if columns is None:
        return pd.DataFrame(columns=["time_seconds", "heading_deg"])
    work = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    work.insert(0, "time_seconds", pd.to_numeric(time_seconds, errors="coerce"))
    work = work.dropna().sort_values("time_seconds")
    if work.empty:
        return pd.DataFrame(columns=["time_seconds", "heading_deg"])
    if work["time_seconds"].duplicated().any():
        records: list[dict[str, float]] = []
        for timestamp, group in work.groupby("time_seconds", sort=True):
            average = _markley_average(group.loc[:, list(columns)].to_numpy(float))
            heading = _heading_from_quaternions(average.reshape(1, 4))[0]
            records.append({"time_seconds": float(timestamp), "heading_deg": float(heading)})
        return pd.DataFrame(records)
    heading = _heading_from_quaternions(work.loc[:, list(columns)].to_numpy(float))
    return pd.DataFrame(
        {"time_seconds": work["time_seconds"].to_numpy(float), "heading_deg": heading}
    )


def _mean_heading_window(
    trajectory: pd.DataFrame,
    start: float,
    end: float,
    minimum_samples: int = 3,
) -> float:
    """Mean baseline-corrected heading in a half-open time window."""

    mask = (
        (trajectory["time_seconds"] >= start)
        & (trajectory["time_seconds"] < end)
        & np.isfinite(trajectory["heading_corrected_deg"])
    )
    values = trajectory.loc[mask, "heading_corrected_deg"]
    return float(values.mean()) if len(values) >= minimum_samples else np.nan


def _smooth_heading_for_feature_extraction(
    time_seconds: np.ndarray,
    heading_degrees: np.ndarray,
    window_seconds: float = 0.22,
) -> np.ndarray:
    """Centred moving mean with an approximately fixed temporal width."""

    time_seconds = np.asarray(time_seconds, dtype=float)
    heading_degrees = np.asarray(heading_degrees, dtype=float)
    finite_times = time_seconds[np.isfinite(time_seconds)]
    differences = np.diff(np.unique(finite_times))
    differences = differences[differences > 0]
    if not len(differences):
        return heading_degrees.copy()
    samples = max(3, int(round(window_seconds / float(np.median(differences)))))
    if samples % 2 == 0:
        samples += 1
    return (
        pd.Series(heading_degrees)
        .rolling(samples, center=True, min_periods=max(2, samples // 3))
        .mean()
        .to_numpy(float)
    )


def _threshold_suffix(threshold: float) -> str:
    return f"t{int(round(threshold * 100)):02d}"


def extract_trial_features(
    trial_file: Path,
    mapping_row: pd.Series,
    settings: AnalysisSettings,
) -> dict[str, Any]:
    """Extract trigger and heading features from one prespecified common window."""

    frame = pd.read_csv(trial_file)
    required = {"Timestamp", "TriggerValueRight"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{trial_file} is missing columns: {', '.join(sorted(missing))}")

    passage = _passage_time_seconds(mapping_row)
    timestamp_scale = _timestamp_scale_to_seconds(frame["Timestamp"], passage)
    time_seconds = pd.to_numeric(frame["Timestamp"], errors="coerce") * timestamp_scale
    trigger, trigger_scale = _normalise_trigger(frame["TriggerValueRight"])
    window_start = passage - settings.window_seconds
    mask = (
        np.isfinite(time_seconds)
        & np.isfinite(trigger)
        & (time_seconds >= window_start)
        & (time_seconds < passage)
    )
    selected = pd.DataFrame(
        {"time_seconds": time_seconds[mask], "trigger": trigger[mask]},
        index=frame.index[mask],
    )
    if selected.empty:
        raise ValueError("No valid trigger samples in the primary common window")

    relative = (selected["time_seconds"] - window_start) / settings.bin_seconds
    selected["bin_index"] = np.floor(relative + 1e-9).astype(int)
    selected = selected[
        (selected["bin_index"] >= 0) & (selected["bin_index"] < settings.expected_bins)
    ]
    grouped = selected.groupby("bin_index", sort=True)["trigger"]
    bin_max = grouped.max()
    valid_bins = int(bin_max.size)

    result: dict[str, Any] = {
        "window_start_s": window_start,
        "window_end_s": passage,
        "timestamp_scale_to_seconds": timestamp_scale,
        "expected_bins": settings.expected_bins,
        "valid_bins": valid_bins,
        "missing_bins": settings.expected_bins - valid_bins,
        "window_complete": valid_bins == settings.expected_bins,
        "trigger_scale": trigger_scale,
        "trigger_samples": int(selected.shape[0]),
        "mean_trigger": float(selected["trigger"].mean()),
        "peak_trigger": float(selected["trigger"].max()),
    }
    for threshold in settings.sensitivity_thresholds:
        unsafe_states = (bin_max > threshold).astype(int)
        unsafe_bins = int(unsafe_states.sum())
        suffix = _threshold_suffix(threshold)
        result[f"unsafe_bins_{suffix}"] = unsafe_bins
        result[f"safe_bins_{suffix}"] = valid_bins - unsafe_bins
        result[f"unsafe_pct_{suffix}"] = 100.0 * unsafe_bins / valid_bins if valid_bins else np.nan
        if len(unsafe_states) > 2 and unsafe_states.iloc[:-1].std() > 0 and unsafe_states.iloc[1:].std() > 0:
            result[f"unsafe_lag1_{suffix}"] = float(
                np.corrcoef(unsafe_states.iloc[:-1], unsafe_states.iloc[1:])[0, 1]
            )
        else:
            result[f"unsafe_lag1_{suffix}"] = np.nan

    primary_suffix = _threshold_suffix(settings.primary_threshold)
    result["unsafe_bins"] = result[f"unsafe_bins_{primary_suffix}"]
    result["safe_bins"] = result[f"safe_bins_{primary_suffix}"]
    result["unsafe_pct"] = result[f"unsafe_pct_{primary_suffix}"]
    # Event-defined trigger outcomes use the same binned state definition as
    # the primary outcome. They are calculated only for complete windows so a
    # missing bin can never be mistaken for a safe state. Latencies are
    # relative to the common-window start. A zero first-active latency is
    # explicitly left-censored: the participant may have begun pressing before
    # the five-second window.
    primary_states = (
        (bin_max.sort_index() > settings.primary_threshold)
        .astype(float)
        .reindex(range(settings.expected_bins))
    )
    if primary_states.notna().all():
        state_values = primary_states.to_numpy(int)
        active_indices = np.flatnonzero(state_values == 1)
        return_indices = np.flatnonzero(np.diff(state_values) == -1) + 1
        result["any_trigger_press"] = int(bool(active_indices.size))
        result["trigger_active_at_window_start"] = int(state_values[0] == 1)
        result["trigger_press_duration_s"] = float(
            state_values.sum() * settings.bin_seconds
        )
        result["trigger_activation_count"] = int(
            state_values[0] + (np.diff(state_values) == 1).sum()
        )
        result["trigger_first_active_latency_s"] = (
            float(active_indices[0] * settings.bin_seconds)
            if active_indices.size
            else np.nan
        )
        if active_indices.size:
            return_after_activation = return_indices[
                return_indices > active_indices[0]
            ]
            result["trigger_return_to_safe"] = int(
                bool(return_after_activation.size)
            )
            result["trigger_first_return_latency_s"] = (
                float(return_after_activation[0] * settings.bin_seconds)
                if return_after_activation.size
                else np.nan
            )
        else:
            result["trigger_return_to_safe"] = np.nan
            result["trigger_first_return_latency_s"] = np.nan
    else:
        result["any_trigger_press"] = np.nan
        result["trigger_active_at_window_start"] = np.nan
        result["trigger_press_duration_s"] = np.nan
        result["trigger_activation_count"] = np.nan
        result["trigger_first_active_latency_s"] = np.nan
        result["trigger_return_to_safe"] = np.nan
        result["trigger_first_return_latency_s"] = np.nan

    trajectory = _unity_heading_trajectory(frame, time_seconds)
    if not trajectory.empty:
        valid = np.isfinite(trajectory["time_seconds"]) & np.isfinite(trajectory["heading_deg"])
        trajectory = trajectory.loc[valid].sort_values("time_seconds").copy()
    if not trajectory.empty:
        trajectory["heading_unwrapped_deg"] = np.degrees(
            np.unwrap(np.radians(trajectory["heading_deg"].to_numpy(float)))
        )
        baseline_mask = (
            (trajectory["time_seconds"] >= 0.02)
            & (trajectory["time_seconds"] < 0.30)
        )
        baseline_values = trajectory.loc[baseline_mask, "heading_unwrapped_deg"]
        baseline = float(baseline_values.mean()) if len(baseline_values) >= 3 else np.nan
        trajectory["heading_corrected_deg"] = trajectory["heading_unwrapped_deg"] - baseline
        passage_mask = (
            (trajectory["time_seconds"] >= passage - 0.10)
            & (trajectory["time_seconds"] < passage + 0.10)
        )
        passage_values = trajectory.loc[passage_mask, "heading_corrected_deg"].dropna()
        common_mask = (
            (trajectory["time_seconds"] >= window_start)
            & (trajectory["time_seconds"] < passage)
        )
        common_values = trajectory.loc[common_mask, "heading_corrected_deg"].dropna()
        result["heading_baseline_deg"] = baseline
        result["heading_baseline_samples"] = int(baseline_values.size)
        result["heading_passage_mean_deg"] = (
            float(passage_values.mean()) if len(passage_values) >= 3 else np.nan
        )
        # Alias the passage-window outcome to the terminology used in the
        # original participant-level head-heading analysis.
        result["heading_at_pass_deg"] = result["heading_passage_mean_deg"]
        result["heading_passage_samples"] = int(passage_values.size)
        result["heading_common_window_mean_deg"] = (
            float(common_values.mean()) if not common_values.empty else np.nan
        )
        result["heading_common_window_mean_abs_deg"] = (
            float(common_values.abs().mean()) if not common_values.empty else np.nan
        )
        result["heading_common_window_sd_deg"] = (
            float(common_values.std(ddof=1)) if len(common_values) > 1 else np.nan
        )
        common_trajectory = trajectory.loc[
            common_mask, ["time_seconds", "heading_corrected_deg"]
        ].dropna()
        if not common_trajectory.empty:
            relative_bin = np.floor(
                (common_trajectory["time_seconds"] - window_start)
                / settings.bin_seconds
                + 1e-9
            ).astype(int)
            common_trajectory = common_trajectory.assign(common_bin=relative_bin)
            common_trajectory = common_trajectory[
                common_trajectory["common_bin"].between(
                    0, settings.expected_bins - 1
                )
            ]
            common_binned = common_trajectory.groupby(
                "common_bin", sort=True
            )["heading_corrected_deg"].mean()
            bin_numbers = common_binned.index.to_numpy(float)
            headings = common_binned.to_numpy(float)
            elapsed = np.diff(bin_numbers) * settings.bin_seconds
            angular_change = np.abs(np.diff(headings))
            valid_velocity = (
                np.isfinite(elapsed)
                & (elapsed > 0)
                & np.isfinite(angular_change)
            )
            result["heading_yaw_activity_deg_s"] = (
                float(np.mean(angular_change[valid_velocity] / elapsed[valid_velocity]))
                if valid_velocity.any()
                else np.nan
            )
            result["heading_yaw_activity_bin_pairs"] = int(valid_velocity.sum())
        else:
            result["heading_yaw_activity_deg_s"] = np.nan
            result["heading_yaw_activity_bin_pairs"] = 0

        far_pre = _mean_heading_window(trajectory, passage - 3.0, passage - 2.0)
        immediate_pre = _mean_heading_window(trajectory, passage - 0.5, passage)
        immediate_post = _mean_heading_window(trajectory, passage, passage + 0.5)
        result["far_pre_heading_deg"] = far_pre
        result["immediate_pre_heading_deg"] = immediate_pre
        result["immediate_post_heading_deg"] = immediate_post
        result["far_to_pre_change_deg"] = (
            immediate_pre - far_pre
            if np.isfinite(immediate_pre) and np.isfinite(far_pre)
            else np.nan
        )
        result["passage_change_deg"] = (
            immediate_post - immediate_pre
            if np.isfinite(immediate_post) and np.isfinite(immediate_pre)
            else np.nan
        )

        smooth_heading = _smooth_heading_for_feature_extraction(
            trajectory["time_seconds"].to_numpy(float),
            trajectory["heading_corrected_deg"].to_numpy(float),
        )
        minimum_mask = (
            (trajectory["time_seconds"].to_numpy(float) >= 0.50)
            & (trajectory["time_seconds"].to_numpy(float) <= passage - 0.20)
            & np.isfinite(smooth_heading)
        )
        if minimum_mask.any():
            candidate_indices = np.flatnonzero(minimum_mask)
            minimum_index = int(candidate_indices[np.argmin(smooth_heading[minimum_mask])])
            minimum_heading = float(smooth_heading[minimum_index])
            result["minimum_heading_deg"] = minimum_heading
            result["minimum_time_rel_pass_s"] = float(
                trajectory["time_seconds"].iloc[minimum_index] - passage
            )
            at_pass = result["heading_at_pass_deg"]
            result["recovered_by_pass_deg"] = (
                float(at_pass - minimum_heading) if np.isfinite(at_pass) else np.nan
            )
        else:
            result["minimum_heading_deg"] = np.nan
            result["minimum_time_rel_pass_s"] = np.nan
            result["recovered_by_pass_deg"] = np.nan
    else:
        result["heading_baseline_deg"] = np.nan
        result["heading_baseline_samples"] = 0
        result["heading_passage_mean_deg"] = np.nan
        result["heading_at_pass_deg"] = np.nan
        result["heading_passage_samples"] = 0
        result["heading_common_window_mean_deg"] = np.nan
        result["heading_common_window_mean_abs_deg"] = np.nan
        result["heading_common_window_sd_deg"] = np.nan
        result["heading_yaw_activity_deg_s"] = np.nan
        result["heading_yaw_activity_bin_pairs"] = 0
        result["far_pre_heading_deg"] = np.nan
        result["immediate_pre_heading_deg"] = np.nan
        result["immediate_post_heading_deg"] = np.nan
        result["far_to_pre_change_deg"] = np.nan
        result["passage_change_deg"] = np.nan
        result["minimum_heading_deg"] = np.nan
        result["minimum_time_rel_pass_s"] = np.nan
        result["recovered_by_pass_deg"] = np.nan
    return result


def _sequence_hash(video_ids: Iterable[str]) -> str:
    text = "|".join(video_ids)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _mapping_content_hash(mapping: pd.DataFrame) -> str:
    """Hash participant condition metadata independently of mapping row order."""

    stable = mapping.copy()
    stable.columns = stable.columns.astype(str)
    stable = stable.reindex(sorted(stable.columns), axis=1)
    stable = stable.sort_values("video_id").reset_index(drop=True)
    text = stable.to_csv(index=False, lineterminator="\n", na_rep="NA")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _participant_mapping_path(
    participant_dir: Path,
    participant_id: str,
    filename_template: str,
) -> Path:
    """Resolve the mapping saved inside one randomised participant folder."""

    rendered = filename_template.format(
        participant_id=participant_id,
        participant_folder=participant_dir.name,
    )
    path = participant_dir / rendered
    if not path.is_file():
        raise FileNotFoundError(
            f"Participant mapping file not found: {path}. Expected template: "
            f"{filename_template}"
        )
    return path


def extract_ordering_group(
    root: Path,
    ordering_group: str,
    settings: AnalysisSettings,
    *,
    shared_mapping: pd.DataFrame | None = None,
    shared_mapping_path: Path | None = None,
    participant_mapping_filename: str | None = None,
    timing_mapping: pd.DataFrame | None = None,
    timing_mapping_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Extract all trial rows and a transparent exclusion audit for one group."""

    participants = discover_participants(root)
    if (shared_mapping is None) == (participant_mapping_filename is None):
        raise ValueError(
            "Provide exactly one mapping strategy: a shared mapping or a "
            "participant mapping filename template"
        )
    rows: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    sequences: list[dict[str, Any]] = []

    for participant_dir in participants:
        participant_id = _participant_number(participant_dir)
        participant_uid = f"{ordering_group}:{participant_id}"
        if participant_mapping_filename is not None:
            try:
                mapping_path = _participant_mapping_path(
                    participant_dir,
                    participant_id,
                    participant_mapping_filename,
                )
                mapping = load_mapping(mapping_path)
            except Exception as exc:
                sequences.append(
                    {
                        "ordering_group": ordering_group,
                        "participant_id": participant_id,
                        "participant_uid": participant_uid,
                        "n_mapping_trials": 0,
                        "n_response_trials": 0,
                        "mapping_source": str(
                            participant_dir
                            / participant_mapping_filename.format(
                                participant_id=participant_id,
                                participant_folder=participant_dir.name,
                            )
                        ),
                        "mapping_content_hash": None,
                        "mapping_sequence_hash": None,
                        "mapping_sequence": "",
                        "mapping_order_matches_response": False,
                        "sequence_hash": _sequence_hash([]),
                        "sequence": "",
                    }
                )
                audit.append(
                    {
                        "ordering_group": ordering_group,
                        "participant_id": participant_id,
                        "participant_uid": participant_uid,
                        "video_id": None,
                        "included": False,
                        "reason": "participant_mapping_error",
                        "detail": str(exc),
                        "mapping_source": str(participant_dir),
                    }
                )
                continue
        else:
            assert shared_mapping is not None
            mapping = shared_mapping
            mapping_path = shared_mapping_path

        if timing_mapping is not None:
            try:
                mapping = attach_passage_timing(mapping, timing_mapping)
            except Exception as exc:
                audit.append(
                    {
                        "ordering_group": ordering_group,
                        "participant_id": participant_id,
                        "participant_uid": participant_uid,
                        "video_id": None,
                        "included": False,
                        "reason": "timing_mapping_error",
                        "detail": str(exc),
                        "mapping_source": str(mapping_path),
                        "timing_mapping_source": str(timing_mapping_path),
                    }
                )
                continue

        mapping_index = mapping.set_index("video_id", drop=False)
        mapping_video_ids = mapping["video_id"].astype(str).tolist()
        mapping_source = str(mapping_path) if mapping_path is not None else "shared_mapping"
        timing_mapping_source = (
            str(timing_mapping_path)
            if timing_mapping_path is not None
            else "mapping_contains_passage_timing"
        )
        mapping_hash = _mapping_content_hash(mapping)
        try:
            responses = read_participant_responses(participant_dir)
        except Exception as exc:
            audit.append(
                {
                    "ordering_group": ordering_group,
                    "participant_id": participant_id,
                    "participant_uid": participant_uid,
                    "video_id": None,
                    "included": False,
                    "reason": "response_file_error",
                    "detail": str(exc),
                    "mapping_source": mapping_source,
                    "timing_mapping_source": timing_mapping_source,
                }
            )
            continue

        experimental = responses[responses["video_id"].isin(mapping_index.index)].copy()
        response_video_ids = experimental["video_id"].astype(str).tolist()
        sequences.append(
            {
                "ordering_group": ordering_group,
                "participant_id": participant_id,
                "participant_uid": participant_uid,
                "n_mapping_trials": len(mapping_video_ids),
                "n_response_trials": len(experimental),
                "mapping_source": mapping_source,
                "timing_mapping_source": timing_mapping_source,
                "mapping_content_hash": mapping_hash,
                "mapping_sequence_hash": _sequence_hash(mapping_video_ids),
                "mapping_sequence": "|".join(mapping_video_ids),
                "mapping_order_matches_response": mapping_video_ids == response_video_ids,
                "sequence_hash": _sequence_hash(response_video_ids),
                "sequence": "|".join(response_video_ids),
            }
        )
        for response in experimental.itertuples(index=False):
            response_dict = response._asdict()
            video_id = str(response_dict["video_id"])
            record = {
                "ordering_group": ordering_group,
                "ordering_group_label": GROUP_LABELS[ordering_group],
                "participant_id": participant_id,
                "participant_uid": participant_uid,
                "participant_dir": str(participant_dir),
                "mapping_source": mapping_source,
                "timing_mapping_source": timing_mapping_source,
                "mapping_content_hash": mapping_hash,
                "video_id": video_id,
                "trial_number": int(response_dict["trial_number"]),
                "Q1": response_dict["Q1"],
                "Q2": response_dict["Q2"],
                "Q3": response_dict["Q3"],
            }
            mapping_row = mapping_index.loc[video_id]
            record.update(mapping_row.to_dict())
            try:
                trial_file = find_trial_file(participant_dir, video_id)
                record["trial_file"] = str(trial_file)
                features = extract_trial_features(trial_file, mapping_row, settings)
                record.update(features)
                if settings.require_complete_window and not features["window_complete"]:
                    raise ValueError(
                        f"Incomplete primary window: {features['valid_bins']} of "
                        f"{features['expected_bins']} bins"
                    )
                rows.append(record)
                audit.append(
                    {
                        "ordering_group": ordering_group,
                        "participant_id": participant_id,
                        "participant_uid": participant_uid,
                        "video_id": video_id,
                        "included": True,
                        "reason": "included",
                        "detail": "",
                        "mapping_source": mapping_source,
                        "timing_mapping_source": timing_mapping_source,
                    }
                )
            except Exception as exc:
                audit.append(
                    {
                        "ordering_group": ordering_group,
                        "participant_id": participant_id,
                        "participant_uid": participant_uid,
                        "video_id": video_id,
                        "included": False,
                        "reason": "trial_extraction_error",
                        "detail": str(exc),
                        "mapping_source": mapping_source,
                        "timing_mapping_source": timing_mapping_source,
                    }
                )

    return pd.DataFrame(rows), pd.DataFrame(audit), pd.DataFrame(sequences)


def validate_sample(
    trials: pd.DataFrame,
    audit: pd.DataFrame,
    sequences: pd.DataFrame,
    settings: AnalysisSettings,
) -> tuple[pd.DataFrame, list[str]]:
    """Apply participant level completeness rule and check order manipulation."""

    warnings_out: list[str] = []
    if trials.empty:
        if audit.empty:
            summary = "No extraction audit rows were produced."
        else:
            failure_counts = (
                audit.groupby(["ordering_group", "reason"], dropna=False)
                .size()
                .sort_values(ascending=False)
            )
            summary = "; ".join(
                f"{group}/{reason}: {int(count)}"
                for (group, reason), count in failure_counts.items()
            )
        raise RuntimeError(
            "No valid trial rows were extracted. Failure counts: "
            f"{summary}. See _output/exclusion_audit.csv for participant and "
            "trial level details."
        )
    counts = (
        trials.groupby(["ordering_group", "participant_uid"], observed=True)
        .size()
        .rename("valid_trials")
        .reset_index()
    )
    eligible = counts.loc[
        counts["valid_trials"] >= settings.minimum_valid_trials_per_participant,
        "participant_uid",
    ]
    excluded = counts.loc[
        counts["valid_trials"] < settings.minimum_valid_trials_per_participant
    ]
    if not excluded.empty:
        warnings_out.append(
            f"Excluded {len(excluded)} participant(s) with fewer than "
            f"{settings.minimum_valid_trials_per_participant} valid trials."
        )
    trials = trials[trials["participant_uid"].isin(set(eligible))].copy()

    sequence_summary = (
        sequences[sequences["participant_uid"].isin(set(eligible))]
        .groupby("ordering_group", observed=True)
        .agg(participants=("participant_uid", "nunique"), unique_sequences=("sequence_hash", "nunique"))
        .reset_index()
    )
    fixed = sequence_summary[sequence_summary["ordering_group"] == GROUP_FIXED]
    randomised = sequence_summary[sequence_summary["ordering_group"] == GROUP_RANDOMISED]
    if not fixed.empty and int(fixed.iloc[0]["unique_sequences"]) != 1:
        warnings_out.append(
            "The fixed sequence group contains more than one realised sequence; verify the manipulation."
        )
    if not randomised.empty and int(randomised.iloc[0]["participants"]) > 1:
        if int(randomised.iloc[0]["unique_sequences"]) <= 1:
            warnings_out.append(
                "The randomised order group contains only one realised sequence; verify the manipulation."
            )
        elif int(randomised.iloc[0]["unique_sequences"]) < int(
            randomised.iloc[0]["participants"]
        ):
            warnings_out.append(
                "At least two randomised-order participants have the same realised "
                "sequence. Verify the sequence-generation record and report the "
                "duplicate transparently."
            )

    eligible_sequences = sequences[
        sequences["participant_uid"].isin(set(eligible))
    ].copy()
    if "mapping_order_matches_response" in eligible_sequences:
        mapping_mismatches = eligible_sequences[
            ~eligible_sequences["mapping_order_matches_response"].fillna(False)
        ]
        if not mapping_mismatches.empty:
            warnings_out.append(
                f"The mapping row order differed from the recorded response order for "
                f"{len(mapping_mismatches)} participant(s). Actual trial position was "
                "taken from the response file; inspect mapping_audit.csv."
            )
    randomised_sequences = eligible_sequences[
        eligible_sequences["ordering_group"] == GROUP_RANDOMISED
    ]
    if not randomised_sequences.empty and (
        randomised_sequences["mapping_source"].nunique()
        != randomised_sequences["participant_uid"].nunique()
    ):
        warnings_out.append(
            "The randomised cohort did not resolve one unique mapping file per "
            "participant; inspect mapping_audit.csv."
        )
    fixed_sequences = eligible_sequences[
        eligible_sequences["ordering_group"] == GROUP_FIXED
    ]
    if not fixed_sequences.empty and fixed_sequences["mapping_source"].nunique() != 1:
        warnings_out.append(
            "The fixed sequence cohort used more than one mapping file; inspect "
            "mapping_audit.csv."
        )

    represented = set(trials["ordering_group"].unique())
    if represented != {GROUP_RANDOMISED, GROUP_FIXED}:
        raise RuntimeError(f"Both ordering groups are required; found {sorted(represented)}")
    if trials["participant_uid"].nunique() < 4:
        warnings_out.append("Fewer than four participants are available after exclusions.")
    return trials, warnings_out


def sample_flow_table(
    trials: pd.DataFrame,
    audit: pd.DataFrame,
    sequences: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group in [GROUP_RANDOMISED, GROUP_FIXED]:
        group_audit = audit[audit["ordering_group"] == group]
        group_trials = trials[trials["ordering_group"] == group]
        group_sequences = sequences[sequences["ordering_group"] == group]
        rows.append(
            {
                "ordering_group": group,
                "ordering_group_label": GROUP_LABELS[group],
                "participant_folders": int(group_sequences["participant_uid"].nunique()),
                "participants_analysed": int(group_trials["participant_uid"].nunique()),
                "trials_attempted": int(len(group_audit)),
                "trials_analysed": int(len(group_trials)),
                "trials_excluded": int((~group_audit["included"].fillna(False)).sum()),
                "unique_sequences": int(group_sequences["sequence_hash"].nunique()),
                "median_valid_bins": float(group_trials["valid_bins"].median())
                if not group_trials.empty
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def mapping_audit_table(sequences: pd.DataFrame) -> pd.DataFrame:
    """Compact provenance and order agreement report for every participant mapping."""

    columns = [
        "ordering_group",
        "participant_id",
        "participant_uid",
        "mapping_source",
        "timing_mapping_source",
        "mapping_content_hash",
        "n_mapping_trials",
        "mapping_sequence_hash",
        "n_response_trials",
        "sequence_hash",
        "mapping_order_matches_response",
    ]
    if sequences.empty:
        return pd.DataFrame(columns=columns)
    available = [column for column in columns if column in sequences.columns]
    return sequences[available].sort_values(
        ["ordering_group", "participant_id"]
    ).reset_index(drop=True)


def duplicate_sequence_audit(sequences: pd.DataFrame) -> pd.DataFrame:
    """List participants sharing a realised sequence within an ordering group."""

    duplicate = sequences[
        sequences.duplicated(["ordering_group", "sequence_hash"], keep=False)
    ].copy()
    if duplicate.empty:
        return pd.DataFrame(
            columns=[
                "ordering_group",
                "sequence_hash",
                "participants_sharing_sequence",
                "participant_id",
                "participant_uid",
                "sequence",
            ]
        )
    duplicate["participants_sharing_sequence"] = duplicate.groupby(
        ["ordering_group", "sequence_hash"], observed=True
    )["participant_uid"].transform("nunique")
    return duplicate[
        [
            "ordering_group",
            "sequence_hash",
            "participants_sharing_sequence",
            "participant_id",
            "participant_uid",
            "sequence",
        ]
    ].sort_values(["ordering_group", "sequence_hash", "participant_id"])


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


def _break_segment(trial_number: pd.Series | np.ndarray | float) -> Any:
    """Scheduled break-opportunity segment for trial positions 1 to 40."""

    values = np.asarray(trial_number, dtype=float)
    result = np.where(
        values <= 14,
        "before_break_14",
        np.where(values <= 26, "after_break_14", "after_break_26"),
    )
    return result.item() if result.ndim == 0 else result


def _prepare_progression_frame(trials: pd.DataFrame) -> pd.DataFrame:
    frame = trials.copy()
    # Ten-trial units make the linear term interpretable without changing the
    # model fit.  The midpoint is fixed by design, not estimated from missingness.
    frame["trial_scaled"] = (pd.to_numeric(frame["trial_number"], errors="coerce") - 20.5) / 10.0
    frame["break_segment"] = _break_segment(frame["trial_number"])
    return frame


def _progression_prediction_table(
    result: Any,
    outcome: str,
    spec: dict[str, Any],
    working_correlation: str,
) -> pd.DataFrame:
    """Equal-cell predictions over trial position for adjusted temporal plots."""

    base = _factorial_prediction_grid()
    link = "logit" if spec["model_family"] == "binomial" else "identity"
    scale = float(spec["scale"])
    critical = st.norm.ppf(0.975)
    rows: list[dict[str, Any]] = []
    design_info = result.model.data.design_info
    for group in [GROUP_RANDOMISED, GROUP_FIXED]:
        for trial_number in range(1, 41):
            grid = base.copy()
            grid["ordering_group"] = group
            grid["trial_scaled"] = (trial_number - 20.5) / 10.0
            grid["break_segment"] = _break_segment(float(trial_number))
            estimate, _, variance = _average_prediction(result, design_info, grid, link)
            standard_error = math.sqrt(variance)
            lower = estimate - critical * standard_error
            upper = estimate + critical * standard_error
            if link == "logit":
                lower, upper = max(0.0, lower), min(1.0, upper)
            rows.append(
                {
                    "outcome": outcome,
                    "outcome_label": spec["label"],
                    "outcome_family": spec["family"],
                    "analysis_role": spec["role"],
                    "units": spec["units"],
                    "ordering_group": group,
                    "ordering_group_label": GROUP_LABELS[group],
                    "trial_number": trial_number,
                    "break_segment": _break_segment(float(trial_number)),
                    "adjusted_estimate": scale * estimate,
                    "ci_low": scale * lower,
                    "ci_high": scale * upper,
                    "model_family": spec["model_family"],
                    "working_correlation": working_correlation,
                    "standardisation": "equal weighting of the 2 by 2 by 2 by 5 condition grid",
                }
            )
    return pd.DataFrame(rows)


def run_adjusted_temporal_models(trials: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Condition-adjusted progression models for trigger, ratings, and heading."""

    frame_all = _prepare_progression_frame(trials)
    group_term = "C(ordering_group, Treatment(reference='randomised_order'))"
    rhs = (
        f"{_design_formula(False)} + trial_scaled + I(trial_scaled ** 2) "
        f"+ C(break_segment) + {group_term}:trial_scaled "
        f"+ {group_term}:I(trial_scaled ** 2)"
    )
    coefficient_tables: list[pd.DataFrame] = []
    test_rows: list[dict[str, Any]] = []
    prediction_tables: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []
    critical = st.norm.ppf(0.975)

    for outcome in TEMPORAL_MODEL_OUTCOMES:
        spec = OUTCOME_SPECS[outcome]
        if outcome not in frame_all.columns:
            failures.append({"outcome": outcome, "reason": "Outcome column was not available"})
            continue
        frame = frame_all.dropna(
            subset=[outcome, "participant_uid", "trial_scaled", "break_segment"]
        ).copy()
        if frame.empty or frame["participant_uid"].nunique() < 3 or frame[outcome].nunique() < 2:
            failures.append({"outcome": outcome, "reason": "Insufficient observations or variation"})
            continue
        try:
            result, working = _fit_gee_with_fallback(
                f"{outcome} ~ {rhs}", frame, spec["model_family"]
            )
        except RuntimeError as exc:
            failures.append({"outcome": outcome, "reason": str(exc)})
            LOGGER.warning("Adjusted temporal model omitted for %s: %s", outcome, exc)
            continue

        coefficients = _secondary_coefficient_table(result, outcome, spec, working)
        coefficients.insert(0, "model", "condition_adjusted_trial_position")
        coefficient_tables.append(coefficients)
        prediction_tables.append(
            _progression_prediction_table(result, outcome, spec, working)
        )

        names = list(pd.Series(result.params).index.astype(str))
        linear = [
            index
            for index, name in enumerate(names)
            if "C(ordering_group" in name
            and ":trial_scaled" in name
            and "I(trial_scaled ** 2)" not in name
        ]
        quadratic = [
            index
            for index, name in enumerate(names)
            if "C(ordering_group" in name and "I(trial_scaled ** 2)" in name
        ]
        for test_name, indexes in [
            ("group by linear trial position", linear),
            ("group by quadratic trial position", quadratic),
            ("joint group by trial position", linear + quadratic),
        ]:
            test = _wald_test(result, indexes, test_name)
            estimate = standard_error = ci_low = ci_high = np.nan
            standardised_estimate = standardised_ci_low = standardised_ci_high = np.nan
            estimate_scale = "joint Wald test"
            if len(indexes) == 1:
                estimate = float(result.params.iloc[indexes[0]])
                covariance = np.asarray(result.cov_params(), dtype=float)
                standard_error = math.sqrt(
                    max(0.0, float(covariance[indexes[0], indexes[0]]))
                )
                ci_low = estimate - critical * standard_error
                ci_high = estimate + critical * standard_error
                if "quadratic" in test_name:
                    estimate_scale = (
                        "log odds per squared ten-trial unit"
                        if spec["model_family"] == "binomial"
                        else f"{spec['units']} per squared ten-trial unit"
                    )
                else:
                    estimate_scale = (
                        "log odds per ten trials"
                        if spec["model_family"] == "binomial"
                        else f"{spec['units']} per ten trials"
                    )
                if spec["model_family"] != "binomial":
                    scale = float(spec["scale"])
                    estimate *= scale
                    standard_error *= scale
                    ci_low *= scale
                    ci_high *= scale
                    outcome_sd = float(pd.to_numeric(frame[outcome], errors="coerce").std(ddof=1)) * scale
                    if np.isfinite(outcome_sd) and outcome_sd > 0:
                        standardised_estimate = estimate / outcome_sd
                        standardised_ci_low = ci_low / outcome_sd
                        standardised_ci_high = ci_high / outcome_sd
            test_rows.append(
                {
                    "outcome": outcome,
                    "outcome_label": spec["label"],
                    "outcome_family": spec["family"],
                    "analysis_role": spec["role"],
                    "test": test_name,
                    "df": test["df"],
                    "chi_square": test["chi_square"],
                    "estimate_fixed_minus_randomised": estimate,
                    "standard_error": standard_error,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "standardised_estimate": standardised_estimate,
                    "standardised_ci_low": standardised_ci_low,
                    "standardised_ci_high": standardised_ci_high,
                    "estimate_scale": estimate_scale,
                    "p_value": test["p_value"],
                    "working_correlation": working,
                    "condition_adjustment": "full factorial condition structure plus scheduled break-opportunity segment",
                }
            )

    tests = pd.DataFrame(test_rows)
    if not tests.empty:
        tests = _holm_within(tests, ["outcome_family", "test"])
        tests["multiplicity_method"] = (
            "Holm across outcomes within family, separately for each temporal test"
        )
        tests = _add_global_holm(tests, ["test"])
    return {
        "temporal_adjusted_coefficients": (
            pd.concat(coefficient_tables, ignore_index=True)
            if coefficient_tables
            else pd.DataFrame()
        ),
        "temporal_adjusted_tests": tests,
        "temporal_adjusted_predictions": (
            pd.concat(prediction_tables, ignore_index=True)
            if prediction_tables
            else pd.DataFrame()
        ),
        "temporal_adjusted_failures": pd.DataFrame(
            failures, columns=["outcome", "reason"]
        ),
    }


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


SESSION_SEGMENTS = (
    ("before_break_14", "Trials 1-14"),
    ("after_break_14", "Trials 15-26"),
    ("after_break_26", "Trials 27-40"),
)


def _delta_contrast(
    result: Any,
    estimate: float,
    gradient: np.ndarray,
    scale: float = 1.0,
) -> dict[str, float]:
    """Delta-method contrast on the response scale."""

    covariance = np.asarray(result.cov_params(), dtype=float)
    variance = float(gradient @ covariance @ gradient)
    standard_error = math.sqrt(max(0.0, variance))
    z_value = estimate / standard_error if standard_error > 0 else np.nan
    critical = st.norm.ppf(0.975)
    return {
        "estimate": scale * estimate,
        "standard_error": scale * standard_error,
        "ci_low": scale * (estimate - critical * standard_error),
        "ci_high": scale * (estimate + critical * standard_error),
        "z": z_value,
        "p_value": (
            float(2.0 * st.norm.sf(abs(z_value)))
            if np.isfinite(z_value)
            else np.nan
        ),
    }


def run_session_segment_models(trials: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Categorical early/middle/late robustness models aligned to break opportunities.

    The polynomial models provide a compact trajectory, but their conclusion can
    depend on the assumed curve. This analysis instead treats trials 1-14,
    15-26, and 27-40 as categories. The segments are labelled by scheduled break
    opportunities; they do not imply that a participant actually took a break.
    """

    frame_all = _prepare_progression_frame(trials)
    group_term = "C(ordering_group, Treatment(reference='randomised_order'))"
    rhs = (
        f"{_design_formula(False)} + C(break_segment) "
        f"+ {group_term}:C(break_segment)"
    )
    coefficient_tables: list[pd.DataFrame] = []
    prediction_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for outcome in TEMPORAL_MODEL_OUTCOMES:
        spec = OUTCOME_SPECS[outcome]
        if outcome not in frame_all.columns:
            failures.append({"outcome": outcome, "reason": "Outcome column was not available"})
            continue
        frame = frame_all.dropna(
            subset=[outcome, "participant_uid", "break_segment"]
        ).copy()
        if frame.empty or frame["participant_uid"].nunique() < 3 or frame[outcome].nunique() < 2:
            failures.append({"outcome": outcome, "reason": "Insufficient observations or variation"})
            continue
        try:
            result, working = _fit_gee_with_fallback(
                f"{outcome} ~ {rhs}", frame, spec["model_family"]
            )
        except RuntimeError as exc:
            failures.append({"outcome": outcome, "reason": str(exc)})
            LOGGER.warning("Session-segment model omitted for %s: %s", outcome, exc)
            continue

        coefficients = _secondary_coefficient_table(result, outcome, spec, working)
        coefficients.insert(0, "model", "categorical_session_segment")
        coefficient_tables.append(coefficients)
        link = "logit" if spec["model_family"] == "binomial" else "identity"
        scale = float(spec["scale"])
        design_info = result.model.data.design_info
        components: dict[tuple[str, str], tuple[float, np.ndarray, float]] = {}
        critical = st.norm.ppf(0.975)
        for ordering_group in [GROUP_RANDOMISED, GROUP_FIXED]:
            for segment, segment_label in SESSION_SEGMENTS:
                grid = _factorial_prediction_grid()
                grid["ordering_group"] = ordering_group
                grid["break_segment"] = segment
                estimate, gradient, variance = _average_prediction(
                    result, design_info, grid, link
                )
                components[(ordering_group, segment)] = (estimate, gradient, variance)
                standard_error = math.sqrt(variance)
                prediction_rows.append(
                    {
                        "outcome": outcome,
                        "outcome_label": spec["label"],
                        "outcome_family": spec["family"],
                        "analysis_role": spec["role"],
                        "units": spec["units"],
                        "ordering_group": ordering_group,
                        "ordering_group_label": GROUP_LABELS[ordering_group],
                        "session_segment": segment,
                        "session_segment_label": segment_label,
                        "adjusted_estimate": scale * estimate,
                        "ci_low": scale * (estimate - critical * standard_error),
                        "ci_high": scale * (estimate + critical * standard_error),
                        "working_correlation": working,
                        "standardisation": "equal weighting of the 2 by 2 by 2 by 5 condition grid",
                    }
                )

        names = list(pd.Series(result.params).index.astype(str))
        interaction_indices = [
            index
            for index, name in enumerate(names)
            if "C(ordering_group" in name and "C(break_segment)" in name
        ]
        joint = _wald_test(
            result, interaction_indices, "joint group by session segment"
        )
        test_rows.append(
            {
                "outcome": outcome,
                "outcome_label": spec["label"],
                "outcome_family": spec["family"],
                "analysis_role": spec["role"],
                "test": joint["test"],
                "df": joint["df"],
                "chi_square": joint["chi_square"],
                "estimate": np.nan,
                "standard_error": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "z": np.nan,
                "p_value": joint["p_value"],
                "estimate_scale": "joint Wald test",
                "working_correlation": working,
            }
        )

        early = SESSION_SEGMENTS[0][0]
        late = SESSION_SEGMENTS[-1][0]
        changes: dict[str, tuple[float, np.ndarray]] = {}
        for ordering_group in [GROUP_RANDOMISED, GROUP_FIXED]:
            late_estimate, late_gradient, _ = components[(ordering_group, late)]
            early_estimate, early_gradient, _ = components[(ordering_group, early)]
            change = late_estimate - early_estimate
            gradient = late_gradient - early_gradient
            changes[ordering_group] = (change, gradient)
            statistics = _delta_contrast(result, change, gradient, scale)
            test_rows.append(
                {
                    "outcome": outcome,
                    "outcome_label": spec["label"],
                    "outcome_family": spec["family"],
                    "analysis_role": spec["role"],
                    "test": f"early-to-late change: {ordering_group}",
                    "df": 1,
                    "chi_square": statistics["z"] ** 2,
                    **statistics,
                    "estimate_scale": spec["units"],
                    "working_correlation": working,
                }
            )

        fixed_change, fixed_gradient = changes[GROUP_FIXED]
        random_change, random_gradient = changes[GROUP_RANDOMISED]
        difference_statistics = _delta_contrast(
            result,
            fixed_change - random_change,
            fixed_gradient - random_gradient,
            scale,
        )
        test_rows.append(
            {
                "outcome": outcome,
                "outcome_label": spec["label"],
                "outcome_family": spec["family"],
                "analysis_role": spec["role"],
                "test": "difference in early-to-late change",
                "df": 1,
                "chi_square": difference_statistics["z"] ** 2,
                **difference_statistics,
                "estimate_scale": spec["units"],
                "working_correlation": working,
            }
        )

    tests = pd.DataFrame(test_rows)
    if not tests.empty:
        tests = _holm_within(tests, ["outcome_family", "test"])
        tests["multiplicity_method"] = (
            "Holm across outcomes within family, separately for each session-segment test"
        )
        tests = _add_global_holm(tests, ["test"])
    return {
        "session_segment_coefficients": (
            pd.concat(coefficient_tables, ignore_index=True)
            if coefficient_tables
            else pd.DataFrame()
        ),
        "session_segment_predictions": pd.DataFrame(prediction_rows),
        "session_segment_tests": tests,
        "session_segment_failures": pd.DataFrame(
            failures, columns=["outcome", "reason"]
        ),
    }


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


def _prepare_prior_exposures(trials: pd.DataFrame) -> pd.DataFrame:
    """Count previous trials at the current level of each factorial variable."""

    frame = trials.sort_values(["participant_uid", "trial_number"]).copy()
    for factor, _ in EXPOSURE_FACTORS:
        name = f"prior_same_{factor}_per10"
        frame[name] = (
            frame.groupby(["participant_uid", factor], observed=True).cumcount() / 10.0
        )
    return frame


def run_prior_exposure_models(trials: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Replace trial position with prior same-level factorial exposure counts."""

    frame_all = _prepare_prior_exposures(trials)
    group_term = "C(ordering_group, Treatment(reference='randomised_order'))"
    exposure_columns = [f"prior_same_{factor}_per10" for factor, _ in EXPOSURE_FACTORS]
    rhs = (
        f"{_design_formula(False)} + "
        + " + ".join(exposure_columns)
        + " + "
        + " + ".join(f"{group_term}:{column}" for column in exposure_columns)
    )
    coefficient_tables: list[pd.DataFrame] = []
    test_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    critical = st.norm.ppf(0.975)

    for outcome in EXPOSURE_MODEL_OUTCOMES:
        spec = OUTCOME_SPECS[outcome]
        if outcome not in frame_all.columns:
            failures.append({"outcome": outcome, "reason": "Outcome column was not available"})
            continue
        frame = frame_all.dropna(subset=[outcome, "participant_uid", *exposure_columns]).copy()
        if frame.empty or frame["participant_uid"].nunique() < 3 or frame[outcome].nunique() < 2:
            failures.append({"outcome": outcome, "reason": "Insufficient observations or variation"})
            continue
        try:
            result, working = _fit_gee_with_fallback(
                f"{outcome} ~ {rhs}", frame, spec["model_family"]
            )
        except RuntimeError as exc:
            failures.append({"outcome": outcome, "reason": str(exc)})
            LOGGER.warning("Prior-exposure model omitted for %s: %s", outcome, exc)
            continue

        coefficients = _secondary_coefficient_table(result, outcome, spec, working)
        coefficients.insert(0, "model", "prior_same_level_exposure")
        coefficient_tables.append(coefficients)
        names = list(pd.Series(result.params).index.astype(str))
        interaction_indexes: list[int] = []
        for (factor, factor_label), column in zip(EXPOSURE_FACTORS, exposure_columns):
            indexes = [
                index
                for index, name in enumerate(names)
                if "C(ordering_group" in name and name.endswith(f":{column}")
            ]
            interaction_indexes.extend(indexes)
            test = _wald_test(result, indexes, f"group by prior {factor_label} exposure")
            estimate = standard_error = ci_low = ci_high = np.nan
            standardised_estimate = np.nan
            estimate_scale = "joint Wald test"
            if len(indexes) == 1:
                estimate = float(result.params.iloc[indexes[0]])
                covariance = np.asarray(result.cov_params(), dtype=float)
                standard_error = math.sqrt(
                    max(0.0, float(covariance[indexes[0], indexes[0]]))
                )
                ci_low = estimate - critical * standard_error
                ci_high = estimate + critical * standard_error
                estimate_scale = (
                    "log odds per ten prior same-level exposures"
                    if spec["model_family"] == "binomial"
                    else f"{spec['units']} per ten prior same-level exposures"
                )
                if spec["model_family"] != "binomial":
                    scale = float(spec["scale"])
                    estimate *= scale
                    standard_error *= scale
                    ci_low *= scale
                    ci_high *= scale
                    outcome_sd = float(pd.to_numeric(frame[outcome], errors="coerce").std(ddof=1)) * scale
                    if np.isfinite(outcome_sd) and outcome_sd > 0:
                        standardised_estimate = estimate / outcome_sd
            test_rows.append(
                {
                    "outcome": outcome,
                    "outcome_label": spec["label"],
                    "outcome_family": spec["family"],
                    "analysis_role": spec["role"],
                    "exposure_factor": factor,
                    "test": test["test"],
                    "df": test["df"],
                    "chi_square": test["chi_square"],
                    "estimate_fixed_minus_randomised": estimate,
                    "standard_error": standard_error,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "standardised_estimate": standardised_estimate,
                    "estimate_scale": estimate_scale,
                    "p_value": test["p_value"],
                    "working_correlation": working,
                }
            )
        joint = _wald_test(result, interaction_indexes, "joint group by prior exposure")
        test_rows.append(
            {
                "outcome": outcome,
                "outcome_label": spec["label"],
                "outcome_family": spec["family"],
                "analysis_role": spec["role"],
                "exposure_factor": "joint",
                "test": joint["test"],
                "df": joint["df"],
                "chi_square": joint["chi_square"],
                "estimate_fixed_minus_randomised": np.nan,
                "standard_error": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "estimate_scale": "joint Wald test",
                "p_value": joint["p_value"],
                "working_correlation": working,
            }
        )

    tests = pd.DataFrame(test_rows)
    if not tests.empty:
        tests = _holm_within(tests, ["outcome_family", "test"])
        tests["multiplicity_method"] = (
            "Holm across outcomes within family, separately for each exposure test"
        )
        tests = _add_global_holm(tests, ["test"])
    return {
        "exposure_adjusted_coefficients": (
            pd.concat(coefficient_tables, ignore_index=True)
            if coefficient_tables
            else pd.DataFrame()
        ),
        "exposure_adjusted_tests": tests,
        "exposure_adjusted_failures": pd.DataFrame(
            failures, columns=["outcome", "reason"]
        ),
    }


def _hedges_g(first: np.ndarray, second: np.ndarray) -> float:
    """Hedges g where positive means fixed sequence is higher."""

    n1, n2 = len(first), len(second)
    if n1 < 2 or n2 < 2:
        return np.nan
    pooled_var = ((n1 - 1) * np.var(first, ddof=1) + (n2 - 1) * np.var(second, ddof=1)) / (
        n1 + n2 - 2
    )
    if pooled_var <= 0:
        return np.nan
    d = (np.mean(second) - np.mean(first)) / math.sqrt(pooled_var)
    correction = 1.0 - 3.0 / (4.0 * (n1 + n2) - 9.0)
    return correction * d


def _welch_difference(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    """Fixed sequence minus randomised order, with Welch confidence interval."""

    n1, n2 = len(first), len(second)
    mean_difference = float(np.mean(second) - np.mean(first))
    v1, v2 = np.var(first, ddof=1), np.var(second, ddof=1)
    se2 = v1 / n1 + v2 / n2
    se = math.sqrt(se2)
    numerator = se2**2
    denominator = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
    df = numerator / denominator if denominator > 0 else np.nan
    critical = st.t.ppf(0.975, df) if np.isfinite(df) else np.nan
    t_value = mean_difference / se if se > 0 else np.nan
    p_value = 2.0 * st.t.sf(abs(t_value), df) if np.isfinite(t_value) else np.nan
    return {
        "difference_fixed_minus_randomised": mean_difference,
        "standard_error": se,
        "degrees_of_freedom": df,
        "ci_low": mean_difference - critical * se,
        "ci_high": mean_difference + critical * se,
        "t": t_value,
        "p_value": p_value,
    }


def participant_level_analysis(
    trials: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outcomes = [outcome for outcome in PARTICIPANT_OUTCOMES if outcome in trials.columns]
    participant_means = (
        trials.groupby(
            ["ordering_group", "ordering_group_label", "participant_uid"], observed=True
        )[outcomes]
        .mean()
        .reset_index()
    )
    if "any_trigger_press" in participant_means:
        participant_means["any_trigger_press"] *= 100.0
    descriptive_records: list[dict[str, Any]] = []
    comparison_records: list[dict[str, Any]] = []
    for outcome in outcomes:
        spec = OUTCOME_SPECS[outcome]
        for group in [GROUP_RANDOMISED, GROUP_FIXED]:
            values = participant_means.loc[
                participant_means["ordering_group"] == group, outcome
            ].dropna()
            descriptive_records.append(
                {
                    "outcome": outcome,
                    "outcome_label": spec["label"],
                    "outcome_family": spec["family"],
                    "analysis_role": spec["role"],
                    "units": spec["units"],
                    "ordering_group": group,
                    "ordering_group_label": GROUP_LABELS[group],
                    "n_participants": int(values.size),
                    "mean": float(values.mean()) if not values.empty else np.nan,
                    "standard_deviation": float(values.std(ddof=1)) if values.size > 1 else np.nan,
                    "median": float(values.median()) if not values.empty else np.nan,
                    "q1": float(values.quantile(0.25)) if not values.empty else np.nan,
                    "q3": float(values.quantile(0.75)) if not values.empty else np.nan,
                }
            )
        randomised = participant_means.loc[
            participant_means["ordering_group"] == GROUP_RANDOMISED, outcome
        ].dropna().to_numpy(float)
        fixed = participant_means.loc[
            participant_means["ordering_group"] == GROUP_FIXED, outcome
        ].dropna().to_numpy(float)
        if len(randomised) < 2 or len(fixed) < 2:
            continue
        comparison = _welch_difference(randomised, fixed)
        comparison.update(
            {
                "outcome": outcome,
                "outcome_label": spec["label"],
                "outcome_family": spec["family"],
                "analysis_role": spec["role"],
                "units": spec["units"],
                "n_randomised": len(randomised),
                "n_fixed": len(fixed),
                "hedges_g_fixed_minus_randomised": _hedges_g(randomised, fixed),
            }
        )
        comparison_records.append(comparison)

    comparisons = pd.DataFrame(comparison_records)
    if not comparisons.empty:
        comparisons = _holm_within(comparisons, ["outcome_family"])
        comparisons["multiplicity_method"] = (
            "Holm across participant-mean outcomes within outcome family"
        )
        comparisons = _add_global_holm(comparisons)
    return (
        participant_means,
        pd.DataFrame(descriptive_records),
        comparisons,
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


def _participant_temporal_metrics(group: pd.DataFrame, outcome: str) -> dict[str, float]:
    frame = group.dropna(subset=[outcome, "trial_number"]).sort_values("trial_number")
    if len(frame) < 4:
        return {"slope_per_trial": np.nan, "early_late_difference": np.nan}
    x = frame["trial_number"].to_numpy(float)
    y = frame[outcome].to_numpy(float)
    slope = float(np.polyfit(x, y, 1)[0])
    early = y[x <= np.quantile(x, 0.25)]
    late = y[x >= np.quantile(x, 0.75)]
    return {
        "slope_per_trial": slope,
        "early_late_difference": float(np.mean(late) - np.mean(early)),
    }


def temporal_descriptives(trials: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Participant slopes and late minus early drift as secondary summaries."""

    records: list[dict[str, Any]] = []
    requested = [
        "unsafe_pct",
        "Q1",
        "Q2",
        "Q3",
        "mean_trigger",
        "peak_trigger",
        "heading_at_pass_deg",
        "minimum_heading_deg",
        "passage_change_deg",
    ]
    outcomes = [outcome for outcome in requested if outcome in trials.columns]
    for (ordering_group, participant_uid), group in trials.groupby(
        ["ordering_group", "participant_uid"], observed=True
    ):
        for outcome in outcomes:
            metrics = _participant_temporal_metrics(group, outcome)
            for metric, value in metrics.items():
                records.append(
                    {
                        "ordering_group": ordering_group,
                        "participant_uid": participant_uid,
                        "outcome": outcome,
                        "metric": metric,
                        "value": value,
                    }
                )
    participant_metrics = pd.DataFrame(records)
    comparisons: list[dict[str, Any]] = []
    for (outcome, metric), group in participant_metrics.groupby(["outcome", "metric"]):
        randomised = group.loc[group["ordering_group"] == GROUP_RANDOMISED, "value"].dropna().to_numpy(float)
        fixed = group.loc[group["ordering_group"] == GROUP_FIXED, "value"].dropna().to_numpy(float)
        if len(randomised) < 2 or len(fixed) < 2:
            continue
        row = _welch_difference(randomised, fixed)
        row.update(
            {
                "outcome": outcome,
                "metric": metric,
                "n_randomised": len(randomised),
                "n_fixed": len(fixed),
                "hedges_g_fixed_minus_randomised": _hedges_g(randomised, fixed),
            }
        )
        comparisons.append(row)
    table = pd.DataFrame(comparisons)
    if not table.empty:
        table["p_value_adjusted_fdr"] = multipletests(table["p_value"], method="fdr_bh")[1]
        table["multiplicity_method"] = "Benjamini Hochberg across drift tests"
    return participant_metrics, table


def _condition_adjusted_residuals(frame: pd.DataFrame, outcome: str) -> pd.Series:
    """Remove current trial factorial condition means within one participant."""

    columns = ["yielding", "eHMIOn", "camera", "distPed_m"]
    work = frame[[outcome, *columns]].copy()
    valid = work.notna().all(axis=1)
    residuals = pd.Series(np.nan, index=frame.index, dtype=float)
    if valid.sum() < 10:
        return residuals
    condition = pd.get_dummies(
        work.loc[valid, columns].astype(
            {"yielding": "int64", "eHMIOn": "int64", "camera": "int64", "distPed_m": "float64"}
        ),
        columns=columns,
        drop_first=True,
        dtype=float,
    )
    design = np.column_stack([np.ones(len(condition)), condition.to_numpy(float)])
    values = work.loc[valid, outcome].to_numpy(float)
    coefficients, _, _, _ = np.linalg.lstsq(design, values, rcond=None)
    residuals.loc[valid] = values - design @ coefficients
    return residuals


def carryover_analysis(trials: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exploratory previous trial effects adjusted for the current condition."""

    requested = [
        "unsafe_pct",
        "Q1",
        "Q2",
        "Q3",
        "mean_trigger",
        "heading_at_pass_deg",
        "passage_change_deg",
    ]
    outcomes = [outcome for outcome in requested if outcome in trials.columns]
    previous_factors = ["yielding", "eHMIOn", "camera", "distPed_m"]
    records: list[dict[str, Any]] = []
    for (ordering_group, participant_uid), participant in trials.groupby(
        ["ordering_group", "participant_uid"], observed=True
    ):
        participant = participant.sort_values("trial_number").copy()
        for factor in previous_factors:
            participant[f"previous_{factor}"] = participant[factor].shift(1)
        for outcome in outcomes:
            participant["adjusted_outcome"] = _condition_adjusted_residuals(participant, outcome)
            for factor in previous_factors:
                previous = f"previous_{factor}"
                frame = participant.dropna(subset=["adjusted_outcome", previous])
                effect = np.nan
                if factor == "distPed_m":
                    if len(frame) >= 10 and frame[previous].nunique() > 1:
                        effect = float(
                            np.polyfit(
                                frame[previous].to_numpy(float),
                                frame["adjusted_outcome"].to_numpy(float),
                                1,
                            )[0]
                        )
                    estimand = "adjusted slope per previous metre"
                else:
                    zero = frame.loc[frame[previous] == 0, "adjusted_outcome"]
                    one = frame.loc[frame[previous] == 1, "adjusted_outcome"]
                    if len(zero) >= 2 and len(one) >= 2:
                        effect = float(one.mean() - zero.mean())
                    estimand = "adjusted previous 1 minus previous 0"
                records.append(
                    {
                        "ordering_group": ordering_group,
                        "participant_uid": participant_uid,
                        "outcome": outcome,
                        "previous_factor": factor,
                        "estimand": estimand,
                        "carryover_effect": effect,
                    }
                )

    participant_effects = pd.DataFrame(records)
    comparisons: list[dict[str, Any]] = []
    for (outcome, factor), frame in participant_effects.groupby(
        ["outcome", "previous_factor"], observed=True
    ):
        randomised = frame.loc[
            frame["ordering_group"] == GROUP_RANDOMISED, "carryover_effect"
        ].dropna().to_numpy(float)
        fixed = frame.loc[
            frame["ordering_group"] == GROUP_FIXED, "carryover_effect"
        ].dropna().to_numpy(float)
        if len(randomised) < 2 or len(fixed) < 2:
            continue
        row = _welch_difference(randomised, fixed)
        row.update(
            {
                "outcome": outcome,
                "previous_factor": factor,
                "n_randomised": len(randomised),
                "n_fixed": len(fixed),
                "hedges_g_fixed_minus_randomised": _hedges_g(randomised, fixed),
            }
        )
        comparisons.append(row)
    table = pd.DataFrame(comparisons)
    if not table.empty:
        table["p_value_adjusted_fdr"] = multipletests(table["p_value"], method="fdr_bh")[1]
        table["multiplicity_method"] = "Benjamini Hochberg across carryover tests"
    return participant_effects, table


def sequence_position_audit(trials: pd.DataFrame) -> pd.DataFrame:
    """Quantify trial position association with each condition within each group."""

    records: list[dict[str, Any]] = []
    for (ordering_group, participant_uid), frame in trials.groupby(
        ["ordering_group", "participant_uid"], observed=True
    ):
        trial = frame["trial_number"].to_numpy(float)
        for factor in ["yielding", "eHMIOn", "camera", "distPed_m"]:
            values = pd.to_numeric(frame[factor], errors="coerce").to_numpy(float)
            valid = np.isfinite(trial) & np.isfinite(values)
            correlation = (
                float(np.corrcoef(trial[valid], values[valid])[0, 1])
                if valid.sum() > 2 and np.std(values[valid]) > 0
                else np.nan
            )
            records.append(
                {
                    "ordering_group": ordering_group,
                    "participant_uid": participant_uid,
                    "factor": factor,
                    "trial_position_correlation": correlation,
                }
            )
    participant = pd.DataFrame(records)
    summary = (
        participant.groupby(["ordering_group", "factor"], observed=True)[
            "trial_position_correlation"
        ]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
    )
    return summary


def _normalise_participant_id(value: Any) -> str:
    digits = re.sub(r"\D+", "", str(value))
    if digits:
        return str(int(digits))
    return str(value).strip().lower()


def _infer_questionnaire_id_column(frame: pd.DataFrame) -> str | None:
    columns = list(frame.columns)
    for token in ["participant", "subject", "respondent"]:
        matches = [column for column in columns if token in str(column).lower()]
        if matches:
            return str(matches[0])
    exact = [column for column in columns if str(column).strip().lower() in {"id", "pid"}]
    return str(exact[0]) if exact else None


def _read_questionnaire(path: Path) -> pd.DataFrame:
    for separator in [",", ";"]:
        try:
            frame = pd.read_csv(path, sep=separator)
        except Exception:
            continue
        if len(frame.columns) > 1:
            return frame
    raise ValueError(f"Questionnaire could not be read: {path}")


def _normalise_gender(value: Any) -> str:
    if pd.isna(value) or not str(value).strip():
        return "Missing"
    text = str(value).strip().lower()
    if text in {"male", "man", "m"}:
        return "Male"
    if text in {"female", "woman", "f"}:
        return "Female"
    if "prefer" in text or "disclos" in text or text in {"other", "non-binary", "nonbinary"}:
        return "Other or not disclosed"
    return str(value).strip()


def _normalise_vr_experience(value: Any) -> str:
    if pd.isna(value) or not str(value).strip():
        return "Missing"
    text = str(value).strip().lower()
    if "never" in text or (("not" in text or "no " in text) and "month" in text):
        return "None in past month"
    if "less than once" in text or "<" in text:
        return "Less than weekly"
    if any(token in text for token in ["week", "daily", "regular", "often", "times"]):
        return "Regular"
    return "Other"


def load_demographics(
    config: StudyConfig,
    trials: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load prespecified baseline covariates and audit participant linkage."""

    records: list[pd.DataFrame] = []
    status_records: list[dict[str, Any]] = []
    for group, path in [
        (GROUP_RANDOMISED, config.randomised_intake),
        (GROUP_FIXED, config.fixed_intake),
    ]:
        if path is None or not path.is_file():
            status_records.append(
                {
                    "ordering_group": group,
                    "status": "unavailable",
                    "detail": f"Intake questionnaire not found: {path}",
                }
            )
            continue
        try:
            frame = _read_questionnaire(path)
            id_column = _infer_questionnaire_id_column(frame)
            if id_column is None:
                raise ValueError(
                    "No participant identifier column was found. Add participant or subject to its heading."
                )
            missing_fields = [
                field
                for field in [INTAKE_AGE, INTAKE_GENDER, INTAKE_VR]
                if field not in frame.columns
            ]
            if missing_fields:
                raise ValueError(f"Missing baseline fields: {missing_fields}")
            selected = pd.DataFrame(
                {
                    "ordering_group": group,
                    "participant_id": frame[id_column].map(_normalise_participant_id),
                    "age": pd.to_numeric(frame[INTAKE_AGE], errors="coerce"),
                    "gender": frame[INTAKE_GENDER].map(_normalise_gender),
                    "vr_experience": frame[INTAKE_VR].map(_normalise_vr_experience),
                    "nationality": frame[INTAKE_NATIONALITY].astype(str).str.strip()
                    if INTAKE_NATIONALITY in frame.columns
                    else "",
                }
            )
            selected["participant_uid"] = (
                selected["ordering_group"] + ":" + selected["participant_id"]
            )
            selected = selected.drop_duplicates("participant_uid", keep="first")
            records.append(selected)
            status_records.append(
                {
                    "ordering_group": group,
                    "status": "loaded",
                    "detail": f"{len(selected)} unique intake records; ID column: {id_column}",
                }
            )
        except Exception as exc:
            status_records.append(
                {"ordering_group": group, "status": "error", "detail": str(exc)}
            )

    demographics = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    analysed = set(trials["participant_uid"].unique())
    linked = set(demographics["participant_uid"].unique()) if not demographics.empty else set()
    for group in [GROUP_RANDOMISED, GROUP_FIXED]:
        group_analysed = {
            value for value in analysed if str(value).startswith(f"{group}:")
        }
        group_linked = group_analysed.intersection(linked)
        status_records.append(
            {
                "ordering_group": group,
                "status": "linkage",
                "detail": (
                    f"{len(group_linked)} of {len(group_analysed)} analysed participants linked; "
                    f"unlinked IDs: {sorted(group_analysed.difference(linked))}"
                ),
            }
        )
    return demographics, pd.DataFrame(status_records)


def baseline_demographic_tables(
    demographics: pd.DataFrame,
    trials: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Describe cohort composition and quantify baseline imbalance."""

    if demographics.empty:
        return pd.DataFrame(), pd.DataFrame()
    analysed = demographics[
        demographics["participant_uid"].isin(set(trials["participant_uid"]))
    ].copy()
    descriptive: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    for group in [GROUP_RANDOMISED, GROUP_FIXED]:
        current = analysed[analysed["ordering_group"] == group]
        age = current["age"].dropna()
        descriptive.append(
            {
                "variable": "age",
                "level": "continuous",
                "ordering_group": group,
                "n": int(age.size),
                "value": float(age.mean()) if not age.empty else np.nan,
                "standard_deviation": float(age.std(ddof=1)) if len(age) > 1 else np.nan,
            }
        )
        for variable in ["gender", "vr_experience", "nationality"]:
            for level, count in current[variable].fillna("Missing").value_counts().items():
                descriptive.append(
                    {
                        "variable": variable,
                        "level": level,
                        "ordering_group": group,
                        "n": int(count),
                        "value": float(count / len(current)) if len(current) else np.nan,
                        "standard_deviation": np.nan,
                    }
                )

    random_age = analysed.loc[
        analysed["ordering_group"] == GROUP_RANDOMISED, "age"
    ].dropna().to_numpy(float)
    fixed_age = analysed.loc[
        analysed["ordering_group"] == GROUP_FIXED, "age"
    ].dropna().to_numpy(float)
    if len(random_age) >= 2 and len(fixed_age) >= 2:
        age_result = _welch_difference(random_age, fixed_age)
        age_result.update(
            {
                "variable": "age",
                "test": "Welch two sample comparison",
                "effect_size": _hedges_g(random_age, fixed_age),
            }
        )
        comparisons.append(age_result)
    for variable in ["gender", "vr_experience"]:
        contingency = pd.crosstab(analysed["ordering_group"], analysed[variable])
        if contingency.shape[0] == 2 and contingency.shape[1] >= 2:
            chi_square, p_value, degrees, _ = st.chi2_contingency(contingency)
            total = contingency.to_numpy().sum()
            denominator = min(contingency.shape) - 1
            cramer_v = math.sqrt(chi_square / (total * denominator)) if denominator > 0 else np.nan
            comparisons.append(
                {
                    "variable": variable,
                    "test": "Pearson chi square",
                    "chi_square": chi_square,
                    "degrees_of_freedom": degrees,
                    "p_value": p_value,
                    "effect_size": cramer_v,
                }
            )
    return pd.DataFrame(descriptive), pd.DataFrame(comparisons)


def _standardised_demographic_marginal_contrast(
    result: Any,
    design_info: Any,
    demographics: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    factorial = _factorial_prediction_grid()
    demographic_values = demographics[
        ["age_centered", "age_missing", "gender", "vr_experience"]
    ].copy()
    estimates: dict[str, tuple[float, np.ndarray, float]] = {}
    for group in [GROUP_RANDOMISED, GROUP_FIXED]:
        grid = factorial.merge(demographic_values, how="cross")
        grid["ordering_group"] = group
        estimates[group] = _average_prediction(result, design_info, grid, "logit")
    fixed_estimate, fixed_gradient, _ = estimates[GROUP_FIXED]
    random_estimate, random_gradient, _ = estimates[GROUP_RANDOMISED]
    difference = fixed_estimate - random_estimate
    gradient = fixed_gradient - random_gradient
    variance = float(gradient @ np.asarray(result.cov_params()) @ gradient)
    standard_error = math.sqrt(max(0.0, variance))
    critical = st.norm.ppf(0.975)
    z_value = difference / standard_error if standard_error > 0 else np.nan
    return pd.DataFrame(
        [
            {
                "threshold": threshold,
                "contrast": "fixed sequence minus randomised order",
                "standardisation": "equal factorial cells and pooled analysed participant demographics",
                "difference_percentage_points": 100.0 * difference,
                "standard_error_percentage_points": 100.0 * standard_error,
                "ci_low": 100.0 * (difference - critical * standard_error),
                "ci_high": 100.0 * (difference + critical * standard_error),
                "z": z_value,
                "p_value": 2.0 * st.norm.sf(abs(z_value)) if np.isfinite(z_value) else np.nan,
            }
        ]
    )


def demographic_adjusted_sensitivity(
    trials: pd.DataFrame,
    demographics: pd.DataFrame,
    settings: AnalysisSettings,
) -> dict[str, pd.DataFrame]:
    """Adjust the primary group contrast for measured cohort composition."""

    empty = {
        "demographic_adjusted_primary_contrast": pd.DataFrame(),
        "demographic_adjusted_coefficients": pd.DataFrame(),
        "demographic_adjustment_status": pd.DataFrame(),
    }
    if demographics.empty:
        empty["demographic_adjustment_status"] = pd.DataFrame(
            [{"status": "not estimated", "reason": "No intake demographics were loaded"}]
        )
        return empty
    linked = demographics[
        demographics["participant_uid"].isin(set(trials["participant_uid"]))
    ].copy()
    counts = linked.groupby("ordering_group")["participant_uid"].nunique()
    if any(counts.get(group, 0) < 10 for group in [GROUP_RANDOMISED, GROUP_FIXED]):
        empty["demographic_adjustment_status"] = pd.DataFrame(
            [{"status": "not estimated", "reason": "Fewer than 10 linked participants in a group"}]
        )
        return empty
    linked["age_missing"] = linked["age"].isna().astype(int)
    pooled_age = float(linked["age"].median())
    linked["age_centered"] = linked["age"].fillna(pooled_age) - pooled_age
    linked["gender"] = linked["gender"].fillna("Missing")
    linked["vr_experience"] = linked["vr_experience"].fillna("Missing")
    model_frame = trials.merge(
        linked[
            [
                "participant_uid",
                "age_centered",
                "age_missing",
                "gender",
                "vr_experience",
            ]
        ],
        on="participant_uid",
        how="inner",
    )
    additional = "age_centered + C(gender) + C(vr_experience)"
    if linked["age_missing"].nunique() > 1:
        additional += " + age_missing"
    try:
        result, design_info, _, _ = _fit_grouped_binomial(
            model_frame,
            settings.primary_threshold,
            False,
            additional_rhs=additional,
        )
        contrast = _standardised_demographic_marginal_contrast(
            result,
            design_info,
            linked,
            settings.primary_threshold,
        )
        coefficients = _coefficient_table(
            result,
            "demographic_adjusted_mean_and_condition",
            settings.primary_threshold,
        )
        status = pd.DataFrame(
            [
                {
                    "status": "estimated",
                    "reason": "Sensitivity analysis only; measured covariates cannot remove run level confounding",
                    "participants": linked["participant_uid"].nunique(),
                }
            ]
        )
        return {
            "demographic_adjusted_primary_contrast": contrast,
            "demographic_adjusted_coefficients": coefficients,
            "demographic_adjustment_status": status,
        }
    except Exception as exc:
        empty["demographic_adjustment_status"] = pd.DataFrame(
            [{"status": "failed", "reason": str(exc)}]
        )
        return empty


def demographic_adjusted_participant_outcomes(
    participant_means: pd.DataFrame,
    demographics: pd.DataFrame,
) -> pd.DataFrame:
    """Independent-participant HC3 sensitivity models for all paper outcomes."""

    if demographics.empty or participant_means.empty:
        return pd.DataFrame()
    linked = demographics[
        demographics["participant_uid"].isin(set(participant_means["participant_uid"]))
    ].copy()
    counts = linked.groupby("ordering_group")["participant_uid"].nunique()
    if any(counts.get(group, 0) < 10 for group in [GROUP_RANDOMISED, GROUP_FIXED]):
        return pd.DataFrame()
    linked["age_missing"] = linked["age"].isna().astype(int)
    pooled_age = float(linked["age"].median())
    linked["age_centered"] = linked["age"].fillna(pooled_age) - pooled_age
    linked["gender"] = linked["gender"].fillna("Missing")
    linked["vr_experience"] = linked["vr_experience"].fillna("Missing")
    frame = participant_means.merge(
        linked[
            [
                "participant_uid",
                "age_centered",
                "age_missing",
                "gender",
                "vr_experience",
            ]
        ],
        on="participant_uid",
        how="inner",
    )
    group_term = (
        "C(ordering_group, Treatment(reference='randomised_order'))[T.fixed_sequence]"
    )
    rows: list[dict[str, Any]] = []
    outcomes = [outcome for outcome in PARTICIPANT_OUTCOMES if outcome in frame.columns]
    for outcome in outcomes:
        spec = OUTCOME_SPECS[outcome]
        model_frame = frame.dropna(subset=[outcome]).copy()
        if model_frame["ordering_group"].nunique() < 2 or len(model_frame) < 20:
            continue
        rhs = (
            "C(ordering_group, Treatment(reference='randomised_order')) "
            "+ age_centered + C(gender) + C(vr_experience)"
        )
        if model_frame["age_missing"].nunique() > 1:
            rhs += " + age_missing"
        try:
            result = smf.ols(f"{outcome} ~ {rhs}", data=model_frame).fit(cov_type="HC3")
            estimate = float(result.params[group_term])
            standard_error = float(result.bse[group_term])
            z_value = estimate / standard_error if standard_error > 0 else np.nan
            critical = st.norm.ppf(0.975)
            rows.append(
                {
                    "outcome": outcome,
                    "outcome_label": spec["label"],
                    "outcome_family": spec["family"],
                    "analysis_role": spec["role"],
                    "units": spec["units"],
                    "contrast": "fixed sequence minus randomised order",
                    "adjusted_difference": estimate,
                    "standard_error_hc3": standard_error,
                    "ci_low": estimate - critical * standard_error,
                    "ci_high": estimate + critical * standard_error,
                    "z": z_value,
                    "p_value": (
                        2.0 * st.norm.sf(abs(z_value))
                        if np.isfinite(z_value)
                        else np.nan
                    ),
                    "participants": int(len(model_frame)),
                    "adjusted_for": "age, age missingness, gender, and recent VR experience",
                    "interpretation": "sensitivity analysis; does not remove unmeasured run-level confounding",
                }
            )
        except Exception as exc:
            LOGGER.warning(
                "Participant-level demographic sensitivity failed for %s: %s",
                outcome,
                exc,
            )
    table = _holm_within(pd.DataFrame(rows), ["outcome_family"])
    if not table.empty:
        table["multiplicity_method"] = (
            "Holm across adjusted participant outcomes within outcome family"
        )
        table = _add_global_holm(table)
    return table


def outcome_dictionary() -> pd.DataFrame:
    definitions = {
        "unsafe_pct": "Percentage of valid 100 ms bins above the primary trigger threshold in the five-second pre-passage window.",
        "Q1": "Trial-level 0 to 100 rating of the behaviour of the other pedestrian.",
        "Q2": "Trial-level 0 to 100 rating of the distance between the pedestrians.",
        "Q3": "Trial-level 0 to 100 rating of the intention of the vehicle.",
        "mean_trigger": "Mean normalised analogue trigger value in the five-second pre-passage window.",
        "peak_trigger": "Maximum normalised analogue trigger value in the five-second pre-passage window.",
        "any_trigger_press": "Indicator that at least one 100 ms bin exceeded the primary trigger threshold.",
        "trigger_first_active_latency_s": "Time from the start of the five-second common window to the first trigger-active 100 ms bin, defined only for trials with activation; zero is left-censored because a press may have begun before the window.",
        "trigger_return_to_safe": "Among trials with trigger activation, indicator that a subsequent 100 ms bin returned below or equal to the primary threshold before passage.",
        "trigger_first_return_latency_s": "Time from the common-window start to the first return-to-safe bin, defined only when activation was followed by a return before passage.",
        "heading_at_pass_deg": "Mean baseline-corrected Unity horizontal head heading from 100 ms before to 100 ms after participant passage.",
        "minimum_heading_deg": "Minimum smoothed baseline-corrected horizontal head heading from 0.5 s after trial onset to 0.2 s before passage.",
        "passage_change_deg": "Mean heading in the first 0.5 s after passage minus the final 0.5 s before passage.",
        "heading_common_window_sd_deg": "Within-trial standard deviation of baseline-corrected heading in the five-second pre-passage window.",
        "heading_yaw_activity_deg_s": "Mean absolute rate of change between successive available 100 ms means of unwrapped, baseline-corrected Unity horizontal HMD heading in the five-second pre-passage window.",
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


def _save_plot(figure: Any, stem: Path, width: int = 1500, height: int = 900) -> None:
    """Save HTML, EPS, and PNG using the original project's Plotly helper pattern."""

    import plotly.io as pio

    # Match FigureExportMixin.save_plotly in the supplied human_analysis code.
    # Disabling MathJax prevents Kaleido from loading an external renderer.
    pio.kaleido.scope.mathjax = None
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(stem.with_suffix(".html")), include_plotlyjs="cdn")
    # Remove the PDF produced by releases before 2026-07-22.4 when rerunning
    # into the same generated-figures directory.
    stem.with_suffix(".pdf").unlink(missing_ok=True)
    try:
        figure.write_image(str(stem.with_suffix(".eps")), width=width, height=height)
    except Exception as exc:
        LOGGER.warning(
            "Skipping EPS export for %s because Plotly/Kaleido could not create it: %s",
            stem.name,
            exc,
        )
    try:
        figure.write_image(
            str(stem.with_suffix(".png")), width=width, height=height, scale=2
        )
    except Exception as exc:
        LOGGER.warning(
            "Skipping PNG export for %s because Plotly/Kaleido could not create it: %s",
            stem.name,
            exc,
        )


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
    """Create publication oriented group, condition, sequence, and outcome figures."""

    try:
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        LOGGER.warning("Plotly is unavailable; figures were not generated")
        return

    config.figures.mkdir(parents=True, exist_ok=True)
    primary = marginal_estimates[
        np.isclose(marginal_estimates["threshold"], config.settings.primary_threshold)
    ].copy()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=primary["ordering_group_label"],
            y=primary["predicted_unsafe_pct"],
            mode="markers",
            marker={"size": 14, "color": ["#0072B2", "#D55E00"]},
            error_y={
                "type": "data",
                "symmetric": False,
                "array": primary["ci_high"] - primary["predicted_unsafe_pct"],
                "arrayminus": primary["predicted_unsafe_pct"] - primary["ci_low"],
            },
        )
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Trial ordering group",
        yaxis_title="Predicted unsafe bins (%)",
        showlegend=False,
        font={"family": "Arial", "size": 22},
    )
    _save_plot(fig, config.figures / "figure_1_primary_marginal_estimates")

    fig = px.box(
        participant_means,
        x="ordering_group_label",
        y="unsafe_pct",
        points="all",
        color="ordering_group_label",
        color_discrete_map={"Randomised order": "#0072B2", "Fixed sequence": "#D55E00"},
        labels={"ordering_group_label": "Trial ordering group", "unsafe_pct": "Participant mean unsafe bins (%)"},
        template="plotly_white",
    )
    fig.update_layout(showlegend=False, font={"family": "Arial", "size": 22})
    _save_plot(fig, config.figures / "figure_2_participant_distributions")

    by_participant_trial = (
        trials.groupby(
            ["ordering_group", "ordering_group_label", "participant_uid", "trial_number"],
            observed=True,
        )["unsafe_pct"]
        .mean()
        .reset_index()
    )
    summary = (
        by_participant_trial.groupby(
            ["ordering_group", "ordering_group_label", "trial_number"], observed=True
        )["unsafe_pct"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary["se"] = summary["std"] / np.sqrt(summary["count"])
    fig = go.Figure()
    for group, colour in [(GROUP_RANDOMISED, "#0072B2"), (GROUP_FIXED, "#D55E00")]:
        subset = summary[summary["ordering_group"] == group]
        fig.add_trace(
            go.Scatter(
                x=subset["trial_number"],
                y=subset["mean"],
                mode="lines+markers",
                name=GROUP_LABELS[group],
                line={"color": colour},
                error_y={"type": "data", "array": 1.96 * subset["se"], "visible": True},
            )
        )
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Trial position",
        yaxis_title="Mean unsafe bins (%)",
        font={"family": "Arial", "size": 22},
        legend_title_text="Trial ordering group",
    )
    _save_plot(fig, config.figures / "figure_3_trial_position_profiles")

    if not condition_effects.empty:
        forest = condition_effects.copy()
        forest["error_plus"] = (
            forest["difference_ci_high"]
            - forest["difference_of_effects_percentage_points"]
        )
        forest["error_minus"] = (
            forest["difference_of_effects_percentage_points"]
            - forest["difference_ci_low"]
        )
        family_colours = {
            "conditional eHMI": "#009E73",
            "relative pedestrian order": "#CC79A7",
            "AV behaviour": "#E69F00",
        }
        fig = px.scatter(
            forest,
            x="difference_of_effects_percentage_points",
            y="contrast",
            color="contrast_family",
            error_x="error_plus",
            error_x_minus="error_minus",
            color_discrete_map=family_colours,
            labels={
                "difference_of_effects_percentage_points": (
                    "Difference of condition effects: fixed minus randomised (percentage points)"
                ),
                "contrast": "Condition contrast",
                "contrast_family": "Contrast family",
            },
            hover_data={
                "p_value_holm": ":.3g",
                "randomised_effect_percentage_points": ":.2f",
                "fixed_effect_percentage_points": ":.2f",
            },
            template="plotly_white",
        )
        fig.add_vline(x=0.0, line_dash="dash", line_color="black", line_width=1)
        fig.update_traces(marker={"size": 11})
        fig.update_layout(
            font={"family": "Arial", "size": 18},
            yaxis={"categoryorder": "array", "categoryarray": forest["contrast"].tolist()[::-1]},
            margin={"l": 360, "r": 40, "t": 40, "b": 90},
        )
        _save_plot(
            fig,
            config.figures / "figure_4_condition_effect_differences",
            width=1800,
            height=1200,
        )

    composition = (
        trials.groupby(["ordering_group", "trial_number"], observed=True)[
            ["yielding", "eHMIOn", "camera", "distPed_m"]
        ]
        .mean()
        .reset_index()
    )
    panels = [
        ("yielding", "Proportion yielding"),
        ("eHMIOn", "Proportion eHMI active"),
        ("camera", "Proportion participant first"),
        ("distPed_m", "Mean spacing (m)"),
    ]
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.055)
    for row_number, (factor, y_label) in enumerate(panels, start=1):
        for group, colour in [(GROUP_RANDOMISED, "#0072B2"), (GROUP_FIXED, "#D55E00")]:
            subset = composition[composition["ordering_group"] == group]
            fig.add_trace(
                go.Scatter(
                    x=subset["trial_number"],
                    y=subset[factor],
                    mode="lines+markers",
                    name=GROUP_LABELS[group],
                    legendgroup=group,
                    showlegend=row_number == 1,
                    line={"color": colour, "width": 2},
                    marker={"size": 5},
                ),
                row=row_number,
                col=1,
            )
        fig.update_yaxes(title_text=y_label, row=row_number, col=1)
        if factor != "distPed_m":
            fig.update_yaxes(range=[-0.05, 1.05], row=row_number, col=1)
    fig.update_xaxes(title_text="Trial position", row=4, col=1)
    fig.update_layout(
        template="plotly_white",
        font={"family": "Arial", "size": 18},
        legend_title_text="Trial ordering group",
        margin={"l": 150, "r": 40, "t": 40, "b": 80},
    )
    _save_plot(
        fig,
        config.figures / "figure_5_sequence_composition",
        width=1600,
        height=1400,
    )

    rating_columns = [column for column in ["Q1", "Q2", "Q3"] if column in participant_means]
    if rating_columns:
        ratings = participant_means.melt(
            id_vars=["ordering_group", "ordering_group_label", "participant_uid"],
            value_vars=rating_columns,
            var_name="rating",
            value_name="participant_mean",
        ).dropna(subset=["participant_mean"])
        if not ratings.empty:
            fig = px.box(
                ratings,
                x="ordering_group_label",
                y="participant_mean",
                color="ordering_group_label",
                facet_col="rating",
                points="all",
                color_discrete_map={
                    "Randomised order": "#0072B2",
                    "Fixed sequence": "#D55E00",
                },
                labels={
                    "ordering_group_label": "Trial ordering group",
                    "participant_mean": "Participant mean rating (0 to 100)",
                },
                template="plotly_white",
            )
            fig.for_each_annotation(lambda annotation: annotation.update(text=annotation.text.split("=")[-1]))
            fig.update_yaxes(range=[0, 100])
            fig.update_layout(
                showlegend=False,
                font={"family": "Arial", "size": 18},
                margin={"l": 90, "r": 30, "t": 60, "b": 90},
            )
            _save_plot(
                fig,
                config.figures / "figure_6_questionnaire_ratings",
                width=1800,
                height=800,
            )

    heading_column = "heading_at_pass_deg"
    if heading_column in participant_means and participant_means[heading_column].notna().any():
        heading = participant_means.dropna(subset=[heading_column]).copy()
        fig = px.box(
            heading,
            x="ordering_group_label",
            y=heading_column,
            color="ordering_group_label",
            points="all",
            color_discrete_map={"Randomised order": "#0072B2", "Fixed sequence": "#D55E00"},
            labels={
                "ordering_group_label": "Trial ordering group",
                heading_column: "Baseline corrected heading at participant passage (degrees)",
            },
            template="plotly_white",
        )
        fig.add_hline(y=0.0, line_dash="dash", line_color="black", line_width=1)
        fig.update_layout(
            showlegend=False,
            font={"family": "Arial", "size": 20},
            margin={"l": 100, "r": 30, "t": 40, "b": 90},
        )
        _save_plot(fig, config.figures / "figure_7_head_heading_at_passage")

    trigger_columns = [
        column
        for column in ["mean_trigger", "peak_trigger", "any_trigger_press"]
        if column in participant_means
    ]
    if trigger_columns:
        trigger = participant_means[
            ["ordering_group_label", "participant_uid", *trigger_columns]
        ].copy()
        for column in ["mean_trigger", "peak_trigger"]:
            if column in trigger:
                trigger[column] *= 100.0
        trigger = trigger.melt(
            id_vars=["ordering_group_label", "participant_uid"],
            value_vars=trigger_columns,
            var_name="trigger_outcome",
            value_name="participant_mean_percent",
        ).dropna(subset=["participant_mean_percent"])
        trigger_labels = {
            "mean_trigger": "Mean trigger (% full scale)",
            "peak_trigger": "Peak trigger (% full scale)",
            "any_trigger_press": "Trials with activation (%)",
        }
        trigger["trigger_outcome"] = trigger["trigger_outcome"].map(trigger_labels)
        fig = px.box(
            trigger,
            x="ordering_group_label",
            y="participant_mean_percent",
            color="ordering_group_label",
            facet_col="trigger_outcome",
            points="all",
            color_discrete_map={
                "Randomised order": "#0072B2",
                "Fixed sequence": "#D55E00",
            },
            labels={
                "ordering_group_label": "Trial ordering group",
                "participant_mean_percent": "Participant mean (%)",
            },
            template="plotly_white",
        )
        fig.for_each_annotation(
            lambda annotation: annotation.update(text=annotation.text.split("=")[-1])
        )
        fig.update_yaxes(range=[0, 105])
        fig.update_layout(
            showlegend=False,
            font={"family": "Arial", "size": 17},
            margin={"l": 90, "r": 30, "t": 70, "b": 90},
        )
        _save_plot(
            fig,
            config.figures / "figure_8_trigger_activation_and_magnitude",
            width=1900,
            height=850,
        )

    head_columns = [
        column
        for column in [
            "heading_at_pass_deg",
            "minimum_heading_deg",
            "passage_change_deg",
            "heading_common_window_sd_deg",
        ]
        if column in participant_means
    ]
    if head_columns:
        head = participant_means[
            ["ordering_group_label", "participant_uid", *head_columns]
        ].melt(
            id_vars=["ordering_group_label", "participant_uid"],
            value_vars=head_columns,
            var_name="head_outcome",
            value_name="participant_mean_deg",
        ).dropna(subset=["participant_mean_deg"])
        head_labels = {
            "heading_at_pass_deg": "Heading at passage",
            "minimum_heading_deg": "Minimum pre-passage heading",
            "passage_change_deg": "Change across passage",
            "heading_common_window_sd_deg": "Pre-passage variability",
        }
        head["head_outcome"] = head["head_outcome"].map(head_labels)
        fig = px.box(
            head,
            x="ordering_group_label",
            y="participant_mean_deg",
            color="ordering_group_label",
            facet_col="head_outcome",
            facet_col_wrap=2,
            points="all",
            color_discrete_map={
                "Randomised order": "#0072B2",
                "Fixed sequence": "#D55E00",
            },
            labels={
                "ordering_group_label": "Trial ordering group",
                "participant_mean_deg": "Participant mean (degrees)",
            },
            template="plotly_white",
        )
        fig.for_each_annotation(
            lambda annotation: annotation.update(text=annotation.text.split("=")[-1])
        )
        fig.add_hline(y=0.0, line_dash="dash", line_color="black", line_width=1)
        fig.update_layout(
            showlegend=False,
            font={"family": "Arial", "size": 17},
            margin={"l": 100, "r": 30, "t": 70, "b": 90},
        )
        _save_plot(
            fig,
            config.figures / "figure_9_head_movement_outcomes",
            width=1800,
            height=1200,
        )

    progression_outcomes = [
        outcome
        for outcome in [
            "unsafe_pct",
            "mean_trigger",
            "any_trigger_press",
            "Q3",
            "heading_at_pass_deg",
            "passage_change_deg",
        ]
        if outcome in set(temporal_predictions.get("outcome", pd.Series(dtype=str)))
    ]
    if progression_outcomes:
        subplot_titles = [OUTCOME_SPECS[outcome]["label"] for outcome in progression_outcomes]
        rows = int(math.ceil(len(progression_outcomes) / 2))
        fig = make_subplots(
            rows=rows,
            cols=2,
            subplot_titles=subplot_titles,
            vertical_spacing=0.11,
            horizontal_spacing=0.10,
        )
        colours = {GROUP_RANDOMISED: "#0072B2", GROUP_FIXED: "#D55E00"}
        fills = {GROUP_RANDOMISED: "rgba(0,114,178,0.15)", GROUP_FIXED: "rgba(213,94,0,0.15)"}
        for panel_index, outcome in enumerate(progression_outcomes):
            row = panel_index // 2 + 1
            column = panel_index % 2 + 1
            outcome_frame = temporal_predictions[temporal_predictions["outcome"] == outcome]
            for group in [GROUP_RANDOMISED, GROUP_FIXED]:
                subset = outcome_frame[outcome_frame["ordering_group"] == group].sort_values("trial_number")
                if subset.empty:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=pd.concat([subset["trial_number"], subset["trial_number"].iloc[::-1]]),
                        y=pd.concat([subset["ci_high"], subset["ci_low"].iloc[::-1]]),
                        fill="toself",
                        fillcolor=fills[group],
                        line={"color": "rgba(255,255,255,0)"},
                        hoverinfo="skip",
                        showlegend=False,
                        legendgroup=group,
                    ),
                    row=row,
                    col=column,
                )
                fig.add_trace(
                    go.Scatter(
                        x=subset["trial_number"],
                        y=subset["adjusted_estimate"],
                        mode="lines",
                        name=GROUP_LABELS[group],
                        legendgroup=group,
                        showlegend=panel_index == 0,
                        line={"color": colours[group], "width": 3},
                    ),
                    row=row,
                    col=column,
                )
            fig.add_vline(x=14.5, line_dash="dot", line_color="#777777", row=row, col=column)
            fig.add_vline(x=26.5, line_dash="dot", line_color="#777777", row=row, col=column)
            fig.update_xaxes(title_text="Trial position", row=row, col=column)
            fig.update_yaxes(title_text=OUTCOME_SPECS[outcome]["units"], row=row, col=column)
        fig.update_layout(
            template="plotly_white",
            font={"family": "Arial", "size": 17},
            legend_title_text="Trial ordering group",
            margin={"l": 110, "r": 30, "t": 90, "b": 80},
        )
        _save_plot(
            fig,
            config.figures / "figure_10_condition_adjusted_trial_profiles",
            width=1900,
            height=max(1000, 500 * rows),
        )

    if not temporal_tests.empty:
        forest = temporal_tests[
            temporal_tests["test"].eq("group by linear trial position")
            & temporal_tests["standardised_estimate"].notna()
        ].copy()
        if not forest.empty:
            forest["error_plus"] = forest["standardised_ci_high"] - forest["standardised_estimate"]
            forest["error_minus"] = forest["standardised_estimate"] - forest["standardised_ci_low"]
            fig = px.scatter(
                forest,
                x="standardised_estimate",
                y="outcome_label",
                color="outcome_family",
                error_x="error_plus",
                error_x_minus="error_minus",
                hover_data={"p_value": ":.3g", "p_value_adjusted": ":.3g"},
                labels={
                    "standardised_estimate": "Fixed minus randomised linear change per 10 trials (SD units)",
                    "outcome_label": "Outcome",
                    "outcome_family": "Outcome family",
                },
                template="plotly_white",
            )
            fig.add_vline(x=0.0, line_dash="dash", line_color="black", line_width=1)
            fig.update_traces(marker={"size": 12})
            fig.update_layout(
                font={"family": "Arial", "size": 17},
                margin={"l": 310, "r": 40, "t": 50, "b": 100},
            )
            _save_plot(
                fig,
                config.figures / "figure_11_adjusted_temporal_effects",
                width=1800,
                height=1000,
            )

    if not exposure_tests.empty:
        heat = exposure_tests[
            exposure_tests["exposure_factor"].ne("joint")
            & exposure_tests["standardised_estimate"].notna()
        ].copy()
        if not heat.empty:
            labels = dict(EXPOSURE_FACTORS)
            heat["factor_label"] = heat["exposure_factor"].map(labels)
            value_matrix = heat.pivot(
                index="outcome_label", columns="factor_label", values="standardised_estimate"
            )
            p_matrix = heat.pivot(
                index="outcome_label", columns="factor_label", values="p_value_adjusted"
            ).reindex(index=value_matrix.index, columns=value_matrix.columns)
            annotations = p_matrix.map(
                lambda value: "" if pd.isna(value) else f"Holm p={value:.3g}"
            )
            limit = float(np.nanmax(np.abs(value_matrix.to_numpy(float))))
            limit = max(limit, 0.10)
            fig = go.Figure(
                go.Heatmap(
                    z=value_matrix.to_numpy(float),
                    x=value_matrix.columns.tolist(),
                    y=value_matrix.index.tolist(),
                    text=annotations.to_numpy(str),
                    texttemplate="%{text}",
                    colorscale="RdBu_r",
                    zmid=0.0,
                    zmin=-limit,
                    zmax=limit,
                    colorbar={"title": "SD units<br>per 10 exposures"},
                    hovertemplate=(
                        "Outcome=%{y}<br>Prior exposure=%{x}<br>"
                        "Standardised interaction=%{z:.3f}<br>%{text}<extra></extra>"
                    ),
                )
            )
            fig.update_layout(
                template="plotly_white",
                font={"family": "Arial", "size": 16},
                xaxis_title="Prior exposure to the current factor level",
                yaxis_title="Outcome",
                margin={"l": 320, "r": 130, "t": 50, "b": 120},
            )
            _save_plot(
                fig,
                config.figures / "figure_12_prior_exposure_interactions",
                width=1900,
                height=1100,
            )

    event_outcomes = [
        outcome
        for outcome in [
            "trigger_first_active_latency_s",
            "trigger_return_to_safe",
            "trigger_first_return_latency_s",
        ]
        if outcome
        in set(secondary_marginal_estimates.get("outcome", pd.Series(dtype=str)))
    ]
    if event_outcomes:
        fig = make_subplots(
            rows=1,
            cols=len(event_outcomes),
            subplot_titles=[OUTCOME_SPECS[outcome]["label"] for outcome in event_outcomes],
            horizontal_spacing=0.10,
        )
        for column, outcome in enumerate(event_outcomes, start=1):
            subset = secondary_marginal_estimates[
                secondary_marginal_estimates["outcome"] == outcome
            ].copy()
            fig.add_trace(
                go.Scatter(
                    x=subset["ordering_group_label"],
                    y=subset["marginal_estimate"],
                    mode="markers",
                    marker={"size": 14, "color": ["#0072B2", "#D55E00"]},
                    error_y={
                        "type": "data",
                        "symmetric": False,
                        "array": subset["ci_high"] - subset["marginal_estimate"],
                        "arrayminus": subset["marginal_estimate"] - subset["ci_low"],
                    },
                    showlegend=False,
                    hovertemplate=(
                        "%{x}<br>Estimate=%{y:.2f}<extra></extra>"
                    ),
                ),
                row=1,
                col=column,
            )
            fig.update_xaxes(title_text="Trial ordering group", row=1, col=column)
            fig.update_yaxes(
                title_text=OUTCOME_SPECS[outcome]["units"], row=1, col=column
            )
        fig.update_layout(
            template="plotly_white",
            font={"family": "Arial", "size": 17},
            margin={"l": 100, "r": 30, "t": 80, "b": 100},
        )
        _save_plot(
            fig,
            config.figures / "figure_13_event_defined_trigger_outcomes",
            width=1900,
            height=800,
        )

    segment_outcomes = [
        outcome
        for outcome in [
            "unsafe_pct",
            "Q3",
            "any_trigger_press",
            "trigger_first_active_latency_s",
            "heading_yaw_activity_deg_s",
        ]
        if outcome
        in set(session_segment_predictions.get("outcome", pd.Series(dtype=str)))
    ]
    if segment_outcomes:
        rows = int(math.ceil(len(segment_outcomes) / 2))
        fig = make_subplots(
            rows=rows,
            cols=2,
            subplot_titles=[OUTCOME_SPECS[outcome]["label"] for outcome in segment_outcomes],
            vertical_spacing=0.12,
            horizontal_spacing=0.11,
        )
        colours = {GROUP_RANDOMISED: "#0072B2", GROUP_FIXED: "#D55E00"}
        segment_order = [label for _, label in SESSION_SEGMENTS]
        for panel_index, outcome in enumerate(segment_outcomes):
            row = panel_index // 2 + 1
            column = panel_index % 2 + 1
            outcome_frame = session_segment_predictions[
                session_segment_predictions["outcome"] == outcome
            ]
            for ordering_group in [GROUP_RANDOMISED, GROUP_FIXED]:
                subset = outcome_frame[
                    outcome_frame["ordering_group"] == ordering_group
                ].copy()
                subset["session_segment_label"] = pd.Categorical(
                    subset["session_segment_label"],
                    categories=segment_order,
                    ordered=True,
                )
                subset = subset.sort_values("session_segment_label")
                fig.add_trace(
                    go.Scatter(
                        x=subset["session_segment_label"],
                        y=subset["adjusted_estimate"],
                        mode="lines+markers",
                        name=GROUP_LABELS[ordering_group],
                        legendgroup=ordering_group,
                        showlegend=panel_index == 0,
                        line={"color": colours[ordering_group], "width": 3},
                        marker={"size": 10},
                        error_y={
                            "type": "data",
                            "symmetric": False,
                            "array": subset["ci_high"] - subset["adjusted_estimate"],
                            "arrayminus": subset["adjusted_estimate"] - subset["ci_low"],
                        },
                    ),
                    row=row,
                    col=column,
                )
            fig.update_xaxes(title_text="Session segment", row=row, col=column)
            fig.update_yaxes(
                title_text=OUTCOME_SPECS[outcome]["units"], row=row, col=column
            )
        fig.update_layout(
            template="plotly_white",
            font={"family": "Arial", "size": 16},
            legend_title_text="Trial ordering group",
            margin={"l": 110, "r": 30, "t": 90, "b": 100},
        )
        _save_plot(
            fig,
            config.figures / "figure_14_adjusted_session_segments",
            width=1900,
            height=max(900, 520 * rows),
        )

    cue_changes = ehmi_learning_change_tests[
        ehmi_learning_change_tests.get(
            "ordering_group", pd.Series(dtype=str)
        ).isin([GROUP_RANDOMISED, GROUP_FIXED])
    ].copy()
    if not cue_changes.empty:
        outcomes = [
            outcome
            for outcome in EHMI_LEARNING_OUTCOMES
            if outcome in set(cue_changes["outcome"])
        ]
        progression_panels = [
            (
                "trial_position",
                "Trial 40 minus trial 1",
            ),
            (
                "prior_yielding_ehmi_exposure",
                "Maximum minus zero prior yielding-eHMI encounters",
            ),
        ]
        titles = [
            f"{progression_label}: {OUTCOME_SPECS[outcome]['label']}"
            for _, progression_label in progression_panels
            for outcome in outcomes
        ]
        fig = make_subplots(
            rows=len(progression_panels),
            cols=len(outcomes),
            subplot_titles=titles,
            horizontal_spacing=0.16,
            vertical_spacing=0.20,
        )
        for row, (progression_metric, progression_label) in enumerate(
            progression_panels, start=1
        ):
            for column, outcome in enumerate(outcomes, start=1):
                subset = cue_changes[
                    (cue_changes["outcome"] == outcome)
                    & (cue_changes["progression_metric"] == progression_metric)
                ]
                for ordering_group, colour in [
                    (GROUP_RANDOMISED, "#0072B2"),
                    (GROUP_FIXED, "#D55E00"),
                ]:
                    row_data = subset[subset["ordering_group"] == ordering_group]
                    if row_data.empty:
                        continue
                    estimate = float(row_data["estimate"].iloc[0])
                    fig.add_trace(
                        go.Scatter(
                            x=[estimate],
                            y=[GROUP_LABELS[ordering_group]],
                            mode="markers",
                            marker={"size": 14, "color": colour},
                            error_x={
                                "type": "data",
                                "symmetric": False,
                                "array": [float(row_data["ci_high"].iloc[0]) - estimate],
                                "arrayminus": [estimate - float(row_data["ci_low"].iloc[0])],
                            },
                            showlegend=False,
                        ),
                        row=row,
                        col=column,
                    )
                fig.add_vline(
                    x=0.0,
                    line_dash="dash",
                    line_color="black",
                    line_width=1,
                    row=row,
                    col=column,
                )
                fig.update_xaxes(
                    title_text=(
                        "Change in yielding-trial eHMI effect "
                        f"({OUTCOME_SPECS[outcome]['units']})"
                    ),
                    row=row,
                    col=column,
                )
        fig.update_layout(
            template="plotly_white",
            font={"family": "Arial", "size": 17},
            margin={"l": 170, "r": 30, "t": 90, "b": 110},
        )
        _save_plot(
            fig,
            config.figures / "figure_15_ehmi_cue_learning_contrasts",
            width=1800,
            height=1200,
        )

    if spline_predictions is not None and not spline_predictions.empty:
        spline_outcomes = [
            outcome
            for outcome in [
                "unsafe_pct",
                "Q3",
                "heading_common_window_sd_deg",
            ]
            if outcome in set(spline_predictions["outcome"])
        ]
        if spline_outcomes:
            fig = make_subplots(
                rows=1,
                cols=len(spline_outcomes),
                subplot_titles=[
                    OUTCOME_SPECS[outcome]["label"]
                    for outcome in spline_outcomes
                ],
                horizontal_spacing=0.09,
            )
            colours = {
                GROUP_RANDOMISED: "#0072B2",
                GROUP_FIXED: "#D55E00",
            }
            fills = {
                GROUP_RANDOMISED: "rgba(0,114,178,0.15)",
                GROUP_FIXED: "rgba(213,94,0,0.15)",
            }
            for column, outcome in enumerate(spline_outcomes, start=1):
                outcome_frame = spline_predictions[
                    spline_predictions["outcome"] == outcome
                ]
                for group in [GROUP_RANDOMISED, GROUP_FIXED]:
                    subset = outcome_frame[
                        outcome_frame["ordering_group"] == group
                    ].sort_values("trial_number")
                    if subset.empty:
                        continue
                    fig.add_trace(
                        go.Scatter(
                            x=pd.concat(
                                [
                                    subset["trial_number"],
                                    subset["trial_number"].iloc[::-1],
                                ]
                            ),
                            y=pd.concat(
                                [
                                    subset["ci_high"],
                                    subset["ci_low"].iloc[::-1],
                                ]
                            ),
                            fill="toself",
                            fillcolor=fills[group],
                            line={"color": "rgba(255,255,255,0)"},
                            hoverinfo="skip",
                            showlegend=False,
                            legendgroup=group,
                        ),
                        row=1,
                        col=column,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=subset["trial_number"],
                            y=subset["adjusted_estimate"],
                            mode="lines",
                            name=GROUP_LABELS[group],
                            legendgroup=group,
                            showlegend=column == 1,
                            line={"color": colours[group], "width": 3},
                        ),
                        row=1,
                        col=column,
                    )
                fig.add_vline(
                    x=14.5,
                    line_dash="dot",
                    line_color="#777777",
                    row=1,
                    col=column,
                )
                fig.add_vline(
                    x=26.5,
                    line_dash="dot",
                    line_color="#777777",
                    row=1,
                    col=column,
                )
                fig.update_xaxes(
                    title_text="Trial position",
                    row=1,
                    col=column,
                )
                fig.update_yaxes(
                    title_text=OUTCOME_SPECS[outcome]["units"],
                    row=1,
                    col=column,
                )
            fig.update_layout(
                template="plotly_white",
                font={"family": "Arial", "size": 17},
                legend_title_text="Trial ordering group",
                margin={"l": 110, "r": 30, "t": 90, "b": 90},
            )
            _save_plot(
                fig,
                config.figures / "figure_16_spline_trial_trajectories",
                width=2000,
                height=850,
            )

    robustness_available = all(
        frame is not None and not frame.empty
        for frame in [
            cluster_bootstrap_replicates,
            leave_one_out,
            bayesian_bootstrap_draws,
        ]
    )
    if robustness_available:
        assert cluster_bootstrap_replicates is not None
        assert leave_one_out is not None
        assert bayesian_bootstrap_draws is not None
        ordered_leave_one_out = leave_one_out.sort_values(
            "difference_fixed_minus_randomised_percentage_points"
        ).reset_index(drop=True)
        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=[
                "Participant cluster bootstrap",
                "Leave one participant out",
                "Bayesian participant bootstrap",
            ],
            horizontal_spacing=0.10,
        )
        fig.add_trace(
            go.Histogram(
                x=cluster_bootstrap_replicates[
                    "difference_percentage_points"
                ],
                nbinsx=50,
                marker={"color": "#0072B2"},
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=np.arange(1, len(ordered_leave_one_out) + 1),
                y=ordered_leave_one_out[
                    "difference_fixed_minus_randomised_percentage_points"
                ],
                mode="markers",
                marker={
                    "color": ordered_leave_one_out[
                        "omitted_ordering_group"
                    ].map(
                        {
                            GROUP_RANDOMISED: "#0072B2",
                            GROUP_FIXED: "#D55E00",
                        }
                    ),
                    "size": 8,
                },
                text=ordered_leave_one_out["omitted_participant_uid"],
                hovertemplate=(
                    "Omitted=%{text}<br>Difference=%{y:.2f} percentage "
                    "points<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Histogram(
                x=bayesian_bootstrap_draws["difference_percentage_points"],
                nbinsx=50,
                marker={"color": "#009E73"},
                showlegend=False,
            ),
            row=1,
            col=3,
        )
        for column in [1, 2, 3]:
            if column == 2:
                fig.add_hline(
                    y=0.0,
                    line_dash="dash",
                    line_color="black",
                    row=1,
                    col=column,
                )
            else:
                fig.add_vline(
                    x=0.0,
                    line_dash="dash",
                    line_color="black",
                    row=1,
                    col=column,
                )
        fig.update_xaxes(
            title_text="Fixed minus randomised (percentage points)",
            row=1,
            col=1,
        )
        fig.update_xaxes(
            title_text="Ordered omission",
            row=1,
            col=2,
        )
        fig.update_xaxes(
            title_text="Fixed minus randomised (percentage points)",
            row=1,
            col=3,
        )
        fig.update_yaxes(title_text="Replicates", row=1, col=1)
        fig.update_yaxes(
            title_text="Primary difference (percentage points)",
            row=1,
            col=2,
        )
        fig.update_yaxes(title_text="Posterior draws", row=1, col=3)
        fig.update_layout(
            template="plotly_white",
            font={"family": "Arial", "size": 16},
            margin={"l": 100, "r": 30, "t": 90, "b": 100},
        )
        _save_plot(
            fig,
            config.figures / "figure_17_primary_robustness",
            width=2000,
            height=850,
        )

    if (
        not condition_cell_contrasts.empty
        and not condition_cell_summary.empty
    ):
        selected_summary = condition_cell_summary[
            condition_cell_summary[
                "omnibus_selected_for_condition_followup_figure"
            ].fillna(False)
        ].copy()
        selected_outcomes = selected_summary["outcome"].tolist()
        if selected_outcomes:
            selected_cells = condition_cell_contrasts[
                condition_cell_contrasts["outcome"].isin(selected_outcomes)
            ].copy()
            condition_order = (
                selected_cells[
                    [
                        "condition_id",
                        "condition_label",
                        "yielding",
                        "eHMIOn",
                        "camera",
                        "distPed_m",
                    ]
                ]
                .drop_duplicates()
                .sort_values(
                    ["yielding", "eHMIOn", "camera", "distPed_m"],
                    kind="stable",
                )
            )
            condition_ids = condition_order["condition_id"].tolist()
            condition_labels = condition_order["condition_label"].tolist()
            short_outcome_labels = {
                "unsafe_pct": "Unsafe bins",
                "Q1": "Q1",
                "Q2": "Q2",
                "Q3": "Q3",
                "mean_trigger": "Mean trigger",
                "peak_trigger": "Peak trigger",
                "any_trigger_press": "Any activation",
                "trigger_first_active_latency_s": "Activation latency",
                "trigger_return_to_safe": "Return to safe",
                "trigger_first_return_latency_s": "Return latency",
                "heading_at_pass_deg": "Heading at passage",
                "minimum_heading_deg": "Minimum heading",
                "passage_change_deg": "Passage change",
                "heading_common_window_sd_deg": "Heading variability",
                "heading_yaw_activity_deg_s": "Yaw activity",
            }
            x_labels = [
                short_outcome_labels.get(outcome, outcome)
                for outcome in selected_outcomes
            ]
            z_matrix = np.full(
                (len(condition_ids), len(selected_outcomes)),
                np.nan,
                dtype=float,
            )
            custom_data = np.empty(
                (len(condition_ids), len(selected_outcomes), 5),
                dtype=object,
            )
            custom_data[:] = np.nan
            significance_annotations: list[dict[str, Any]] = []
            for column, outcome in enumerate(selected_outcomes):
                outcome_cells = (
                    selected_cells[selected_cells["outcome"] == outcome]
                    .set_index("condition_id")
                    .reindex(condition_ids)
                )
                z_matrix[:, column] = pd.to_numeric(
                    outcome_cells["z"],
                    errors="coerce",
                ).to_numpy(float)
                custom_data[:, column, 0] = pd.to_numeric(
                    outcome_cells["difference_fixed_minus_randomised"],
                    errors="coerce",
                ).to_numpy(float)
                custom_data[:, column, 1] = pd.to_numeric(
                    outcome_cells["simultaneous_ci_low"],
                    errors="coerce",
                ).to_numpy(float)
                custom_data[:, column, 2] = pd.to_numeric(
                    outcome_cells["simultaneous_ci_high"],
                    errors="coerce",
                ).to_numpy(float)
                custom_data[:, column, 3] = pd.to_numeric(
                    outcome_cells["p_value_holm_within_outcome"],
                    errors="coerce",
                ).to_numpy(float)
                custom_data[:, column, 4] = outcome_cells["units"].to_numpy()
                significant = (
                    pd.to_numeric(
                        outcome_cells["p_value_holm_within_outcome"],
                        errors="coerce",
                    )
                    < 0.05
                ).fillna(False)
                for row_number in np.flatnonzero(significant.to_numpy()):
                    significance_annotations.append(
                        {
                            "x": x_labels[column],
                            "y": condition_labels[row_number],
                            "text": "●",
                            "showarrow": False,
                            "font": {"color": "black", "size": 15},
                        }
                    )

            finite_z = np.abs(z_matrix[np.isfinite(z_matrix)])
            colour_limit = (
                max(2.0, min(5.0, float(np.quantile(finite_z, 0.98))))
                if finite_z.size
                else 3.0
            )
            fig = go.Figure(
                data=[
                    go.Heatmap(
                        z=z_matrix,
                        x=x_labels,
                        y=condition_labels,
                        customdata=custom_data,
                        colorscale="RdBu",
                        reversescale=True,
                        zmid=0.0,
                        zmin=-colour_limit,
                        zmax=colour_limit,
                        colorbar={"title": "Robust z"},
                        hovertemplate=(
                            "%{y}<br>%{x}<br>"
                            "Difference=%{customdata[0]:.2f} %{customdata[4]}<br>"
                            "Simultaneous 95% CI=[%{customdata[1]:.2f}, "
                            "%{customdata[2]:.2f}]<br>"
                            "Holm p=%{customdata[3]:.3g}<extra></extra>"
                        ),
                    )
                ]
            )
            fig.update_layout(
                template="plotly_white",
                title=(
                    "Condition cell follow up after multiplicity controlled omnibus tests"
                    "<br><sup>Colour shows fixed minus randomised robust z; "
                    "dot denotes Holm p &lt; .05 within outcome</sup>"
                ),
                font={"family": "Arial", "size": 15},
                xaxis_title="Outcome",
                yaxis_title="Factorial condition",
                annotations=significance_annotations,
                margin={"l": 420, "r": 80, "t": 120, "b": 180},
            )
            fig.update_xaxes(tickangle=-35)
            _save_plot(
                fig,
                config.figures
                / "figure_18_condition_cell_followup_heatmap",
                width=1900,
                height=1550,
            )

        primary_cells = condition_cell_contrasts[
            condition_cell_contrasts["outcome"] == "unsafe_pct"
        ].copy()
        if not primary_cells.empty:
            primary_cells = primary_cells.sort_values(
                ["yielding", "eHMIOn", "camera", "distPed_m"],
                kind="stable",
            )
            category_order = primary_cells["condition_label"].tolist()
            fig = go.Figure()
            for yielding_value, colour in [(0, "#56B4E9"), (1, "#D55E00")]:
                subset = primary_cells[
                    primary_cells["yielding"] == yielding_value
                ]
                fig.add_trace(
                    go.Scatter(
                        x=subset["difference_fixed_minus_randomised"],
                        y=subset["condition_label"],
                        mode="markers",
                        name=(
                            "Yielding"
                            if yielding_value
                            else "Non yielding"
                        ),
                        marker={"size": 10, "color": colour},
                        error_x={
                            "type": "data",
                            "symmetric": False,
                            "array": (
                                subset["simultaneous_ci_high"]
                                - subset[
                                    "difference_fixed_minus_randomised"
                                ]
                            ),
                            "arrayminus": (
                                subset[
                                    "difference_fixed_minus_randomised"
                                ]
                                - subset["simultaneous_ci_low"]
                            ),
                        },
                        customdata=np.column_stack(
                            [
                                subset["randomised_estimate"],
                                subset["fixed_estimate"],
                                subset[
                                    "p_value_holm_within_outcome"
                                ],
                            ]
                        ),
                        hovertemplate=(
                            "%{y}<br>Randomised=%{customdata[0]:.2f}%<br>"
                            "Fixed=%{customdata[1]:.2f}%<br>"
                            "Difference=%{x:.2f} percentage points<br>"
                            "Holm p=%{customdata[2]:.3g}<extra></extra>"
                        ),
                    )
                )
            fig.add_vline(
                x=0.0,
                line_dash="dash",
                line_color="black",
                line_width=1,
            )
            fig.update_layout(
                template="plotly_white",
                title=(
                    "Primary outcome differences across factorial conditions"
                    "<br><sup>Fixed minus randomised with Bonferroni "
                    "simultaneous 95% confidence intervals</sup>"
                ),
                xaxis_title=(
                    "Difference in trigger active bins "
                    "(percentage points)"
                ),
                yaxis_title="Factorial condition",
                yaxis={
                    "categoryorder": "array",
                    "categoryarray": category_order[::-1],
                },
                legend_title_text="AV behaviour",
                font={"family": "Arial", "size": 15},
                margin={"l": 430, "r": 50, "t": 115, "b": 100},
            )
            _save_plot(
                fig,
                config.figures
                / "figure_19_primary_condition_cell_forest",
                width=1800,
                height=1650,
            )


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

Every figure is exported as HTML, high-resolution PNG, and vector EPS. The
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


def run_analysis(config: StudyConfig) -> dict[str, pd.DataFrame]:
    """Run the full prespecified comparison and return every output table."""

    config.output.mkdir(parents=True, exist_ok=True)
    config.figures.mkdir(parents=True, exist_ok=True)
    fixed_mapping = load_mapping(config.mapping)
    timing_mapping_path = config.timing_mapping or config.mapping
    timing_mapping = load_mapping(timing_mapping_path)
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
    create_figures(
        trials,
        participant_means,
        primary["primary_marginal_estimates"],
        primary["primary_condition_effect_contrasts"],
        condition_cells["condition_cell_contrasts"],
        condition_cells["condition_cell_followup_summary"],
        temporal_adjusted["temporal_adjusted_predictions"],
        temporal_adjusted["temporal_adjusted_tests"],
        secondary["secondary_marginal_estimates"],
        session_segments["session_segment_predictions"],
        session_segments["session_segment_tests"],
        ehmi_learning["ehmi_learning_effects"],
        ehmi_learning["ehmi_learning_change_tests"],
        exposure_adjusted["exposure_adjusted_tests"],
        config,
        spline_predictions=spline_temporal["spline_temporal_predictions"],
        cluster_bootstrap_replicates=primary_robustness[
            "primary_cluster_bootstrap_replicates"
        ],
        leave_one_out=primary_robustness[
            "primary_leave_one_participant_out"
        ],
        bayesian_bootstrap_draws=primary_robustness[
            "primary_bayesian_bootstrap_draws"
        ],
    )
    write_method_note(config, warnings_out)
    write_and_emit_results_log(tables, config.output, warnings_out)
    LOGGER.info("Analysis complete. Outputs written to %s", config.output)
    return tables


def read_command_line(argv: Sequence[str] | None = None) -> tuple[Path, str]:
    """Read the deliberately small command line interface without extra dependencies."""

    values = list(sys.argv[1:] if argv is None else argv)
    if values == ["--version"]:
        print(f"ordering_comparison.py {SCRIPT_VERSION}")
        raise SystemExit(0)
    if values and values[0] in {"help", "--help", "-h"}:
        print(
            "Usage: python ordering_comparison.py [CONFIG] [--log-level LEVEL]\n"
            "Default CONFIG: extensionless config beside this script; "
            "config.comparison.json is also recognised"
        )
        raise SystemExit(0)

    # Prefer the repository's extensionless `config`, which is also read by
    # common.get_configs. Keep config.comparison.json as a portable fallback.
    script_directory = Path(__file__).resolve().parent
    config_path = next(
        (
            script_directory / filename
            for filename in DEFAULT_CONFIG_FILENAMES
            if (script_directory / filename).is_file()
        ),
        script_directory / DEFAULT_CONFIG_FILENAMES[0],
    )
    level = "INFO"
    if values and values[0] != "--log-level":
        config_path = Path(values.pop(0))
    if values:
        if len(values) != 2 or values[0] != "--log-level":
            raise ValueError("Expected only: [CONFIG] [--log-level LEVEL]")
        level = values[1].upper()
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
            raise ValueError("Log level must be DEBUG, INFO, WARNING, or ERROR")
    return config_path, level


def main(argv: Sequence[str] | None = None) -> int:
    config_path, log_level = read_command_line(argv)
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    LOGGER.info("Starting ordering comparison, version %s", SCRIPT_VERSION)
    LOGGER.info("Using configuration: %s", config_path)
    warnings.filterwarnings("once", category=RuntimeWarning)
    try:
        config = load_config(config_path)
        run_analysis(config)
    except Exception as exc:
        LOGGER.exception("Analysis failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
