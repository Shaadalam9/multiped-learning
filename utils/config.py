"""Read configuration files and resolve the study settings and data locations."""

from __future__ import annotations

from typing import Any
from pathlib import Path
import common
from dataclasses import dataclass
from dataclasses import field
import json
import numpy as np
from utils.constants import (
    COMMON_CONFIG_KEYS,
    LOGGER,
)


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
    final_figures: Path | None = None
    save_final: bool = False
    auto_open: bool = True


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
    """Load flat configuration fields and resolve analysis settings."""

    config_path = config_path.expanduser().resolve()
    raw = _load_raw_config(config_path)
    base = config_path.parent

    primary_threshold = float(raw.get("primary_threshold", 0.10))
    sensitivity_thresholds = [
        float(x) for x in raw.get("sensitivity_thresholds", [0.05, 0.10, 0.50])
    ]
    if not any(np.isclose(primary_threshold, value) for value in sensitivity_thresholds):
        sensitivity_thresholds.append(primary_threshold)
    sensitivity_thresholds = sorted(set(sensitivity_thresholds))
    settings = AnalysisSettings(
        window_seconds=float(raw.get("window_seconds", 5.0)),
        bin_seconds=float(raw.get("bin_seconds", 0.1)),
        primary_threshold=primary_threshold,
        sensitivity_thresholds=tuple(sensitivity_thresholds),
        require_complete_window=bool(raw.get("require_complete_window", True)),
        minimum_valid_trials_per_participant=int(
            raw.get("minimum_valid_trials_per_participant", 32)
        ),
        bootstrap_replicates=int(
            raw.get("bootstrap_replicates", 5000)
        ),
        bootstrap_seed=int(raw.get("bootstrap_seed", 2901)),
        bayesian_bootstrap_draws=int(
            raw.get("bayesian_bootstrap_draws", 20000)
        ),
        spline_degrees_of_freedom=int(
            raw.get("spline_degrees_of_freedom", 4)
        ),
        planning_effect_sizes_percentage_points=tuple(
            float(value)
            for value in raw.get(
                "planning_effect_sizes_percentage_points",
                [2.5, 5.0, 7.5, 10.0],
            )
        ),
        planning_power=tuple(
            float(value)
            for value in raw.get("planning_power", [0.80, 0.90])
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
    # Generated outputs always live under the result directory.
    figures = output / "figures"
    final_figures = _resolve_path(raw.get("figures", "figures"), base)
    save_final = raw.get("save_final", False)
    auto_open = raw.get("auto_open", True)
    for name, value in (("save_final", save_final), ("auto_open", auto_open)):
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be a JSON boolean (true or false)")
    if save_final and final_figures is None:
        raise ValueError("Set figures to a destination folder when save_final is true")

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
                    "beside main.py."
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
        final_figures=final_figures,
        save_final=save_final,
        auto_open=auto_open,
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
