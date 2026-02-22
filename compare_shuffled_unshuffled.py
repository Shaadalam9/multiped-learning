"""Compare shuffled vs unshuffled datasets.

Entrypoint for the refactored codebase.

1) Update the six paths below (3 shuffled + 3 unshuffled).
2) Run:

    python3 compare_shuffled_unshuffled.py

This script always runs:
- the A–F comparison pipeline
- the mixed models analysis

Modules
-------
- csu_core.py        : shared stats/CSV/plot helpers
- csu_features.py    : feature extraction helpers (trigger, yaw, Q1–Q3)
- csu_pipeline.py    : A–F pipeline (reporting + plots)
- csu_mixed_models.py: mixed models / sensitivity analyses
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import common
from custom_logger import CustomLogger
from logmod import logs

import csu_pipeline as pipeline
import csu_mixed_models as mm

logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)  # use custom logger

MAPPING_CSV = common.get_configs("mapping")
OUTPUT_ROOT = common.get_configs("output")


def run_compare() -> None:
    """Run the A–F comparison pipeline."""
    runner = pipeline.ComparisonPipeline(
        shuffled_data=common.get_configs("shuffled_data"),
        shuffled_intake_questionnaire=common.get_configs("shuffled_intake_questionnaire"),
        shuffled_post_experiment_questionnaire=common.get_configs("shuffled_post_experiment_questionnaire"),
        unshuffled_data=common.get_configs("unshuffled_data"),
        unshuffled_intake_questionnaire=common.get_configs("unshuffled_intake_questionnaire"),
        unshuffled_post_experiment_questionnaire=common.get_configs("unshuffled_post_experiment_questionnaire"),
        mapping_csv=MAPPING_CSV,
        output_root=OUTPUT_ROOT,
    )
    runner.run()


def run_mixed(trial_df: Optional[pd.DataFrame] = None) -> None:
    """Run the mixed-models analysis."""
    runner = mm.MixedModelsPipeline(output_root=OUTPUT_ROOT)
    runner.run(trial_df=trial_df)


if __name__ == "__main__":
    run_compare()
    try:
        run_mixed()
    except Exception as e:
        # Mixed models are "best effort"; keep the overall pipeline usable.
        logger.error(f"[MM] failed: {e}")
