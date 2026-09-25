"""Plots for the trial-order comparison."""

from __future__ import annotations

import shutil
import webbrowser

from utils.constants import (
    LOGGER,
)
from utils.plots.analysis import (
    create_figures,
)


def regenerate_all_figures(tables, config):
    """One shared figure path for fresh analysis, CSV migration and pickle reuse."""
    destination = config.figures
    destination.mkdir(parents=True, exist_ok=True)
    before = {path.name: path.stat().st_mtime_ns for path in destination.iterdir() if path.is_file()}
    create_figures(
        trials=tables["trial_level_common_window"],
        participant_means=tables["participant_level_means"],
        marginal_estimates=tables["primary_marginal_estimates"],
        condition_effects=tables["primary_condition_effect_contrasts"],
        condition_cell_contrasts=tables["condition_cell_contrasts"],
        condition_cell_summary=tables["condition_cell_followup_summary"],
        temporal_predictions=tables["temporal_adjusted_predictions"],
        temporal_tests=tables["temporal_adjusted_tests"],
        secondary_marginal_estimates=tables["secondary_marginal_estimates"],
        session_segment_predictions=tables["session_segment_predictions"],
        session_segment_tests=tables["session_segment_tests"],
        ehmi_learning_effects=tables["ehmi_learning_effects"],
        ehmi_learning_change_tests=tables["ehmi_learning_change_tests"],
        exposure_tests=tables["exposure_adjusted_tests"],
        spline_predictions=tables["spline_temporal_predictions"],
        cluster_bootstrap_replicates=tables["primary_cluster_bootstrap_replicates"],
        leave_one_out=tables["primary_leave_one_participant_out"],
        bayesian_bootstrap_draws=tables["primary_bayesian_bootstrap_draws"],
        config=config,
    )
    from utils.plots.manuscript import generate
    generate(config.output, destination)
    # Include only files produced by this run, not stale optional-model plots.
    generated = [path for path in sorted(destination.iterdir())
                 if path.is_file() and path.suffix in {'.pdf', '.png', '.html', '.pickle', '.eps', '.json'}
                 and path.stat().st_mtime_ns != before.get(path.name)]
    if config.save_final:
        config.final_figures.mkdir(parents=True, exist_ok=True)
        for source in generated:
            if source.suffix not in {".html", ".png"}:
                continue
            target = config.final_figures / source.name
            if source.resolve() != target.resolve():
                shutil.copy2(source, target)
        LOGGER.info("Saved final HTML and PNG copies to %s", config.final_figures)
    if config.auto_open:
        for path in generated:
            if path.suffix == '.html':
                try:
                    if not webbrowser.open(path.resolve().as_uri(), new=2):
                        LOGGER.warning("Could not open HTML automatically: %s", path)
                except Exception as exc:
                    LOGGER.warning("Could not open HTML %s: %s", path, exc)
    LOGGER.info("Generated analysis and manuscript figures in %s", destination)
