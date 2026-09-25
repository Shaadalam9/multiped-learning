"""Parse run options, configure logging and select the data-processing route."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence
import argparse
from custom_logger import configure_logging
import pandas as pd
import warnings
from utils.config import (
    StudyConfig,
    load_config,
)
from utils.constants import (
    DEFAULT_CONFIG_FILENAMES,
    LOGGER,
    PROJECT_ROOT,
    SCRIPT_VERSION,
)
from utils.plots import (
    regenerate_all_figures,
)
from utils.reporting import (
    write_tables,
)
from utils.snapshot import (
    load_result_snapshot,
    save_result_snapshot,
)


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Read command-line options; do not load data or start logging."""
    parser = argparse.ArgumentParser(
        description="Run all analyses, tables, pickle snapshots and figures from one entry point.")
    root = PROJECT_ROOT
    default = next((root / name for name in DEFAULT_CONFIG_FILENAMES if (root / name).is_file()), root / 'config')
    parser.add_argument('config', nargs='?', type=Path, default=default)
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--refresh', action='store_true',
                       help='Re-read raw data and refit all analyses, replacing the saved snapshot')
    modes.add_argument('--from-saved', action='store_true',
                       help=(
                           'Explicitly adopt existing CSV results without refitting; save pickle and '
                           'regenerate all figures'
                       ))
    parser.add_argument('--version', action='store_true', help='Log the program version and exit')
    return parser.parse_args(argv)


def load_saved_csv_tables(output: Path) -> dict[str, pd.DataFrame]:
    """Adopt previously exported tables without pretending to refit their models."""
    paths = sorted(output.glob("*.csv"))
    if not paths:
        raise ValueError("No saved CSV tables found; use --refresh")
    tables = {}
    for path in paths:
        try:
            tables[path.stem] = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            tables[path.stem] = pd.DataFrame()
    return tables


def run_workflow(config: StudyConfig, *, refresh: bool = False, from_saved: bool = False) -> None:
    """Choose exactly one route: adopt CSVs, reuse a snapshot, or analyse raw data."""
    if refresh and from_saved:
        raise ValueError("Choose either refresh or from_saved, not both")
    if from_saved:
        LOGGER.info("Adopting saved CSV results; no raw-data extraction or model refitting")
        tables = load_saved_csv_tables(config.output)
        regenerate_all_figures(tables, config)
        save_result_snapshot(tables, config, "explicit import of saved CSV results (8-decimal exported precision)")
    elif not refresh and (config.output / "analysis_results.pickle").exists():
        tables = load_result_snapshot(config)
        write_tables(tables, config.output)
        regenerate_all_figures(tables, config)
    else:
        LOGGER.info("Reading raw data and running the full analysis")
        from utils.pipeline import run_analysis
        run_analysis(config)


def main(argv: Sequence[str] | None = None) -> int:
    """Set up logging, load configuration, and report workflow success or failure."""
    args = parse_arguments(argv)
    try:
        log_path = configure_logging(args.log_level)
        if args.version:
            LOGGER.info("main.py %s", SCRIPT_VERSION)
            return 0
        LOGGER.info("Run log: %s", log_path)
        warnings.filterwarnings("once", category=RuntimeWarning)
        config = load_config(args.config)
        run_workflow(config, refresh=args.refresh, from_saved=args.from_saved)
        LOGGER.info("Finished successfully. Results: %s; analysis figures: %s", config.output, config.figures)
    except Exception:
        LOGGER.exception("Analysis/output generation failed")
        return 1
    return 0
