"""Save and validate result-table snapshots for reproducible figure regeneration."""

from __future__ import annotations

import hashlib
import json
import numpy as np
import os
import pandas as pd
import pickle
import platform
from pathlib import Path
from utils.constants import (
    CACHE_SCHEMA,
    LOGGER,
    PROJECT_ROOT,
)


def analysis_code_hash() -> str:
    """Fingerprint the entry point and every Python source used by the pipeline."""
    # Discover project scripts and this package instead of hard-coding old names.
    package_root = Path(__file__).resolve().parent
    paths = sorted(set(PROJECT_ROOT.glob("*.py")) | set(package_root.rglob("*.py")))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(PROJECT_ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _snapshot_signature(config):
    import statsmodels
    settings = json.loads(config.config_path.read_text())
    # Export preferences do not change the saved scientific results.
    for key in ("figures", "save_final", "auto_open"):
        settings.pop(key, None)
    config_hash = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    return {
        "schema": CACHE_SCHEMA,
        "analysis_code_sha256": analysis_code_hash(),
        "config_sha256": config_hash,
        "python": platform.python_version(),
        "pandas": pd.__version__, "numpy": np.__version__,
        "statsmodels": statsmodels.__version__,
    }


def save_result_snapshot(tables, config, provenance):
    """Atomically save dataframes, not fitted models; preserve full in-memory precision."""
    # Validate source/config access before replacing an existing snapshot.
    signature = _snapshot_signature(config)
    target = config.output / "analysis_results.pickle"
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(".tmp")
    with temp.open("wb") as handle:
        pickle.dump(tables, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temp, target)
    metadata = {**signature, "provenance": provenance,
                "pickle_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "table_names": sorted(tables),
                "csv_sha256": {name: hashlib.sha256((config.output / (name + ".csv")).read_bytes()).hexdigest()
                               for name in tables}}
    meta_path = target.with_suffix(".json")
    temp = meta_path.with_suffix(".tmp")
    temp.write_text(json.dumps(metadata, indent=2) + "\n")
    os.replace(temp, meta_path)


def load_result_snapshot(config):
    """Load only this project's locally generated, checksum-verified snapshot.

    Pickles can execute code: do not substitute downloaded/untrusted pickle files.
    The cache is a saved-result snapshot, not a claim that raw files are unchanged.
    Use --refresh after changing raw input data.
    """
    target = config.output / "analysis_results.pickle"
    metadata = json.loads(target.with_suffix(".json").read_text())
    if any(metadata.get(k) != v for k, v in _snapshot_signature(config).items()):
        raise ValueError(
            (
                'Result snapshot is incompatible with code/config/environment. Use '
                '--refresh, or explicitly --from-saved for existing CSV results.'
            ))
    if hashlib.sha256(target.read_bytes()).hexdigest() != metadata['pickle_sha256']:
        raise ValueError("Result snapshot checksum mismatch; use --refresh or --from-saved.")
    with target.open("rb") as handle:
        tables = pickle.load(handle)
    if not isinstance(tables, dict) or sorted(tables) != metadata['table_names'] or not all(
            isinstance(v, pd.DataFrame) for v in tables.values()):
        raise ValueError("Invalid table snapshot")
    LOGGER.info("Reusing saved results (%s); raw data are not re-read. Use --refresh for changed raw data.",
                metadata['provenance'])
    return tables
