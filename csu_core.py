"""Core utilities for the shuffled-vs-unshuffled comparison pipeline.

This file is auto-refactored out of the original `compare_shuffled_unshuffled.py` to improve readability.
It contains:
- Robust statistical helpers
- CSV probing helpers
- Plot saving utilities (Plotly HTML/PNG/EPS + an index page)
- Generic helpers used by feature extraction, pipeline, and mixed-model modules

Public API:
- compare_shuffled_unshuffled
- compare_participant_metrics
"""

from __future__ import annotations

import os
import warnings
import math
from typing import Optional, List, Tuple, Any

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.stats import ttest_ind, linregress, pearsonr, mannwhitneyu

import plotly.express as px  # noqa:F401
import plotly.graph_objects as go  # noqa:F401

import statsmodels.formula.api as smf  # noqa:F401
import statsmodels.api as sm  # noqa:F401
from statsmodels.stats.multitest import multipletests  # noqa:F401

# Project helpers (used for plot output directories).
from helper import HMD_helper
import common  # project module used by HMD_helper.save_plotly
from custom_logger import CustomLogger

# Shared yaw/quaternion column heuristics.
# These are used by file-selection helpers (e.g., `_score_trial_file`) and yaw feature extraction.
# Kept in a standalone module to avoid circular imports between `csu_core` and `csu_features`.
from csu_yaw_constants import _YAW_CANDIDATES, _QUAT_REGEX, _QUAT_LIST_COL_PAT  # noqa:F401

HAVE_SM = True
logger = CustomLogger(__name__)  # use custom logger


# -----------------------
# Plot saving utilities
# -----------------------

_PLOTS_WRITTEN: List[Tuple[str, str]] = []  # (name, html_path) in primary output dir
_KALEIDO_WARNED = False


def _np_trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    # Avoid numpy deprecation warnings (np.trapz -> np.trapezoid in newer numpy).
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))  # type: ignore[attr-defined]
    return float(np.trapz(y, x))  # type: ignore


def _safe_corr(x: pd.Series, y: pd.Series, min_n: int = 3) -> float:
    """Pearson correlation with guards for NaNs / constants."""
    mask = (~x.isna()) & (~y.isna())
    if mask.sum() < min_n:
        return np.nan
    xx = x.loc[mask]
    yy = y.loc[mask]
    if xx.nunique(dropna=True) < 2 or yy.nunique(dropna=True) < 2:
        return np.nan
    return float(xx.corr(yy))


def _is_near_constant(arr: NDArray[Any] | None, range_thr: float = 1e-6, sd_thr: float = 1e-6) -> bool:
    """True if array is (almost) constant."""
    if arr is None:
        return True

    a = np.asarray(arr, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size < 2:
        return True

    rng: float = float(np.nanmax(a) - np.nanmin(a))
    sd: float = float(np.nanstd(a, ddof=1))

    return bool((rng <= range_thr) or (sd <= sd_thr))


def _perm_pvalue_mean_diff(xa: np.ndarray, xb: np.ndarray, n_resamples: int = 20000, seed: int = 0) -> float:
    """Two-sided permutation p-value for difference in means."""
    xa = np.asarray(xa, dtype=float)
    xb = np.asarray(xb, dtype=float)
    xa = xa[np.isfinite(xa)]
    xb = xb[np.isfinite(xb)]
    if xa.size == 0 or xb.size == 0:
        return np.nan
    obs = float(np.mean(xa) - np.mean(xb))
    pooled = np.concatenate([xa, xb])
    n_a = xa.size
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(int(n_resamples)):
        perm = rng.permutation(pooled)
        diff = float(np.mean(perm[:n_a]) - np.mean(perm[n_a:]))
        if abs(diff) >= abs(obs) - 1e-15:
            count += 1
    return float((count + 1) / (n_resamples + 1))


def _safe_two_sample_test(xa_in: Any, xb_in: Any, *, equal_var: bool = False, min_n: int = 2,
                          const_range_thr: float = 1e-6, const_sd_thr: float = 1e-6,
                          n_perm: int = 20000, seed: int = 0) -> Tuple[float, float, str, str]:
    """Robust 2-sample comparison returning (t_or_nan, p, test_name, note).

    - If both groups are (near) constant and equal -> p=1 (identical).
    - If (near) constant or precision-loss warning occurs -> permutation p-value.
    - Otherwise Welch t-test.
    Also tries Mann–Whitney U when appropriate (heavy ties).
    """
    xa = np.asarray(pd.to_numeric(pd.Series(xa_in), errors="coerce"), dtype=float)
    xb = np.asarray(pd.to_numeric(pd.Series(xb_in), errors="coerce"), dtype=float)
    xa = xa[np.isfinite(xa)]
    xb = xb[np.isfinite(xb)]
    if xa.size < min_n or xb.size < min_n:
        return (np.nan, np.nan, "insufficient_n", "n<2 in one group")

    mean_a = float(np.mean(xa))
    mean_b = float(np.mean(xb))
    sd_a = float(np.std(xa, ddof=1)) if xa.size > 1 else 0.0
    sd_b = float(np.std(xb, ddof=1)) if xb.size > 1 else 0.0

    a_const = _is_near_constant(xa, const_range_thr, const_sd_thr)
    b_const = _is_near_constant(xb, const_range_thr, const_sd_thr)

    if a_const and b_const:
        if np.isclose(mean_a, mean_b, atol=const_range_thr, rtol=0.0):
            return (0.0, 1.0, "constant_equal", "both groups (near) constant and equal")
        p_perm = _perm_pvalue_mean_diff(xa, xb, n_resamples=max(2000, n_perm // 5), seed=seed)
        return (np.nan, float(p_perm), "perm_mean_diff", "both groups (near) constant; Welch t-test unstable")

    # If one group constant or very small variance, Welch can be unstable; use Mann–Whitney or permutation
    if a_const or b_const or (sd_a < const_sd_thr * 10) or (sd_b < const_sd_thr * 10):
        # Try Mann–Whitney U (handles non-normal + ties); fall back to permutation.
        try:
            u_stat, p_u = mannwhitneyu(xa, xb, alternative="two-sided", method="auto")
            return (np.nan, float(p_u), "mannwhitneyu", "one group low-variance/constant; used MWU")
        except Exception:
            p_perm = _perm_pvalue_mean_diff(xa, xb, n_resamples=n_perm, seed=seed)
            return (np.nan, float(p_perm), "perm_mean_diff", "one group low-variance/constant; MWU failed")

    # Welch t-test, but capture precision-loss warning and fall back to permutation p-value if it happens
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", RuntimeWarning)
        t_stat, p_val = ttest_ind(xa, xb, equal_var=equal_var)
    warn_text = "; ".join([str(ww.message) for ww in w]) if w else ""
    if "Precision loss occurred" in warn_text:
        p_perm = _perm_pvalue_mean_diff(xa, xb, n_resamples=n_perm, seed=seed)
        return (float(t_stat), float(p_perm), "welch+perm", "precision-loss warning; using permutation p-value")

    return (float(t_stat), float(p_val), "welch", "")


def _zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    m = float(s.mean())
    sd = float(s.std(ddof=0))
    if sd == 0 or np.isnan(sd):
        return s * np.nan
    return (s - m) / sd


def _fisher_z(r: float) -> float:
    if r is None or np.isnan(r):
        return np.nan
    if abs(r) >= 1:
        return np.nan
    return float(np.arctanh(r))


def _xcorr_max_r_lag(x: np.ndarray, y: np.ndarray, dt_s: float, max_lag_s: float = 2.0, min_n: int = 10) -> tuple:
    """Max (absolute) Pearson cross-correlation and its lag.

    We compute r(k) = corr(x[t], y[t+k]). Positive lag means y occurs *after* x.
    In our usage: x=yaw_speed, y=trigger_derivative -> positive lag suggests head movement precedes risk updates.
    """
    if x is None or y is None:
        return (np.nan, np.nan)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(x.size, y.size)
    if n < min_n or (not np.isfinite(dt_s)) or dt_s <= 0:
        return (np.nan, np.nan)
    x = x[:n]
    y = y[:n]

    max_k = int(round(max_lag_s / dt_s))
    if max_k < 1:
        return (np.nan, np.nan)

    best_r = np.nan
    best_k = 0

    for k in range(-max_k, max_k + 1):
        if k == 0:
            continue
        if k > 0:
            xx = x[:-k]
            yy = y[k:]
        else:
            kk = -k
            xx = x[kk:]
            yy = y[:-kk]

        if xx.size < min_n:
            continue
        mask = np.isfinite(xx) & np.isfinite(yy)
        if mask.sum() < min_n:
            continue

        xxm = xx[mask]
        yym = yy[mask]
        if np.unique(xxm).size < 2 or np.unique(yym).size < 2:
            continue

        r = np.corrcoef(xxm, yym)[0, 1]
        if not np.isfinite(r):
            continue

        if (not np.isfinite(best_r)) or (abs(r) > abs(best_r)):
            best_r = float(r)
            best_k = int(k)

    if not np.isfinite(best_r):
        return (np.nan, np.nan)
    return (best_r, float(best_k) * float(dt_s))


def _trial_num_display(x: pd.Series) -> pd.Series:
    """Convert 0-based trial_index to 1-based trial number for plotting.

    Many internal computations use 0-based indexing (0..39). For presentation,
    we display trials as 1..40.
    """
    xs = pd.to_numeric(x, errors="coerce")
    try:
        mn = float(np.nanmin(xs.to_numpy(dtype=float)))
    except Exception:
        return xs
    if np.isfinite(mn) and mn == 0.0:
        return xs + 1
    return xs


def _read_csv_flexible(fp: str) -> Optional[pd.DataFrame]:
    for sep in [",", ";", "\t"]:
        try:
            return pd.read_csv(fp, sep=sep)
        except Exception:
            continue
    return None


def _probe_csv_columns(fp: str) -> Optional[List[str]]:
    """Read only the header (and optionally 1 row) to get column names.

    This is used to avoid accidentally selecting processed/summary CSVs that share the same
    video_id in their filename but do not contain the raw HMD rotation columns.
    """
    for sep in [",", ";", "\t"]:
        try:
            df = pd.read_csv(fp, sep=sep, nrows=1)
            return [str(c) for c in df.columns]
        except Exception:
            continue
    return None


def _score_trial_file(fp: str) -> int:
    """Heuristic score for selecting the *raw* trial CSV for yaw extraction."""
    cols = _probe_csv_columns(fp)
    if not cols:
        return 0
    colset = set(cols)
    score = 0

    # Highest priority: the exact raw HMD quaternion columns used by the prior pipeline
    req = {"HMDRotationW", "HMDRotationX", "HMDRotationY", "HMDRotationZ"}
    if req.issubset(colset):
        score += 100

    # General quaternion columns (list-like or separate) – moderate priority
    if any(_QUAT_LIST_COL_PAT.search(c) for c in colset):
        score += 20
    # If we can match all 4 components via regex, add score
    comp_hits = 0
    for comp in ["w", "x", "y", "z"]:
        if any(_QUAT_REGEX[comp].search(c) for c in colset):
            comp_hits += 1
    if comp_hits == 4:
        score += 20

    # Yaw columns are helpful but lower priority (can be non-HMD yaw)
    if any(c in colset for c in _YAW_CANDIDATES):
        score += 10

    if "Timestamp" in colset:
        score += 5

    return score


def _choose_trial_file(csv_files: List[str], vid: str, folder: str) -> Optional[str]:
    """Pick the best candidate CSV for a given video_id within a participant folder.

    Replicates the behaviour of the previous pipeline (helper.export_participant_quaternion_matrix):
    prefer an exact file named '{video_id}.csv'. Otherwise, pick the file whose *columns* most
    resemble the raw trial log (contains HMDRotationW/X/Y/Z).
    """
    # 1) Exact match: <folder>/<vid>.csv
    exact = os.path.join(folder, f"{vid}.csv")
    if os.path.isfile(exact):
        return exact

    # 2) Candidates containing the vid in the basename
    candidates = [fp for fp in csv_files if vid in os.path.basename(fp)]
    if not candidates:
        return None

    # 3) If any basename exactly matches, prefer that
    exact2 = [fp for fp in candidates if os.path.basename(fp) == f"{vid}.csv"]
    if exact2:
        return exact2[0]

    # 4) Score by header contents
    scored = sorted(((_score_trial_file(fp), fp) for fp in candidates), reverse=True)
    best_score, best_fp = scored[0]

    # Tie-break: shorter basename tends to be the raw file (fewer prefixes/suffixes)
    top = [fp for sc, fp in scored if sc == best_score]
    if len(top) > 1:
        top = sorted(top, key=lambda x: (len(os.path.basename(x)), os.path.basename(x)))
        best_fp = top[0]

    return best_fp


def _bh_fdr(pvals: pd.Series) -> pd.Series:
    """Benjamini–Hochberg FDR correction (returns q-values aligned to input index)."""
    p = pd.to_numeric(pvals, errors="coerce")
    q = pd.Series(np.nan, index=p.index, dtype=float)

    p_non = p.dropna()
    n = int(p_non.shape[0])
    if n == 0:
        return q

    # Force a real numpy float array so np.argsort / arithmetic are well-typed
    pv: NDArray[np.float64] = p_non.to_numpy(dtype=np.float64, copy=False)

    order = np.argsort(pv)                # NDArray[int]
    ranked = pv[order]                    # NDArray[float]
    qs = ranked * (n / np.arange(1, n + 1, dtype=np.float64))
    qs = np.minimum.accumulate(qs[::-1])[::-1]

    q.loc[p_non.index[order]] = qs
    return q


def compare_participant_metrics(df: pd.DataFrame, metrics: List[str], group_col: str = "dataset", a: str = "shuffled",
                                b: str = "unshuffled", fdr: bool = True) -> pd.DataFrame:
    """Welch t-test + Cohen's d on participant-level metrics."""
    rows = []
    for m in metrics:
        if m not in df.columns:
            continue
        x = pd.to_numeric(df[df[group_col] == a][m], errors="coerce").dropna()
        y = pd.to_numeric(df[df[group_col] == b][m], errors="coerce").dropna()
        if len(x) < 2 or len(y) < 2:
            continue

        t, p, test_name, test_note = _safe_two_sample_test(x, y, equal_var=False)

        nx, ny = len(x), len(y)
        sx, sy = float(x.std(ddof=1)), float(y.std(ddof=1))
        pooled = np.sqrt(((nx - 1) * sx * sx + (ny - 1) * sy * sy) / (nx + ny - 2)) if (nx + ny - 2) > 0 else np.nan
        d = (float(x.mean()) - float(y.mean())) / pooled if pooled and not np.isnan(pooled) else np.nan

        rows.append({
            "metric": m,
            "n_shuffled": nx,
            "n_unshuffled": ny,
            "mean_shuffled": float(x.mean()),
            "mean_unshuffled": float(y.mean()),
            "t_stat": float(t),
            "p_value": float(p) if p is not None else np.nan,
            "test": test_name,
            "note": test_note,
            "cohens_d": float(d) if d is not None else np.nan,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_value"] = _bh_fdr(out["p_value"]) if fdr else np.nan
    out = out.sort_values(["q_value", "p_value"]) if fdr else out.sort_values("p_value")
    return out


def _cohens_d(xa: np.ndarray, xb: np.ndarray) -> float:
    """Cohen's d using pooled SD (ignores NaNs; expects 1D arrays)."""
    xa = np.asarray(xa, dtype=float)
    xb = np.asarray(xb, dtype=float)
    xa = xa[~np.isnan(xa)]
    xb = xb[~np.isnan(xb)]
    if len(xa) < 2 or len(xb) < 2:
        return np.nan
    nx, ny = len(xa), len(xb)
    sx, sy = np.std(xa, ddof=1), np.std(xb, ddof=1)
    denom = (nx + ny - 2)
    if denom <= 0:
        return np.nan
    pooled = np.sqrt(((nx - 1) * sx * sx + (ny - 1) * sy * sy) / denom)
    if pooled == 0 or np.isnan(pooled):
        return np.nan
    return float((np.mean(xa) - np.mean(xb)) / pooled)


def _build_cond_key(df: pd.DataFrame) -> pd.Series:
    """Create a compact condition key from available mapping factors.

    Used when a feature table lacks a precomputed `condition_name` but comparisons
    request grouping by it.
    """
    cols: List[str] = []
    for c in ["yielding", "eHMIOn", "camera", "distPed"]:
        if c in df.columns:
            cols.append(c)

    if cols:
        parts = []
        for c in cols:
            parts.append(c + "=" + df[c].astype(str))
        return parts[0] if len(parts) == 1 else parts[0].str.cat(parts[1:], sep="|")

    if "condition_name" in df.columns:
        return df["condition_name"].astype(str)
    if "video_id" in df.columns:
        return df["video_id"].astype(str)
    return pd.Series(["cond"] * len(df), index=df.index)


def compare_shuffled_unshuffled(
    features_df: pd.DataFrame,
    dataset_col: str = "dataset",
    shuffled_label: str = "shuffled",
    unshuffled_label: str = "unshuffled",
    metrics: Optional[List[str]] = None,
    groupby: Optional[List[str]] = None,
    equal_var: bool = False,
    fdr: bool = True,
    *,
    unit: str = "trial",
    participant_col: str = "participant_id",
) -> pd.DataFrame:
    """Statistical comparison between shuffled vs unshuffled.

    IMPORTANT
    - If unit is "trial" (default), the function treats rows as independent observations.
      This is fine for between subject tables where each participant contributes at most
      one row per group (eg grouped by condition_name).
    - If unit is "participant", the function aggregates each metric to a participant level
      mean within each group (dataset + groupby + participant_id) before running the test.
      This avoids the clustering violation that occurs when many trials per participant
      are analysed as if they were independent.

    Output
    Produces a tidy table with group keys (if any), metric name, n mean sd per dataset,
    t test p values, Cohen's d, and optional BH FDR corrected p values.

    This is implemented locally so the script does not depend on helper.HMD_helper having
    the method (repo versions may differ).
    """
    if groupby is None:
        groupby = []
    df = features_df.copy()
    if dataset_col not in df.columns:
        return pd.DataFrame()

    # Robustness: some feature tables (e.g., yaw-only exports) may not carry the
    # mapping-derived condition_name. If requested, synthesize a stable key from
    # available factors before grouping.
    missing_gb = [c for c in groupby if c not in df.columns]
    if missing_gb:
        if (missing_gb == ["condition_name"]) and ("condition_name" not in df.columns):
            df["condition_name"] = _build_cond_key(df)
            missing_gb = []
        else:
            # Drop unavailable groupers rather than raising.
            keep = [c for c in groupby if c in df.columns]
            if keep != groupby:
                logger.warning(f"[compare] Dropping missing group-by columns: {missing_gb}. Using: {keep if keep else 'overall'}")  # noqa: E501
            groupby = keep

    df = df[df[dataset_col].isin([shuffled_label, unshuffled_label])]
    if df.empty:
        return pd.DataFrame()

    # Default: compare a sensible set of metrics (trigger + yaw), falling back to numeric columns.
    if metrics is None:
        default_metrics = [
            # trigger
            "frac_time_unsafe",
            "trigger_mean",
            "trigger_peak",
            "trigger_p95",
            "max_ramp_rate",
            "trigger_sd",
            "dtrigger_sd",
            "latency_first_press_s",
            "latency_first_release_s",
            "n_safe_to_unsafe",
            "n_unsafe_to_safe",
            "mean_unsafe_bout_s",
            "longest_unsafe_bout_s",
            "press_release_hysteresis",
            "time_to_brake_first_press_s",
            "anticipation_index",
            # yaw
            "yaw_forward_frac_15",
            "yaw_mean",
            "yaw_sd",
            "yaw_abs_mean",
            "yaw_range_deg",
            "yaw_turn_rate_mean_dps",
            "yaw_turn_rate_sd_dps",
        ]
        metrics = [m for m in default_metrics if m in df.columns]
        if not metrics:
            # Fall back to numeric columns excluding obvious IDs/factors.
            exclude = set([dataset_col, "participant_id", "video_id", "condition_name", "source_file"])
            metrics = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

    # Force numeric where possible
    for m in metrics:
        if m in df.columns:
            df[m] = pd.to_numeric(df[m], errors="coerce")

    # Optional: participant level aggregation to avoid clustering violations when many
    # trials per participant are present.
    unit_norm = str(unit).strip().lower()
    if unit_norm in {"participant", "participant_mean", "cluster", "clustered"}:
        if participant_col in df.columns:
            agg_cols: List[str] = []
            if groupby:
                agg_cols.extend(list(groupby))
            agg_cols.extend([dataset_col, participant_col])
            # Each metric becomes the participant mean within each group
            df = (
                df.groupby(agg_cols, dropna=False)[metrics]
                .mean()
                .reset_index()
            )
        else:
            logger.warning(f"[compare] unit=participant requested but {participant_col} missing; using trial level")
            unit_norm = "trial"
    else:
        unit_norm = "trial"

    if groupby:
        groups = df.groupby(groupby, dropna=False)
    else:
        groups = [((), df)]

    rows = []
    for gkey, gdf in groups:
        gdict = {}
        if groupby:
            if not isinstance(gkey, tuple):
                gkey = (gkey,)
            gdict = dict(zip(groupby, gkey))

        a = gdf[gdf[dataset_col] == shuffled_label]
        b = gdf[gdf[dataset_col] == unshuffled_label]

        for m in metrics:
            if m not in gdf.columns:
                continue
            xa = a[m].to_numpy(dtype=float)
            xb = b[m].to_numpy(dtype=float)
            xa = xa[~np.isnan(xa)]
            xb = xb[~np.isnan(xb)]

            if len(xa) < 2 or len(xb) < 2:
                t_stat, p_val, d = np.nan, np.nan, np.nan
                test_name, test_note = "insufficient_n", "n<2 in one group"
            else:
                t_stat, p_val, test_name, test_note = _safe_two_sample_test(xa, xb, equal_var=equal_var)
                d = _cohens_d(xa, xb)

            rows.append({
                **gdict,
                "unit": unit_norm,
                "metric": m,
                "n_shuffled": int(len(xa)),
                "mean_shuffled": float(np.mean(xa)) if len(xa) else np.nan,
                "sd_shuffled": float(np.std(xa, ddof=1)) if len(xa) > 1 else np.nan,
                "n_unshuffled": int(len(xb)),
                "mean_unshuffled": float(np.mean(xb)) if len(xb) else np.nan,
                "sd_unshuffled": float(np.std(xb, ddof=1)) if len(xb) > 1 else np.nan,
                "t": float(t_stat) if not np.isnan(t_stat) else np.nan,
                "p": float(p_val) if not np.isnan(p_val) else np.nan,
                "test": test_name,
                "note": test_note,
                "cohens_d": float(d) if not np.isnan(d) else np.nan,
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    if fdr:
        out["p_fdr"] = np.nan
        if groupby:
            for _, idxs in out.groupby(groupby, dropna=False).groups.items():
                out.loc[list(idxs), "p_fdr"] = _bh_fdr(out.loc[list(idxs), "p"]).values
        else:
            out["p_fdr"] = _bh_fdr(out["p"]).values

    return out


def _print_table(df: pd.DataFrame, title: str, max_rows: int = 12) -> None:
    logger.info("\n" + title)
    if df is None or df.empty:
        logger.info("(empty)")
        return
    with pd.option_context("display.max_rows", max_rows, "display.max_columns", 50, "display.width", 160):
        logger.info(df.head(max_rows).to_string(index=False))


def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _get_output_dir_for_logs() -> str:
    try:
        if common is not None:
            return str(common.get_configs("output"))
    except Exception:
        pass
    return "(see common.get_configs('output'))"


def _resolve_plot_dirs(h: HMD_helper, out_root: Optional[str] = None) -> List[str]:
    out_dirs: List[str] = []
    # 1) common output (usually "_output")
    try:
        if common is not None:
            out_dirs.append(str(common.get_configs("output")))
    except Exception:
        pass
    # 2) explicit out_root (usually "_compare_output")
    if out_root:
        out_dirs.append(str(out_root))
    # 3) figures folder (often where the user is looking)
    try:
        if getattr(h, "folder_figures", None):
            out_dirs.append(str(h.folder_figures))
    except Exception:
        pass

    # de-dup while keeping order
    seen = set()
    return [d for d in out_dirs if d and not (d in seen or seen.add(d))]


def _save_plot(h: HMD_helper, fig, name: str, out_root: Optional[str] = None, record_index: bool = True,
               open_browser: bool = True) -> None:
    """
    Save a Plotly figure.

    Preferred behaviour (when available): use `HMD_helper.save_plotly(...)` so figures can pop up
    one by one (interactive Plotly view).

    Fallback behaviour: write HTML via `fig.write_html(...)` and export PNG and EPS (best effort;
    requires Kaleido).

    Notes
    - Set environment variable `CSU_OPEN_BROWSER=1` to force opening each plot in your browser.
    - The function still writes plots to the relevant output folders.
    """
    global _KALEIDO_WARNED
    if fig is None:
        return

    # env var override (handy when you do not want to touch call sites)
    env_open = os.environ.get('CSU_OPEN_BROWSER', None)
    if env_open is not None and str(env_open).strip() != '':
        open_browser = str(env_open).strip().lower() in {'1', 'true', 'yes', 'y'}

    # If we have the project's helper, prefer it. It handles Kaleido MathJax quirks and browser opening.
    # NOTE: Some variants of HMD_helper.save_plotly also save PDF outputs. This project does not need PDF.
    # We therefore:
    # 1) Pass flags to disable PDF where supported (eg save_pdf, save_final).
    # 2) Always do an explicit EPS export ourselves as a backstop.
    try:
        if hasattr(h, "save_plotly"):
            # HMD_helper.save_plotly saves to `common.get_configs("output")` (and optionally to h.folder_figures).
            # Only use it when that matches the requested out_root, otherwise we fall back to multi dir saving.
            out_dirs = _resolve_plot_dirs(h, out_root=out_root)
            common_out = out_dirs[0] if out_dirs else None

            def _same_dir(a: Optional[str], b: Optional[str]) -> bool:
                if not a or not b:
                    return False
                try:
                    return os.path.abspath(str(a)) == os.path.abspath(str(b))
                except Exception:
                    return False

            if (out_root is None) or _same_dir(out_root, common_out):
                # Build kwargs defensively across helper versions.
                # Only pass parameters that exist in the helper signature.
                import inspect

                params = set()
                try:
                    params = set(inspect.signature(h.save_plotly).parameters.keys())
                except Exception:
                    params = set()

                kwargs = {}
                if 'name' in params:
                    kwargs['name'] = name
                if 'remove_margins' in params:
                    kwargs['remove_margins'] = False
                if 'width' in params:
                    kwargs['width'] = 1320
                if 'height' in params:
                    kwargs['height'] = 680
                if 'save_eps' in params:
                    kwargs['save_eps'] = True
                if 'save_png' in params:
                    kwargs['save_png'] = True
                if 'save_html' in params:
                    kwargs['save_html'] = True
                if 'open_browser' in params:
                    kwargs['open_browser'] = bool(open_browser)
                if 'save_mp4' in params:
                    kwargs['save_mp4'] = False
                # Disable PDF like outputs where the helper supports it.
                if 'save_pdf' in params:
                    kwargs['save_pdf'] = False
                # Many helper versions use save_final to create additional final exports (often PDF).
                if 'save_final' in params:
                    kwargs['save_final'] = False

                h.save_plotly(fig, **kwargs)

                # Even if the helper ran, ensure EPS exists (and remove any PDF created).
                out_dirs = _resolve_plot_dirs(h, out_root=out_root)
                for d in out_dirs:
                    try:
                        os.makedirs(d, exist_ok=True)
                    except Exception:
                        continue

                    # EPS backstop
                    try:
                        fig.write_image(os.path.join(d, f"{name}.eps"), width=1320, height=680)
                    except Exception as e:
                        if not _KALEIDO_WARNED:
                            _KALEIDO_WARNED = True
                            logger.warning(f"[plot] NOTE: EPS export needs Kaleido. If missing, run: pip install -U kaleido. (first error: {e})")  # noqa: E501

                    # Remove PDF if the helper created one
                    try:
                        import glob
                        patterns = [
                            os.path.join(d, f"{name}*.pdf"),
                            os.path.join(d, "final", f"{name}*.pdf"),
                            os.path.join(d, "pdf", f"{name}*.pdf"),
                        ]
                        for pat in patterns:
                            for pdf_path in glob.glob(pat):
                                try:
                                    if os.path.isfile(pdf_path):
                                        os.remove(pdf_path)
                                except Exception:
                                    pass
                    except Exception:
                        pass

                return
    except Exception as e:
        logger.error(f"[plot] NOTE: HMD_helper.save_plotly failed for {name}: {e}. Falling back to fig.write_html.")

    # ---- Fallback multi directory saving ----
    out_dirs = _resolve_plot_dirs(h, out_root=out_root)
    if not out_dirs:
        logger.warning(f"[plot] SKIP {name}: no output directories resolved")
        return

    wrote_html: List[str] = []
    primary_out = out_dirs[0]

    for d in out_dirs:
        try:
            os.makedirs(d, exist_ok=True)
        except Exception:
            continue

        # HTML
        html_path = os.path.join(d, f"{name}.html")
        try:
            fig.write_html(html_path, include_plotlyjs="cdn", full_html=True, auto_open=False)
            wrote_html.append(html_path)
            if record_index and (d == primary_out):
                _PLOTS_WRITTEN.append((name, html_path))
        except Exception as e:
            logger.error(f"[plot] FAILED html {html_path}: {e}")

        # Make Kaleido more robust (MathJax is a frequent culprit).
        try:
            import plotly.io as pio  # local import
            try:
                pio.kaleido.scope.mathjax = None
            except Exception:
                pass
        except Exception:
            pass

        # PNG
        try:
            fig.write_image(os.path.join(d, f"{name}.png"), width=1320, height=680)
        except Exception as e:
            if not _KALEIDO_WARNED:
                _KALEIDO_WARNED = True
                logger.warning(f"[plot] NOTE: PNG and EPS export needs Kaleido. If missing, run: pip install -U kaleido. (first error: {e})")  # noqa: E501

        # EPS
        try:
            fig.write_image(os.path.join(d, f"{name}.eps"), width=1320, height=680)
        except Exception:
            pass

    # Open the primary HTML for this figure if requested
    if wrote_html and open_browser:
        try:
            import webbrowser
            webbrowser.open("file://" + os.path.abspath(wrote_html[0]))
        except Exception:
            pass

    if wrote_html:
        logger.info(f"[plot] wrote: {name} -> " + ", ".join(wrote_html))
    else:
        logger.info(f"[plot] FAILED to write any HTML for {name}")


def _write_plot_index_and_open(h: HMD_helper) -> None:
    """Write a single HTML index linking all recorded plots, then auto-open it (one tab)."""
    if not _PLOTS_WRITTEN:
        return

    out_dirs = _resolve_plot_dirs(h, out_root=None)
    if not out_dirs:
        return
    out_dir = out_dirs[0]
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception:
        return

    # build simple index
    rows = []
    for name, path in _PLOTS_WRITTEN:
        fn = os.path.basename(path)
        rows.append(f'<li><a href="{fn}">{name}</a></li>')
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Multiped plots</title></head>
<body>
<h2>Multiped plots index</h2>
<p>Saved in: {out_dir}</p>
<ul>
{''.join(rows)}
</ul>
</body></html>"""

    index_path = os.path.join(out_dir, "index_plots.html")
    try:
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(html)
        # auto-open (one tab)
        try:
            import webbrowser
            webbrowser.open("file://" + os.path.abspath(index_path))
        except Exception:
            pass
    except Exception as e:
        logger.error(f"[plot] FAILED to write index: {e}")


def _lin_slope(x: np.ndarray, y: np.ndarray) -> dict:
    """Return slope/intercept/r/p/n using scipy.stats.linregress; robust to NaNs."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 5 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return {"n": int(x.size), "slope": np.nan, "r": np.nan, "p": np.nan}
    res = linregress(x, y)
    return {"n": int(x.size), "slope": float(res.slope), "r": float(res.rvalue), "p": float(res.pvalue)}


def _fisher_ci(r: float, n: int) -> Tuple[float, float]:
    """95% CI for Pearson r via Fisher z transform."""
    if n is None or n < 4 or r is None or np.isnan(r) or abs(r) >= 1.0:
        return (np.nan, np.nan)
    z = np.arctanh(r)
    se = 1.0 / math.sqrt(n - 3)
    lo = np.tanh(z - 1.96 * se)
    hi = np.tanh(z + 1.96 * se)
    return (float(lo), float(hi))


def _p_from_z(z: float) -> float:
    """Two-sided p-value from z using erfc (no extra deps)."""
    if z is None or np.isnan(z):
        return np.nan
    return float(math.erfc(abs(z) / math.sqrt(2.0)))


def _compare_independent_corr(r1: float, n1: int, r2: float, n2: int) -> Tuple[float, float]:
    """Fisher r-to-z test for difference between independent correlations."""
    if any(v is None or np.isnan(v) for v in [r1, r2]) or n1 < 4 or n2 < 4:
        return (np.nan, np.nan)
    z1 = np.arctanh(r1)
    z2 = np.arctanh(r2)
    se = math.sqrt(1.0 / (n1 - 3) + 1.0 / (n2 - 3))
    if se <= 0:
        return (np.nan, np.nan)
    z = (z1 - z2) / se
    return (float(z), _p_from_z(z))


def _read_csv_loose(path: str) -> Optional[pd.DataFrame]:
    if path is None or (not isinstance(path, str)) or (not os.path.exists(path)):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_csv(path, sep=";")
        except Exception as e:
            logger.error(f"[Q] Could not read CSV {path}: {e}")
            return None


def _roc_curve_and_auc(y: np.ndarray, score: np.ndarray):
    # Simple ROC curve + AUC (no sklearn dependency)
    y = np.asarray(y).astype(int)
    score = np.asarray(score).astype(float)
    mask = (~np.isnan(score)) & (~np.isnan(y))
    y = y[mask]
    score = score[mask]
    # Need both classes
    if y.size < 3 or len(np.unique(y)) < 2:
        return None, None, np.nan

    order = np.argsort(-score)  # descending score
    y_sorted = y[order]
    score_sorted = score[order]  # noqa: F841

    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    tp_total = tp[-1]
    fp_total = fp[-1]
    if tp_total == 0 or fp_total == 0:
        return None, None, np.nan

    tpr = tp / tp_total
    fpr = fp / fp_total

    # Add (0,0) at start
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])

    auc = _np_trapezoid(tpr, fpr)
    return fpr, tpr, auc


def _corr_with_p(x: pd.Series, y: pd.Series, min_n: int = 5) -> Tuple[float, float, int]:
    mask = (~x.isna()) & (~y.isna())
    n0 = int(mask.sum())
    if n0 < min_n:
        return (np.nan, np.nan, n0)

    xx = pd.to_numeric(x.loc[mask], errors="coerce")
    yy = pd.to_numeric(y.loc[mask], errors="coerce")

    mask2 = (~xx.isna()) & (~yy.isna())
    xx = xx.loc[mask2]
    yy = yy.loc[mask2]

    n = int(xx.shape[0])
    if n < min_n or xx.nunique(dropna=True) < 2 or yy.nunique(dropna=True) < 2:
        return (np.nan, np.nan, n)

    # Force real numpy float arrays for SciPy typing + runtime consistency
    x_arr = xx.to_numpy(dtype=np.float64, copy=False)
    y_arr = yy.to_numpy(dtype=np.float64, copy=False)

    r, p = pearsonr(x_arr, y_arr)
    return (float(r), float(p), n)


def _ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# Star-import compatibility
# ---------------------------------------------------------------------------
# The refactor keeps many helpers with leading underscores to minimize changes
# to the original script bodies. Other modules import this file with
# `from csu_core import *`, but Python's default star-import behavior excludes
# underscore-prefixed names unless they are listed in __all__.
#
# To preserve the original code behavior without editing thousands of call
# sites, we export all public and underscore-prefixed symbols here.
__all__ = [k for k in globals().keys() if not k.startswith("__")]  # pyright: ignore[reportUnsupportedDunderAll]
