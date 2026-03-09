"""High-level analysis pipeline for shuffled-vs-unshuffled comparison.

This module holds the "A–F" pipeline from the original script (plots + reporting),
but relies on:
- `csu_core` for shared utilities (stats, plot saving, CSV helpers)
- `csu_features` for per-trial feature extraction

Configuration
-------------
Set these globals before calling `main()` (the entrypoint wrapper does this for you):
- DATASETS: dict with 'shuffled' and 'unshuffled' paths
- MAPPING_CSV: path to mapping.csv
- OUTPUT_ROOT: output folder

"""

from __future__ import annotations

import os
import re
from typing import Optional, List, Dict, Tuple, Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, linregress

from custom_logger import CustomLogger

# Import helpers + features (kept as names to avoid editing the original function bodies)
from csu_core import (  # noqa: F401
    HAVE_SM,
    _compare_independent_corr,
    _corr_with_p,
    _ensure_dir,
    _fisher_ci,
    _fisher_z,
    _lin_slope,
    _pick_col,
    _print_table,
    _read_csv_loose,
    _roc_curve_and_auc,
    _safe_corr,
    _safe_two_sample_test,
    _save_plot,
    _trial_num_display,
    common,
    compare_participant_metrics,
    compare_shuffled_unshuffled,
    sm,
    smf,
)
from csu_features import (
    load_trial_q123_from_responses,
    compute_participant_q_behavior_metrics,
    compute_trigger_features_dataset,
    compute_yaw_features_dataset,
    summarize_and_plot_yaw_results,
)  # noqa: F401


# Plotly is optional (some pipeline plots require it).
import plotly.express as px  # noqa:F401
import plotly.graph_objects as go  # noqa:F401

from helper import HMD_helper

# ----------------------
# Runtime configuration
# ----------------------
# NOTE:
# - Prefer using `ComparisonPipeline(...)` from `compare_shuffled_unshuffled.py`,
#   where you manually set the 6 path variables (3 shuffled + 3 unshuffled).
# - The globals below exist only for backwards compatibility with the original
#   script-style `main()` function.

DATASETS: Dict[str, Dict[str, str]] = {}
MAPPING_CSV: str = "mapping.csv"
OUTPUT_ROOT: str = "_compare_output"
logger = CustomLogger(__name__)  # use custom logger


INTAKE_COLUMNS_CATEGORICAL = [
    "Do you consent to participate in this study as described in the information provided above?",
    "Have you read and understood the above instructions?",
    "What is your gender?",
    "Do you have problems with hearing?",
    "How often in the last month have you experienced virtual reality?",
    "I am comfortable with walking in areas with dense traffic.",
    "The presence of another pedestrian reduces my willingness to cross the street when a car is driving towards me.",
    "What is your primary mode of transportation?",
    "On average, how often did you drive a vehicle in the last 12 months?",
    "About how many kilometers did you drive in last 12 months?",
    "How often do you do the following?: Becoming angered by a particular type of driver, and indicate your hostility by whatever means you can.",  # noqa: E501
    "How often do you do the following?: Disregarding the speed limit on a motorway.",
    "How often do you do the following?: Disregarding the speed limit on a residential road. ",
    "How many accidents were you involved in when driving a car in the last 3 years? (please include all accidents, regardless of how they were caused, how slight they were, or where they happened)",  # noqa: E501
    "How often do you do the following?: Driving so close to the car in front that it would be difficult to stop in an emergency. ",  # noqa: E501
    "How often do you do the following?: Racing away from traffic lights with the intention of beating the driver next to you. ",  # noqa: E501
    "How often do you do the following?: Sounding your horn to indicate your annoyance with another road user. ",
    "How often do you do the following?: Using a mobile phone without a hands free kit.",
    "How often do you do the following?: Doing my best not to be obstacle for other drivers.",
    "I would like to communicate with other road users while crossing the road (for instance, using eye contact, gestures, verbal communication, etc.).",  # noqa: E501
    "I trust an automated car more than a manually driven car.",
]

POST_COLUMNS_CATEGORICAL = [
    "The presence of another pedestrian influenced my willingness to cross the road.",
    "The type of car (with eHMI or without eHMI) affected my decision to cross the road.",
    "I trust an automated car more than a manually driven car.",
]

INTAKE_COLUMNS_NUMERIC = [
    "What is your age (in years)?",
    "At what age did you obtain your first license for driving a car or motorcycle?",
]

POST_COLUMNS_NUMERIC = [
    "How stressful did you feel during the experiment?",
    "How anxious did you feel during the experiment?",
    "How realistic did you find the experiment?",
    "How would you rate your overall experience in this experiment?",
]

# Which metrics to plot (must exist in trigger_trial_features.csv)
PLOT_METRICS = [
    "frac_time_unsafe",
    "trigger_mean",
    "trigger_p95",
    "max_ramp_rate",
    "latency_first_press_s",
    "latency_first_release_s",
    "press_release_hysteresis",
    "mean_unsafe_bout_s",
    "yaw_abs_mean",
    "yaw_forward_frac_15",
    "yaw_sd",
    "yaw_entropy",
    "yaw_speed_mean",
    "head_turn_count_15",
    "head_turn_dwell_mean_s_15",
    "yaw_speed_pre_press_mean_1s",
    "lag_turn_to_press_s_15",
    "yaw_pre_press_delta_1s",
    "yaw_speed_pre_press_mean_2s",
    "yaw_pre_press_mean_2s",
    "yaw_pre_press_mean_2to1s",
    "yaw_around_release_mean_pm1s",
    "xcorr_yawspd_dtrig_max_r",
    "xcorr_yawspd_dtrig_lag_s",
    "yaw_pre_release_delta_1s",
]


# ---------------------------------------------------------------------------
# Questionnaire columns of interest (copied from analysis.py)
# Note: The questionnaire loader prefixes raw CSV column names as:
#   intake__<question text>  and  post__<question text>
# These lists should therefore contain the *raw* column names from the CSVs.
# ---------------------------------------------------------------------------
INTAKE_COLUMNS_TO_EXTRACT = [
    "Do you consent to participate in this study as described in the information provided above?",
    "Have you read and understood the above instructions?",
    "What is your gender?",
    "Do you have problems with hearing?",
    "How often in the last month have you experienced virtual reality?",
    "I am comfortable with walking in areas with dense traffic.",
    "The presence of another pedestrian reduces my willingness to cross the street when a car is driving towards me.",
    "What is your primary mode of transportation?",
    "On average, how often did you drive a vehicle in the last 12 months?",
    "About how many kilometers did you drive in last 12 months?",
    "How often do you do the following?: Becoming angered by a particular type of driver, and indicate your hostility by whatever means you can.",  # noqa: E501
    "How often do you do the following?: Disregarding the speed limit on a motorway.",
    "How often do you do the following?: Disregarding the speed limit on a residential road. ",
    "How many accidents were you involved in when driving a car in the last 3 years? (please include all accidents, regardless of how they were caused, how slight they were, or where they happened)",  # noqa: E501
    "How often do you do the following?: Driving so close to the car in front that it would be difficult to stop in an emergency. ",  # noqa: E501
    "How often do you do the following?: Racing away from traffic lights with the intention of beating the driver next to you. ",  # noqa: E501
    "How often do you do the following?: Sounding your horn to indicate your annoyance with another road user. ",
    "How often do you do the following?: Using a mobile phone without a hands free kit.",
    "How often do you do the following?: Doing my best not to be obstacle for other drivers.",
    "I would like to communicate with other road users while crossing the road (for instance, using eye contact, gestures, verbal communication, etc.).",  # noqa: E501
    "I trust an automated car more than a manually driven car.",
]

POST_COLUMNS_TO_EXTRACT = [
    "The presence of another pedestrian influenced my willingness to cross the road.",
    "The type of car (with eHMI or without eHMI) affected my decision to cross the road.",
    "I trust an automated car more than a manually driven car.",
]

# Numeric questions (summarize with mean/sd etc.)
INTAKE_NUMERIC_COLUMNS_TO_EXTRACT = [
    "What is your age (in years)?",
    "At what age did you obtain your first license for driving a car or motorcycle?",
]

POST_NUMERIC_COLUMNS_TO_EXTRACT = [
    "How stressful did you feel during the experiment?",
    "How anxious did you feel during the experiment?",
    "How realistic did you find the experiment?",
    "How would you rate your overall experience in this experiment?",
]


def _print_dataset_overview(all_features: pd.DataFrame) -> None:
    if all_features.empty:
        return
    logger.info("\n=== Dataset overview ===")
    if "participant_id" in all_features.columns:
        pcount = all_features.groupby("dataset")["participant_id"].nunique().to_dict()
        logger.info("Unique participants:", pcount)
    tcount = all_features.groupby("dataset").size().to_dict()
    logger.info("Participant×trial rows:", tcount)

    miss_metrics = [m for m in PLOT_METRICS if m in all_features.columns]
    if miss_metrics:
        miss = (
            all_features.groupby("dataset")[miss_metrics]
            .apply(lambda g: g.isna().mean())
            .round(3)
        )
        logger.info("\nMissingness (fraction NaN) for plotted metrics:")
        logger.info(miss.to_string())


def _print_metric_descriptives(all_features: pd.DataFrame, metrics: List[str]) -> None:
    logger.info("\n=== Descriptive stats by dataset (non-NaN rows) ===")
    rows = []
    for metric in metrics:
        if metric not in all_features.columns:
            continue
        for ds, g in all_features.groupby("dataset"):
            s = pd.to_numeric(g[metric], errors="coerce").dropna()
            if len(s) == 0:
                continue
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            rows.append(
                {
                    "metric": metric,
                    "dataset": ds,
                    "n": int(s.shape[0]),
                    "mean": float(s.mean()),
                    "sd": float(s.std(ddof=1)) if s.shape[0] > 1 else float("nan"),
                    "median": float(s.median()),
                    "q1": float(q1),
                    "q3": float(q3),
                    "iqr": float(q3 - q1),
                }
            )
    if not rows:
        logger.info("(no metrics found)")
        return
    df = pd.DataFrame(rows)
    with pd.option_context("display.max_rows", 200, "display.max_columns", 50):
        logger.info(df.sort_values(["metric", "dataset"]).round(4).to_string(index=False))


def _print_top_results(res: pd.DataFrame, title: str, top_k: int = 15) -> None:
    if res is None or res.empty:
        logger.info(f"\n=== {title} ===\n(no rows)")
        return

    metric_col = _pick_col(res, ["metric", "variable", "var", "measure"])
    p_col = _pick_col(res, ["p_fdr", "q_value", "p_adj", "p_adjusted", "p_value", "p"])
    d_col = _pick_col(res, ["cohen_d", "effect_size_d", "d"])

    logger.info(f"\n=== {title} ===")
    cols_to_show = [c for c in [metric_col, d_col, p_col] if c]
    for c in ["t_stat", "t", "df", "n_shuffled", "n_unshuffled", "mean_shuffled", "mean_unshuffled"]:
        if c in res.columns and c not in cols_to_show:
            cols_to_show.append(c)

    if p_col:
        out = res.sort_values(p_col, ascending=True).head(top_k)
    elif d_col:
        out = res.reindex(res[d_col].abs().sort_values(ascending=False).index).head(top_k)
    else:
        out = res.head(top_k)

    with pd.option_context("display.max_rows", 200, "display.max_columns", 100):
        logger.info(out[cols_to_show].to_string(index=False))


def _factor_drift_curve(df: pd.DataFrame, factor_col: str) -> pd.DataFrame:
    """Compute proportion (mean) of a binary factor over trial_index by dataset.

    Returns columns: dataset, trial_index, prop, n, ci_lo, ci_hi.
    CI is a normal-approx binomial 95% interval (clipped to [0,1]).
    """
    if df is None or df.empty:
        return pd.DataFrame()
    if "dataset" not in df.columns or "trial_index" not in df.columns or factor_col not in df.columns:
        return pd.DataFrame()

    tmp = df[["dataset", "trial_index", factor_col]].copy()
    tmp["trial_index"] = pd.to_numeric(tmp["trial_index"], errors="coerce")
    tmp[factor_col] = pd.to_numeric(tmp[factor_col], errors="coerce")
    tmp = tmp.dropna(subset=["dataset", "trial_index", factor_col])
    if tmp.empty:
        return pd.DataFrame()

    grp = tmp.groupby(["dataset", "trial_index"])[factor_col].agg(["mean", "count"]).reset_index()
    grp = grp.rename(columns={"mean": "prop", "count": "n"})

    # Normal-approx CI (fine here because n≈50 per index; clip for safety)
    se = np.sqrt(np.clip(grp["prop"] * (1.0 - grp["prop"]) / grp["n"].replace(0, np.nan), 0, np.inf))
    grp["ci_lo"] = np.clip(grp["prop"] - 1.96 * se, 0.0, 1.0)
    grp["ci_hi"] = np.clip(grp["prop"] + 1.96 * se, 0.0, 1.0)
    return grp


def _plot_factor_drift_by_trial_index(df: pd.DataFrame, factor_col: str, title: str, name: str, h: HMD_helper) -> None:
    """Plot factor composition drift over trial_index (Randomised vs Fixed-order).

    Style is tuned for paper figures:
    - Dots + connecting lines
    - Fixed-order can be 0/1 by construction (same trial for everyone)
    - 95% CI band shown for Randomised (shuffled) only
    """
    if go is None:
        logger.warning("[plot] plotly.graph_objects not available; skipping factor drift plots.")
        return

    curve = _factor_drift_curve(df, factor_col=factor_col)
    if curve is None or curve.empty:
        logger.info(f"[plot] factor drift: no data for {factor_col}; skipping.")
        return

    # Display names + colors (match typical paper styling)
    label_map = {"shuffled": "Randomised", "unshuffled": "Fixed-order"}
    color_map = {
        "shuffled": "#1f77b4",    # blue
        "unshuffled": "#ff7f0e",  # orange
    }
    band_color = "rgba(31,119,180,0.18)"  # light blue

    # Order traces for legend (Randomised first)
    ds_order = [ds for ds in ["shuffled", "unshuffled"] if ds in set(curve["dataset"])]
    ds_order += [ds for ds in sorted(curve["dataset"].unique()) if ds not in ds_order]

    fig = go.Figure()

    for ds in ds_order:
        sub = curve[curve["dataset"] == ds].sort_values("trial_index")
        disp = label_map.get(ds, str(ds))
        col = color_map.get(ds, None)

        # CI band (Randomised only)
        if ds == "shuffled":
            fig.add_trace(
                go.Scatter(
                    x=_trial_num_display(sub["trial_index"]),
                    y=sub["ci_hi"],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=_trial_num_display(sub["trial_index"]),
                    y=sub["ci_lo"],
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=band_color,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        # Mean line with dots
        fig.add_trace(
            go.Scatter(
                x=_trial_num_display(sub["trial_index"]),
                y=sub["prop"],
                mode="lines+markers",
                name=disp,
                line=dict(width=3, color=col),
                marker=dict(size=9, symbol="circle", color=col),
                hovertemplate="dataset=%{text}<br>trial=%{x}<br>prop=%{y:.3f}<extra></extra>",
                text=[disp] * len(sub),
            )
        )

    fig.update_layout(
        title=title,
        template="simple_white",
        xaxis_title="Trial number",
        yaxis_title="Proportion",
        yaxis=dict(range=[-0.05, 1.05]),
        legend=dict(x=0.98, y=0.98, xanchor="right", yanchor="top", bgcolor="rgba(255,255,255,0.7)",
                    bordercolor="rgba(0,0,0,0.15)", borderwidth=1),
        margin=dict(l=60, r=20, t=60, b=55),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)")

    _save_plot(h, fig, name=name)


def _make_trial_pos(df: pd.DataFrame, order_col: Optional[str]) -> pd.DataFrame:
    """Ensure df has trial_pos and trial_pos_norm (0..1) within dataset×participant."""
    if "participant_id" not in df.columns:
        return df
    sort_cols = ["dataset", "participant_id"]
    if order_col is not None and order_col in df.columns:
        sort_cols.append(order_col)
    else:
        # fallback: preserve current order within each participant
        df = df.copy()
        df["_row_id_tmp"] = np.arange(len(df), dtype=int)
        sort_cols.append("_row_id_tmp")

    df = df.sort_values(sort_cols).copy()
    df["trial_pos"] = df.groupby(["dataset", "participant_id"]).cumcount()
    n_trials = df.groupby(["dataset", "participant_id"])["trial_pos"].transform("max") + 1
    df["trial_pos_norm"] = np.where(n_trials > 1, df["trial_pos"] / (n_trials - 1), 0.0)
    if "_row_id_tmp" in df.columns:
        df = df.drop(columns=["_row_id_tmp"])
    return df


def _split_half_means(df: pd.DataFrame, value_col: str, split: str) -> pd.DataFrame:
    """Return per-participant split-half means for `value_col`.

    Output columns: dataset, participant_id, half_A, half_B
    split:
      - 'odd_even': odd vs even trial numbers (1-based display)
      - 'early_late': first half vs second half by trial_index
    """
    need = {"dataset", "participant_id", "trial_index", value_col}
    if not need.issubset(df.columns):
        return pd.DataFrame()

    tmp = df[["dataset", "participant_id", "trial_index", value_col]].copy()
    tmp["trial_index"] = pd.to_numeric(tmp["trial_index"], errors="coerce")
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
    tmp = tmp.dropna(subset=["dataset", "participant_id", "trial_index"])
    if tmp.empty:
        return pd.DataFrame()

    out_rows = []
    for (dataset, pid), g in tmp.groupby(["dataset", "participant_id"]):
        g = g.sort_values("trial_index")
        n_total = int(np.nanmax(g["trial_index"].to_numpy(dtype=float)) + 1) if len(g) else 0
        if n_total <= 0:
            continue

        if split == "odd_even":
            trial_num = g["trial_index"] + 1  # display numbering
            a = g.loc[(trial_num % 2) == 1, value_col]
            b = g.loc[(trial_num % 2) == 0, value_col]
        elif split == "early_late":
            mid = max(1, n_total // 2)
            a = g.loc[g["trial_index"] < mid, value_col]
            b = g.loc[g["trial_index"] >= mid, value_col]
        else:
            continue

        # require at least a few observations on each side
        if a.notna().sum() < 3 or b.notna().sum() < 3:
            continue

        out_rows.append(
            {
                "dataset": dataset,
                "participant_id": pid,
                "half_A": float(a.mean()),
                "half_B": float(b.mean()),
            }
        )

    return pd.DataFrame(out_rows)


def _within_participant_reliability(df: pd.DataFrame, outcomes: List[str]) -> pd.DataFrame:
    """Compute split-half reliability (odd/even and early/late) per dataset for each outcome.

    Reliability is computed as correlation of split-half participant means across participants
    (rank-order stability / individual-differences reliability).
    """
    splits = ["odd_even", "early_late"]
    rows: List[Dict[str, object]] = []

    for outcome in outcomes:
        if outcome not in df.columns:
            continue
        for split in splits:
            halves = _split_half_means(df, outcome, split)
            if halves.empty:
                continue

            # per dataset
            for dataset, g in halves.groupby("dataset"):
                if len(g) < 4:
                    continue
                r, p = pearsonr(g["half_A"], g["half_B"])
                n = int(len(g))
                ci_lo, ci_hi = _fisher_ci(r, n)
                r_sb = (2.0 * r / (1.0 + r)) if r > -0.999 else np.nan
                rows.append(
                    {
                        "outcome": outcome,
                        "split": split,
                        "dataset": dataset,
                        "n_participants": n,
                        "r": float(r),
                        "r_spearman_brown": float(r_sb),
                        "ci_lo": ci_lo,
                        "ci_hi": ci_hi,
                        "p_value": float(p),
                    }
                )

            # add dataset-difference test (shuffled vs unshuffled) if both present
            if set(halves["dataset"].unique()) >= {"shuffled", "unshuffled"}:
                g1 = halves.loc[halves["dataset"] == "shuffled"]
                g2 = halves.loc[halves["dataset"] == "unshuffled"]
                if len(g1) >= 4 and len(g2) >= 4:
                    r1, _ = pearsonr(g1["half_A"], g1["half_B"])
                    r2, _ = pearsonr(g2["half_A"], g2["half_B"])
                    z, pz = _compare_independent_corr(r1, len(g1), r2, len(g2))
                    rows.append(
                        {
                            "outcome": outcome,
                            "split": split,
                            "dataset": "diff(shuffled-unshuffled)",
                            "n_participants": f"{len(g1)} vs {len(g2)}",
                            "r": float(r1 - r2),
                            "r_spearman_brown": np.nan,
                            "ci_lo": np.nan,
                            "ci_hi": np.nan,
                            "p_value": float(pz),
                            "z_diff": float(z),
                        }
                    )

    return pd.DataFrame(rows)


def _plot_reliability_scatter(df: pd.DataFrame, outcome: str, split: str):
    """Scatter of split-half participant means (two datasets in one panel)."""
    if go is None:
        return None
    halves = _split_half_means(df, outcome, split)
    if halves.empty:
        return None

    fig = go.Figure()
    color_map = {"shuffled": "#1f77b4", "unshuffled": "#ff7f0e"}

    for dataset, g in halves.groupby("dataset"):
        fig.add_trace(
            go.Scatter(
                x=g["half_A"],
                y=g["half_B"],
                mode="markers",
                name=str(dataset),
                marker=dict(size=7, opacity=0.85, color=color_map.get(str(dataset), None)),
                text=g["participant_id"],
                hovertemplate="participant=%{text}<br>half A=%{x:.3f}<br>half B=%{y:.3f}<extra></extra>",
            )
        )

    # identity line
    x_all = halves["half_A"].to_numpy(dtype=float)
    y_all = halves["half_B"].to_numpy(dtype=float)
    lo = float(np.nanmin(np.concatenate([x_all, y_all])))
    hi = float(np.nanmax(np.concatenate([x_all, y_all])))
    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            name="y=x",
            line=dict(dash="dash"),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    split_lbl = "Odd vs even trials" if split == "odd_even" else "Early vs late halves"
    fig.update_layout(
        title=f"Split-half stability: {outcome} ({split_lbl})",
        xaxis_title="Half A mean",
        yaxis_title="Half B mean",
        legend_title="Dataset",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    fig.update_xaxes(zeroline=False)
    fig.update_yaxes(zeroline=False, scaleanchor="x", scaleratio=1)
    return fig


def _infer_break_end_positions(n_trials: int) -> List[int]:
    """Infer end-of-block indices (0-based) for breaks.

    Heuristic: assume 4 blocks of roughly equal size.
    Backward-compatible: if a participant DataFrame/Series is passed by mistake,
    infer n_trials from its trial_pos / trial_index.
    """
    # Backward compatibility if caller passes a DataFrame/Series
    if isinstance(n_trials, (pd.DataFrame, pd.Series)):
        g = n_trials
        if isinstance(g, pd.Series):
            ser = pd.to_numeric(g, errors="coerce")
        else:
            col = "trial_pos" if "trial_pos" in g.columns else ("trial_index" if "trial_index" in g.columns else None)
            if col is None:
                ser = pd.Series(np.arange(len(g)))
            else:
                ser = pd.to_numeric(g[col], errors="coerce")
        if ser.notna().any():
            n_trials = int(ser.max()) + 1
        else:
            n_trials = int(len(ser))

    try:
        n_trials = int(n_trials)
    except Exception:
        return []

    if n_trials < 12:
        return []
    blocks = 4
    block_size = int(n_trials // blocks)
    if block_size < 5:
        return []
    ends = [block_size - 1, 2 * block_size - 1, 3 * block_size - 1]
    # keep only valid where we can take 2 trials before and 2 after
    ends = [e for e in ends if (e - 1) >= 0 and (e + 2) < n_trials]
    return ends


def _participant_learning_metrics(df: pd.DataFrame, outcomes: List[str]) -> pd.DataFrame:
    """Compute time-on-task slope + early/late drift + break reset per participant."""
    rows = []
    if df.empty:
        return pd.DataFrame()

    # ensure trial_pos exists
    if "trial_pos" not in df.columns:
        df = _make_trial_pos(df, order_col=None)

    for (ds, pid), g in df.groupby(["dataset", "participant_id"], dropna=False):
        g = g.sort_values("trial_pos")
        n_trials = int(g["trial_pos"].max()) + 1 if g["trial_pos"].notna().any() else len(g)
        break_ends = _infer_break_end_positions(n_trials)

        for col in outcomes:
            y = pd.to_numeric(g[col], errors="coerce")
            x = pd.to_numeric(g["trial_pos"], errors="coerce")
            lr = _lin_slope(x.values, y.values)
            slope = lr["slope"]

            # early/late drift (last third - first third)
            n = int(np.isfinite(y).sum())  # noqa:F841
            drift = np.nan
            if n_trials >= 6:
                k = max(2, n_trials // 3)
                early = y.iloc[:k]
                late = y.iloc[-k:]
                if np.isfinite(early).sum() >= 2 and np.isfinite(late).sum() >= 2:
                    drift = float(np.nanmean(late) - np.nanmean(early))

            # break reset: average (post2 - pre2) across inferred breaks
            reset = np.nan
            if break_ends:
                diffs = []
                for e in break_ends:
                    pre = y.iloc[max(0, e - 1): e + 1]          # two trials ending at e
                    post = y.iloc[e + 1: min(n_trials, e + 3)]  # two trials after e
                    if np.isfinite(pre).sum() >= 1 and np.isfinite(post).sum() >= 1:
                        diffs.append(float(np.nanmean(post) - np.nanmean(pre)))
                if diffs:
                    reset = float(np.nanmean(diffs))

            rows.append(
                {
                    "dataset": ds,
                    "participant_id": pid,
                    "metric": col,
                    "slope": slope,
                    "drift_late_minus_early": drift,
                    "post_break_reset": reset,
                    "n_trials": n_trials,
                    "n_valid": int(np.isfinite(y).sum()),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # pivot to wide participant table
    wide = out.pivot_table(
        index=["dataset", "participant_id"],
        columns="metric",
        values=["slope", "drift_late_minus_early", "post_break_reset"],
        aggfunc="first",
    )
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    return wide


def _build_cond_key(df: pd.DataFrame) -> pd.Series:
    """Create a compact condition key from available mapping factors."""
    cols = []
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


def _participant_sequential_metrics(df: pd.DataFrame, outcomes: List[str]) -> pd.DataFrame:
    """Compute per-participant sequential-dependence metrics.

    Metrics (per outcome):
      - switch_cost / switch_cost_condadj: outcome difference on switch vs repeat trials
      - carryover_prev_yielding / _condadj: effect of previous-trial yielding (binary)
      - carryover_prev_eHMIOn / _condadj: effect of previous-trial eHMI (binary)
      - carryover_prev_camera / _condadj: effect of previous-trial camera (binary)
      - carryover_prev_distPed / _condadj: slope of outcome vs previous-trial distance (ordinal)
      - autocorr_lag1: lag-1 autocorrelation of the outcome

    Rationale: in fixed-order designs, previous-trial factors are partially predictable from the sequence,
    which can inflate (or distort) naive carryover estimates.
    """
    if df.empty:
        return pd.DataFrame()
    if "trial_pos" not in df.columns:
        df = _make_trial_pos(df, order_col=None)

    df = df.copy()
    df["cond_key"] = _build_cond_key(df)

    def _binary_carryover(y: pd.Series, prev: pd.Series, min_n: int = 3) -> float:
        m0 = (prev == 0)
        m1 = (prev == 1)
        if np.isfinite(y[m0]).sum() >= min_n and np.isfinite(y[m1]).sum() >= min_n:
            return float(np.nanmean(y[m1]) - np.nanmean(y[m0]))
        return np.nan

    def _binary_carryover_condadj(
        g: pd.DataFrame,
        col: str,
        prev_col: str,
        min_n: int = 2,
    ) -> float:
        diffs = []
        weights = []
        for ck, gg in g.groupby("cond_key"):
            y2 = pd.to_numeric(gg[col], errors="coerce")
            pv = pd.to_numeric(gg[prev_col], errors="coerce")
            m0 = (pv == 0)
            m1 = (pv == 1)
            if np.isfinite(y2[m0]).sum() >= min_n and np.isfinite(y2[m1]).sum() >= min_n:
                diffs.append(float(np.nanmean(y2[m1]) - np.nanmean(y2[m0])))
                weights.append(int(len(gg)))
        if diffs:
            w = np.asarray(weights, dtype=float)
            return float(np.average(np.asarray(diffs, dtype=float), weights=w))
        return np.nan

    def _slope(x: pd.Series, y: pd.Series, min_n: int = 6) -> float:
        xx = pd.to_numeric(x, errors="coerce")
        yy = pd.to_numeric(y, errors="coerce")
        mask = np.isfinite(xx) & np.isfinite(yy)
        if mask.sum() < min_n:
            return np.nan
        x2 = xx[mask]
        y2 = yy[mask]
        if x2.nunique(dropna=True) < 2:
            return np.nan
        try:
            return float(linregress(x2.to_numpy(dtype=float), y2.to_numpy(dtype=float)).slope)
        except Exception:
            return np.nan

    def _slope_condadj(g: pd.DataFrame, col: str, prev_col: str) -> float:
        slopes = []
        weights = []
        for ck, gg in g.groupby("cond_key"):
            s = _slope(gg[prev_col], gg[col], min_n=5)
            if np.isfinite(s):
                slopes.append(float(s))
                weights.append(int(len(gg)))
        if slopes:
            w = np.asarray(weights, dtype=float)
            return float(np.average(np.asarray(slopes, dtype=float), weights=w))
        return np.nan

    rows = []
    for (ds, pid), g in df.groupby(["dataset", "participant_id"], dropna=False):
        g = g.sort_values("trial_pos").copy()
        g["prev_cond_key"] = g["cond_key"].shift(1)
        g["switch"] = (g["cond_key"] != g["prev_cond_key"]).astype(float)

        # previous-trial factors (shifted)
        g["prev_yielding"] = pd.to_numeric(g["yielding"], errors="coerce").shift(1) if "yielding" in g.columns else np.nan  # noqa: E501
        g["prev_eHMIOn"] = pd.to_numeric(g["eHMIOn"], errors="coerce").shift(1) if "eHMIOn" in g.columns else np.nan
        g["prev_camera"] = pd.to_numeric(g["camera"], errors="coerce").shift(1) if "camera" in g.columns else np.nan
        g["prev_distPed"] = pd.to_numeric(g["distPed"], errors="coerce").shift(1) if "distPed" in g.columns else np.nan

        for col in outcomes:
            y = pd.to_numeric(g[col], errors="coerce")

            # switch cost (simple within-person)
            sw = (g["switch"] == 1)
            rp = (g["switch"] == 0)
            sc = np.nan
            if np.isfinite(y[sw]).sum() >= 3 and np.isfinite(y[rp]).sum() >= 3:
                sc = float(np.nanmean(y[sw]) - np.nanmean(y[rp]))

            # condition-adjusted switch cost (within each current condition)
            sc_adj = np.nan
            diffs = []
            weights = []
            for ck, gg in g.groupby("cond_key"):
                y2 = pd.to_numeric(gg[col], errors="coerce")
                sw2 = (gg["switch"] == 1)
                rp2 = (gg["switch"] == 0)
                if np.isfinite(y2[sw2]).sum() >= 2 and np.isfinite(y2[rp2]).sum() >= 2:
                    diffs.append(float(np.nanmean(y2[sw2]) - np.nanmean(y2[rp2])))
                    weights.append(int(len(gg)))
            if diffs:
                w = np.asarray(weights, dtype=float)
                sc_adj = float(np.average(np.asarray(diffs, dtype=float), weights=w))

            # carryover effects (previous-trial factors)
            co_y = _binary_carryover(y, g["prev_yielding"])
            co_y_adj = _binary_carryover_condadj(g, col, "prev_yielding")

            co_e = _binary_carryover(y, g["prev_eHMIOn"])
            co_e_adj = _binary_carryover_condadj(g, col, "prev_eHMIOn")

            co_c = _binary_carryover(y, g["prev_camera"])
            co_c_adj = _binary_carryover_condadj(g, col, "prev_camera")

            co_d = _slope(g["prev_distPed"], y)
            co_d_adj = _slope_condadj(g, col, "prev_distPed")

            # autocorr with lag-1
            ac = np.nan
            if len(y) >= 6:
                ac = _safe_corr(y.iloc[1:], y.shift(1).iloc[1:], min_n=5)

            rows.append(
                {
                    "dataset": ds,
                    "participant_id": pid,
                    "metric": col,
                    "switch_cost": sc,
                    "switch_cost_condadj": sc_adj,
                    "carryover_prev_yielding": co_y,
                    "carryover_prev_yielding_condadj": co_y_adj,
                    "carryover_prev_eHMIOn": co_e,
                    "carryover_prev_eHMIOn_condadj": co_e_adj,
                    "carryover_prev_camera": co_c,
                    "carryover_prev_camera_condadj": co_c_adj,
                    "carryover_prev_distPed": co_d,
                    "carryover_prev_distPed_condadj": co_d_adj,
                    "autocorr_lag1": ac,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    wide = out.pivot_table(
        index=["dataset", "participant_id"],
        columns="metric",
        values=[
            "switch_cost",
            "switch_cost_condadj",
            "carryover_prev_yielding",
            "carryover_prev_yielding_condadj",
            "carryover_prev_eHMIOn",
            "carryover_prev_eHMIOn_condadj",
            "carryover_prev_camera",
            "carryover_prev_camera_condadj",
            "carryover_prev_distPed",
            "carryover_prev_distPed_condadj",
            "autocorr_lag1",
        ],
        aggfunc="first",
    )
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    return wide


def _participant_break_matched_metrics(
    df: pd.DataFrame,
    outcomes: List[str],
    window_k: int = 5,
    match_cols: Tuple[str, str] = ("yielding", "eHMIOn"),
) -> pd.DataFrame:
    """Participant-level break reset controlling for factor composition.

    For each inferred break boundary (end-of-block index), compare the last `window_k`
    trials before the break to the first `window_k` trials after the break.
    To reduce confounding from factor drift, compute post--pre within matched
    cells defined by `match_cols` (default yielding×eHMI), then average within-break
    using weights proportional to the minimum cell sample size, and finally average
    across breaks.

    Returns a participant table with columns:
        break_reset_matched_<outcome>
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # ensure trial_pos exists (0-based order within participant)
    if "trial_pos" not in df.columns:
        order_col = _pick_col(df, ["trial_index", "no", "trial_no", "trial", "order"])
        df = _make_trial_pos(df, order_col=order_col)

    rows: List[Dict[str, float]] = []

    for (ds, pid), g in df.groupby(["dataset", "participant_id"], dropna=False):
        g = g.sort_values("trial_pos")
        # infer n_trials robustly
        if g["trial_pos"].notna().any():
            n_trials = int(pd.to_numeric(g["trial_pos"], errors="coerce").max()) + 1
        else:
            n_trials = int(len(g))

        break_ends = _infer_break_end_positions(n_trials)
        if not break_ends:
            continue

        out_row: Dict[str, float] = {"dataset": ds, "participant_id": pid}

        for col in outcomes:
            if col not in g.columns:
                continue
            y_all = pd.to_numeric(g[col], errors="coerce")
            if y_all.notna().sum() < 3:
                continue

            resets: List[float] = []
            weights_break: List[int] = []

            for e in break_ends:
                pre_idx = list(range(max(0, e - window_k + 1), e + 1))
                post_idx = list(range(e + 1, min(n_trials, e + 1 + window_k)))

                pre = g[g["trial_pos"].isin(pre_idx)]
                post = g[g["trial_pos"].isin(post_idx)]
                if pre.empty or post.empty:
                    continue

                diffs: List[float] = []
                weights: List[int] = []

                # cells present in either side
                cells = (
                    pd.MultiIndex.from_frame(pre[list(match_cols)])
                    .union(pd.MultiIndex.from_frame(post[list(match_cols)]))
                    .unique()
                )

                for cell in cells:
                    pre_c = pre[(pre[match_cols[0]] == cell[0]) & (pre[match_cols[1]] == cell[1])]
                    post_c = post[(post[match_cols[0]] == cell[0]) & (post[match_cols[1]] == cell[1])]
                    if pre_c.empty or post_c.empty:
                        continue

                    pre_y = pd.to_numeric(pre_c[col], errors="coerce").dropna()
                    post_y = pd.to_numeric(post_c[col], errors="coerce").dropna()
                    if pre_y.empty or post_y.empty:
                        continue

                    diffs.append(float(post_y.mean() - pre_y.mean()))
                    weights.append(int(min(len(pre_y), len(post_y))))

                if diffs:
                    wsum = int(np.sum(weights))
                    if wsum <= 0:
                        continue
                    resets.append(float(np.average(diffs, weights=weights)))
                    weights_break.append(wsum)

            if resets:
                out_row[f"break_reset_matched_{col}"] = float(np.average(resets,
                                                                         weights=weights_break)) if weights_break else float(np.mean(resets))  # noqa: E501
            else:
                out_row[f"break_reset_matched_{col}"] = np.nan

        rows.append(out_row)

    out = pd.DataFrame(rows)
    return out


def _binned_curves(df: pd.DataFrame, outcome_col: str, n_bins: int = 10) -> pd.DataFrame:
    """Participant-averaged binned curves over trial_pos_norm."""
    if df.empty or "trial_pos_norm" not in df.columns:
        return pd.DataFrame()
    tmp = df[["dataset", "participant_id", "trial_pos_norm", outcome_col]].copy()
    tmp[outcome_col] = pd.to_numeric(tmp[outcome_col], errors="coerce")
    tmp = tmp.dropna(subset=["trial_pos_norm"])
    tmp["bin"] = pd.cut(tmp["trial_pos_norm"], bins=np.linspace(0, 1, n_bins + 1), include_lowest=True, labels=False)
    # participant mean per bin
    pb = tmp.groupby(["dataset", "participant_id", "bin"], dropna=False)[outcome_col].mean().reset_index()
    # dataset mean + sem across participants
    agg = pb.groupby(["dataset", "bin"], dropna=False)[outcome_col].agg(["count", "mean", "std"]).reset_index()
    agg["sem"] = agg["std"] / np.sqrt(agg["count"].replace(0, np.nan))
    agg["bin_center"] = (agg["bin"].astype(float) + 0.5) / float(n_bins)
    agg = agg.rename(columns={"mean": "y_mean", "count": "n_participants"})
    return agg


def _normalize_pid(pid: str) -> str:
    if pid is None:
        return ""
    s = str(pid)
    # Keep digits (works for 'Participant_001', 'P1', etc.)
    d = re.sub(r"\D+", "", s)
    if d != "":
        # remove leading zeros robustly so '001' and 1 match
        try:
            return str(int(d))
        except Exception:
            dd = d.lstrip("0")
            return dd if dd != "" else "0"
    return s.strip().lower()


def _infer_id_col(df: pd.DataFrame) -> Optional[str]:
    cols = list(df.columns)
    # Prefer explicit participant/id columns
    patterns = ["participant", "subject", "subj", "id"]
    for p in patterns:
        for c in cols:
            if p in str(c).lower():
                return c
    return cols[0] if cols else None


def _load_questionnaire_csv(path: str, prefix: str) -> Optional[pd.DataFrame]:
    if path is None or (not isinstance(path, str)) or (not os.path.exists(path)):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        try:
            df = pd.read_csv(path, sep=";")
        except Exception as e:
            logger.error(f"[F] Could not read questionnaire CSV {path}: {e}")
            return None

    if df.shape[0] == 0:
        return None

    id_col = _infer_id_col(df)
    if id_col is None:
        return None

    out = df.copy()
    out["participant_key"] = out[id_col].apply(_normalize_pid)

    def _maybe_to_numeric(s: pd.Series, min_frac_numeric: float = 0.7) -> pd.Series:
        """Convert to numeric only if conversion retains enough non-missing values.

        Avoids pandas FutureWarning about errors='ignore' and prevents
        categorical columns from being wiped to NaN.
        """
        if not (s.dtype == object or pd.api.types.is_string_dtype(s)):
            return s
        sn = pd.to_numeric(s, errors="coerce")
        denom = max(int(s.notna().sum()), 1)
        frac = float(sn.notna().sum()) / float(denom)
        return sn if frac >= min_frac_numeric else s

    # Keep numeric columns and also try to coerce object columns that look numeric
    for c in out.columns:
        if c in [id_col, "participant_key"]:
            continue
        out[c] = _maybe_to_numeric(out[c])

    # Rename columns with a prefix to avoid collisions (except participant_key)
    rename = {}
    for c in out.columns:
        if c in [id_col, "participant_key"]:
            continue
        rename[c] = f"{prefix}__{c}"
    out = out.rename(columns=rename)

    # Drop duplicate participant rows by keeping the first (if any duplicates)
    out = out.drop_duplicates(subset=["participant_key"])
    return out


# ---------------------------------------------------------------------------
# Questionnaire extraction (values) + sample/demographic summary
# ---------------------------------------------------------------------------

# In the original monolithic script this map lived next to
# `_normalize_nationality`. Keep it at module scope so downstream calls work.
_NATIONALITY_MAP = {
    "Pakistani": "Pakistan",
    "Yemini": "Yemen",
    "Yemeni": "Yemen",
    "Nepalese": "Nepal",
    "Chinese": "China",
    "chinese": "China",
    " Chinese": "China",
    "Polish": "Poland",
    "Indian ": "India",
    "Indian": "India",
    "indian": "India",
    "INDIA": "India",
    "Dutch ": "Netherlands",
    "Dutch": "Netherlands",
    "Nederlandse": "Netherlands",
    "Iranian": "Iran",
    "Romanian": "Romania",
    "Spanish": "Spain",
    "Colombian": "Colombia",
    "portuguese": "Portugal",
    "Taiwanese": "Taiwan",
    "German": "Germany",
    "dutch": "Netherlands",
    "Austrian": "Austria",
    "Brazilian ": "Brazil",
    "Bulgarian": "Bulgaria",
    "bulgarian": "Bulgaria",
    "Italian": "Italy",
    "Indonesian": "Indonesia",
    "Dutch/Moroccan": "Morocco",
    "Maltese": "Malta",
    "Canadian": "Canada",
    "Portuguese": "Portugal",
    "Brazilian": "Brazil",
    "GREEK": "Greece"
}


def _normalize_nationality(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip()
    if s in _NATIONALITY_MAP:
        return _NATIONALITY_MAP[s]
    s2 = s.strip()
    if not s2:
        return np.nan
    return s2[:1].upper() + s2[1:]


def _bucket_vr_experience(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "Missing"
    s = str(x).strip().lower()
    if not s:
        return "Missing"
    if ("not" in s and "month" in s) or ("no" in s and "month" in s) or ("never" in s):
        return "Not in past month"
    if "less than once" in s or "<" in s:
        return "<1/week"
    if "week" in s or "daily" in s or "often" in s or "times" in s or "regular" in s:
        return "Regular"
    return "Other"


def _parse_seeing_aids(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "Missing"
    s = str(x).strip().lower()
    if not s:
        return "Missing"
    if "glass" in s:
        return "Glasses"
    if "contact" in s:
        return "Contact lenses"
    if s in ["no", "none", "not", "na", "n/a"]:
        return "None"
    if "yes" in s:
        return "Yes (unspecified)"
    return "Other"


def _extract_questionnaire_selected_values(
    dataset: str,
    intake_path: str,
    post_path: str,
    out_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Extract selected intake/post questionnaire values (raw), and save tables."""
    _ensure_dir(out_dir)

    long_rows = []
    missing_rows = []
    cat_rows = []
    num_rows = []

    # Intake
    intake = _read_csv_loose(intake_path)
    if intake is not None and intake.shape[0] > 0:
        id_col = _infer_id_col(intake)
        if id_col is not None:
            intake = intake.copy()
            intake["participant_key"] = intake[id_col].apply(_normalize_pid)
            intake_keep = [c for c in (["participant_key"] + INTAKE_COLUMNS_CATEGORICAL + INTAKE_COLUMNS_NUMERIC) if c in intake.columns]  # noqa: E501
            intake_wide = intake[intake_keep].drop_duplicates(subset=["participant_key"])
            intake_wide.to_csv(os.path.join(out_dir, f"intake_selected_values_wide_{dataset}.csv"), index=False)

            for col in INTAKE_COLUMNS_CATEGORICAL + INTAKE_COLUMNS_NUMERIC:
                if col not in intake.columns:
                    missing_rows.append({"dataset": dataset, "questionnaire": "intake", "question": col})
                    continue
                for _, r in intake_wide[["participant_key", col]].iterrows():
                    long_rows.append({"dataset": dataset, "questionnaire": "intake",
                                      "participant_key": r["participant_key"], "question": col, "value": r[col]})

            for col in INTAKE_COLUMNS_CATEGORICAL:
                if col not in intake.columns:
                    continue
                s = intake_wide[col].dropna().astype(str).str.strip()
                vc = s.value_counts(dropna=False)
                tot = float(vc.sum()) if vc.sum() > 0 else 0.0
                for val, cnt in vc.items():
                    cat_rows.append({"dataset": dataset, "questionnaire": "intake", "question": col, "value": val,
                                     "count": int(cnt), "pct": (float(cnt) / tot * 100.0) if tot > 0 else np.nan})

            for col in INTAKE_COLUMNS_NUMERIC:
                if col not in intake.columns:
                    continue
                x = pd.to_numeric(intake_wide[col], errors="coerce").dropna()
                num_rows.append({"dataset": dataset, "questionnaire": "intake", "question": col, "n": int(x.shape[0]),
                                 "mean": float(x.mean()) if x.shape[0] else np.nan,
                                 "sd": float(x.std(ddof=1)) if x.shape[0] > 1 else np.nan,
                                 "median": float(x.median()) if x.shape[0] else np.nan, "min": float(x.min()) if x.shape[0] else np.nan,  # noqa: E501
                                 "max": float(x.max()) if x.shape[0] else np.nan})

    # Post
    post = _read_csv_loose(post_path)
    if post is not None and post.shape[0] > 0:
        id_col = _infer_id_col(post)
        if id_col is not None:
            post = post.copy()
            post["participant_key"] = post[id_col].apply(_normalize_pid)
            post_keep = [c for c in (["participant_key"] + POST_COLUMNS_CATEGORICAL + POST_COLUMNS_NUMERIC) if c in post.columns]  # noqa: E501
            post_wide = post[post_keep].drop_duplicates(subset=["participant_key"])
            post_wide.to_csv(os.path.join(out_dir, f"post_selected_values_wide_{dataset}.csv"), index=False)

            for col in POST_COLUMNS_CATEGORICAL + POST_COLUMNS_NUMERIC:
                if col not in post.columns:
                    missing_rows.append({"dataset": dataset, "questionnaire": "post", "question": col})
                    continue
                for _, r in post_wide[["participant_key", col]].iterrows():
                    long_rows.append({"dataset": dataset, "questionnaire": "post",
                                      "participant_key": r["participant_key"], "question": col, "value": r[col]})

            for col in POST_COLUMNS_CATEGORICAL:
                if col not in post.columns:
                    continue
                s = post_wide[col].dropna().astype(str).str.strip()
                vc = s.value_counts(dropna=False)
                tot = float(vc.sum()) if vc.sum() > 0 else 0.0
                for val, cnt in vc.items():
                    cat_rows.append({"dataset": dataset, "questionnaire": "post", "question": col, "value": val,
                                     "count": int(cnt), "pct": (float(cnt) / tot * 100.0) if tot > 0 else np.nan})

            for col in POST_COLUMNS_NUMERIC:
                if col not in post.columns:
                    continue
                x = pd.to_numeric(post_wide[col], errors="coerce").dropna()
                num_rows.append({"dataset": dataset, "questionnaire": "post", "question": col, "n": int(x.shape[0]),
                                 "mean": float(x.mean()) if x.shape[0] else np.nan,
                                 "sd": float(x.std(ddof=1)) if x.shape[0] > 1 else np.nan,
                                 "median": float(x.median()) if x.shape[0] else np.nan,
                                 "min": float(x.min()) if x.shape[0] else np.nan,
                                 "max": float(x.max()) if x.shape[0] else np.nan})

    long_df = pd.DataFrame(long_rows)
    cat_dist = pd.DataFrame(cat_rows)
    num_sum = pd.DataFrame(num_rows)
    missing_df = pd.DataFrame(missing_rows)

    if long_df.shape[0] > 0:
        long_df.to_csv(os.path.join(out_dir, "questionnaire_selected_values_long.csv"), index=False)
    if cat_dist.shape[0] > 0:
        cat_dist.to_csv(os.path.join(out_dir, "questionnaire_categorical_distribution.csv"), index=False)
    if num_sum.shape[0] > 0:
        num_sum.to_csv(os.path.join(out_dir, "questionnaire_numeric_summary.csv"), index=False)
    if missing_df.shape[0] > 0:
        missing_df.to_csv(os.path.join(out_dir, "questionnaire_missing_questions_report.csv"), index=False)

    return long_df, cat_dist, num_sum


def _summarize_demographics_from_intake(dataset: str, intake_path: str, features_df: Optional[pd.DataFrame],
                                        mapping_df: Optional[pd.DataFrame], out_dir: str) -> None:
    """Print and save sample/demographics summary (if the fields exist)."""
    intake = _read_csv_loose(intake_path)
    if intake is None or intake.shape[0] == 0:
        logger.warning(f"[DEM] {dataset}: intake questionnaire not found or empty -> {intake_path}")
        return

    id_col = _infer_id_col(intake)
    if id_col is None:
        logger.warning(f"[DEM] {dataset}: could not infer participant id column in intake.")
        return

    df = intake.copy()
    df["participant_key"] = df[id_col].apply(_normalize_pid)
    df = df.drop_duplicates(subset=["participant_key"])

    col_gender = "What is your gender?"
    col_age = "What is your age (in years)?"
    col_nat = "What is your nationality?"
    col_aids = "Are you wearing any seeing aids during the experiments?"
    col_vr = "How often in the last month have you experienced virtual reality?"

    n_intake = int(df.shape[0])

    # Gender
    gender_counts = pd.DataFrame(columns=["gender_raw", "count"])
    n_no_disclose = np.nan
    if col_gender in df.columns:
        g = df[col_gender].astype(str).str.strip()
        g = g.replace({"nan": np.nan, "": np.nan})
        gender_counts = g.value_counts(dropna=False).reset_index()
        gender_counts.columns = ["gender_raw", "count"]
        mask_nd = g.isna() | g.str.lower().isin(["prefer not to say", "prefer not to disclose",
                                                 "not disclosed", "dont want to say", "don't want to say"])
        n_no_disclose = int(mask_nd.sum())

    # Age
    age_mean = np.nan
    age_sd = np.nan
    n_age = 0
    if col_age in df.columns:
        age = pd.to_numeric(df[col_age], errors="coerce").dropna()
        n_age = int(age.shape[0])
        age_mean = float(age.mean()) if n_age else np.nan
        age_sd = float(age.std(ddof=1)) if n_age > 1 else np.nan

    # Nationality
    nat_counts = pd.DataFrame(columns=["nationality", "count"])
    if col_nat in df.columns:
        nat = df[col_nat].apply(_normalize_nationality)
        nat_counts = nat.dropna().value_counts().reset_index()
        nat_counts.columns = ["nationality", "count"]

    # Seeing aids
    aids_counts = pd.DataFrame(columns=["seeing_aids_bucket", "count"])
    if col_aids in df.columns:
        aids_bucket = df[col_aids].apply(_parse_seeing_aids)
        aids_counts = aids_bucket.value_counts(dropna=False).reset_index()
        aids_counts.columns = ["seeing_aids_bucket", "count"]

    # VR experience
    vr_counts_raw = pd.DataFrame(columns=["vr_raw", "count"])
    vr_counts_bucket = pd.DataFrame(columns=["vr_bucket", "count"])
    vr_other_detail = None
    vr_missing_detail = None
    if col_vr in df.columns:
        vr_raw = df[col_vr].astype(str).str.strip()
        vr_counts_raw = vr_raw.value_counts(dropna=False).reset_index()
        vr_counts_raw.columns = ["vr_raw", "count"]
        vr_bucket = df[col_vr].apply(_bucket_vr_experience)
        vr_counts_bucket = vr_bucket.value_counts(dropna=False).reset_index()
        vr_counts_bucket.columns = ["vr_bucket", "count"]

        # Capture exact responses that ended up in "Other" (and "Missing") so you can report them.
        other_mask = vr_bucket.astype(str).str.strip().eq("Other")
        if bool(other_mask.any()):
            vr_other_detail = df.loc[other_mask, ["participant_key", col_vr]].copy()
            vr_other_detail.columns = ["participant_key", "vr_experience_raw"]

        missing_mask = vr_bucket.astype(str).str.strip().eq("Missing")
        if bool(missing_mask.any()):
            vr_missing_detail = df.loc[missing_mask, ["participant_key", col_vr]].copy()
            vr_missing_detail.columns = ["participant_key", "vr_experience_raw"]

    # Completion (from extracted features)
    n_feat_p = np.nan
    n_complete = np.nan
    expected_trials = np.nan
    completeness = None
    if features_df is not None and (not features_df.empty) and ("participant_id" in features_df.columns) and ("video_id" in features_df.columns):  # noqa: E501
        feat = features_df.copy()
        if "trial_index" in feat.columns:
            ti = pd.to_numeric(feat["trial_index"], errors="coerce")
            if ti.notna().any():
                # Main trials only: by convention indices < 2 are practice.
                feat = feat.loc[ti >= 2].copy()

        if mapping_df is not None and "video_id" in mapping_df.columns:
            expected_trials = int(feat["video_id"].nunique()) if not feat.empty else int(mapping_df["video_id"].nunique())  # noqa: E501
        else:
            expected_trials = int(feat["video_id"].nunique())

        per = feat.groupby("participant_id")["video_id"].nunique().reset_index()
        per.columns = ["participant_id", "n_trials"]
        n_feat_p = int(per.shape[0])
        n_complete = int((per["n_trials"] >= expected_trials).sum()) if not np.isnan(expected_trials) else np.nan
        completeness = per

    # Print (logs)
    logger.info(f"\n=== Sample summary (from intake + trial logs): {dataset} ===")
    logger.info(f"Participants (intake questionnaire rows): {n_intake}")
    if not np.isnan(n_feat_p):
        logger.info(f"Participants (with extracted trial logs): {int(n_feat_p)}")
        logger.info(f"Complete datasets (>= {int(expected_trials)} trials): {int(n_complete)}/{int(n_feat_p)}")

    if gender_counts.shape[0] > 0:
        top_g = ", ".join([f"{r['gender_raw']}: {int(r['count'])}" for _, r in gender_counts.iterrows()])
        logger.info(f"Gender counts: {top_g}")
        if not np.isnan(n_no_disclose):
            logger.info(f"Gender not disclosed (heuristic): {int(n_no_disclose)}")
    else:
        logger.error("Gender: (column not found)")

    if n_age > 0:
        logger.info(f"Age: mean={age_mean:.2f}, SD={age_sd:.2f} (n={n_age})")
    else:
        logger.error("Age: (column not found / no numeric values)")

    if nat_counts.shape[0] > 0:
        top_nat = ", ".join([f"{r['nationality']}: {int(r['count'])}" for _, r in nat_counts.iterrows()])
        logger.info(f"Nationalities: {top_nat}")
    else:
        logger.error("Nationality: (column not found)")

    if aids_counts.shape[0] > 0:
        top_a = ", ".join([f"{r['seeing_aids_bucket']}: {int(r['count'])}" for _, r in aids_counts.iterrows()])
        logger.info(f"Seeing aids (bucketed): {top_a}")
    else:
        logger.error("Seeing aids: (column not found)")

    if vr_counts_bucket.shape[0] > 0:
        top_v = ", ".join([f"{r['vr_bucket']}: {int(r['count'])}" for _, r in vr_counts_bucket.iterrows()])
        logger.info(f"VR experience (bucketed): {top_v}")
        if vr_other_detail is not None and vr_other_detail.shape[0] > 0:
            logger.info("VR experience -> Other (exact raw responses):")
            for _, r in vr_other_detail.iterrows():
                logger.info(f"  {r['participant_key']}: {r['vr_experience_raw']}")
        if vr_missing_detail is not None and vr_missing_detail.shape[0] > 0:
            logger.info("VR experience -> Missing (exact raw responses):")
            for _, r in vr_missing_detail.iterrows():
                logger.info(f"  {r['participant_key']}: {r['vr_experience_raw']}")
    elif vr_counts_raw.shape[0] > 0:
        top_v = ", ".join([f"{r['vr_raw']}: {int(r['count'])}" for _, r in vr_counts_raw.iterrows()])
        logger.info(f"VR experience (raw): {top_v}")
    else:
        logger.info("VR experience: (column not found)")

    # Save
    _ensure_dir(out_dir)
    summ = pd.DataFrame([{
        "dataset": dataset,
        "n_intake": n_intake,
        "n_features_participants": int(n_feat_p) if not np.isnan(n_feat_p) else np.nan,
        "expected_trials": int(expected_trials) if not np.isnan(expected_trials) else np.nan,
        "n_complete": int(n_complete) if not np.isnan(n_complete) else np.nan,
        "age_mean": age_mean,
        "age_sd": age_sd,
        "n_age": n_age,
        "gender_not_disclosed": int(n_no_disclose) if not np.isnan(n_no_disclose) else np.nan,
    }])
    summ.to_csv(os.path.join(out_dir, f"demographics_summary_{dataset}.csv"), index=False)
    if gender_counts.shape[0] > 0:
        gender_counts.to_csv(os.path.join(out_dir, f"demographics_gender_{dataset}.csv"), index=False)
    if nat_counts.shape[0] > 0:
        nat_counts.to_csv(os.path.join(out_dir, f"demographics_nationality_{dataset}.csv"), index=False)
    if aids_counts.shape[0] > 0:
        aids_counts.to_csv(os.path.join(out_dir, f"demographics_seeing_aids_{dataset}.csv"), index=False)
    if vr_counts_raw.shape[0] > 0:
        vr_counts_raw.to_csv(os.path.join(out_dir, f"demographics_vr_experience_raw_{dataset}.csv"), index=False)
    if vr_counts_bucket.shape[0] > 0:
        vr_counts_bucket.to_csv(os.path.join(out_dir, f"demographics_vr_experience_bucket_{dataset}.csv"), index=False)
    if vr_other_detail is not None and vr_other_detail.shape[0] > 0:
        vr_other_detail.to_csv(os.path.join(out_dir, f"demographics_vr_experience_other_{dataset}.csv"), index=False)
    if vr_missing_detail is not None and vr_missing_detail.shape[0] > 0:
        vr_missing_detail.to_csv(os.path.join(out_dir, f"demographics_vr_experience_missing_{dataset}.csv"),
                                 index=False)
    if completeness is not None and completeness.shape[0] > 0:
        completeness.to_csv(os.path.join(out_dir, f"participant_trial_counts_{dataset}.csv"), index=False)


def _qcol(prefix: str, raw_question: str) -> str:
    """Return the prefixed column name produced by _load_questionnaire_csv."""
    return f"{prefix}__{raw_question}"


def extract_selected_questionnaire_values(
    datasets: Dict[str, Dict[str, str]],
    output_root: str,
    intake_cols: List[str] = INTAKE_COLUMNS_TO_EXTRACT,
    post_cols: List[str] = POST_COLUMNS_TO_EXTRACT,
    intake_numeric_cols: List[str] = INTAKE_NUMERIC_COLUMNS_TO_EXTRACT,
    post_numeric_cols: List[str] = POST_NUMERIC_COLUMNS_TO_EXTRACT,
) -> Dict[str, pd.DataFrame]:
    """Extract raw participant-level values (no plots) for selected questionnaire columns.

    Writes three CSVs into output_root:
      1) questionnaire_selected_values_long.csv
      2) questionnaire_categorical_distribution.csv
      3) questionnaire_numeric_summary.csv

    Returns a dict with DataFrames: {"values_long": ..., "cat_dist": ..., "num_summary": ...}.
    """
    _ensure_dir(output_root)

    rows_long: List[pd.DataFrame] = []
    missing_rows: List[Dict[str, Any]] = []

    def _add_long(df: pd.DataFrame, ds: str, qtype: str, prefix: str, questions: List[str]) -> None:
        if df is None or df.empty:
            for q in questions:
                missing_rows.append({"dataset": ds, "questionnaire": qtype, "question": q,
                                     "reason": "file-empty-or-missing"})
            return

        for q in questions:
            c = _qcol(prefix, q)
            if c not in df.columns:
                missing_rows.append({"dataset": ds, "questionnaire": qtype, "question": q,
                                     "reason": "column-not-found"})
                continue
            tmp = pd.DataFrame({
                "dataset": ds,
                "questionnaire": qtype,
                "participant_key": df["participant_key"],
                "question": q,
                "value": df[c],
            })
            rows_long.append(tmp)

    for ds, paths in datasets.items():
        intake = _load_questionnaire_csv(paths.get("intake_questionnaire"), prefix="intake")
        post = _load_questionnaire_csv(paths.get("post_experiment_questionnaire"), prefix="post")

        _add_long(intake, ds, "intake", prefix="intake", questions=intake_cols)
        _add_long(post, ds, "post", prefix="post", questions=post_cols)
        _add_long(intake, ds, "intake", prefix="intake", questions=intake_numeric_cols)
        _add_long(post, ds, "post", prefix="post", questions=post_numeric_cols)

        # Optional: dataset-specific wide exports (handy for quick inspection)
        try:
            if intake is not None and not intake.empty:
                cols = ["participant_key"] + [c for c in [_qcol("intake", q) for q in (intake_cols + intake_numeric_cols)] if c in intake.columns]  # noqa: E501
                wide = intake[cols].copy()
                wide = wide.rename(columns={c: c.replace("intake__", "") for c in wide.columns if c.startswith("intake__")})  # noqa: E501
                wide.to_csv(os.path.join(output_root, f"intake_selected_values_wide_{ds}.csv"), index=False)
            if post is not None and not post.empty:
                cols = ["participant_key"] + [c for c in [_qcol("post", q) for q in (post_cols + post_numeric_cols)] if c in post.columns]  # noqa: E501
                wide = post[cols].copy()
                wide = wide.rename(columns={c: c.replace("post__", "") for c in wide.columns if c.startswith("post__")})  # noqa: E501
                wide.to_csv(os.path.join(output_root, f"post_selected_values_wide_{ds}.csv"), index=False)
        except Exception:
            # Non-fatal convenience export
            pass

    values_long = pd.concat(rows_long, ignore_index=True) if rows_long else pd.DataFrame(columns=["dataset",
                                                                                                  "questionnaire",
                                                                                                  "participant_key",
                                                                                                  "question", "value"])
    values_path = os.path.join(output_root, "questionnaire_selected_values_long.csv")
    values_long.to_csv(values_path, index=False)
    logger.info(f"[Q] wrote: {values_path} (rows={len(values_long)})")

    # Categorical distribution (for intake_cols + post_cols only)
    if not values_long.empty:
        is_cat = values_long["question"].isin(intake_cols + post_cols)
        cat_long = values_long.loc[is_cat].copy()
        # Stable labels, include missing
        cat_long["value_label"] = cat_long["value"].astype(str)
        cat_long.loc[cat_long["value"].isna(), "value_label"] = "(missing)"
        cat_counts = (
            cat_long
            .groupby(["dataset", "questionnaire", "question", "value_label"], dropna=False)
            .size()
            .reset_index(name="n")
        )
        totals = (
            cat_long
            .groupby(["dataset", "questionnaire", "question"], dropna=False)
            .size()
            .reset_index(name="n_total")
        )
        cat_dist = cat_counts.merge(totals, on=["dataset", "questionnaire", "question"], how="left")
        cat_dist["pct"] = (cat_dist["n"] / cat_dist["n_total"].replace(0, np.nan)).astype(float)
        cat_path = os.path.join(output_root, "questionnaire_categorical_distribution.csv")
        cat_dist.to_csv(cat_path, index=False)
        logger.info(f"[Q] wrote: {cat_path} (rows={len(cat_dist)})")
    else:
        cat_dist = pd.DataFrame(columns=["dataset", "questionnaire", "question", "value_label", "n", "n_total", "pct"])

    # Numeric summary (for intake_numeric_cols + post_numeric_cols)
    if not values_long.empty:
        is_num = values_long["question"].isin(intake_numeric_cols + post_numeric_cols)
        num_long = values_long.loc[is_num].copy()
        num_long["value_num"] = pd.to_numeric(num_long["value"], errors="coerce")
        g = num_long.groupby(["dataset", "questionnaire", "question"], dropna=False)["value_num"]
        num_summary = g.agg(
            n="count",
            mean="mean",
            sd=lambda x: float(x.std(ddof=1)) if x.dropna().shape[0] > 1 else np.nan,
            median="median",
            min="min",
            max="max",
        ).reset_index()
        num_path = os.path.join(output_root, "questionnaire_numeric_summary.csv")
        num_summary.to_csv(num_path, index=False)
        logger.info(f"[Q] wrote: {num_path} (rows={len(num_summary)})")
    else:
        num_summary = pd.DataFrame(columns=["dataset", "questionnaire", "question", "n", "mean",
                                            "sd", "median", "min", "max"])

    # Missing questions report (helps when column strings differ between versions)
    if missing_rows:
        miss_df = pd.DataFrame(missing_rows)
        miss_path = os.path.join(output_root, "questionnaire_missing_questions_report.csv")
        miss_df.to_csv(miss_path, index=False)
        logger.info(f"[Q] wrote: {miss_path} (rows={len(miss_df)})")

    return {"values_long": values_long, "cat_dist": cat_dist, "num_summary": num_summary}


def _select_moderator_columns(df: pd.DataFrame) -> List[str]:
    # pick numeric columns whose names suggest relevant constructs
    keywords = [
        "trust", "automation", "av", "vehicle", "vr", "experience", "comfort", "nausea",
        "sickness", "simulator", "presence", "confidence", "understand", "safety", "risk"
    ]
    cols = []
    for c in df.columns:
        if c in ["dataset", "participant_id", "participant_key"]:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        name = str(c).lower()
        if any(k in name for k in keywords):
            cols.append(c)
    # If nothing matched, fall back to all numeric columns (excluding IDs)
    if not cols:
        cols = [c for c in df.columns if c not in ["dataset", "participant_id",
                                                   "participant_key"] and pd.api.types.is_numeric_dtype(df[c])]
    return cols


def _gee_binomial(formula: str, df: pd.DataFrame, group_col: str = "participant_id"):
    """Fit a participant-clustered binomial GEE (exchangeable) with robust SEs."""
    if not HAVE_SM:
        raise RuntimeError("statsmodels not available")
    model = smf.gee(
        formula,
        groups=group_col,
        data=df,
        family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Exchangeable(),
    )
    return model.fit()


def _summarize_gee(res, outcome: str, model_name: str) -> pd.DataFrame:
    """Return tidy coefficient table with OR and 95% CI."""
    params = res.params
    bse = res.bse
    pvals = res.pvalues
    ci = res.conf_int()
    out = pd.DataFrame({
        "outcome": outcome,
        "model": model_name,
        "term": params.index,
        "coef": params.values,
        "se": bse.values,
        "p": pvals.values,
        "ci_lo": ci[0].values,
        "ci_hi": ci[1].values,
    })
    out["OR"] = np.exp(out["coef"])
    out["OR_lo"] = np.exp(out["ci_lo"])
    out["OR_hi"] = np.exp(out["ci_hi"])
    return out


def _latency_missingness_analysis(trial_df: pd.DataFrame, out_root: str, h: HMD_helper) -> None:
    """Analyze whether latency metrics are missing at random.

    We treat missing latency as informative behavior:
      - miss_press: no unsafe press occurred in the window
      - miss_release: no release observed (includes never-pressed)
      - miss_release_given_press: among pressed trials, no release observed

    Models are participant-clustered binomial GEEs (exchangeable), which approximate
    a random-intercept logistic mixed model while remaining stable under factor drift.
    """
    if not HAVE_SM:
        logger.warning("[Missingness] statsmodels not available; skipping missingness models.")
        return

    needed = ["dataset", "participant_id", "trial_index", "yielding", "eHMIOn", "camera", "distPed",
              "latency_first_press_s", "latency_first_release_s"]
    missing_cols = [c for c in needed if c not in trial_df.columns]
    if missing_cols:
        logger.warning(f"[Missingness] missing required columns; skipping: {missing_cols}")
        return

    d = trial_df.copy()
    d = d[d["trial_index"].notna()].copy()
    # ensure integer trial index
    d["trial_index"] = pd.to_numeric(d["trial_index"], errors="coerce")
    d = d[d["trial_index"].notna()].copy()
    d["trial_index"] = d["trial_index"].astype(int)

    # indicators
    d["miss_press"] = d["latency_first_press_s"].isna().astype(int)
    d["miss_release"] = d["latency_first_release_s"].isna().astype(int)
    d["had_press"] = d["latency_first_press_s"].notna().astype(int)
    d["miss_release_given_press"] = ((d["latency_first_release_s"].isna()) & (d["latency_first_press_s"].notna())).astype(int)  # noqa: E501

    # cast factors
    for c in ["dataset", "yielding", "eHMIOn", "camera", "distPed"]:
        d[c] = d[c].astype("category")

    # Descriptives
    desc = d.groupby("dataset").agg(
        n_trials=("participant_id", "size"),
        miss_press=("miss_press", "mean"),
        miss_release=("miss_release", "mean"),
        press_rate=("had_press", "mean"),
    ).reset_index()
    # conditional missing release (given press)
    tmp = d[d["had_press"] == 1].groupby("dataset").agg(
        miss_release_given_press=("miss_release_given_press", "mean"),
        n_pressed=("miss_release_given_press", "size"),
    ).reset_index()
    desc = desc.merge(tmp, on="dataset", how="left")

    desc_path = os.path.join(out_root, "missingness_latency_descriptives.csv")
    desc.to_csv(desc_path, index=False)

    logger.info("\n=== Missingness: latency metrics ===")
    logger.info(desc.to_string(index=False))
    logger.info(f"[Missingness] wrote descriptives -> {desc_path}")

    # Models
    # 1) Simple: dataset + trial index
    f_simple_press = "miss_press ~ C(dataset) + trial_index"
    f_simple_release = "miss_release ~ C(dataset) + trial_index"

    # 2) Adjusted: dataset interacts with factors + trial index; plus yielding×eHMI (strategy)
    f_adj_press = (
        "miss_press ~ C(dataset)*(C(yielding)+C(eHMIOn)+C(camera)+C(distPed)+trial_index) "
        "+ C(yielding)*C(eHMIOn)"
    )
    f_adj_release = (
        "miss_release ~ C(dataset)*(C(yielding)+C(eHMIOn)+C(camera)+C(distPed)+trial_index) "
        "+ C(yielding)*C(eHMIOn)"
    )

    rows = []
    try:
        r = _gee_binomial(f_simple_press, d)
        rows.append(_summarize_gee(r, "miss_press", "GEE_simple"))
    except Exception as e:
        logger.error(f"[Missingness] press simple model failed: {e}")

    try:
        r = _gee_binomial(f_adj_press, d)
        rows.append(_summarize_gee(r, "miss_press", "GEE_adjusted"))
    except Exception as e:
        logger.error(f"[Missingness] press adjusted model failed: {e}")

    try:
        r = _gee_binomial(f_simple_release, d)
        rows.append(_summarize_gee(r, "miss_release", "GEE_simple"))
    except Exception as e:
        logger.error(f"[Missingness] release simple model failed: {e}")

    try:
        r = _gee_binomial(f_adj_release, d)
        rows.append(_summarize_gee(r, "miss_release", "GEE_adjusted"))
    except Exception as e:
        logger.error(f"[Missingness] release adjusted model failed: {e}")

    # Conditional release missing (given press)
    d_press = d[d["had_press"] == 1].copy()
    f_adj_rel_cond = (
        "miss_release_given_press ~ C(dataset)*(C(yielding)+C(eHMIOn)+C(camera)+C(distPed)+trial_index) "
        "+ C(yielding)*C(eHMIOn)"
    )
    try:
        r = _gee_binomial(f_adj_rel_cond, d_press)
        rows.append(_summarize_gee(r, "miss_release_given_press", "GEE_adjusted"))
    except Exception as e:
        logger.error(f"[Missingness] conditional release model failed: {e}")

    if rows:
        out = pd.concat(rows, ignore_index=True)
        out_path = os.path.join(out_root, "missingness_latency_models.csv")
        out.to_csv(out_path, index=False)
        logger.info(f"[Missingness] wrote models -> {out_path}")

        # quick highlight of dataset-related terms
        highlight = out[(out["model"] == "GEE_adjusted") & (out["term"].str.contains("dataset"))].copy()
        if not highlight.empty:
            highlight = highlight.sort_values(["outcome", "p"]).head(20)
            logger.info("\n[Missingness] adjusted model: top dataset-related terms (by p)")
            logger.info(highlight[["outcome", "term", "OR", "OR_lo", "OR_hi", "p"]].to_string(index=False))

    # Optional plot: missingness over trial number (1-based for display)
    try:
        if go is not None:
            plot_df = d.copy()
            plot_df["trial_num"] = plot_df["trial_index"] + 1
            curve = plot_df.groupby(["dataset", "trial_num"]).agg(
                miss_press=("miss_press", "mean"),
                miss_release=("miss_release", "mean"),
            ).reset_index()

            def _plot_one(metric: str, title: str, fname: str):
                fig = go.Figure()
                for ds in ["shuffled", "unshuffled"]:
                    sub = curve[curve["dataset"] == ds]
                    fig.add_trace(go.Scatter(
                        x=sub["trial_num"], y=sub[metric],
                        mode="lines+markers",
                        name=("Randomised" if ds == "shuffled" else "Fixed-order"),
                    ))
                fig.update_layout(
                    title=title,
                    xaxis_title="Trial number",
                    yaxis_title="Missingness proportion",
                    template="plotly_white",
                )
                _save_plot(h, fig, fname)

            _plot_one("miss_press", "Missingness of first-press latency by trial number",
                      "missingness_press_over_trial")
            _plot_one("miss_release", "Missingness of first-release latency by trial number",
                      "missingness_release_over_trial")
    except Exception as e:
        logger.error(f"[Missingness] plotting failed: {e}")


def main() -> None:
    _ensure_dir(OUTPUT_ROOT)
    # ------------------------------------------------------------
    # A0) Questionnaire extraction (selected columns; values only)
    # ------------------------------------------------------------
    try:
        extract_selected_questionnaire_values(DATASETS, OUTPUT_ROOT)
    except Exception as e:
        logger.error(f"[Q] extraction failed (non-fatal): {e}")

    mapping = pd.read_csv(MAPPING_CSV)

    # Plotting helper (used throughout; must be defined before any _save_plot calls)
    h = HMD_helper()

    feature_dfs = []
    features_by_dataset = {}
    merged = None  # trigger + Q1/Q2/Q3 trial table
    part_metrics = None  # participant-level Q-behavior metrics

    for label, paths in DATASETS.items():
        out_dir = os.path.join(OUTPUT_ROOT, label)
        _ensure_dir(out_dir)

        out_csv = os.path.join(out_dir, "trigger_trial_features.csv")
        df = compute_trigger_features_dataset(
            data_folder=paths["data"],
            mapping_df=mapping,
            out_csv=out_csv,
            dataset_label=label,
            trigger_col="TriggerValueRight",
            time_col="Timestamp",
            thresholds=(0.10, 0.30, 0.50),
            analysis_window="crossing",
        )
        logger.info(f"{label}: extracted {len(df)} participant×trial rows -> {out_csv}")
        features_by_dataset[label] = df
        if not df.empty:
            feature_dfs.append(df)

    if not feature_dfs:
        logger.warning("No features extracted. Check your paths and filename matching.")
        return

    all_features = pd.concat(feature_dfs, ignore_index=True)
    all_csv = os.path.join(OUTPUT_ROOT, "trigger_trial_features_all.csv")
    all_features.to_csv(all_csv, index=False)
    logger.info(f"Wrote combined features -> {all_csv}")

    # ----------------------
    # Questionnaires: selected values (no plots) + demographics summary (printed)
    # ----------------------
    for ds_name, paths in DATASETS.items():
        ds_out = os.path.join(OUTPUT_ROOT, ds_name)
        _ensure_dir(ds_out)

        _extract_questionnaire_selected_values(
            dataset=ds_name,
            intake_path=paths.get("intake_questionnaire"),
            post_path=paths.get("post_experiment_questionnaire"),
            out_dir=ds_out,
        )

        _summarize_demographics_from_intake(
            dataset=ds_name,
            intake_path=paths.get("intake_questionnaire"),
            features_df=features_by_dataset.get(ds_name),
            mapping_df=mapping,
            out_dir=ds_out,
        )

    # ----------------------
    # Yaw/head-orientation features (optional, trial-level)
    # ----------------------
    yaw_dfs = []
    for label, paths in DATASETS.items():
        out_yaw_csv = os.path.join(OUTPUT_ROOT, label, "yaw_trial_features.csv")
        ydf = compute_yaw_features_dataset(
            data_folder=paths["data"],
            mapping_df=mapping,
            dataset_label=label,
            out_csv=out_yaw_csv,
            time_col="Timestamp",
            trigger_col="TriggerValueRight",
        )
        if ydf is not None and (not ydf.empty):
            logger.info(f"{label}: extracted {len(ydf)} yaw participant×trial rows -> {out_yaw_csv}")
            yaw_dfs.append(ydf)

    if yaw_dfs:
        yaw_all = pd.concat(yaw_dfs, ignore_index=True)
        yaw_all_csv = os.path.join(OUTPUT_ROOT, "yaw_trial_features_all.csv")
        yaw_all.to_csv(yaw_all_csv, index=False)
        logger.info(f"Wrote yaw features -> {yaw_all_csv}")

        key_cols = ["dataset", "participant_id", "video_id"]
        if all(c in all_features.columns for c in key_cols) and all(c in yaw_all.columns for c in key_cols):
            # Avoid clobbering / suffixing design columns (e.g., condition_name) that already exist in trigger features.  # noqa: E501
            overlap = [c for c in yaw_all.columns if (c in all_features.columns) and (c not in key_cols)]
            yaw_merge = yaw_all.drop(columns=overlap, errors="ignore")
            all_features = all_features.merge(yaw_merge, on=key_cols, how="left")
            all_csv2 = os.path.join(OUTPUT_ROOT, "trigger_trial_features_all_with_yaw.csv")
            all_features.to_csv(all_csv2, index=False)
            logger.info(f"Wrote combined trigger+yaw features -> {all_csv2}")
        else:
            logger.warning("[YAW] Could not merge yaw features (missing keys).")

        logger.info("\n=== Yaw ↔ trigger pooled correlations by dataset (trial-level) ===")
        yaw_abs_col = _pick_col(all_features, ["yaw_abs_mean"])
        yaw_fwd_col = _pick_col(all_features, ["yaw_forward_frac_15", "yaw_forward_frac_10"])
        trig_mean_col = _pick_col(all_features, ["trigger_mean", "avg_trigger", "mean_trigger"])
        unsafe_col = _pick_col(all_features, ["frac_time_unsafe", "unsafe_time_frac", "frac_unsafe"])
        if yaw_abs_col and trig_mean_col:
            for ds, g in all_features.groupby("dataset"):
                r = _safe_corr(g[yaw_abs_col], g[trig_mean_col], min_n=10)
                logger.info(f"[stats] pooled corr({yaw_abs_col}, {trig_mean_col}) in {ds}: r={r:.3f}" if (r is not None and not np.isnan(r)) else f"[stats] pooled corr({yaw_abs_col}, {trig_mean_col}) in {ds}: n/a")  # noqa: E501
        if yaw_fwd_col and unsafe_col:
            for ds, g in all_features.groupby("dataset"):
                r = _safe_corr(g[yaw_fwd_col], g[unsafe_col], min_n=10)
                logger.info(f"[stats] pooled corr({yaw_fwd_col}, {unsafe_col}) in {ds}: r={r:.3f}" if (r is not None and not np.isnan(r)) else f"[stats] pooled corr({yaw_fwd_col}, {unsafe_col}) in {ds}: n/a")  # noqa: E501

        # Yaw-focused reporting tables + plots
        try:
            summarize_and_plot_yaw_results(
                yaw_df=yaw_all,
                mapping_df=mapping,
                out_root=OUTPUT_ROOT,
                h=h,
            )
        except Exception as e:
            logger.error(f"[YAW] summarize/plot failed (non-fatal): {e}")
    else:
        # Fallback: if yaw features were computed in an earlier run, reuse them so yaw summaries/plots are still produced.  # noqa: E501
        yaw_all_csv = os.path.join(OUTPUT_ROOT, "yaw_trial_features_all.csv")
        if os.path.exists(yaw_all_csv):
            try:
                yaw_all = pd.read_csv(yaw_all_csv)
                logger.info(f"[YAW] Re-using existing yaw features -> {yaw_all_csv} (rows={len(yaw_all)})")

                # Try merging into the current trigger table if needed (best effort).
                key_cols = ["dataset", "participant_id", "video_id"]
                if all(c in all_features.columns for c in key_cols) and all(c in yaw_all.columns for c in key_cols):
                    overlap = [c for c in yaw_all.columns if (c in all_features.columns) and (c not in key_cols)]
                    yaw_merge = yaw_all.drop(columns=overlap, errors="ignore")
                    all_features = all_features.merge(yaw_merge, on=key_cols, how="left")
                    all_csv2 = os.path.join(OUTPUT_ROOT, "trigger_trial_features_all_with_yaw.csv")
                    all_features.to_csv(all_csv2, index=False)
                    logger.info(f"[YAW] Wrote combined trigger+yaw features -> {all_csv2}")

                try:
                    summarize_and_plot_yaw_results(
                        yaw_df=yaw_all,
                        mapping_df=mapping,
                        out_root=OUTPUT_ROOT,
                        h=h,
                    )
                except Exception as e:
                    logger.error(f"[YAW] summarize/plot failed (non-fatal): {e}")
            except Exception as e:
                logger.error(f"[YAW] Existing yaw features file found but could not be read: {yaw_all_csv} ({e})")
        else:
            logger.error("[YAW] No yaw columns found in time-series CSVs; skipping yaw features.")

    _print_dataset_overview(all_features)
    _print_metric_descriptives(all_features, PLOT_METRICS)

    # ----------------------
    # Q1–Q3 trial responses + derived participant metrics
    # ----------------------
    q_dfs = []
    for label, paths in DATASETS.items():
        qdf = load_trial_q123_from_responses(
            responses_root=paths["data"],
            dataset_label=label,
            response_col_index=2,  # Q2 is in the middle by default
        )
        if qdf is None or qdf.empty:
            logger.info(f"[Q123] {label}: no Q1/Q2/Q3 trial data found (skipping).")
        else:
            logger.info(f"[Q123] {label}: loaded {len(qdf)} Q123 trial rows")
            q_dfs.append(qdf)

    if q_dfs:
        q_all = pd.concat(q_dfs, ignore_index=True)

        merged = all_features.merge(
            q_all,
            on=["dataset", "participant_id", "video_id"],
            how="left",
        )
        merged_csv = os.path.join(OUTPUT_ROOT, "trigger_trial_features_with_Q123_all.csv")
        merged.to_csv(merged_csv, index=False)
        logger.info(f"Wrote trigger+Q trial table -> {merged_csv}")

        # Missingness mechanism analysis for latency metrics (press/release)
        _latency_missingness_analysis(merged, OUTPUT_ROOT, h)

        part_metrics = compute_participant_q_behavior_metrics(
            merged,
            group_cols=["yielding", "eHMIOn"],
        )
        part_csv = os.path.join(OUTPUT_ROOT, "participant_q_behavior_metrics.csv")
        part_metrics.to_csv(part_csv, index=False)
        logger.info(f"Wrote participant Q–behavior metrics -> {part_csv}")

        # quick descriptives for a few high-signal participant metrics
        key_pm = [c for c in [
            "Q3_within_group_sd_mean",
            "Q3_sd_late_minus_early",
            "Q3_iqr_late_minus_early",
            "dissoc_mean_abs_z_Q2_minus_unsafe",
            "z_corr_Q3_volatility",
            "z_corr_Q3_transitions",
            "z_corr_Q3_release_yielding",
        ] if c in part_metrics.columns]
        if key_pm:
            desc = part_metrics.groupby("dataset")[key_pm].agg(["count", "mean", "std"])
            logger.info("\nParticipant-metric descriptives (count/mean/std) for key metrics:")
            with pd.option_context("display.width", 160):
                logger.info(desc.to_string())

        metric_cols = [c for c in part_metrics.columns if c not in ["dataset", "participant_id"]]
        comp_part = compare_participant_metrics(part_metrics, metric_cols, fdr=True)
        comp_csv = os.path.join(OUTPUT_ROOT, "comparison_participant_q_behavior_metrics.csv")
        comp_part.to_csv(comp_csv, index=False)
        logger.info(f"Wrote participant-metric comparison -> {comp_csv}")
        _print_table(comp_part, title="Top participant-level Q–behavior differences (sorted by p/q)", max_rows=15)

        # ----------------------
        # C. Factor drift plots (design confounding fingerprint)
        # ----------------------
        # These are *not* outcome drift; they visualize how the experimental factors
        # themselves vary across trial_index in the fixed-order dataset.
        if go is not None and "trial_index" in merged.columns:
            try:
                if "yielding" in merged.columns:
                    _plot_factor_drift_by_trial_index(
                        merged,
                        factor_col="yielding",
                        title="Yielding proportion by trial index",
                        name="factor_drift_yielding_over_trial_index",
                        h=h,
                    )
                if "eHMIOn" in merged.columns:
                    _plot_factor_drift_by_trial_index(
                        merged,
                        factor_col="eHMIOn",
                        title="eHMI-on proportion by trial index",
                        name="factor_drift_eHMIOn_over_trial_index",
                        h=h,
                    )
            except Exception as e:
                logger.error(f"[plot] factor drift plotting failed: {e}")

    else:
        logger.warning("[Q123] No Q1/Q2/Q3 data found in participant response folders; skipping Q–behavior metrics.")

    # ----------------------
    # E. Learning / expectation / sequential effects
    # ----------------------
    if merged is not None and isinstance(merged, pd.DataFrame) and (not merged.empty):
        # determine an order column (if present)
        order_col = _pick_col(merged, ["trial_index", "no", "trial_no", "trial", "order"])
        merged = _make_trial_pos(merged, order_col=order_col)

        # outcomes to analyze for sequential effects
        e_outcomes = []
        trig_int = _pick_col(merged, ["trigger_mean", "trigger_auc", "trigger_avg", "mean_trigger"])
        if trig_int is not None:
            e_outcomes.append(trig_int)
        transitions = _pick_col(merged, ["n_transitions", "transitions"])
        if transitions is not None:
            e_outcomes.append(transitions)
        if "Q3" in merged.columns:
            e_outcomes.append("Q3")
        yaw_disp = _pick_col(merged, ["yaw_sd", "yaw_iqr"])
        if yaw_disp is not None:
            e_outcomes.append(yaw_disp)

        # also useful: volatility
        vol_col = _pick_col(merged, ["dtrigger_sd", "trigger_sd", "dtrigger_dt_sd"])
        if vol_col is not None:
            e_outcomes.append(vol_col)

        # make unique while preserving order
        seen = set()
        e_outcomes = [c for c in e_outcomes if (c not in seen and not seen.add(c))]

        if e_outcomes:
            logger.info("\n=== E: learning/sequential outcomes ===")
            logger.info("Using outcomes:", e_outcomes)

            # time-on-task slopes + drift + break reset
            part_learn = _participant_learning_metrics(merged, outcomes=e_outcomes)
            learn_csv = os.path.join(OUTPUT_ROOT, "participant_learning_drift_metrics.csv")
            part_learn.to_csv(learn_csv, index=False)
            logger.info(f"Wrote participant learning/drift metrics -> {learn_csv}")

            # sequential dependency (switch cost / carryover / autocorr)
            part_seq = _participant_sequential_metrics(merged, outcomes=e_outcomes)
            seq_csv = os.path.join(OUTPUT_ROOT, "participant_sequential_metrics.csv")
            part_seq.to_csv(seq_csv, index=False)
            logger.info(f"Wrote participant sequential metrics -> {seq_csv}")

            # merge into a single participant table for comparisons/plots
            part_E = part_learn.merge(part_seq, on=["dataset", "participant_id"], how="outer")
            part_E_csv = os.path.join(OUTPUT_ROOT, "participant_learning_sequential_metrics.csv")
            part_E.to_csv(part_E_csv, index=False)
            logger.info(f"Wrote combined participant E metrics -> {part_E_csv}")

            # log descriptives
            key_cols = [c for c in part_E.columns if c not in ["dataset", "participant_id"]]
            desc = part_E.groupby("dataset")[key_cols].agg(["count", "mean", "std"])
            logger.info("\nE-metric descriptives (count/mean/std):")
            with pd.option_context("display.width", 180):
                logger.info(desc.to_string())

            # compare shuffled vs unshuffled for participant E metrics
            comp_E = compare_participant_metrics(part_E, key_cols, fdr=True)
            comp_E_csv = os.path.join(OUTPUT_ROOT, "comparison_participant_learning_sequential_metrics.csv")
            comp_E.to_csv(comp_E_csv, index=False)
            logger.info(f"Wrote participant E-metric comparison -> {comp_E_csv}")
            _print_table(comp_E, title="Top participant-level Learning/Sequential differences (sorted by p/q)",
                         max_rows=15)

            # Carryover by previous-trial factors (compact report for the logs)
            carry_mask = comp_E["metric"].astype(str).str.startswith("carryover_prev_")
            comp_carry = comp_E.loc[carry_mask].copy()
            if not comp_carry.empty:
                comp_carry = comp_carry.sort_values(["q_value", "p_value"], na_position="last")
                _print_table(comp_carry, title="Participant-level carryover by previous-trial factors (sorted by p/q)",
                             max_rows=25)

            # Within-participant reliability / stability (split-half)
            # Reliability here is rank-order stability of participant means across two trial subsets.
            rel_outcomes = [c for c in ["trigger_mean", "Q3"] if c in merged.columns]
            if rel_outcomes:
                rel = _within_participant_reliability(merged, outcomes=rel_outcomes)
                if not rel.empty:
                    rel_csv = os.path.join(OUTPUT_ROOT, "within_participant_reliability.csv")
                    rel.to_csv(rel_csv, index=False)
                    logger.info(f"Wrote within-participant reliability -> {rel_csv}")
                    _print_table(rel, title="Within-participant reliability (split-half; participant means)",
                                 max_rows=50)

                    if go is not None:
                        for oc in rel_outcomes:
                            for sp in ["odd_even", "early_late"]:
                                fig = _plot_reliability_scatter(merged, outcome=oc, split=sp)
                                if fig is not None:
                                    _save_plot(h, fig, name=f"reliability_{oc}_{sp}")

            # Deeper break analysis: matched-composition pre/post comparisons
            # (last `window_k` vs first `window_k` trials around each break).
            # This reduces confounding from factor drift in the fixed-order dataset.
            break_outcomes = []
            if trig_int is not None:
                break_outcomes.append(trig_int)
            if vol_col is not None and vol_col not in break_outcomes:
                break_outcomes.append(vol_col)
            if "Q3" in merged.columns and "Q3" not in break_outcomes:
                break_outcomes.append("Q3")

            if break_outcomes:
                part_break = _participant_break_matched_metrics(
                    merged,
                    outcomes=break_outcomes,
                    window_k=5,
                    match_cols=("yielding", "eHMIOn"),
                )
                if not part_break.empty:
                    break_csv = os.path.join(OUTPUT_ROOT, "participant_break_matched_metrics.csv")
                    part_break.to_csv(break_csv, index=False)
                    logger.info(f"Wrote participant break-matched metrics -> {break_csv}")

                    break_cols = [c for c in part_break.columns if c.startswith("break_reset_matched_")]
                    comp_break = compare_participant_metrics(part_break, break_cols, fdr=True)
                    comp_break_csv = os.path.join(OUTPUT_ROOT, "comparison_participant_break_matched_metrics.csv")
                    comp_break.to_csv(comp_break_csv, index=False)
                    logger.info(f"Wrote break-matched metric comparison -> {comp_break_csv}")
                    _print_table(comp_break, title="Top participant-level break-matched differences (sorted by p/q)",
                                 max_rows=15)

                    if px is not None:
                        for mc in break_cols:
                            fig = px.violin(
                                part_break,
                                x="dataset",
                                color="dataset",
                                y=mc,
                                box=True,
                                points="all",
                                hover_data=["participant_id"],
                            )
                            short = mc.replace("break_reset_matched_", "")
                            fig.update_layout(title=f"Break-matched reset: {short}")
                            _save_plot(h, fig, name=f"compare_participant_violin_breakmatched_{short}")

            # Plots for E (participant-level)
            if px is not None:
                for col in e_outcomes:
                    slope_c = f"slope_{col}"
                    drift_c = f"drift_late_minus_early_{col}"
                    reset_c = f"post_break_reset_{col}"
                    sc_c = f"switch_cost_{col}"
                    co_y = f"carryover_prev_yielding_{col}"
                    co_e = f"carryover_prev_eHMIOn_{col}"
                    co_cam = f"carryover_prev_camera_{col}"
                    co_dist = f"carryover_prev_distPed_{col}"
                    for metric_name, mc in [
                        ("slope", slope_c),
                        ("drift", drift_c),
                        ("breakreset", reset_c),
                        ("switchcost", sc_c),
                        # keep legacy naming for prev-yielding carryover (used in the paper)
                        ("carryover", co_y),
                        ("carryover_prev_eHMIOn", co_e),
                        ("carryover_prev_camera", co_cam),
                        ("carryover_prev_distPed", co_dist),
                    ]:
                        if mc in part_E.columns:
                            fig = px.violin(
                                part_E,
                                x="dataset",
                                color="dataset",
                                y=mc,
                                box=True,
                                points="all",
                                hover_data=["participant_id"],
                            )
                            fig.update_layout(title=f"E: {metric_name} ({col}) shuffled vs unshuffled (participant-level)")  # noqa: E501
                            _save_plot(h, fig, name=f"compare_participant_violin_E_{metric_name}_{col}")

                for col in e_outcomes:
                    curve = _binned_curves(merged, outcome_col=col, n_bins=10)
                    if curve is None or curve.empty:
                        continue
                    fig = px.line(
                        curve,
                        x="bin_center",
                        y="y_mean",
                        color="dataset",
                        error_y="sem",
                    )
                    fig.update_layout(
                        title=f"E: time-on-task drift curve for {col} (binned over trial position)",
                        xaxis_title="Normalized trial position (0=start, 1=end)",
                        yaxis_title=f"Mean {col}",
                    )
                    _save_plot(h, fig, name=f"curve_time_on_task_{col}")

            if "dataset" in merged.columns and "shuffled" in merged["dataset"].unique():
                gsh = merged[merged["dataset"] == "shuffled"].copy()
                logger.info("\n[E] Shuffled-only quick checks:")
                if transitions is not None and trig_int is not None:
                    r = _safe_corr(pd.to_numeric(gsh[transitions], errors="coerce"),
                                   pd.to_numeric(gsh[trig_int], errors="coerce"), min_n=20)
                    logger.info(f"pooled corr({transitions}, {trig_int}) in shuffled: r={r:.3f}" if not np.isnan(r) else " pooled corr: n/a")  # noqa: E501
        else:
            logger.error("[E] merged table exists but none of the E outcomes were found; skipping learning/sequential effects.")  # noqa: E501
    else:
        logger.error("[E] No merged trial table available; skipping learning/sequential effects.")

    # ----------------------
    # Statistical comparison
    # ----------------------
    # Overall comparisons: use participant level aggregation to avoid treating repeated
    # trials within participant as independent observations.
    overall_trial = compare_shuffled_unshuffled(
        all_features,
        shuffled_label="shuffled",
        unshuffled_label="unshuffled",
        groupby=[],
        fdr=True,
        unit="trial",
    )
    overall_trial_csv = os.path.join(OUTPUT_ROOT, "comparison_overall_trial_level.csv")
    overall_trial.to_csv(overall_trial_csv, index=False)
    logger.info(f"Wrote overall trial level comparison -> {overall_trial_csv}")

    overall = compare_shuffled_unshuffled(
        all_features,
        shuffled_label="shuffled",
        unshuffled_label="unshuffled",
        groupby=[],
        fdr=True,
        unit="participant",
    )
    overall_csv = os.path.join(OUTPUT_ROOT, "comparison_overall.csv")
    overall.to_csv(overall_csv, index=False)
    logger.info(f"Wrote overall participant level comparison -> {overall_csv}")
    _print_top_results(overall, title='Top overall differences (sorted by p/q)')

    by_cond = compare_shuffled_unshuffled(
        all_features,
        shuffled_label="shuffled",
        unshuffled_label="unshuffled",
        groupby=["condition_name"],
        fdr=True,
    )
    by_cond_csv = os.path.join(OUTPUT_ROOT, "comparison_by_condition.csv")
    by_cond.to_csv(by_cond_csv, index=False)
    logger.info(f"Wrote condition-level comparison -> {by_cond_csv}")
    _print_top_results(by_cond, title='Top condition-level differences (sorted by p/q)')

    # Useful factor breakdown (adjust as you like)
    factor_cols = [c for c in ["yielding", "eHMIOn", "camera", "distPed"] if c in all_features.columns]
    if factor_cols:
        by_factors = compare_shuffled_unshuffled(
            all_features,
            shuffled_label="shuffled",
            unshuffled_label="unshuffled",
            groupby=factor_cols,
            fdr=True,
        )
        by_factors_csv = os.path.join(OUTPUT_ROOT, "comparison_by_factors.csv")
        by_factors.to_csv(by_factors_csv, index=False)
        logger.info(f"Wrote factor-level comparison -> {by_factors_csv}")
        _print_top_results(by_factors, title='Top factor-level differences (sorted by p/q)')

    # Use your project's preferred saving helper (saves eps/png/html)

    for metric in PLOT_METRICS:
        if metric not in all_features.columns:
            continue

        # Simple violin (dataset on x)
        fig = px.violin(
            all_features,
            x="dataset",
            color="dataset",
            y=metric,
            box=True,
            points="outliers",
            hover_data=["participant_id", "video_id", "condition_name"],
        )
        fig.update_layout(title=f"{metric}: shuffled vs unshuffled")

        # Saved to common.get_configs("output") (and optionally the final-figure folder)
        _save_plot(h, fig, name=f"compare_violin_{metric}")

    # ----------------------
    # Extra plots: Q1–Q3 and participant-level metrics
    # ----------------------
    if merged is not None and isinstance(merged, pd.DataFrame) and (not merged.empty):
        # Trial-level distributions for Q1/Q2/Q3
        hover_cols = [c for c in ["participant_id", "video_id", "condition_name",
                                  "yielding", "eHMIOn", "trial_index"] if c in merged.columns]
        for q in ["Q1", "Q2", "Q3"]:
            if q not in merged.columns:
                continue
            fig = px.violin(
                merged,
                x="dataset",
                color="dataset",
                y=q,
                box=True,
                points="outliers",
                hover_data=hover_cols,
            )
            fig.update_layout(title=f"{q}: shuffled vs unshuffled (trial-level)")
            _save_plot(h, fig, name=f"compare_violin_{q}")

        # Scatter plots for Q–behavior alignment (pooled across trials)
        vol_col = _pick_col(merged, ["dtrigger_sd", "trigger_sd", "volatility"])
        unsafe_col = _pick_col(merged, ["frac_time_unsafe", "unsafe_time_frac", "frac_unsafe"])
        trans_col = _pick_col(merged, ["n_transitions", "transitions", "num_transitions"])

        if vol_col is not None and "Q3" in merged.columns:
            # Print pooled correlations by dataset
            for ds, g in merged.groupby("dataset"):
                r = _safe_corr(g["Q3"], g[vol_col], min_n=10)
                logger.info(f"[stats] pooled corr(Q3, {vol_col}) in {ds}: r={r:.3f}" if (r is not None and not np.isnan(r)) else f"[stats] pooled corr(Q3, {vol_col}) in {ds}: n/a")  # noqa: E501
            fig = px.scatter(
                merged,
                x="Q3",
                y=vol_col,
                color="dataset",
                hover_data=hover_cols,
            )
            fig.update_layout(title=f"Q3 vs {vol_col} (trial-level, colored by dataset)")
            _save_plot(h, fig, name=f"scatter_Q3_vs_{vol_col}")

        if trans_col is not None and "Q3" in merged.columns:
            for ds, g in merged.groupby("dataset"):
                r = _safe_corr(g["Q3"], g[trans_col], min_n=10)
                logger.info(f"[stats] pooled corr(Q3, {trans_col}) in {ds}: r={r:.3f}" if (r is not None and not np.isnan(r)) else f"[stats] pooled corr(Q3, {trans_col}) in {ds}: n/a")  # noqa:E501
            fig = px.scatter(
                merged,
                x="Q3",
                y=trans_col,
                color="dataset",
                hover_data=hover_cols,
            )
            fig.update_layout(title=f"Q3 vs {trans_col} (trial-level, colored by dataset)")
            _save_plot(h, fig, name=f"scatter_Q3_vs_{trans_col}")

        if unsafe_col is not None and "Q2" in merged.columns:
            for ds, g in merged.groupby("dataset"):
                r = _safe_corr(g["Q2"], g[unsafe_col], min_n=10)
                logger.info(f"[stats] pooled corr(Q2, {unsafe_col}) in {ds}: r={r:.3f}" if (r is not None and not np.isnan(r)) else f"[stats] pooled corr(Q2, {unsafe_col}) in {ds}: n/a")  # noqa:E501
            fig = px.scatter(
                merged,
                x="Q2",
                y=unsafe_col,
                color="dataset",
                hover_data=hover_cols,
            )
            fig.update_layout(title=f"Q2 vs {unsafe_col} (trial-level, colored by dataset)")
            _save_plot(h, fig, name=f"scatter_Q2_vs_{unsafe_col}")

    if part_metrics is not None and isinstance(part_metrics, pd.DataFrame) and (not part_metrics.empty):
        # Add participant-level coupling (within-person correlations across trials), if possible
        if merged is not None and isinstance(merged, pd.DataFrame) and (not merged.empty):
            need_cols = ("participant_id" in merged.columns) and ("dataset" in merged.columns)
            if need_cols:
                vol_trial = _pick_col(merged, ["dtrigger_sd", "trigger_sd", "dtrigger_dt_sd"])
                if vol_trial is not None and "yaw_sd" in merged.columns:
                    pc = []
                    for (ds, pid), g in merged.groupby(["dataset", "participant_id"]):
                        r = _safe_corr(pd.to_numeric(g["yaw_sd"],
                                                     errors="coerce"), pd.to_numeric(g[vol_trial],
                                                                                     errors="coerce"), min_n=6)
                        pc.append({"dataset": ds, "participant_id": pid, "corr_yaw_sd_vs_volatility": r,
                                   "z_corr_yaw_sd_vs_volatility": _fisher_z(r)})
                    pc = pd.DataFrame(pc)
                    if not pc.empty:
                        part_metrics = part_metrics.merge(pc, on=["dataset", "participant_id"], how="left")

                if "yaw_forward_frac_15" in merged.columns and "Q3" in merged.columns:
                    pc2 = []
                    for (ds, pid), g in merged.groupby(["dataset", "participant_id"]):
                        r = _safe_corr(pd.to_numeric(g["yaw_forward_frac_15"], errors="coerce"),
                                       pd.to_numeric(g["Q3"], errors="coerce"), min_n=6)
                        pc2.append({"dataset": ds, "participant_id": pid, "corr_yaw_forward_vs_Q3": r,
                                    "z_corr_yaw_forward_vs_Q3": _fisher_z(r)})
                    pc2 = pd.DataFrame(pc2)
                    if not pc2.empty:
                        part_metrics = part_metrics.merge(pc2, on=["dataset", "participant_id"], how="left")

                for col in ["z_corr_yaw_sd_vs_volatility", "z_corr_yaw_forward_vs_Q3"]:
                    if col in part_metrics.columns:
                        rows = []
                        for ds, g in part_metrics.groupby("dataset"):
                            s = pd.to_numeric(g[col], errors="coerce").dropna()
                            if s.shape[0] == 0:
                                continue
                            rows.append({"metric": col, "dataset": ds, "n": int(s.shape[0]),
                                         "mean": float(s.mean()),
                                         "sd": float(s.std(ddof=1)) if s.shape[0] > 1 else float("nan")})
                        _print_table(pd.DataFrame(rows), title=f"=== Participant coupling summary: {col} ===",
                                     max_rows=10)

        # Participant-level distributions for key derived metrics
        pm_cols = [c for c in part_metrics.columns if c not in ["dataset", "participant_id"]]
        # plot a focused set first (more interpretable)
        preferred = [c for c in [
            "Q3_within_group_sd_mean",
            "Q3_within_group_sd_weighted",
            "Q3_sd_late_minus_early",
            "Q3_iqr_late_minus_early",
            "dissoc_mean_abs_z_Q2_minus_unsafe",
            "z_corr_Q3_volatility",
            "z_corr_Q3_transitions",
            "z_corr_Q3_release_yielding",
        ] if c in pm_cols]
        # then add any remaining correlation/dissociation columns (limited)
        extras = [c for c in pm_cols if c.startswith("z_corr_") and c not in preferred]
        metrics_to_plot = preferred + extras[:6]

        for m in metrics_to_plot:
            fig = px.violin(
                part_metrics,
                x="dataset",
                color="dataset",
                y=m,
                box=True,
                points="all",
                hover_data=["participant_id"],
            )
            fig.update_layout(title=f"{m}: shuffled vs unshuffled (participant-level)")
            _save_plot(h, fig, name=f"compare_participant_violin_{m}")

    # ----------------------
    # Extra plots: Yaw / head-orientation (trial-level)
    # ----------------------
    if all_features is not None and isinstance(all_features, pd.DataFrame) and (not all_features.empty):
        yaw_cols = [c for c in [
            "yaw_abs_mean",
            "yaw_forward_frac_15",
            "yaw_sd",
            "yaw_entropy",
            "yaw_speed_mean",
            "head_turn_count_15",
            "head_turn_dwell_mean_s_15",
            "yaw_speed_pre_press_mean_1s",
            "lag_turn_to_press_s_15",
        ] if c in all_features.columns]

        if yaw_cols:
            hover_cols2 = [c for c in ["participant_id", "video_id", "condition_name",
                                       "yielding", "eHMIOn", "camera", "distPed"] if c in all_features.columns]

            for m in yaw_cols:
                fig = px.violin(
                    all_features,
                    x="dataset",
                    color="dataset",
                    y=m,
                    box=True,
                    points="outliers",
                    hover_data=hover_cols2,
                )
                fig.update_layout(title=f"{m}: shuffled vs unshuffled (trial-level)")
                _save_plot(h, fig, name=f"compare_violin_{m}")

            trig_mean_col = _pick_col(all_features, ["trigger_mean", "avg_trigger", "mean_trigger"])
            unsafe_col = _pick_col(all_features, ["frac_time_unsafe", "unsafe_time_frac", "frac_unsafe"])
            ramp_col = _pick_col(all_features, ["max_ramp_rate"])
            press_lat_col = _pick_col(all_features, ["latency_first_press_s"])

            if "yaw_abs_mean" in all_features.columns and trig_mean_col is not None:
                fig = px.scatter(all_features, x="yaw_abs_mean", y=trig_mean_col,
                                 color="dataset", hover_data=hover_cols2)
                fig.update_layout(title=f"yaw_abs_mean vs {trig_mean_col} (trial-level)")
                _save_plot(h, fig, name=f"scatter_yaw_abs_mean_vs_{trig_mean_col}")

            if "yaw_forward_frac_15" in all_features.columns and unsafe_col is not None:
                fig = px.scatter(all_features, x="yaw_forward_frac_15", y=unsafe_col,
                                 color="dataset", hover_data=hover_cols2)
                fig.update_layout(title=f"yaw_forward_frac_15 vs {unsafe_col} (trial-level)")
                _save_plot(h, fig, name=f"scatter_yaw_forward_frac_15_vs_{unsafe_col}")

            if "yaw_speed_mean" in all_features.columns and ramp_col is not None:
                fig = px.scatter(all_features, x="yaw_speed_mean", y=ramp_col, color="dataset", hover_data=hover_cols2)
                fig.update_layout(title=f"yaw_speed_mean vs {ramp_col} (trial-level)")
                _save_plot(h, fig, name=f"scatter_yaw_speed_mean_vs_{ramp_col}")

            if "yaw_speed_pre_press_mean_1s" in all_features.columns and press_lat_col is not None:
                fig = px.scatter(all_features, x="yaw_speed_pre_press_mean_1s", y=press_lat_col,
                                 color="dataset", hover_data=hover_cols2)
                fig.update_layout(title=f"yaw_speed_pre_press_mean_1s vs {press_lat_col} (trial-level)")
                _save_plot(h, fig, name=f"scatter_yaw_speed_pre_press_mean_1s_vs_{press_lat_col}")

            # ----------------------
            # Cross-signal coupling (D): yaw ↔ trigger, yaw ↔ Q
            # ----------------------
            vol_col2 = _pick_col(all_features, ["dtrigger_sd", "trigger_sd", "dtrigger_dt_sd"])
            if "yaw_sd" in all_features.columns and vol_col2 is not None:
                for ds, g in all_features.groupby("dataset"):
                    r = _safe_corr(pd.to_numeric(g["yaw_sd"], errors="coerce"),
                                   pd.to_numeric(g[vol_col2], errors="coerce"), min_n=10)
                    logger.info(f"[stats] pooled corr(yaw_sd, {vol_col2}) in {ds}: r={r:.3f}" if (r is not None and not np.isnan(r)) else f"[stats] pooled corr(yaw_sd, {vol_col2}) in {ds}: n/a")  # noqa: E501
                fig = px.scatter(all_features, x="yaw_sd", y=vol_col2, color="dataset", hover_data=hover_cols2)
                fig.update_layout(title=f"Trial-level coupling: yaw_sd vs {vol_col2}")
                _save_plot(h, fig, name=f"scatter_yaw_sd_vs_{vol_col2}")

            if "yaw_forward_frac_15" in all_features.columns and "Q3" in all_features.columns:
                for ds, g in all_features.groupby("dataset"):
                    r = _safe_corr(pd.to_numeric(g["yaw_forward_frac_15"], errors="coerce"),
                                   pd.to_numeric(g["Q3"], errors="coerce"), min_n=10)
                    logger.info(f"[stats] pooled corr(yaw_forward_frac_15, Q3) in {ds}: r={r:.3f}" if (r is not None and not np.isnan(r)) else f"[stats] pooled corr(yaw_forward_frac_15, Q3) in {ds}: n/a")  # noqa: E501
                fig = px.scatter(all_features, x="yaw_forward_frac_15", y="Q3", color="dataset",
                                 hover_data=hover_cols2)
                fig.update_layout(title="Trial-level coupling: yaw_forward_frac_15 vs Q3 (understanding)")
                _save_plot(h, fig, name="scatter_yaw_forward_frac_15_vs_Q3")

            coupling_cols = [c for c in [
                "xcorr_yawspd_dtrig_max_r",
                "xcorr_yawspd_dtrig_lag_s",
                "yaw_pre_press_mean_2s",
                "yaw_pre_press_mean_2to1s",
                "yaw_around_release_mean_pm1s",
            ] if c in all_features.columns]

            if coupling_cols:
                rows = []
                for c in coupling_cols:
                    for ds, g in all_features.groupby("dataset"):
                        s = pd.to_numeric(g[c], errors="coerce").dropna()
                        if s.shape[0] == 0:
                            continue
                        rows.append({
                            "metric": c,
                            "dataset": ds,
                            "n": int(s.shape[0]),
                            "mean": float(s.mean()),
                            "median": float(s.median()),
                            "sd": float(s.std(ddof=1)) if s.shape[0] > 1 else float("nan"),
                        })
                _print_table(pd.DataFrame(rows), title="=== Coupling metrics summary (trial-level) ===", max_rows=30)

                for c in coupling_cols:
                    fig = px.violin(
                        all_features,
                        x="dataset",
                        color="dataset",
                        y=c,
                        box=True,
                        points="outliers",
                        hover_data=hover_cols2,
                    )
                    fig.update_layout(title=f"{c}: shuffled vs unshuffled (within-trial coupling)")
                    _save_plot(h, fig, name=f"compare_violin_{c}")

    # -----------------------------------------------------------------------
    # F1) Individual differences (questionnaires) as moderators
    # -----------------------------------------------------------------------
    try:
        # Build participant-level outcomes from trial table (all_features has trigger+yaw and mapping cols;
        # merged additionally has Q1/Q2/Q3 when available)
        base_trials = all_features.copy()
        if merged is not None and isinstance(merged, pd.DataFrame) and (not merged.empty):
            # merged contains Q1/Q2/Q3; prefer it for Q3 outcomes
            base_trials = merged.copy()

        # Key outcomes to moderate (participant-level)
        outcome_defs = {}
        # earlier release in yielding
        if "latency_first_release_s" in base_trials.columns and "yielding" in base_trials.columns:
            outcome_defs["release_latency_yielding_mean"] = ("latency_first_release_s", {"yielding": 1})
        # yaw volatility proxy
        yaw_vol_col = _pick_col(base_trials, ["yaw_speed_p95", "yaw_speed_mean", "yaw_sd", "yaw_iqr"])
        if yaw_vol_col is not None:
            outcome_defs["yaw_volatility_mean"] = (yaw_vol_col, None)
        # trigger volatility
        trig_vol_col = _pick_col(base_trials, ["dtrigger_sd", "trigger_sd"])
        if trig_vol_col is not None:
            outcome_defs["trigger_volatility_mean"] = (trig_vol_col, None)
        # understanding (Q3)
        if "Q3" in base_trials.columns:
            outcome_defs["Q3_mean"] = ("Q3", None)

        # Aggregate outcomes per participant
        part_out_rows = []
        if outcome_defs:
            for (ds, pid), g in base_trials.groupby(["dataset", "participant_id"]):
                row = {"dataset": ds, "participant_id": pid, "participant_key": _normalize_pid(pid)}
                for out_name, (col, filt) in outcome_defs.items():
                    gg = g
                    if isinstance(filt, dict):
                        for k, v in filt.items():
                            if k in gg.columns:
                                gg = gg.loc[gg[k] == v]
                    s = pd.to_numeric(gg[col], errors="coerce")
                    row[out_name] = float(s.mean()) if s.dropna().shape[0] > 0 else np.nan
                part_out_rows.append(row)

        part_out = pd.DataFrame(part_out_rows)
        if part_out.shape[0] == 0:
            raise RuntimeError("No participant outcomes computed; skipping questionnaire moderation.")

        # Load questionnaires per dataset and merge
        q_merged_all = []
        for ds_name, paths in DATASETS.items():
            intake = _load_questionnaire_csv(paths.get("intake_questionnaire"), prefix="intake")
            post = _load_questionnaire_csv(paths.get("post_experiment_questionnaire"), prefix="post")
            if intake is None and post is None:
                continue
            qdf = None
            if intake is not None and post is not None:
                qdf = intake.merge(post, on="participant_key", how="outer")
            else:
                qdf = intake if intake is not None else post
            qdf["dataset"] = ds_name
            q_merged_all.append(qdf)

        if not q_merged_all:
            raise RuntimeError("No questionnaire files loaded; skipping moderation.")

        q_all = pd.concat(q_merged_all, ignore_index=True)
        # Merge with participant outcomes
        mod_df = part_out.merge(q_all, on=["dataset", "participant_key"], how="left")

        # Select moderators
        moderator_cols = _select_moderator_columns(mod_df)
        outcome_cols = [c for c in part_out.columns if c not in ["dataset", "participant_id", "participant_key"]]

        # Run moderation stats
        mod_results = []
        logger.info("\n=== F1 Moderation (questionnaires) ===")
        logger.info(f"[F1] moderators: {len(moderator_cols)} | outcomes: {outcome_cols}")
        for ds, g in mod_df.groupby("dataset"):
            logger.info(f"[F1] dataset={ds}: n_participants_with_outcomes={g['participant_id'].nunique()}")
            for mcol in moderator_cols:
                for ocol in outcome_cols:
                    r, p, n = _corr_with_p(g[mcol], g[ocol], min_n=6)
                    # median split (within dataset) for easy interpretability
                    mm = pd.to_numeric(g[mcol], errors="coerce")
                    med = float(mm.median()) if mm.dropna().shape[0] > 0 else np.nan
                    if np.isnan(med):
                        continue
                    group = np.where(mm >= med, "high", "low")
                    gg = g.copy()
                    gg["_group"] = group
                    hi = pd.to_numeric(gg.loc[gg["_group"] == "high", ocol], errors="coerce").dropna()
                    lo = pd.to_numeric(gg.loc[gg["_group"] == "low", ocol], errors="coerce").dropna()
                    tstat = np.nan
                    pt = np.nan
                    tname = ""
                    tnote = ""
                    if hi.shape[0] >= 2 and lo.shape[0] >= 2:
                        try:
                            tstat, pt, tname, tnote = _safe_two_sample_test(hi, lo, equal_var=False)
                            tstat = float(tstat) if tstat is not None and np.isfinite(tstat) else np.nan
                            pt = float(pt) if pt is not None and np.isfinite(pt) else np.nan
                        except Exception:
                            pass

                    mod_results.append({
                        "dataset": ds,
                        "moderator": mcol,
                        "outcome": ocol,
                        "n": int(n),
                        "r": float(r) if r is not None else np.nan,
                        "p_corr": float(p) if p is not None else np.nan,
                        "t_high_vs_low": tstat,
                        "p_high_vs_low": pt,
                        "test_high_vs_low": tname,
                        "note_high_vs_low": tnote,
                        "median_split": med,
                        "mean_high": float(hi.mean()) if hi.shape[0] > 0 else np.nan,
                        "mean_low": float(lo.mean()) if lo.shape[0] > 0 else np.nan,
                    })

        mod_res = pd.DataFrame(mod_results)
        if mod_res.shape[0] > 0:
            mod_path = os.path.join(OUTPUT_ROOT, "moderation_questionnaire_results.csv")
            mod_res.to_csv(mod_path, index=False)
            logger.info(f"[F1] wrote: {mod_path}")

            # Log: top correlations (abs r) per outcome
            for ocol in outcome_cols:
                tmp = mod_res.loc[mod_res["outcome"] == ocol].copy()
                tmp["abs_r"] = tmp["r"].abs()
                tmp = tmp.sort_values(["abs_r", "p_corr"], ascending=[False, True]).head(10)
                _print_table(tmp[["dataset", "moderator", "outcome", "n", "r", "p_corr",
                                  "mean_high", "mean_low", "p_high_vs_low"]],
                             title=f"=== F1 top questionnaire links for outcome: {ocol} ===", max_rows=10)

            # Plots: scatter moderator vs outcome (focus on best few per outcome)
            if px is not None:
                hoverP = ["participant_id"]
                for ocol in outcome_cols:
                    tmp = mod_res.loc[mod_res["outcome"] == ocol].copy()
                    tmp["abs_r"] = tmp["r"].abs()
                    best = tmp.sort_values("abs_r", ascending=False).head(2)["moderator"].tolist()
                    for mcol in best:
                        if mcol not in mod_df.columns:
                            continue
                        fig = px.scatter(
                            mod_df,
                            x=mcol,
                            y=ocol,
                            color="dataset",
                            hover_data=hoverP,
                        )
                        fig.update_layout(title=f"Moderator: {mcol} vs {ocol}")
                        _save_plot(h, fig, name=f"F1_scatter_{mcol}_vs_{ocol}")

                # Violin: high/low groups for a few interpretable pairings if they exist
                # These keyword picks match your examples.
                key_pairs = []
                for mcol in moderator_cols:
                    mn = str(mcol).lower()
                    if ("trust" in mn or "automation" in mn) and "release_latency_yielding_mean" in outcome_cols:
                        key_pairs.append((mcol, "release_latency_yielding_mean"))
                    if ("vr" in mn or "experience" in mn) and "yaw_volatility_mean" in outcome_cols:
                        key_pairs.append((mcol, "yaw_volatility_mean"))
                key_pairs = key_pairs[:4]

                for mcol, ocol in key_pairs:
                    gdf = mod_df.copy()
                    mm = pd.to_numeric(gdf[mcol], errors="coerce")
                    med = float(mm.median()) if mm.dropna().shape[0] > 0 else np.nan
                    if np.isnan(med):
                        continue
                    gdf["group"] = np.where(mm >= med, "high", "low")
                    fig = px.violin(
                        gdf,
                        x="group",
                        y=ocol,
                        color="dataset",
                        box=True,
                        points="all",
                        hover_data=["participant_id"],
                    )
                    fig.update_layout(title=f"{ocol} by {mcol} median split (high/low)")
                    _save_plot(h, fig, name=f"F1_violin_{ocol}_by_{mcol}_median_split")

    except Exception as e:
        logger.error(f"[F1] skipped questionnaire moderation: {e}")

    # -----------------------------------------------------------------------
    # F2) Condition discriminability (signal detection framing): yielding vs not
    # -----------------------------------------------------------------------
    try:
        dfD = all_features.copy()
        if merged is not None and isinstance(merged, pd.DataFrame) and (not merged.empty):
            # If Q3 exists, merged has it; use merged so Q3 can be tested as a signal too
            dfD = merged.copy()

        if "yielding" not in dfD.columns:
            raise RuntimeError("No 'yielding' column found (from mapping).")

        # Candidate signals
        candidates = [
            "trigger_mean", "frac_time_unsafe", "n_transitions", "dtrigger_sd", "trigger_sd",
            "yaw_abs_mean", "yaw_forward_frac_15", "yaw_sd", "yaw_entropy", "yaw_speed_mean", "yaw_speed_p95",
            "Q3"
        ]
        signals = [c for c in candidates if c in dfD.columns]
        if not signals:
            raise RuntimeError("No discriminability signals found in trial table.")

        auc_rows = []
        roc_curves = {}  # (dataset, signal) -> (fpr,tpr)
        logger.info("\n=== F2 Discriminability (yielding vs not) ===")
        for ds, g in dfD.groupby("dataset"):
            y = pd.to_numeric(g["yielding"],  # noqa:F841
                              errors="coerce").dropna().astype(int).values
            # use same mask per signal inside loop
            for sig in signals:
                s = pd.to_numeric(g[sig], errors="coerce").values
                # align mask
                yy = pd.to_numeric(g["yielding"], errors="coerce").values
                mask = (~np.isnan(yy)) & (~np.isnan(s))
                if mask.sum() < 10:
                    continue
                yv = yy[mask].astype(int)
                sv = s[mask].astype(float)
                fpr, tpr, auc = _roc_curve_and_auc(yv, sv)
                if np.isnan(auc):
                    continue
                auc_rows.append({"dataset": ds, "signal": sig, "n": int(mask.sum()), "auc": float(auc)})
                if fpr is not None and tpr is not None:
                    roc_curves[(ds, sig)] = (fpr, tpr, auc)

            # Log top AUC signals for dataset
            tmp = pd.DataFrame([r for r in auc_rows if r["dataset"] == ds]).sort_values("auc", ascending=False).head(8)
            _print_table(tmp, title=f"=== F2 top AUC signals in {ds} ===", max_rows=8)

        auc_df = pd.DataFrame(auc_rows)
        if auc_df.shape[0] == 0:
            raise RuntimeError("No AUCs computed (maybe missing classes or too many NaNs).")

        auc_path = os.path.join(OUTPUT_ROOT, "discriminability_yielding_auc.csv")
        auc_df.to_csv(auc_path, index=False)
        logger.info(f"[F2] wrote: {auc_path}")

        # Compare AUC shuffled vs unshuffled (delta)
        if set(auc_df["dataset"].unique()) >= set(["shuffled", "unshuffled"]):
            piv = auc_df.pivot_table(index="signal", columns="dataset", values="auc", aggfunc="mean")
            piv = piv.reset_index()
            if "shuffled" in piv.columns and "unshuffled" in piv.columns:
                piv["delta_shuffled_minus_unshuffled"] = piv["shuffled"] - piv["unshuffled"]
                piv = piv.sort_values("delta_shuffled_minus_unshuffled", ascending=False)
                delta_path = os.path.join(OUTPUT_ROOT, "discriminability_yielding_auc_delta.csv")
                piv.to_csv(delta_path, index=False)
                _print_table(piv.head(15), title="=== F2 AUC delta (shuffled - unshuffled) ===", max_rows=15)
                logger.info(f"[F2] wrote: {delta_path}")

        # Plots
        if px is not None:
            # Bar plot: AUC by signal and dataset
            fig = px.bar(
                auc_df,
                x="signal",
                y="auc",
                color="dataset",
                barmode="group",
                hover_data=["n"],
            )
            fig.update_layout(title="Yielding discriminability (ROC AUC) by signal")
            _save_plot(h, fig, name="F2_bar_auc_by_signal")

            # ROC curves for top 3 signals (pooled across datasets by mean AUC)
            top_signals = (
                auc_df.groupby("signal")["auc"].mean().sort_values(ascending=False).head(3).index.tolist()
            )
            # Build a long dataframe for curves
            curve_rows = []
            for (ds, sig), (fpr, tpr, auc) in roc_curves.items():
                if sig not in top_signals:
                    continue
                for a, b in zip(fpr, tpr):
                    curve_rows.append({"dataset": ds, "signal": sig, "fpr": float(a),
                                       "tpr": float(b), "auc": float(auc)})
            if curve_rows:
                cdf = pd.DataFrame(curve_rows)
                fig = px.line(
                    cdf,
                    x="fpr",
                    y="tpr",
                    color="dataset",
                    line_dash="signal",
                    hover_data=["auc"],
                )
                fig.update_layout(title="ROC curves (top signals), yielding vs not",
                                  xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
                _save_plot(h, fig, name="F2_roc_curves_top_signals")

    except Exception as e:
        logger.error(f"[F2] skipped discriminability analysis: {e}")


# ---------------------------------------------------------------------------
# Class-based pipeline wrapper
# ---------------------------------------------------------------------------

class ComparisonPipeline:
    """Small OO wrapper around `main()`.

    This does *not* change any computation; it only wires runtime configuration
    (DATASETS/MAPPING_CSV/OUTPUT_ROOT) before invoking the original pipeline.

    You can configure it in two equivalent ways:

    1) Pass a full `datasets` dict (backwards compatible):

        datasets={
            "shuffled": {"data": "...", "intake_questionnaire": "...", "post_experiment_questionnaire": "..."},
            "unshuffled": {"data": "...", "intake_questionnaire": "...", "post_experiment_questionnaire": "..."},
        }

    2) Pass the six explicit paths (recommended for manual editing):

        shuffled_data=..., shuffled_intake_questionnaire=..., shuffled_post_experiment_questionnaire=...,
        unshuffled_data=..., unshuffled_intake_questionnaire=..., unshuffled_post_experiment_questionnaire=...
    """

    def __init__(
        self,
        datasets: Optional[Dict[str, Dict[str, str]]] = None,
        *,
        shuffled_data: Optional[str] = None,
        shuffled_intake_questionnaire: Optional[str] = None,
        shuffled_post_experiment_questionnaire: Optional[str] = None,
        unshuffled_data: Optional[str] = None,
        unshuffled_intake_questionnaire: Optional[str] = None,
        unshuffled_post_experiment_questionnaire: Optional[str] = None,
        mapping_csv: str = "mapping.csv",
        output_root: str = "_compare_output",
    ) -> None:
        if datasets is None:
            missing = [
                name
                for name, val in [
                    ("shuffled_data", shuffled_data),
                    ("shuffled_intake_questionnaire", shuffled_intake_questionnaire),
                    ("shuffled_post_experiment_questionnaire", shuffled_post_experiment_questionnaire),
                    ("unshuffled_data", unshuffled_data),
                    ("unshuffled_intake_questionnaire", unshuffled_intake_questionnaire),
                    ("unshuffled_post_experiment_questionnaire", unshuffled_post_experiment_questionnaire),
                ]
                if not val
            ]
            if missing:
                raise ValueError(
                    "ComparisonPipeline requires either `datasets=...` or all six explicit path arguments. "
                    f"Missing: {', '.join(missing)}"
                )
            datasets = {
                "shuffled": {
                    "data": str(shuffled_data),
                    "intake_questionnaire": str(shuffled_intake_questionnaire),
                    "post_experiment_questionnaire": str(shuffled_post_experiment_questionnaire),
                },
                "unshuffled": {
                    "data": str(unshuffled_data),
                    "intake_questionnaire": str(unshuffled_intake_questionnaire),
                    "post_experiment_questionnaire": str(unshuffled_post_experiment_questionnaire),
                },
            }

        self.datasets = datasets
        self.mapping_csv = mapping_csv
        self.output_root = output_root

    def run(self) -> None:
        global DATASETS, MAPPING_CSV, OUTPUT_ROOT
        DATASETS = self.datasets
        MAPPING_CSV = self.mapping_csv
        OUTPUT_ROOT = self.output_root
        main()
