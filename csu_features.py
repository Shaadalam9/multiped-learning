"""Feature extraction for the shuffled-vs-unshuffled comparison pipeline.

This module contains the per-trial extraction code:
- Q1/Q2/Q3 response extraction and participant-level Q-behavior metrics
- Trigger-signal features (press/release latency, unsafe bouts, etc.)
- Yaw/head-orientation features (optional; derived from yaw or quaternion columns)

All helper utilities (stats, plot saving, CSV probing) come from `csu_core`.
"""

from __future__ import annotations

import os
import glob
import re
import ast
from typing import Optional, List, Dict, Tuple, Any

import numpy as np
from numpy.typing import NDArray
import pandas as pd

import plotly.express as px  # noqa:F401
import plotly.graph_objects as go  # noqa:F401

from custom_logger import CustomLogger
from helper import HMD_helper

try:
    from utils.HMD_helper import HMD_yaw  # type: ignore
except Exception:  # pragma: no cover
    from HMD_helper import HMD_yaw  # type: ignore

# Import helpers (kept as names to avoid editing the original function bodies)
from csu_core import (
    _choose_trial_file,
    _fisher_z,
    _np_trapezoid,
    _pick_col,
    _read_csv_flexible,
    _safe_corr,
    _save_plot,
    _xcorr_max_r_lag,
    _zscore,
)

# Shared yaw or quaternion column heuristics.
# Kept in a standalone module so both `csu_features` and `csu_core` can use them
# without creating circular imports.
from csu_yaw_constants import _YAW_CANDIDATES, _QUAT_REGEX, _QUAT_LIST_COL_PAT

_HMD_YAW = HMD_yaw() if HMD_yaw is not None else None
logger = CustomLogger(__name__)  # use custom logger


# ---------------------------------------------------------------------------
# Yaw / head-orientation features (trial-level)
# ---------------------------------------------------------------------------


def load_trial_q123_from_responses(responses_root: str, dataset_label: str,
                                   response_col_index: int = 2) -> pd.DataFrame:
    """
    Read per-trial Q1/Q2/Q3 ratings from Participant_* response CSVs.

    Structure:
    - participant folders: <responses_root>/Participant_<id>/
    - response files: Participant_<id>_*.csv (no strict header requirement)
    - columns: col0=video_id, and Q1/Q2/Q3 adjacent (Q2 at response_col_index)
    """
    q1_idx = response_col_index - 1
    q2_idx = response_col_index
    q3_idx = response_col_index + 1

    records = []
    if not os.path.isdir(responses_root):
        logger.warning(f"[Q123] responses_root not found: {responses_root}")
        return pd.DataFrame()

    part_dirs = [
        d for d in os.listdir(responses_root)
        if os.path.isdir(os.path.join(responses_root, d)) and d.startswith("Participant_")
    ]
    part_dirs = sorted(part_dirs)

    for d in part_dirs:
        try:
            pid = int(d.split("_")[1])
        except Exception:
            continue

        folder = os.path.join(responses_root, d)
        files = sorted(glob.glob(os.path.join(folder, f"{d}_*.csv")))
        if not files:
            continue

        trial_idx = 0
        seen = set()
        for fp in files:
            df = None
            for sep in [",", ";"]:
                try:
                    df = pd.read_csv(fp, header=None, sep=sep)
                    break
                except Exception:
                    df = None
            if df is None:
                continue

            if df.shape[1] <= q3_idx:
                continue

            tmp = df[[0, q1_idx, q2_idx, q3_idx]].copy()
            tmp.columns = ["video_id", "Q1", "Q2", "Q3"]
            tmp["video_id"] = tmp["video_id"].astype(str)

            tmp = tmp[tmp["video_id"].str.startswith("video_")].copy()
            if tmp.empty:
                continue

            for c in ["Q1", "Q2", "Q3"]:
                tmp[c] = pd.to_numeric(tmp[c], errors="coerce")

            tmp["trial_index"] = np.arange(trial_idx, trial_idx + len(tmp), dtype=int)
            trial_idx += len(tmp)

            tmp["participant_id"] = pid
            tmp["dataset"] = dataset_label

            keep_rows = []
            for _, row in tmp.iterrows():
                vid = row["video_id"]
                key = (pid, dataset_label, vid)
                if key in seen:
                    continue
                seen.add(key)
                keep_rows.append(row)
            if keep_rows:
                records.append(pd.DataFrame(keep_rows))

    if not records:
        return pd.DataFrame()

    return pd.concat(records, ignore_index=True)


def compute_participant_q_behavior_metrics(
    trial_df: pd.DataFrame,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Participant-level metrics:
    - within-person Q3 consistency across repeated (yielding,eHMI) conditions
    - scale-use drift over time (early vs late SD/IQR)
    - Q–behavior dissociation (Q2 vs trigger measures)
    - metacognitive alignment (Q3 vs volatility/transitions/release timing)

    Returns one row per (dataset, participant_id).
    """
    if group_cols is None:
        group_cols = ["yielding", "eHMIOn"]

    trig_mean_col = _pick_col(trial_df, ["trigger_mean", "avg_trigger", "mean_trigger"])
    unsafe_col = _pick_col(trial_df, ["frac_time_unsafe", "unsafe_time_frac", "frac_unsafe"])
    vol_col = _pick_col(trial_df, ["dtrigger_sd", "trigger_sd", "volatility"])
    trans_col = _pick_col(trial_df, ["n_transitions", "transitions", "num_transitions"])
    release_col = _pick_col(trial_df, ["latency_first_release_s", "press_release_hysteresis",
                                       "time_to_yield_stop_release_s", "time_to_yield_start_release_s"])

    out_rows = []
    for (dataset, pid), g in trial_df.groupby(["dataset", "participant_id"]):
        rec = {"dataset": dataset, "participant_id": pid}

        rec["n_trials"] = int(len(g))
        rec["n_Q3"] = int(g["Q3"].notna().sum()) if "Q3" in g.columns else 0

        if all(c in g.columns for c in group_cols) and "Q3" in g.columns:
            sds = []
            weights = []
            for _, gg in g.dropna(subset=["Q3"]).groupby(group_cols):
                n = len(gg)
                if n >= 2:
                    sd = float(gg["Q3"].std())
                    sds.append(sd)
                    weights.append(n)
            rec["Q3_within_group_sd_mean"] = float(np.mean(sds)) if sds else np.nan
            rec["Q3_within_group_sd_weighted"] = (
                float(np.sum(np.array(sds) * np.array(weights)) / np.sum(weights))
                if sds else np.nan
            )

        if "trial_index" in g.columns and "Q3" in g.columns:
            g2 = g.sort_values("trial_index")
            mid = int(len(g2) / 2)
            early = g2.iloc[:mid]
            late = g2.iloc[mid:]

            rec["Q3_sd_early"] = float(early["Q3"].std()) if early["Q3"].notna().sum() >= 2 else np.nan
            rec["Q3_sd_late"] = float(late["Q3"].std()) if late["Q3"].notna().sum() >= 2 else np.nan
            rec["Q3_sd_late_minus_early"] = (
                rec["Q3_sd_late"] - rec["Q3_sd_early"]
                if (not np.isnan(rec["Q3_sd_late"]) and not np.isnan(rec["Q3_sd_early"]))
                else np.nan
            )

            def _iqr(s: pd.Series) -> float:
                s = s.dropna()
                if len(s) < 2:
                    return np.nan
                return float(s.quantile(0.75) - s.quantile(0.25))

            rec["Q3_iqr_early"] = _iqr(early["Q3"])
            rec["Q3_iqr_late"] = _iqr(late["Q3"])
            rec["Q3_iqr_late_minus_early"] = (
                rec["Q3_iqr_late"] - rec["Q3_iqr_early"]
                if (not np.isnan(rec["Q3_iqr_late"]) and not np.isnan(rec["Q3_iqr_early"]))
                else np.nan
            )

        if "Q2" in g.columns and unsafe_col is not None:
            rec["corr_Q2_unsafe"] = _safe_corr(g["Q2"], g[unsafe_col])
            rec["z_corr_Q2_unsafe"] = _fisher_z(rec["corr_Q2_unsafe"])

            zQ2 = _zscore(g["Q2"])
            zUnsafe = _zscore(g[unsafe_col])
            diff = zQ2 - zUnsafe
            rec["dissoc_mean_z_Q2_minus_unsafe"] = float(diff.mean()) if diff.notna().any() else np.nan
            rec["dissoc_mean_abs_z_Q2_minus_unsafe"] = float(diff.abs().mean()) if diff.notna().any() else np.nan

        if "Q2" in g.columns and trig_mean_col is not None:
            rec["corr_Q2_trigger_mean"] = _safe_corr(g["Q2"], g[trig_mean_col])
            rec["z_corr_Q2_trigger_mean"] = _fisher_z(rec["corr_Q2_trigger_mean"])

        if "Q3" in g.columns and vol_col is not None:
            rec["corr_Q3_volatility"] = _safe_corr(g["Q3"], g[vol_col])
            rec["z_corr_Q3_volatility"] = _fisher_z(rec["corr_Q3_volatility"])

        if "Q3" in g.columns and trans_col is not None:
            rec["corr_Q3_transitions"] = _safe_corr(g["Q3"], g[trans_col])
            rec["z_corr_Q3_transitions"] = _fisher_z(rec["corr_Q3_transitions"])

        if "Q3" in g.columns and release_col is not None and "yielding" in g.columns:
            gy = g[g["yielding"] == 1].copy()
            rec["corr_Q3_release_yielding"] = _safe_corr(gy["Q3"], gy[release_col])
            rec["z_corr_Q3_release_yielding"] = _fisher_z(rec["corr_Q3_release_yielding"])

        out_rows.append(rec)

    return pd.DataFrame(out_rows)


def _compute_yaw_from_quaternion_columns(raw_df: pd.DataFrame) -> Optional[np.ndarray]:
    """Compute yaw (degrees) from quaternion columns in a time-series dataframe.

    Supports either:
      1) four separate columns for w/x/y/z
      2) a single column containing a string-encoded list/tuple [w,x,y,z]

    Returns
    -------
    yaw_deg : np.ndarray or None
        Yaw in degrees (wrapped to [-180, 180]).
    """
    if raw_df is None or raw_df.empty:
        return None
    if _HMD_YAW is None:
        return None

    cols = list(raw_df.columns)
    # Strong preference: use the exact HMD quaternion columns from the prior pipeline when present.
    # This avoids accidentally picking other rotation quaternions (e.g., vehicle/body) that match the regex.
    preferred = ["HMDRotationW", "HMDRotationX", "HMDRotationY", "HMDRotationZ"]
    if all(c in raw_df.columns for c in preferred):
        w = pd.to_numeric(raw_df["HMDRotationW"], errors="coerce").to_numpy(dtype=float)
        x = pd.to_numeric(raw_df["HMDRotationX"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(raw_df["HMDRotationY"], errors="coerce").to_numpy(dtype=float)
        z = pd.to_numeric(raw_df["HMDRotationZ"], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(w) & np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        if ok.sum() >= 5:
            yaw = np.full(w.shape, np.nan, dtype=float)
            for i in np.where(ok)[0]:
                try:
                    yaw[i] = _HMD_YAW.quaternion_to_euler(w[i], x[i], y[i], z[i])[2]
                except Exception:
                    yaw[i] = np.nan
            yaw_deg = np.degrees(yaw)
            yaw_deg = (yaw_deg + 180.0) % 360.0 - 180.0
            return yaw_deg

    # 1) Try to find separate component columns
    comp_cols = {}

    def _quat_col_rank(col_name: str) -> tuple:
        s = str(col_name)
        sl = s.lower()
        score = 0
        # Prefer explicit head/HMD orientation columns.
        if "hmd" in sl or "head" in sl:
            score += 10
        # Slight preference for rotation/orientation naming.
        if "rotation" in sl or "orient" in sl or "quat" in sl:
            score += 3
        # De-prioritize obvious non-head rotations if they exist.
        if "car" in sl or "vehicle" in sl or "ped" in sl or "body" in sl:
            score -= 5
        # Sort: highest score first, then shortest name.
        return (-score, len(s), s)

    for comp in ["w", "x", "y", "z"]:
        cands = [c for c in cols if _QUAT_REGEX[comp].search(str(c))]
        if not cands:
            continue
        cands = sorted(cands, key=_quat_col_rank)
        comp_cols[comp] = cands[0]
    if len(comp_cols) == 4:
        w = pd.to_numeric(raw_df[comp_cols["w"]], errors="coerce").to_numpy(dtype=float)
        x = pd.to_numeric(raw_df[comp_cols["x"]], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(raw_df[comp_cols["y"]], errors="coerce").to_numpy(dtype=float)
        z = pd.to_numeric(raw_df[comp_cols["z"]], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(w) & np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        if ok.sum() < 5:
            return None
        yaw = np.full(w.shape, np.nan, dtype=float)
        # row-wise conversion (fast enough at 50 Hz)
        for i in np.where(ok)[0]:
            try:
                yaw[i] = _HMD_YAW.quaternion_to_euler(w[i], x[i], y[i], z[i])[2]
            except Exception:
                yaw[i] = np.nan
        yaw_deg = np.degrees(yaw)
        yaw_deg = (yaw_deg + 180.0) % 360.0 - 180.0
        return yaw_deg

    # 2) Try a single list-like quaternion column
    cand_cols = [c for c in cols if _QUAT_LIST_COL_PAT.search(str(c))]
    for c in cand_cols:
        ser = raw_df[c]
        # quick check: are there list-ish strings?
        sample = ser.dropna().astype(str).head(10).tolist()
        if not any(("[" in s and "]" in s) or ("(" in s and ")" in s) for s in sample):
            continue
        yaw = np.full((len(raw_df),), np.nan, dtype=float)
        for i, v in enumerate(ser.astype(str).tolist()):
            try:
                q = ast.literal_eval(v)
                if isinstance(q, (list, tuple)) and len(q) == 4:
                    yaw[i] = _HMD_YAW.quaternion_to_euler(float(q[0]), float(q[1]), float(q[2]), float(q[3]))[2]
            except Exception:
                continue
        if np.isfinite(yaw).sum() >= 5:
            yaw_deg = np.degrees(yaw)
            yaw_deg = (yaw_deg + 180.0) % 360.0 - 180.0
            return yaw_deg

    return None


def _infer_degrees(y: NDArray[Any] | None) -> NDArray[np.float64] | None:
    if y is None:
        return None

    yy: NDArray[np.float64] = np.asarray(y, dtype=np.float64)
    a = np.abs(yy[np.isfinite(yy)])
    if a.size == 0:
        return yy

    p99 = float(np.nanpercentile(a, 99))
    if p99 <= 6.5:
        return np.rad2deg(yy)
    return yy


def _yaw_entropy_deg(yaw_deg: np.ndarray, clip: float = 90.0, bin_width: float = 10.0) -> float:
    y = yaw_deg[np.isfinite(yaw_deg)]
    if y.size < 5:
        return float("nan")
    y = np.clip(y, -clip, clip)
    bins = np.arange(-clip, clip + bin_width, bin_width)
    hist, _ = np.histogram(y, bins=bins)
    if hist.sum() == 0:
        return float("nan")
    p = hist.astype(float) / hist.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _extract_yaw_features_from_timeseries(raw_df: pd.DataFrame, video_id: str, participant_id: int, dataset_label: str,
                                          time_col: str = "Timestamp", trigger_col: str = "TriggerValueRight",
                                          yaw_col_candidates: list[str] | None = None,
                                          forward_cone_degs: list[float] | None = None,
                                          turn_threshold_degs: list[float] | None = None,
                                          coupling_window_s: float = 1.0) -> dict:
    if yaw_col_candidates is None:
        yaw_col_candidates = _YAW_CANDIDATES
    if forward_cone_degs is None:
        forward_cone_degs = [10.0, 15.0]
    if turn_threshold_degs is None:
        turn_threshold_degs = [15.0, 30.0]

    rec = {
        "dataset": dataset_label,
        "participant_id": participant_id,
        "video_id": str(video_id),
    }

    if raw_df is None or raw_df.empty:
        return rec

    yaw_col = None
    for c in yaw_col_candidates:
        if c in raw_df.columns:
            yaw_col = c
            break
    if time_col not in raw_df.columns:
        return rec

    t = pd.to_numeric(raw_df[time_col], errors="coerce").to_numpy(dtype=float)

    t = _infer_time_seconds(t)

    # --- Yaw source selection ---
    # Many Unity logs include a generic 'Yaw' that is unrelated to the head/HMD orientation (e.g., vehicle yaw).
    # If quaternion columns exist, we *prefer* computing yaw from those.
    yaw_is_deg = False
    yaw_source = None

    yaw_from_col = None
    if yaw_col is not None:
        yaw_from_col = pd.to_numeric(raw_df[yaw_col], errors="coerce").to_numpy(dtype=float)

    yaw_from_quat = _compute_yaw_from_quaternion_columns(raw_df)

    def _usable(arr: NDArray[Any] | None, tt: NDArray[Any] | None) -> bool:
        if arr is None or tt is None:
            return False

        a = np.asarray(arr, dtype=np.float64)
        t = np.asarray(tt, dtype=np.float64)

        # optional, but often helpful if you expect same-length signals
        if a.shape != t.shape:
            return False

        ok = np.isfinite(t) & np.isfinite(a)
        n_ok = int(ok.sum())
        if n_ok < 5:
            return False

        y = a[ok]
        # wrap for range check
        y = (y + 180.0) % 360.0 - 180.0

        rng = float(np.nanmax(y) - np.nanmin(y))          # safe because n_ok >= 5
        sd = float(np.nanstd(y, ddof=1)) if n_ok > 1 else 0.0

        # If the signal is essentially constant, it is usually a logging artifact.
        return (rng >= 0.1) or (sd >= 0.05)

    # Prefer quaternion-derived yaw when available and usable
    if _usable(yaw_from_quat, t):
        yaw = yaw_from_quat
        yaw_is_deg = True
        yaw_source = "quaternion"
    elif yaw_from_col is not None and _usable(yaw_from_col, t):
        yaw = yaw_from_col
        yaw_source = f"column:{yaw_col}"
    elif yaw_from_quat is not None:
        # Fall back to quaternion even if it is near-constant (will be marked invalid later)
        yaw = yaw_from_quat
        yaw_is_deg = True
        yaw_source = "quaternion_constant"
    elif yaw_from_col is not None:
        yaw = yaw_from_col
        yaw_source = f"column:{yaw_col}_constant_or_sparse"
    else:
        return rec

    rec["yaw_source"] = yaw_source

    valid = np.isfinite(t)
    if valid.sum() < 5:
        return rec
    t = t[valid]
    if yaw is not None:
        yaw = yaw[valid]

    order = np.argsort(t)
    t = t[order]
    if yaw is not None:
        yaw = yaw[order]

    if t.size > 1:
        uniq = np.r_[True, np.diff(t) > 1e-9]
        t = t[uniq]
        if yaw is not None:
            yaw = yaw[uniq]

    # If yaw came from quaternions, it is already in degrees.
    # Otherwise, infer whether the data are in radians or degrees.
    yaw = yaw if yaw_is_deg else _infer_degrees(yaw)
    # Wrap to [-180, 180] for stability across logs
    assert yaw is not None
    yaw = (yaw + 180.0) % 360.0 - 180.0
    abs_yaw = np.abs(yaw)

    # Diagnostics: how much yaw data do we actually have?
    rec["yaw_n_samples"] = int(np.isfinite(yaw).sum())
    rec["yaw_duration_s"] = float(t[-1] - t[0]) if (t.size >= 2 and np.isfinite(t[0]) and np.isfinite(t[-1])) else float("nan")  # noqa:E501
    rec["yaw_range_deg"] = float(np.nanmax(yaw) - np.nanmin(yaw)) if rec["yaw_n_samples"] > 0 else float("nan")

    # Data quality flags (aligned with the prior pipeline: compute yaw whenever HMD rotation exists).
    rec["yaw_has_data"] = int(rec["yaw_n_samples"] >= 5)
    # "Constant" should be interpreted as *near*-constant. With real HMD streams,
    # sensor noise often makes sub-0.1° ranges unrealistic, so we use slightly
    # looser thresholds and expose them in the output for transparency.
    const_range_thr = 0.5  # deg
    const_sd_thr = 0.25    # deg
    _yaw_sd_tmp = float(np.nanstd(yaw, ddof=1)) if np.isfinite(yaw).sum() > 1 else float("nan")
    rec["yaw_constant_range_thr_deg"] = float(const_range_thr)
    rec["yaw_constant_sd_thr_deg"] = float(const_sd_thr)
    rec["yaw_is_constant"] = int(
        rec["yaw_has_data"] == 1
        and np.isfinite(rec["yaw_range_deg"])
        and (rec["yaw_range_deg"] < const_range_thr)
        and np.isfinite(_yaw_sd_tmp)
        and (_yaw_sd_tmp < const_sd_thr)
    )
    # 'Valid' means we had enough yaw samples to compute features (not that the participant moved their head).
    rec["yaw_is_valid"] = int(rec["yaw_has_data"] == 1)

    rec["yaw_abs_mean"] = float(np.nanmean(abs_yaw))
    rec["yaw_mean"] = float(np.nanmean(yaw))
    rec["yaw_sd"] = float(np.nanstd(yaw, ddof=1)) if np.isfinite(yaw).sum() > 1 else float("nan")
    try:
        q1 = float(np.nanpercentile(yaw, 25))
        q3 = float(np.nanpercentile(yaw, 75))
        rec["yaw_iqr"] = q3 - q1
    except Exception:
        rec["yaw_iqr"] = float("nan")

    rec["yaw_entropy"] = _yaw_entropy_deg(yaw)

    for cone in forward_cone_degs:
        rec[f"yaw_forward_frac_{int(cone)}"] = float(np.nanmean(abs_yaw <= cone))

    rec["yaw_right_frac_gt5"] = float(np.nanmean(yaw > 5.0))
    rec["yaw_left_frac_lt-5"] = float(np.nanmean(yaw < -5.0))
    rec["yaw_bias_right_minus_left"] = rec["yaw_right_frac_gt5"] - rec["yaw_left_frac_lt-5"]

    # yaw speed
    if t.size >= 2:
        dt = np.diff(t)
        dy = np.diff(yaw)
        ok = np.isfinite(dt) & np.isfinite(dy) & (dt > 1e-6)
        if ok.any():
            yaw_speed = np.abs(dy[ok] / dt[ok])
            rec["yaw_speed_mean"] = float(np.nanmean(yaw_speed))
            rec["yaw_speed_peak"] = float(np.nanmax(yaw_speed))
            try:
                rec["yaw_speed_p95"] = float(np.nanpercentile(yaw_speed, 95))
            except Exception:
                rec["yaw_speed_p95"] = float("nan")
        else:
            rec["yaw_speed_mean"] = float("nan")
            rec["yaw_speed_peak"] = float("nan")
            rec["yaw_speed_p95"] = float("nan")
    else:
        rec["yaw_speed_mean"] = float("nan")
        rec["yaw_speed_peak"] = float("nan")
        rec["yaw_speed_p95"] = float("nan")

    # head turns
    for thr in turn_threshold_degs:
        turned = abs_yaw >= thr
        onsets = list(np.where(np.diff(turned.astype(int)) == 1)[0] + 1) if turned.size >= 2 else []
        offsets = list(np.where(np.diff(turned.astype(int)) == -1)[0] + 1) if turned.size >= 2 else []
        if turned.size and turned[0]:
            onsets = [0] + onsets
        if turned.size and turned[-1]:
            offsets = offsets + [len(turned)]

        rec[f"head_turn_count_{int(thr)}"] = int(len(onsets))

        durs = []
        for on, off in zip(onsets, offsets):
            off_idx = min(max(off - 1, 0), len(t) - 1)
            on_idx = min(max(on, 0), len(t) - 1)
            if off_idx >= on_idx:
                durs.append(float(t[off_idx] - t[on_idx]))
        rec[f"head_turn_dwell_mean_s_{int(thr)}"] = float(np.mean(durs)) if durs else float("nan")
        rec[f"head_turn_dwell_max_s_{int(thr)}"] = float(np.max(durs)) if durs else float("nan")

    # coupling with trigger press/release
    if trigger_col in raw_df.columns:
        trig = pd.to_numeric(raw_df[trigger_col], errors="coerce").to_numpy(dtype=float)
        trig = trig[valid][order]
        trig = trig[uniq] if trig.size >= t.size else trig[:t.size]

        unsafe = (trig > 0).astype(int)
        if unsafe.size >= 2:
            changes = np.diff(unsafe)
            starts = list(np.where(changes == 1)[0] + 1)
            ends_events = list(np.where(changes == -1)[0] + 1)
            if unsafe[0] == 1:
                starts = [0] + starts

            press_t = float(t[starts[0]]) if starts else float("nan")
            release_t = float(t[ends_events[0]]) if (starts and ends_events) else float("nan")

            # yaw speed time midpoints
            if t.size >= 2:
                dt = np.diff(t)
                dy = np.diff(yaw)
                ok = np.isfinite(dt) & np.isfinite(dy) & (dt > 1e-6)
                t_mid = (t[:-1] + t[1:]) / 2.0
                yaw_speed = np.full(dt.shape, np.nan)
                yaw_speed[ok] = np.abs(dy[ok] / dt[ok])

                def _win_mean(arr, t_arr, end_t, win_s):
                    if not np.isfinite(end_t):
                        return float("nan")
                    mask = (t_arr >= (end_t - win_s)) & (t_arr < end_t)
                    if mask.any():
                        return float(np.nanmean(arr[mask]))
                    return float("nan")

                rec["yaw_speed_pre_press_mean_1s"] = _win_mean(yaw_speed, t_mid, press_t, 1.0)
                rec["yaw_speed_pre_release_mean_1s"] = _win_mean(yaw_speed, t_mid, release_t, 1.0)
                rec["yaw_speed_pre_press_mean_2s"] = _win_mean(yaw_speed, t_mid, press_t, 2.0)
                rec["yaw_speed_pre_release_mean_2s"] = _win_mean(yaw_speed, t_mid, release_t, 2.0)

                # Trigger derivative (dtrigger/dt) on same midpoints, for cross-correlation with yaw_speed
                dtrig_dt = None
                if trig.size >= 2:
                    dtr = np.diff(trig)
                    ok_tr = np.isfinite(dt) & np.isfinite(dtr) & (dt > 1e-6)
                    dtrig_dt = np.full(dt.shape, np.nan)
                    dtrig_dt[ok_tr] = dtr[ok_tr] / dt[ok_tr]
                    rec["dtrigger_dt_sd"] = float(np.nanstd(dtrig_dt, ddof=1)) if np.isfinite(dtrig_dt).sum() > 1 else float("nan")  # noqa: E501
                    rec["dtrigger_dt_p95"] = float(np.nanpercentile(np.abs(dtrig_dt[np.isfinite(dtrig_dt)]), 95)) if np.isfinite(dtrig_dt).sum() > 5 else float("nan")  # noqa: E501

                # Within-trial coupling: cross-correlation yaw_speed ↔ dtrigger/dt
                try:
                    dt_med = float(np.nanmedian(np.diff(t_mid))) if t_mid.size >= 3 else float("nan")
                    rmax, lag_s = _xcorr_max_r_lag(yaw_speed, dtrig_dt, dt_s=dt_med, max_lag_s=2.0, min_n=10)
                    rec["xcorr_yawspd_dtrig_max_r"] = float(rmax)
                    rec["xcorr_yawspd_dtrig_lag_s"] = float(lag_s)
                except Exception:
                    rec["xcorr_yawspd_dtrig_max_r"] = float("nan")
                    rec["xcorr_yawspd_dtrig_lag_s"] = float("nan")

            # lag from last head-turn onset (thr=15) to press/release
            thr = 15.0
            turned = abs_yaw >= thr
            onsets = list(np.where(np.diff(turned.astype(int)) == 1)[0] + 1) if turned.size >= 2 else []
            if turned.size and turned[0]:
                onsets = [0] + onsets
            onset_times = t[onsets] if onsets else np.array([], dtype=float)

            def _lag_to_event(event_t):
                if not np.isfinite(event_t) or onset_times.size == 0:
                    return float("nan")
                prior = onset_times[onset_times <= event_t]
                if prior.size == 0:
                    return float("nan")
                return float(event_t - prior[-1])

            rec["lag_turn_to_press_s_15"] = _lag_to_event(press_t)
            rec["lag_turn_to_release_s_15"] = _lag_to_event(release_t)

            if np.isfinite(press_t):
                idx = int(np.clip(np.searchsorted(t, press_t), 0, len(t) - 1))
                rec["yaw_at_first_press"] = float(yaw[idx])
            if np.isfinite(release_t):
                idx = int(np.clip(np.searchsorted(t, release_t), 0, len(t) - 1))
                rec["yaw_at_first_release"] = float(yaw[idx])

            # yaw change preceding press/release (window mean + delta to event value)
            def _yaw_win_mean(end_t):
                if not np.isfinite(end_t):
                    return float("nan")
                mask = (t >= (end_t - coupling_window_s)) & (t < end_t)
                if mask.any():
                    return float(np.nanmean(yaw[mask]))
                return float("nan")

            rec["yaw_pre_press_mean_1s"] = _yaw_win_mean(press_t)
            rec["yaw_pre_release_mean_1s"] = _yaw_win_mean(release_t)

            # Event-triggered averages (scalar summaries):
            # mean yaw in the 2s before event; and specifically 1–2s before the first press (anticipation).
            def _yaw_win_mean_custom(end_t, win_s, start_offset_s=0.0):
                if not np.isfinite(end_t):
                    return float("nan")
                start_t = end_t - win_s - start_offset_s
                end_tt = end_t - start_offset_s
                mask = (t >= start_t) & (t < end_tt)
                if mask.any():
                    return float(np.nanmean(yaw[mask]))
                return float("nan")

            rec["yaw_pre_press_mean_2s"] = _yaw_win_mean_custom(press_t, win_s=2.0, start_offset_s=0.0)
            rec["yaw_pre_release_mean_2s"] = _yaw_win_mean_custom(release_t, win_s=2.0, start_offset_s=0.0)
            rec["yaw_pre_press_mean_2to1s"] = _yaw_win_mean_custom(press_t, win_s=1.0, start_offset_s=1.0)

            def _yaw_around_mean(center_t, win_s):
                if not np.isfinite(center_t):
                    return float("nan")
                mask = (t >= (center_t - win_s)) & (t <= (center_t + win_s))
                if mask.any():
                    return float(np.nanmean(yaw[mask]))
                return float("nan")

            rec["yaw_around_press_mean_pm1s"] = _yaw_around_mean(press_t, 1.0)
            rec["yaw_around_release_mean_pm1s"] = _yaw_around_mean(release_t, 1.0)
            if "yaw_at_first_press" in rec and np.isfinite(rec.get("yaw_at_first_press", np.nan)) and np.isfinite(rec["yaw_pre_press_mean_1s"]):  # noqa: E501
                rec["yaw_pre_press_delta_1s"] = float(rec["yaw_at_first_press"] - rec["yaw_pre_press_mean_1s"])
            if "yaw_at_first_release" in rec and np.isfinite(rec.get("yaw_at_first_release", np.nan)) and np.isfinite(rec["yaw_pre_release_mean_1s"]):  # noqa: E501
                rec["yaw_pre_release_delta_1s"] = float(rec["yaw_at_first_release"] - rec["yaw_pre_release_mean_1s"])

    # If invalid, null out the derived metrics (keep only IDs + diagnostics).
    if rec.get("yaw_is_valid", 1) == 0:
        for k in list(rec.keys()):
            if k in {"dataset", "participant_id", "video_id", "yaw_source", "yaw_n_samples",
                     "yaw_duration_s", "yaw_range_deg", "yaw_has_data", "yaw_is_constant", "yaw_is_valid"}:
                continue
            if k.startswith("yaw_") or k.startswith("head_turn_") or k.startswith("lag_turn_"):
                rec[k] = float("nan")

    return rec


def compute_yaw_features_dataset(data_folder: str, mapping_df: pd.DataFrame, dataset_label: str,
                                 out_csv: Optional[str] = None, time_col: str = "Timestamp",
                                 trigger_col: str = "TriggerValueRight") -> pd.DataFrame:
    if not os.path.isdir(data_folder):
        logger.warning(f"[YAW] data_folder not found: {data_folder}")
        return pd.DataFrame()

    mapping_ids = set(mapping_df["video_id"].astype(str).tolist()) if "video_id" in mapping_df.columns else set()
    part_dirs = [
        d for d in os.listdir(data_folder)
        if os.path.isdir(os.path.join(data_folder, d)) and d.startswith("Participant_")
    ]
    part_dirs = sorted(part_dirs)

    rows = []
    for d in part_dirs:
        try:
            pid = int(d.split("_")[1])
        except Exception:
            continue
        folder = os.path.join(data_folder, d)
        csv_files = glob.glob(os.path.join(folder, "*.csv"))

        # Pick the correct raw trial file for each (participant, video_id).
        # Participant folders often contain multiple CSVs with the same video_id substring
        # (processed summaries, matrices, etc.). Selecting the wrong file makes yaw appear
        # constant and collapses yaw_valid_fraction. The previous yaw pipeline always used
        # an exact '{video_id}.csv' when present (see helper.export_participant_quaternion_matrix).
        if "video_id" in mapping_df.columns:
            video_ids = mapping_df["video_id"].astype(str).tolist()
        else:
            vids = []
            for fp in csv_files:
                bn = os.path.basename(fp)
                mm = re.search(r"(video_\d+|baseline_\d+)", bn)
                if mm:
                    vids.append(mm.group(1))
            video_ids = sorted(set(vids))

        for vid in video_ids:
            if mapping_ids and (str(vid) not in mapping_ids):
                continue

            fp = _choose_trial_file(csv_files, str(vid), folder)
            if fp is None:
                continue

            raw_df = _read_csv_flexible(fp)
            if raw_df is None or raw_df.empty:
                continue
            if time_col not in raw_df.columns:
                continue

            rec = _extract_yaw_features_from_timeseries(
                raw_df=raw_df,
                video_id=str(vid),
                participant_id=pid,
                dataset_label=dataset_label,
                time_col=time_col,
                trigger_col=trigger_col,
            )
            # keep only rows where yaw metrics were actually computed
            if any(k.startswith("yaw_") or k.startswith("head_turn_") for k in rec.keys()):
                rows.append(rec)

    out = pd.DataFrame(rows)

    # Merge design factors (yielding/eHMI/camera/distPed/etc.) so downstream
    # comparisons can group by condition_name consistently.
    if (out is not None) and (not out.empty) and (mapping_df is not None) and (not mapping_df.empty):
        md = mapping_df.copy()
        if "video_id" in md.columns:
            md["video_id"] = md["video_id"].astype(str)
        if "video_id" in out.columns:
            out["video_id"] = out["video_id"].astype(str)
        fac_cols = [
            c
            for c in [
                "video_id",
                "condition_name",
                "yielding",
                "eHMIOn",
                "camera",
                "distPed",
                "p1",
                "p2",
                "group",
            ]
            if c in md.columns
        ]
        if fac_cols and ("video_id" in fac_cols):
            out = out.merge(md[fac_cols].drop_duplicates("video_id"), on="video_id", how="left")
    if out_csv and (not out.empty):
        try:
            out.to_csv(out_csv, index=False)
        except Exception as e:
            logger.error(f"[YAW] Could not write {out_csv}: {e}")
    return out


def summarize_and_plot_yaw_results(
    yaw_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    out_root: str,
    h: HMD_helper,
    forward_frac_col_candidates: list[str] | None = None,
    mean_col_candidates: list[str] | None = None,
    sd_col_candidates: list[str] | None = None,
) -> None:
    """
    Build participant-aggregated yaw summaries and plots.

    - yaw_summary_by_context.csv: participant × (yielding, eHMIOn, camera) means (trial-averaged)
    - yaw_summary_by_context_group.csv: dataset × (yielding, eHMIOn, camera) mean±SEM across participants
    - yaw_summary_by_dataset.csv: participant means (all trials)
    - yaw_summary_by_dataset_group.csv: dataset means (all participants)

    Plots:
    - yaw_forward_fraction_by_context: 4 contexts (yielding × eHMIOn), bars are shuffled vs unshuffled (8 bars total)
    - yaw_forward_fraction_by_context_camera: 8 contexts (camera × yielding × eHMIOn), bars are shuffled vs unshuffled
    - yaw_mean_by_dataset, yaw_sd_by_dataset, yaw_forward_frac_by_dataset (+ any extra yaw_* metrics if present)
    """
    if yaw_df is None or len(yaw_df) == 0:
        logger.info("[YAW] No yaw rows; skipping yaw summaries/plots.")
        return

    # Ensure output directory exists
    try:
        os.makedirs(out_root, exist_ok=True)
    except Exception:
        pass

    # ---------------------------------------------------------------------
    # 0) Prepare columns + merge trial mapping factors (yielding/eHMI/camera)
    # ---------------------------------------------------------------------
    d = yaw_df.copy()

    if forward_frac_col_candidates is None:
        forward_frac_col_candidates = ["yaw_forward_frac_15", "yaw_forward_frac", "yaw_forward_frac_15deg"]
    if mean_col_candidates is None:
        mean_col_candidates = ["yaw_mean", "yaw_mean_deg"]
    if sd_col_candidates is None:
        sd_col_candidates = ["yaw_sd", "yaw_std", "yaw_sd_deg"]

    def _first_existing(cols: List[str]) -> Optional[str]:
        for c in cols:
            if c in d.columns:
                return c
        return None

    forward_col = _first_existing(forward_frac_col_candidates)
    mean_col = _first_existing(mean_col_candidates)
    sd_col = _first_existing(sd_col_candidates)

    # Standardize canonical names if needed
    if forward_col and forward_col != "yaw_forward_frac_15":
        d["yaw_forward_frac_15"] = d[forward_col]
    if mean_col and mean_col != "yaw_mean":
        d["yaw_mean"] = d[mean_col]
    if sd_col and sd_col != "yaw_sd":
        d["yaw_sd"] = d[sd_col]

    # Merge mapping factors (video_id -> yielding/eHMIOn/camera/distPed if present)
    if "video_id" in d.columns and mapping_df is not None and len(mapping_df) > 0 and "video_id" in mapping_df.columns:
        factors = ["video_id", "yielding", "eHMIOn", "camera", "distPed"]
        factors = [c for c in factors if c in mapping_df.columns]

        def _norm_vid(x) -> str:
            if pd.isna(x):
                return ""
            s = str(x).strip().lower()
            s = re.sub(r"\s+", "", s)
            return s

        d["_video_norm"] = d["video_id"].map(_norm_vid)
        m = mapping_df[factors].copy()
        m["_video_norm"] = m["video_id"].map(_norm_vid)

        # Deduplicate mapping on _video_norm (keep first)
        m = m.drop_duplicates(subset=["_video_norm"], keep="first")

        merge_cols = ["_video_norm"]
        factor_cols = []
        for c in factors:
            if c == "video_id":
                continue
            if c not in d.columns:
                merge_cols.append(c)
                factor_cols.append(c)
            else:
                fill_col = f"{c}__mapfill"
                m = m.rename(columns={c: fill_col})
                merge_cols.append(fill_col)
                factor_cols.append(c)

        d = d.merge(m[merge_cols], on="_video_norm", how="left")

        for c in factors:
            if c == "video_id":
                continue
            fill_col = f"{c}__mapfill"
            if fill_col in d.columns:
                d[c] = d[c] if c in d.columns else np.nan
                d[c] = d[c].where(d[c].notna(), d[fill_col])
                d = d.drop(columns=[fill_col], errors="ignore")

        present_factor_cols = [c for c in ["yielding", "eHMIOn", "camera", "distPed"] if c in d.columns]
        match_rate = float(d[present_factor_cols].notna().any(axis=1).mean()) if present_factor_cols else 1.0
        if match_rate < 0.95:
            bad = d.loc[d[present_factor_cols].isna().all(axis=1), "video_id"].astype(str).unique()[:10] if present_factor_cols else []  # noqa: E501
            logger.warning(f"[YAW] Warning: mapping merge matched only {match_rate*100:.1f}% of yaw rows. Example unmapped video_id: {list(bad)}")  # noqa: E501
        d = d.drop(columns=["_video_norm"], errors="ignore")
    else:
        logger.warning("[YAW] Warning: cannot merge mapping factors (missing video_id or mapping). Context plots may be reduced.")  # noqa: E501

    # Coerce factor columns (nice ordering/labels)
    for c in ["yielding", "eHMIOn", "camera"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    # ---------------------------------------------------------------------
    # 1) Choose yaw metrics to summarize (keep robust if some are missing)
    # ---------------------------------------------------------------------
    metric_cols = []
    for c in [
        "yaw_forward_frac_15",
        "yaw_mean",
        "yaw_abs_mean",
        "yaw_sd",
        "yaw_entropy",
        "yaw_speed_mean",
        "head_turn_count_15",
        "head_turn_dwell_mean_s_15",
        "yaw_valid_fraction",
        "yaw_constant_fraction",
    ]:
        if c in d.columns:
            metric_cols.append(c)

    if len(metric_cols) == 0:
        logger.info("[YAW] No known yaw metrics found; skipping yaw summaries/plots.")
        return

    # ---------------------------------------------------------------------
    # 2) Participant-level summaries
    # ---------------------------------------------------------------------
    # 2a) participant × context × camera
    group_cols_cam = ["dataset", "participant_id"]
    for c in ["yielding", "eHMIOn", "camera"]:
        if c in d.columns:
            group_cols_cam.append(c)

    by_p_cam = (
        d.groupby(group_cols_cam, dropna=False)[metric_cols]
        .mean(numeric_only=True)
        .reset_index()
    )

    # Write participant-level table (context+camera)
    ctx_csv = os.path.join(out_root, "yaw_summary_by_context.csv")
    by_p_cam.to_csv(ctx_csv, index=False)
    logger.info(f"[YAW] wrote: {ctx_csv}")

    # 2b) participant means by dataset (all trials)
    by_p_ds = (
        d.groupby(["dataset", "participant_id"], dropna=False)[metric_cols]
        .mean(numeric_only=True)
        .reset_index()
    )
    ds_csv = os.path.join(out_root, "yaw_summary_by_dataset.csv")
    by_p_ds.to_csv(ds_csv, index=False)
    logger.info(f"[YAW] wrote: {ds_csv}")

    # ---------------------------------------------------------------------
    # 3) Group-level summaries across participants (mean ± SEM)
    # ---------------------------------------------------------------------
    def _sem(x: pd.Series) -> float:
        x = pd.to_numeric(x, errors="coerce").dropna()
        n = len(x)
        if n <= 1:
            return float("nan")
        return float(x.std(ddof=1) / np.sqrt(n))

    def _group_summary(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
        agg = {}
        for mcol in metric_cols:
            agg[mcol] = ["count", "mean", "std", _sem]
        g = df.groupby(group_cols, dropna=False).agg(agg)
        g.columns = [f"{a}__sem" if str(b) == "_sem" else f"{a}_{b}" for a, b in g.columns]
        return g.reset_index()

    # 3a) context+camera group
    group_cols_cam_g = [c for c in group_cols_cam if c != "participant_id"]
    grp_cam = _group_summary(by_p_cam, group_cols_cam_g)
    grp_cam_csv = os.path.join(out_root, "yaw_summary_by_context_group.csv")
    grp_cam.to_csv(grp_cam_csv, index=False)
    logger.info(f"[YAW] wrote: {grp_cam_csv}")

    grp_cam_txt = os.path.join(out_root, "yaw_summary_by_context.txt")
    with open(grp_cam_txt, "w") as f:
        f.write(grp_cam.to_string(index=False))
        f.write("\n")
    logger.info(f"[YAW] wrote: {grp_cam_txt}")

    # 3b) dataset group
    grp_ds = _group_summary(by_p_ds, ["dataset"])
    grp_ds_csv = os.path.join(out_root, "yaw_summary_by_dataset_group.csv")
    grp_ds.to_csv(grp_ds_csv, index=False)
    # (optional) keep the historical txt name
    grp_ds_txt = os.path.join(out_root, "yaw_summary_by_dataset.txt")
    with open(grp_ds_txt, "w") as f:
        f.write(grp_ds.to_string(index=False))
        f.write("\n")
    logger.info(f"[YAW] wrote: {grp_ds_txt}")

    # ---------------------------------------------------------------------
    # 4) Plots
    # ---------------------------------------------------------------------

    # Helper label for yielding/eHMI contexts (4 contexts)
    def _ctx_label(y, e) -> str:
        try:
            if pd.isna(y) or pd.isna(e):
                return "unknown"
            y = int(y)
            e = int(e)
        except Exception:
            return "unknown"

        if y == 1 and e == 1:
            return "Yielding + eHMI"
        if y == 0 and e == 1:
            return "No yielding + eHMI"
        if y == 1 and e == 0:
            return "Yielding + no eHMI"
        if y == 0 and e == 0:
            return "No yielding + no eHMI"
        return "unknown"

    def _axis_margin(ymin: float, ymax: float, min_margin: float = 0.5) -> Tuple[float, float]:
        if not np.isfinite(ymin) or not np.isfinite(ymax):
            return ymin, ymax
        if ymin == ymax:
            return ymin - min_margin, ymax + min_margin
        m = max(min_margin, 0.06 * (ymax - ymin))
        return ymin - m, ymax + m

    # 4a) Forward-looking fraction by context (yielding×eHMI; collapse across camera)
    if "yaw_forward_frac_15" in metric_cols and ("yielding" in by_p_cam.columns) and ("eHMIOn" in by_p_cam.columns):
        # Collapse camera within-participant (equal weight per camera if both present)
        if "camera" in by_p_cam.columns and by_p_cam["camera"].notna().any():
            by_p_ctx = (
                by_p_cam.groupby(["dataset", "participant_id",
                                  "yielding", "eHMIOn"], dropna=False)[["yaw_forward_frac_15"]]
                .mean(numeric_only=True)
                .reset_index()
            )
        else:
            by_p_ctx = by_p_cam[["dataset", "participant_id", "yielding", "eHMIOn", "yaw_forward_frac_15"]].copy()

        grp_ctx = (
            by_p_ctx.groupby(["dataset", "yielding", "eHMIOn"], dropna=False)["yaw_forward_frac_15"]
            .agg(["count", "mean", "std", _sem])
            .reset_index()
            .rename(columns={"mean": "yaw_forward_frac_15_mean",
                             "std": "yaw_forward_frac_15_std", "_sem": "yaw_forward_frac_15__sem"})
        )
        grp_ctx["context"] = grp_ctx.apply(lambda r: _ctx_label(r["yielding"], r["eHMIOn"]), axis=1)
        grp_ctx["pct_within"] = 100.0 * grp_ctx["yaw_forward_frac_15_mean"]
        grp_ctx["sem_pct"] = 100.0 * grp_ctx["yaw_forward_frac_15__sem"]

        ctx_pretty_order = ["Yielding + eHMI", "No yielding + eHMI",
                            "Yielding + no eHMI", "No yielding + no eHMI", "unknown"]
        fig = px.bar(
            grp_ctx,
            x="context",
            y="pct_within",
            color="dataset",
            barmode="group",
            error_y="sem_pct",
            category_orders={"context": ctx_pretty_order},
            text="pct_within",
            title="Head-yaw forward-looking (pct_within) by context (mean ± SEM, participant-aggregated)",
        )
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside", cliponaxis=False)
        ymin = float(np.nanmin(grp_ctx["pct_within"])) if grp_ctx["pct_within"].notna().any() else 0.0
        ymax = float(np.nanmax(grp_ctx["pct_within"])) if grp_ctx["pct_within"].notna().any() else 1.0
        y0, y1 = _axis_margin(ymin, ymax, min_margin=0.8)
        fig.update_yaxes(title_text="Percent within ±15°", range=[y0, y1])
        fig.update_xaxes(title_text="Context (yielding/eHMI)")
        fig.update_layout(legend_title_text="dataset", margin=dict(t=70, l=70, r=20, b=70), height=550)

        logger.info("[YAW] plotting yaw_forward_fraction_by_context")
        _save_plot(h, fig, "yaw_forward_fraction_by_context", out_root=out_root)

        # 4b) Optional: Forward-looking fraction by context WITH camera (8 contexts)
        if "camera" in by_p_cam.columns and by_p_cam["camera"].notna().any():
            grp_cam2 = (
                by_p_cam.groupby(["dataset", "yielding", "eHMIOn", "camera"], dropna=False)["yaw_forward_frac_15"]
                .agg(["count", "mean", "std", _sem])
                .reset_index()
                .rename(columns={"mean": "yaw_forward_frac_15_mean",
                                 "std": "yaw_forward_frac_15_std", "_sem": "yaw_forward_frac_15__sem"})
            )
            grp_cam2["context"] = grp_cam2.apply(lambda r: _ctx_label(r["yielding"], r["eHMIOn"]), axis=1)
            grp_cam2["cam_context"] = grp_cam2.apply(
                lambda r: f"cam={int(r['camera']) if pd.notna(r['camera']) else 'NA'}<br>{r['context']}",
                axis=1,
            )
            grp_cam2["pct_within"] = 100.0 * grp_cam2["yaw_forward_frac_15_mean"]
            grp_cam2["sem_pct"] = 100.0 * grp_cam2["yaw_forward_frac_15__sem"]

            cam_levels = [int(x) for x in sorted(grp_cam2["camera"].dropna().unique())]
            cam_ctx_order = []
            for cam in cam_levels:
                for ctx in ctx_pretty_order:
                    if ctx == "unknown":
                        continue
                    cam_ctx_order.append(f"cam={cam} | {ctx}")
            # add unknown last
            cam_ctx_order.append("cam=NA | unknown")

            fig2 = px.bar(
                grp_cam2,
                x="cam_context",
                y="pct_within",
                color="dataset",
                barmode="group",
                error_y="sem_pct",
                category_orders={"cam_context": cam_ctx_order},
                text="pct_within",
                title="Head-yaw forward-looking (pct_within) by context × camera (mean ± SEM, participant-aggregated)",
            )
            fig2.update_traces(texttemplate="%{text:.2f}", textposition="outside", cliponaxis=False)
            ymin2 = float(np.nanmin(grp_cam2["pct_within"])) if grp_cam2["pct_within"].notna().any() else 0.0
            ymax2 = float(np.nanmax(grp_cam2["pct_within"])) if grp_cam2["pct_within"].notna().any() else 1.0
            y0, y1 = _axis_margin(ymin2, ymax2, min_margin=0.8)
            fig2.update_yaxes(title_text="Percent within ±15°", range=[y0, y1])
            fig2.update_xaxes(title_text="Context (camera | yielding/eHMI)", tickangle=-25)
            fig2.update_layout(legend_title_text="dataset", margin=dict(t=70, l=70, r=20, b=110), height=620)

            logger.info("[YAW] plotting yaw_forward_fraction_by_context_camera (8 contexts)")
            _save_plot(h, fig2, "yaw_forward_fraction_by_context_camera", out_root=out_root)

    # 4c) Dataset-level scatter/strip plots for common yaw metrics
    # Each point = participant mean, overlaid with mean ± SEM per dataset
    def _plot_metric_by_dataset(col: str, title: str, ylab: str, out_name: str):
        if col not in by_p_ds.columns:
            return
        # participant points
        fig = px.strip(
            by_p_ds,
            x="dataset",
            y=col,
            color="dataset",
            stripmode="overlay",
            title=title,
        )
        # add mean±SEM
        dd = (
            by_p_ds.groupby("dataset")[col]
            .agg(["count", "mean", "std", _sem])
            .reset_index()
        )
        dd = dd.rename(columns={"_sem": "sem"})
        fig.add_scatter(
            x=dd["dataset"],
            y=dd["mean"],
            mode="markers",
            marker=dict(size=12, symbol="x"),
            showlegend=False,
            error_y=dict(type="data", array=dd["sem"], visible=True),
        )
        ymin = float(np.nanmin(by_p_ds[col])) if by_p_ds[col].notna().any() else 0.0
        ymax = float(np.nanmax(by_p_ds[col])) if by_p_ds[col].notna().any() else 1.0
        y0, y1 = _axis_margin(ymin, ymax, min_margin=0.02 * (abs(ymax) + 1.0))
        fig.update_yaxes(title_text=ylab, range=[y0, y1])
        fig.update_xaxes(title_text="dataset")
        fig.update_layout(legend_title_text="dataset", margin=dict(t=70, l=70, r=20, b=70), height=520)
        _save_plot(h, fig, out_name, out_root=out_root)

    _plot_metric_by_dataset(
        "yaw_sd",
        "Head-yaw variability (SD) by dataset (participant means)",
        "yaw_sd",
        "yaw_sd_by_dataset",
    )
    _plot_metric_by_dataset(
        "yaw_mean",
        "Mean head yaw (deg) by dataset (participant means)",
        "yaw_mean",
        "yaw_mean_by_dataset",
    )
    _plot_metric_by_dataset(
        "yaw_forward_frac_15",
        "Forward-looking fraction (|yaw| ≤ 15°) by dataset (participant means)",
        "yaw_forward_frac_15",
        "yaw_forward_frac_by_dataset",
    )

    # Extra plots if present in the data
    _plot_metric_by_dataset(
        "yaw_valid_fraction",
        "Yaw valid-sample fraction by dataset (participant means)",
        "yaw_valid_fraction",
        "yaw_valid_fraction_by_dataset",
    )
    _plot_metric_by_dataset(
        "yaw_constant_fraction",
        "Yaw constant-sample fraction by dataset (participant means)",
        "yaw_constant_fraction",
        "yaw_constant_fraction_by_dataset",
    )


def _infer_time_seconds(t: np.ndarray) -> np.ndarray:
    """Heuristic: convert timestamps that look like milliseconds to seconds."""
    tt = np.array(t, dtype=float)
    tt = tt[np.isfinite(tt)]
    if tt.size == 0:
        return np.array(t, dtype=float)
    # Typical trials are ~10–20 seconds; if timestamps exceed a few hundred,
    # they are almost certainly in milliseconds.
    if np.nanmax(tt) > 200.0:
        return np.array(t, dtype=float) / 1000.0
    return np.array(t, dtype=float)


def _trigger_scale_guess(vals: np.ndarray) -> float:
    """Guess whether trigger is in [0,1] or [0,100]."""
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 100.0
    p99 = float(np.nanpercentile(np.abs(v), 99))
    return 1.0 if p99 <= 1.5 else 100.0


def _first_press_release_times(t_rel: np.ndarray, unsafe: np.ndarray) -> Tuple[float, float]:
    """Return (first_press_t, first_release_t) in seconds relative to start."""
    if t_rel.size < 2 or unsafe.size != t_rel.size:
        return float("nan"), float("nan")
    u = unsafe.astype(int)
    # press: 0->1
    press_idx = np.where(np.diff(u) == 1)[0] + 1
    if u[0] == 1:
        press_idx = np.r_[0, press_idx]
    if press_idx.size == 0:
        return float("nan"), float("nan")
    p0 = int(press_idx[0])
    press_t = float(t_rel[p0])

    # release: 1->0 after press
    release_idx = np.where(np.diff(u) == -1)[0] + 1
    release_idx = release_idx[release_idx > p0]
    release_t = float(t_rel[release_idx[0]]) if release_idx.size else float("nan")
    return press_t, release_t


def _unsafe_bout_durations(t_rel: np.ndarray, unsafe: np.ndarray) -> List[float]:
    if t_rel.size < 2 or unsafe.size != t_rel.size:
        return []
    u = unsafe.astype(int)
    onsets = np.where(np.diff(u) == 1)[0] + 1
    offsets = np.where(np.diff(u) == -1)[0] + 1
    if u[0] == 1:
        onsets = np.r_[0, onsets]
    if u[-1] == 1:
        offsets = np.r_[offsets, len(u) - 1]

    durs = []
    for on, off in zip(onsets, offsets):
        on = int(np.clip(on, 0, len(t_rel) - 1))
        off = int(np.clip(off, 0, len(t_rel) - 1))
        if off >= on:
            durs.append(float(t_rel[off] - t_rel[on]))
    return durs


def _compute_trigger_features_one_trial(
    raw_df: pd.DataFrame,
    mapping_row: Optional[pd.Series],
    dataset_label: str,
    participant_id: int,
    video_id: str,
    trigger_col: str,
    time_col: str,
    thresholds: Tuple[float, ...] = (0.10, 0.30, 0.50),
    analysis_window: str = "crossing",
) -> dict:
    """Compute a compact set of trial-level trigger features."""

    rec = {
        "dataset": dataset_label,
        "participant_id": int(participant_id),
        "video_id": str(video_id),
    }

    if raw_df is None or raw_df.empty:
        return rec
    if time_col not in raw_df.columns or trigger_col not in raw_df.columns:
        return rec

    t = pd.to_numeric(raw_df[time_col], errors="coerce").to_numpy(dtype=float)
    trig = pd.to_numeric(raw_df[trigger_col], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(t) & np.isfinite(trig)
    if ok.sum() < 5:
        return rec
    t = _infer_time_seconds(t[ok])
    trig = trig[ok]

    order = np.argsort(t)
    t = t[order]
    trig = trig[order]

    # drop duplicate timestamps
    if t.size > 1:
        uniq = np.r_[True, np.diff(t) > 1e-9]
        t = t[uniq]
        trig = trig[uniq]

    # choose analysis window
    cutoff = None
    if analysis_window == "crossing" and mapping_row is not None:
        try:
            cam = int(mapping_row.get("camera")) if pd.notna(mapping_row.get("camera")) else None
            if cam == 0 and pd.notna(mapping_row.get("cross_p1_time_s")):
                cutoff = float(mapping_row.get("cross_p1_time_s"))
            elif cam == 1 and pd.notna(mapping_row.get("cross_p2_time_s")):
                cutoff = float(mapping_row.get("cross_p2_time_s"))
        except Exception:
            cutoff = None
    if cutoff is not None and np.isfinite(cutoff):
        mask = t <= cutoff
        if mask.sum() >= 5:
            t = t[mask]
            trig = trig[mask]

    # relative time (start at 0)
    t0 = float(t[0])
    t_rel = t - t0

    # primary unsafe definition: any nonzero trigger
    unsafe = trig > 0

    # time-weighted unsafe fraction
    if t_rel.size >= 2:
        dt = np.diff(t_rel)
        dt = np.where(np.isfinite(dt) & (dt > 1e-6), dt, np.nan)
        total = float(np.nansum(dt))
        unsafe_time = float(np.nansum(dt * unsafe[:-1].astype(float))) if total > 0 else float("nan")
        rec["frac_time_unsafe"] = unsafe_time / total if total and not np.isnan(total) else float("nan")
    else:
        rec["frac_time_unsafe"] = float("nan")

    rec["n_samples"] = int(len(trig))
    rec["duration_s"] = float(t_rel[-1]) if t_rel.size else float("nan")

    # summary of trigger magnitude
    rec["trigger_mean"] = float(np.nanmean(trig))
    rec["trigger_sd"] = float(np.nanstd(trig, ddof=1)) if np.isfinite(trig).sum() > 1 else float("nan")
    rec["trigger_p95"] = float(np.nanpercentile(trig, 95)) if np.isfinite(trig).sum() > 5 else float("nan")
    rec["trigger_max"] = float(np.nanmax(trig)) if np.isfinite(trig).sum() else float("nan")

    # transitions / volatility
    if unsafe.size >= 2:
        rec["n_transitions"] = int(np.nansum(np.abs(np.diff(unsafe.astype(int)))))
    else:
        rec["n_transitions"] = 0

    if t_rel.size >= 2:
        dt = np.diff(t_rel)
        dtr = np.diff(trig)
        okd = np.isfinite(dt) & np.isfinite(dtr) & (dt > 1e-6)
        rate = np.full(dt.shape, np.nan)
        rate[okd] = dtr[okd] / dt[okd]
        rec["dtrigger_sd"] = float(np.nanstd(rate, ddof=1)) if np.isfinite(rate).sum() > 1 else float("nan")
        rec["max_ramp_rate"] = float(np.nanmax(rate)) if np.isfinite(rate).sum() else float("nan")
        # AUC (trapezoid)
        try:
            rec["trigger_auc"] = _np_trapezoid(trig, t_rel)
        except Exception:
            rec["trigger_auc"] = float("nan")
    else:
        rec["dtrigger_sd"] = float("nan")
        rec["max_ramp_rate"] = float("nan")
        rec["trigger_auc"] = float("nan")

    # press/release timing
    press_t, release_t = _first_press_release_times(t_rel, unsafe)
    rec["latency_first_press_s"] = press_t
    rec["latency_first_release_s"] = release_t
    rec["press_release_hysteresis"] = (release_t - press_t) if (np.isfinite(press_t) and np.isfinite(release_t)) else float("nan")  # noqa: E501

    durs = _unsafe_bout_durations(t_rel, unsafe)
    rec["unsafe_bout_count"] = int(len(durs))
    rec["mean_unsafe_bout_s"] = float(np.mean(durs)) if durs else float("nan")
    rec["max_unsafe_bout_s"] = float(np.max(durs)) if durs else float("nan")

    # thresholded occupancy (robust to 0–1 vs 0–100 trigger scaling)
    scale = _trigger_scale_guess(trig)
    rec["trigger_scale_inferred"] = float(scale)
    for thr in thresholds:
        cut = float(thr) * float(scale)
        rec[f"frac_time_above_{int(thr*100)}"] = float(np.nanmean(trig >= cut))

    # event-relative release timing (if mapping available)
    if mapping_row is not None and np.isfinite(release_t):
        for col, outcol in [
            ("yield_start_time_s", "time_to_yield_start_release_s"),
            ("yield_stop_time_s", "time_to_yield_stop_release_s"),
            ("yield_resume_time_s", "time_to_yield_resume_release_s"),
        ]:
            if col in mapping_row.index and pd.notna(mapping_row.get(col)):
                try:
                    rec[outcol] = float(release_t - float(mapping_row.get(col)))
                except Exception:
                    rec[outcol] = float("nan")

    return rec


def compute_trigger_features_dataset(
    data_folder: str,
    mapping_df: pd.DataFrame,
    out_csv: str,
    dataset_label: str,
    trigger_col: str = "TriggerValueRight",
    time_col: str = "Timestamp",
    thresholds: Tuple[float, ...] = (0.10, 0.30, 0.50),
    analysis_window: str = "crossing",
    include_baseline: bool = False,
) -> pd.DataFrame:
    """Extract per-participant × per-video trigger (unsafety) features.

    This is intentionally conservative about which CSVs it treats as time-series.
    It will only use files that include both `time_col` and `trigger_col`.
    """
    if not os.path.isdir(data_folder):
        logger.warning(f"[trigger] data_folder not found: {data_folder}")
        return pd.DataFrame()

    md = mapping_df.copy()
    md["video_id"] = md["video_id"].astype(str)
    if not include_baseline:
        md = md[md["video_id"].str.startswith("video_")].copy()

    mapping_ids = md["video_id"].astype(str).tolist()

    part_dirs = [
        d for d in os.listdir(data_folder)
        if os.path.isdir(os.path.join(data_folder, d)) and d.startswith("Participant_")
    ]
    part_dirs = sorted(part_dirs)

    rows: List[dict] = []
    for d in part_dirs:
        try:
            pid = int(d.split("_")[1])
        except Exception:
            continue
        folder = os.path.join(data_folder, d)
        csv_files = glob.glob(os.path.join(folder, "*.csv"))
        if not csv_files:
            continue

        # Pre-index candidate files by video_id substring (fast lookup)
        by_vid: Dict[str, List[str]] = {vid: [] for vid in mapping_ids}
        for fp in csv_files:
            bn = os.path.basename(fp)
            for vid in mapping_ids:
                if vid in bn:
                    by_vid[vid].append(fp)
        # Prefer larger files first (often full time-series)
        for vid in by_vid:
            by_vid[vid] = sorted(by_vid[vid], key=lambda p: os.path.getsize(p), reverse=True)

        for vid in mapping_ids:
            candidates = by_vid.get(vid, [])
            if not candidates:
                # fallback: regex match
                candidates = [fp for fp in csv_files if re.search(rf"\b{re.escape(vid)}\b", os.path.basename(fp))]
                candidates = sorted(candidates, key=lambda p: os.path.getsize(p), reverse=True)
            if not candidates:
                continue

            raw_df = None
            used_fp = None
            for fp in candidates[:4]:  # cap tries
                df_try = _read_csv_flexible(fp)
                if df_try is None or df_try.empty:
                    continue
                if (time_col in df_try.columns) and (trigger_col in df_try.columns):
                    raw_df = df_try
                    used_fp = fp
                    break
            if raw_df is None:
                continue

            mrow = None
            try:
                s = md.loc[md["video_id"] == vid]
                if not s.empty:
                    mrow = s.iloc[0]
            except Exception:
                mrow = None

            rec = _compute_trigger_features_one_trial(
                raw_df=raw_df,
                mapping_row=mrow,
                dataset_label=dataset_label,
                participant_id=pid,
                video_id=vid,
                trigger_col=trigger_col,
                time_col=time_col,
                thresholds=thresholds,
                analysis_window=analysis_window,
            )
            if used_fp:
                rec["source_file"] = os.path.basename(used_fp)
            rows.append(rec)

    out = pd.DataFrame(rows)
    if out.empty:
        logger.warning(f"[trigger] No usable time-series CSVs found for {dataset_label} under {data_folder}")
        return out

    # Merge design factors (yielding/eHMI/camera/distPed etc.)
    fac_cols = [
        c for c in ["video_id", "condition_name", "yielding", "eHMIOn", "camera", "distPed", "p1", "p2", "group"]
        if c in md.columns
    ]
    out = out.merge(md[fac_cols].drop_duplicates("video_id"), on="video_id", how="left")

    try:
        out.to_csv(out_csv, index=False)
    except Exception as e:
        logger.error(f"[trigger] Could not write {out_csv}: {e}")

    return out


# ---------------------------------------------------------------------------
# Star-import compatibility
# ---------------------------------------------------------------------------
# This module also keeps several underscore-prefixed helper functions to avoid
# rewriting the original logic. The pipeline imports it with
# `from csu_features import *`, so we expose underscore names via __all__.
__all__ = [k for k in globals().keys() if not k.startswith("__")]  # pyright: ignore[reportUnsupportedDunderAll]
