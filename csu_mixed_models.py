"""Mixed models + sensitivity analyses for shuffled-vs-unshuffled comparison.

This module is split out of the original `compare_shuffled_unshuffled.py` to keep the entrypoint small.
It depends on:
- `csu_core` for shared helpers (plot saving, printing, FDR helper)
"""

from __future__ import annotations

import os
import re
import warnings
import math
from typing import Optional, Dict

import numpy as np
import pandas as pd

import plotly.express as px  # noqa:F401
import plotly.graph_objects as go  # noqa:F401


# Statsmodels is optional; used for mixed-effects / robustness models.
import statsmodels.formula.api as smf  # noqa:F401
from statsmodels.stats.multitest import multipletests  # noqa:F401

from helper import HMD_helper
from custom_logger import CustomLogger

# Shared helpers (kept as names to avoid editing the original function bodies)
from csu_core import (  # noqa: F401
    _ensure_dir,
    _get_output_dir_for_logs,
    _pick_col,
    _print_table,
    _save_plot,
    _trial_num_display,
    _write_plot_index_and_open,
    _to_string_3dp,
    DATASET_COLOR_MAP,
    DATASET_LABEL_MAP,
    compare_participant_metrics,
)

HAVE_SM = True
logger = CustomLogger(__name__)  # use custom logger


# ----------------------
# Runtime configuration
# ----------------------
OUTPUT_ROOT: str = "_compare_output"

# Default SESOI for equivalence testing in MixedLM results.
# Interpreted as an equivalence bound of (SESOI_SD_MULT × SD(DV)).
SESOI_SD_MULT: float = 0.2


def _mm_bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """BH-FDR correction with fallback."""
    pvals = np.asarray(pvals, dtype=float)
    if (not HAVE_SM) or multipletests is None:
        return pvals
    try:
        _, q, _, _ = multipletests(pvals, method="fdr_bh")
        return q
    except Exception:
        return pvals


def _mm_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _mm_fit_mixedlm(formula: str, df: pd.DataFrame, group_col: str = "participant_id",
                    re_formula: str = "1") -> Optional[object]:
    if not HAVE_SM or smf is None:
        logger.warning("[MM] statsmodels not available; skipping MixedLM.")
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = smf.mixedlm(formula, df, groups=df[group_col], re_formula=re_formula)
            res = model.fit(reml=False, method="lbfgs", maxiter=800, disp=False)
        return res
    except Exception as e:
        logger.error(f"[MM] MixedLM FAILED: {e}\n  formula: {formula}")
        return None


def _mm_extract_fixed_effects(res, dv: str, model: str, keep_regex: Optional[str] = None) -> pd.DataFrame:
    if res is None:
        return pd.DataFrame()
    try:
        params = res.params
        bse = res.bse
        pvals = res.pvalues if hasattr(res, "pvalues") else None
    except Exception:
        return pd.DataFrame()

    rows = []
    for term in params.index:
        if keep_regex and re.search(keep_regex, term) is None:
            continue
        est = float(params[term])
        se = float(bse[term]) if term in bse.index else np.nan
        pv = float(pvals[term]) if (pvals is not None and term in pvals.index) else np.nan
        ci_lo = est - 1.96 * se if (not np.isnan(se)) else np.nan
        ci_hi = est + 1.96 * se if (not np.isnan(se)) else np.nan
        rows.append({
            "dv": dv,
            "model": model,
            "term": term,
            "coef": est,
            "se": se,
            "p": pv,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
        })
    return pd.DataFrame(rows)


def _mm_tost_equivalence(b: float, se: float, delta: float) -> Dict[str, float]:
    """Normal-approx TOST on a coefficient."""
    if np.isnan(b) or np.isnan(se) or se <= 0 or np.isnan(delta) or delta <= 0:
        return {"delta": float(delta) if delta is not None else np.nan,
                "p_tost": np.nan, "p_lower": np.nan, "p_upper": np.nan}

    # z tests
    z_lower = (b + delta) / se  # test b > -delta
    z_upper = (b - delta) / se  # test b < +delta

    # Phi via erf
    Phi = lambda z: 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))  # noqa:E731
    p_lower = 1 - Phi(z_lower)
    p_upper = Phi(z_upper)
    p_tost = max(p_lower, p_upper)
    return {"delta": float(delta), "p_tost": float(p_tost), "p_lower": float(p_lower), "p_upper": float(p_upper)}


def _mm_forest_plot(coef_df: pd.DataFrame, title: str, name: str, h: HMD_helper) -> None:
    if go is None or coef_df is None or coef_df.empty:
        return
    d = coef_df.copy()
    d["label"] = d["dv"].astype(str) + " | " + d["term"].astype(str)
    d = d.sort_values(["dv", "term"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d["coef"],
        y=d["label"],
        mode="markers",
        error_x=dict(
            type="data",
            array=(d["ci_hi"] - d["coef"]).clip(lower=0),
            arrayminus=(d["coef"] - d["ci_lo"]).clip(lower=0),
        ),
        hovertext=[
            f"dv={r.dv}<br>term={r.term}<br>coef={r.coef:.3g}<br>p={r.p:.3g}<br>q={getattr(r, 'q_fdr', np.nan):.3g}"
            for r in d.itertuples(index=False)
        ],
        hoverinfo="text",
    ))
    fig.add_vline(x=0)
    fig.update_layout(title=title, xaxis_title="Coefficient (95% CI)", yaxis_title="")
    _save_plot(h, fig, name=name)


def _mm_means_plot(df: pd.DataFrame, dv: str, factor: str, name: str, h: HMD_helper) -> None:
    if px is None:
        return
    if dv not in df.columns or factor not in df.columns or "dataset" not in df.columns:
        return
    d = df[["dataset", factor, dv]].dropna()
    if d.empty:
        return
    grp = d.groupby(["dataset", factor])[dv]
    out = grp.mean().reset_index(name="mean")
    out["sem"] = grp.sem().values
    out["ci_lo"] = out["mean"] - 1.96 * out["sem"]
    out["ci_hi"] = out["mean"] + 1.96 * out["sem"]

    if go is not None:
        fig = go.Figure()
        for ds in out["dataset"].unique():
            sub = out[out["dataset"] == ds].sort_values(factor)
            fig.add_trace(go.Scatter(x=sub[factor], y=sub["mean"], mode="lines+markers", name=str(ds)))
            fig.add_trace(go.Scatter(x=sub[factor], y=sub["ci_hi"], mode="lines",
                                     line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=sub[factor], y=sub["ci_lo"], mode="lines",
                                     line=dict(width=0), fill="tonexty", showlegend=False))
        fig.update_layout(title=f"{dv}: dataset × {factor} (means ±95% CI)", xaxis_title=factor, yaxis_title=dv)
        _save_plot(h, fig, name=name)
    else:
        fig = px.line(out,
                      x=factor,
                      y="mean",
                      color="dataset",
                      color_discrete_map=DATASET_COLOR_MAP,
                      category_orders={"dataset": ["shuffled", "unshuffled"]},
                      markers=True)

        fig.update_layout(title=f"{dv}: dataset × {factor} (means ±95% CI)", yaxis_title=dv)
        _save_plot(h, fig, name=name)


def _mm_make_exposure_var(df: pd.DataFrame, source_col: str, out_col: str) -> pd.DataFrame:
    """Create cumulative prior exposure count within participant for a binary factor.

    The value at trial j is the number of earlier trials for the same participant in which
    `source_col == 1`. This is a prior-exposure count, so the current trial is excluded.
    """
    d = df.copy()
    if source_col not in d.columns or "participant_id" not in d.columns or "trial_index" not in d.columns:
        d[out_col] = np.nan
        return d
    d = d.sort_values(["dataset", "participant_id", "trial_index"]).copy()
    x = pd.to_numeric(d[source_col], errors="coerce").fillna(0.0)
    # prior exposure count, excluding current trial
    d[out_col] = x.groupby([d["dataset"], d["participant_id"]]).cumsum() - x
    return d


TIME_ON_TASK_YLABELS = {
    "trigger_mean": "Mean unsafety",
    "Q3": "Mean Q3",
    "dtrigger_sd": "Mean unsafety volatility",
    "frac_time_unsafe": "Mean fraction time unsafe",
}


def _mm_exposure_curve_plot(df: pd.DataFrame, dv: str, exposure_col: str, name: str, h: HMD_helper) -> None:
    if go is None or dv not in df.columns or exposure_col not in df.columns:
        return
    d = df[["dataset", exposure_col, dv]].dropna().copy()
    if d.empty:
        return
    d[exposure_col] = pd.to_numeric(d[exposure_col], errors="coerce")
    d[dv] = pd.to_numeric(d[dv], errors="coerce")
    d = d.dropna()
    if d.empty:
        return

    grp = d.groupby(["dataset", exposure_col])[dv]
    out = grp.mean().reset_index(name="mean")
    out["sem"] = grp.sem().values
    out["ci_lo"] = out["mean"] - 1.96 * out["sem"]
    out["ci_hi"] = out["mean"] + 1.96 * out["sem"]

    fig = go.Figure()
    for ds in [x for x in ["shuffled", "unshuffled"] if x in out["dataset"].unique()]:
        sub = out[out["dataset"] == ds].sort_values(exposure_col)
        label = DATASET_LABEL_MAP.get(ds, str(ds))
        colour = DATASET_COLOR_MAP.get(ds)
        # Use pointwise error bars instead of a translucent confidence ribbon.
        # In this environment, Plotly EPS export is much more reliable without filled alpha bands.
        err_plus = (sub["ci_hi"] - sub["mean"]).clip(lower=0)
        err_minus = (sub["mean"] - sub["ci_lo"]).clip(lower=0)
        fig.add_trace(go.Scatter(
            x=sub[exposure_col],
            y=sub["mean"],
            mode="lines+markers",
            name=label,
            line=dict(color=colour),
            marker=dict(color=colour),
            error_y=dict(
                type="data",
                array=err_plus,
                arrayminus=err_minus,
                visible=True,
                color=colour,
                thickness=1.2,
                width=3,
            ),
        ))
    fig.update_layout(
        title="",
        xaxis_title="Cumulative prior exposure",
        yaxis_title=TIME_ON_TASK_YLABELS.get(dv, f"Mean {dv}"),
    )

    # Use the same project-wide Plotly saving path as the other figures.
    # This keeps HTML, PNG, EPS sizing, margins and font handling consistent
    # with the rest of the pipeline instead of using the MM5-only EPS helper.
    _save_plot(h, fig, name=name)


def _mm_exposure_forest_plot(coef_df: pd.DataFrame, name: str, h: HMD_helper) -> None:
    """Plot the MM5 exposure interaction forest plot using compact manuscript labels."""
    if go is None or coef_df is None or coef_df.empty:
        return

    d = coef_df.copy()
    if "dv_clean" not in d.columns:
        d["dv_clean"] = d.get("dv_base", d.get("dv", "Outcome"))
    if "term_clean" not in d.columns:
        d["term_clean"] = d.get("exposure_family", d.get("term", "Exposure"))

    # The full labels were too long for a two-column manuscript figure and made
    # the coefficient panel very narrow. Keep the y tick labels compact here;
    # the caption/table text can still explain that these are prior-exposure
    # interaction coefficients.
    outcome_short = {
        "Mean unsafety": "Mean unsafety",
        "Q3": "Q3",
        "Unsafety volatility": "Volatility",
        "trigger_mean": "Mean unsafety",
        "dtrigger_sd": "Volatility",
    }
    exposure_short = {
        "Prior exposure to yielding": "yielding",
        "Prior exposure to eHMI on": "eHMI",
        "yielding": "yielding",
        "eHMI": "eHMI",
    }
    d["outcome_label"] = d["dv_clean"].map(outcome_short).fillna(d["dv_clean"]).astype(str)
    d["exposure_label"] = d["term_clean"].map(exposure_short).fillna(d["term_clean"]).astype(str)
    d["label"] = d["outcome_label"] + " | " + d["exposure_label"]
    d = d.sort_values(["outcome_label", "exposure_label"])

    err_plus = (d["ci_hi"] - d["coef"]).clip(lower=0)
    err_minus = (d["coef"] - d["ci_lo"]).clip(lower=0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d["coef"],
        y=d["label"],
        mode="markers",
        error_x=dict(
            type="data",
            array=err_plus,
            arrayminus=err_minus,
        ),
        hovertext=[
            (
                f"outcome={r.dv_clean}<br>"
                f"exposure={r.term_clean}<br>"
                f"coef={r.coef:.3g}<br>"
                f"p={r.p:.3g}<br>"
                f"q={getattr(r, 'q_fdr', np.nan):.3g}"
            )
            for r in d.itertuples(index=False)
        ],
        hoverinfo="text",
        showlegend=False,
    ))
    fig.add_vline(x=0, line_width=1.5)

    x_lo = float(np.nanmin([d["ci_lo"].min(), 0.0]))
    x_hi = float(np.nanmax([d["ci_hi"].max(), 0.0]))
    pad = max((x_hi - x_lo) * 0.12, 0.05)

    fig.update_layout(
        title="",
        xaxis_title="Coefficient (95% CI)",
        yaxis_title="",
        showlegend=False,
        template="plotly_white",
    )
    fig.update_xaxes(range=[x_lo - pad, x_hi + pad], zeroline=False, automargin=True)
    fig.update_yaxes(categoryorder="array", categoryarray=list(d["label"]), automargin=True)

    _save_plot(h, fig, name=name)


def _mm_make_full_formula(df: pd.DataFrame, dv: str) -> str:
    # Full factorial (may fail; we fallback if needed).
    parts = []
    parts.append("C(dataset)")
    if "yielding" in df.columns:
        parts.append("yielding")
    if "eHMIOn" in df.columns:
        parts.append("eHMIOn")
    if "camera" in df.columns:
        parts.append("C(camera)")
    if "distPed" in df.columns:
        parts.append("C(distPed)")
    rhs = " * ".join(parts)
    return f"{dv} ~ {rhs}"


def _mm_make_stable_formula(df: pd.DataFrame, dv: str) -> str:
    facs = []
    for f in ["yielding", "eHMIOn", "camera", "distPed"]:
        if f in df.columns and df[f].notna().any():
            facs.append(f)

    def F(f: str) -> str:
        return f"C({f})" if f in ["camera", "distPed"] else f

    base = "C(dataset)"
    rhs = [base] + [F(f) for f in facs] + [f"{base}:{F(f)}" for f in facs]

    # include within-factor 2-ways to reduce confounding without exploding
    for i, a in enumerate(facs):
        for b in facs[i+1:]:
            rhs.append(f"{F(a)}:{F(b)}")

    return f"{dv} ~ " + " + ".join(rhs)


def _mm_prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "participant_id" not in d.columns:
        for cand in ["participant", "pid", "Participant", "ParticipantID"]:
            if cand in d.columns:
                d["participant_id"] = d[cand]
                break
    if "trial_index" not in d.columns:
        for cand in ["no", "trial", "trial_no", "trialNumber"]:
            if cand in d.columns:
                d["trial_index"] = d[cand]
                break
    d["trial_index"] = pd.to_numeric(d.get("trial_index", np.nan), errors="coerce")
    d["yielding"] = pd.to_numeric(d.get("yielding", np.nan), errors="coerce")
    d["eHMIOn"] = pd.to_numeric(d.get("eHMIOn", np.nan), errors="coerce")
    return d


def _between_subject_balance_and_sensitivity(
    merged: Optional[pd.DataFrame],
    part_E: Optional[pd.DataFrame],
    out_root: str,
    h: Optional[HMD_helper] = None,
) -> None:
    """Between-subject balance checks.

    This block keeps only the diagnostics that can vary meaningfully in the
    current analysis:
    - participant counts by dataset
    - baseline early-trial outcome summaries and group comparisons
    - optional participant-level E-metric comparisons

    Constant trial-completion and missingness diagnostics were removed because
    they produced uninformative violin plots when all participants had complete
    trial records and outcome availability was constant.
    """
    logger.info("\n=== [C] Between-subject balance checks ===")
    _ensure_dir(out_root)

    if merged is None or merged.empty:
        logger.info("[C] merged trial table missing/empty; skipping between-subject checks.")
        return

    required = [c for c in ["dataset", "participant_id", "video_id"] if c in merged.columns]
    if len(required) < 2:
        logger.info(f"[C] merged missing required id cols (have {required}); skipping.")
        return

    # Participant counts by dataset.
    parts = merged[["dataset", "participant_id"]].drop_duplicates()
    part_counts = parts.groupby("dataset").size().reset_index(name="n_participants")
    part_counts_csv = os.path.join(out_root, "between_subject_participants.csv")
    part_counts.to_csv(part_counts_csv, index=False)
    logger.info(f"[C] wrote: {part_counts_csv} ({len(part_counts)} rows)")

    # Main-trial mask used for baseline summaries. Practice trials are excluded
    # when a trial_index column is available.
    main_mask = pd.Series(True, index=merged.index)
    if "trial_index" in merged.columns:
        main_mask = pd.to_numeric(merged["trial_index"], errors="coerce") >= 2

    trig_mean = _pick_col(merged, ["trigger_mean", "avg_trigger", "mean_trigger"])
    unsafe = _pick_col(merged, ["frac_time_unsafe", "unsafe_time_frac", "frac_unsafe"])
    q3 = "Q3" if "Q3" in merged.columns else None
    yaw_sd = _pick_col(merged, ["yaw_sd", "yaw_sd_deg", "yaw_sd_deg_mean"])
    yaw_fwd = _pick_col(merged, ["yaw_forward_frac_15", "yaw_forward_frac_10"])
    outcome_cols = [c for c in [trig_mean, unsafe, q3, yaw_sd, yaw_fwd] if c is not None and c in merged.columns]

    # Baseline participant means: first K main trials.
    K = 8
    base = merged.loc[main_mask].copy()
    if "trial_index" in base.columns:
        base["trial_index_num"] = pd.to_numeric(base["trial_index"], errors="coerce")
        base = base.sort_values(["dataset", "participant_id", "trial_index_num"])
    else:
        base = base.sort_values(["dataset", "participant_id"])
    base["_rank_trial"] = base.groupby(["dataset", "participant_id"]).cumcount() + 1
    base = base[base["_rank_trial"] <= K]

    base_rows = []
    for (ds, pid), g in base.groupby(["dataset", "participant_id"], dropna=False):
        rec = {"dataset": ds, "participant_id": pid, "baseline_n_trials": int(g["video_id"].nunique())}
        for oc in outcome_cols:
            rec[f"baseline_mean_{oc}"] = float(pd.to_numeric(g[oc], errors="coerce").mean())
        base_rows.append(rec)
    base_pp = pd.DataFrame(base_rows) if base_rows else pd.DataFrame(columns=["dataset", "participant_id"])
    base_csv = os.path.join(out_root, "between_subject_baseline_metrics.csv")
    base_pp.to_csv(base_csv, index=False)
    logger.info(f"[C] wrote: {base_csv} ({len(base_pp)} rows)")

    base_metric_cols = [
        c for c in base_pp.columns
        if c.startswith("baseline_mean_") and c not in ("dataset", "participant_id")
    ]
    if base_metric_cols:
        comp_base = compare_participant_metrics(base_pp, base_metric_cols, fdr=True)
        comp_base_csv = os.path.join(out_root, "between_subject_baseline_comparison.csv")
        comp_base.to_csv(comp_base_csv, index=False)
        logger.info(f"[C] wrote: {comp_base_csv} ({len(comp_base)} rows)")

    if part_E is not None and isinstance(part_E, pd.DataFrame) and (not part_E.empty):
        em_cols = [c for c in part_E.columns if c not in ["dataset", "participant_id"]]
        if em_cols:
            comp_E = compare_participant_metrics(part_E, em_cols, fdr=True)
            comp_E_csv = os.path.join(out_root, "between_subject_participantE_comparison.csv")
            comp_E.to_csv(comp_E_csv, index=False)
            logger.info(f"[C] wrote: {comp_E_csv} ({len(comp_E)} rows)")

    report_lines = []
    report_lines.append("Between-subject balance checks\n")
    report_lines.append("Participant counts by dataset:\n" + _to_string_3dp(part_counts, index=False) + "\n")
    report_lines.append(f"\nBaseline window: first {K} main trials per participant.\n")
    report_lines.append(
        "\nRemoved constant trial-completion and missingness diagnostics from this block; "
        "they are not plotted or written as separate CSV files.\n"
    )
    report_path = os.path.join(out_root, "between_subject_balance_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    logger.info(f"[C] wrote: {report_path}")

    # Baseline outcome violins are retained because these can show meaningful
    # early-session group differences.
    try:
        if px is not None and not base_pp.empty:
            for oc in outcome_cols:
                bc = f"baseline_mean_{oc}"
                if bc in base_pp.columns:
                    figb = px.violin(
                        base_pp,
                        x="dataset",
                        color="dataset",
                        color_discrete_map=DATASET_COLOR_MAP,
                        category_orders={"dataset": ["shuffled", "unshuffled"]},
                        y=bc,
                        box=True,
                        points="all",
                        hover_data=["participant_id"],
                    )
                    figb.update_layout(
                        title=f"Baseline (first {K} main trials) mean: {oc}",
                        yaxis_title=f"baseline mean {oc}",
                    )
                    _save_plot(h, figb, name=f"between_subject_baseline_mean_{oc}")  # type: ignore
    except Exception as e:
        logger.info(f"[C] baseline plotting failed (non-fatal): {e}")


def run_mixed_models_analysis(trial_df: Optional[pd.DataFrame] = None) -> None:
    """Run mixed-effects + robustness analyses using an existing combined trial table."""
    h = HMD_helper()

    if trial_df is None:
        # read from disk (created by the main pipeline)
        p1 = os.path.join(OUTPUT_ROOT, "trigger_trial_features_with_Q123_all.csv")
        p2 = os.path.join(OUTPUT_ROOT, "trigger_trial_features_all.csv")
        if os.path.exists(p1):
            logger.info(f"[MM] reading {p1}")
            trial_df = pd.read_csv(p1)
        elif os.path.exists(p2):
            logger.info(f"[MM] reading {p2}")
            trial_df = pd.read_csv(p2)
        else:
            raise FileNotFoundError(f"[MM] Could not find combined CSV in {OUTPUT_ROOT}. Run the pipeline first.")

    df = _mm_prepare_df(trial_df)
    if "dataset" not in df.columns:
        raise ValueError("[MM] 'dataset' column missing.")
    if "participant_id" not in df.columns:
        raise ValueError("[MM] 'participant_id' column missing.")

    # DVs
    dv_trigger = "trigger_mean" if "trigger_mean" in df.columns else None
    dv_trans = "n_transitions" if "n_transitions" in df.columns else None
    dv_q3 = "Q3" if "Q3" in df.columns else None
    dv_yaw = "yaw_sd" if "yaw_sd" in df.columns else ("yaw_iqr" if "yaw_iqr" in df.columns else None)
    dv_unsafe = "frac_time_unsafe" if "frac_time_unsafe" in df.columns else None

    dvs = [x for x in [dv_trigger, dv_trans, dv_q3, dv_yaw, dv_unsafe] if x is not None]
    logger.info(f"[MM] DVs found: {dvs}")

    work = df.copy()
    if dv_unsafe is not None:
        work["unsafe_logit"] = _mm_logit(pd.to_numeric(work[dv_unsafe], errors="coerce").to_numpy())
    if dv_trans is not None:
        work["trans_log1p"] = np.log1p(pd.to_numeric(work[dv_trans], errors="coerce").to_numpy())

    # ------------------------------------------------------------
    # 1) Primary test: dataset × condition interactions (MixedLM)
    # ------------------------------------------------------------
    logger.info("\n=== [MM1] dataset × condition interactions (MixedLM) ===")
    if not HAVE_SM:
        logger.info("[MM1] statsmodels not installed; skipping MixedLM fits.")
    else:
        coef_rows = []
        tost_rows = []

        for dv in dvs:
            dv_model = dv
            if dv == dv_unsafe:
                dv_model = "unsafe_logit"
            elif dv == dv_trans:
                dv_model = "trans_log1p"

            dfit = work[[dv_model, "dataset", "participant_id", "trial_index",
                         "yielding", "eHMIOn", "camera", "distPed"]].copy()
            dfit = dfit.dropna(subset=[dv_model, "dataset", "participant_id"])
            if dfit.empty:
                continue

            full_formula = _mm_make_full_formula(dfit, dv_model)
            stable_formula = _mm_make_stable_formula(dfit, dv_model)

            logger.info(f"\n[MM1] DV={dv} (model DV={dv_model})")
            logger.info(f"  try full:   {full_formula}")
            res = _mm_fit_mixedlm(full_formula, dfit, group_col="participant_id", re_formula="1")
            used_formula = "full"
            if res is None:
                logger.info(f"  -> fallback stable: {stable_formula}")
                res = _mm_fit_mixedlm(stable_formula, dfit, group_col="participant_id", re_formula="1")
                used_formula = "stable"

            if res is None:
                continue

            # keep dataset interaction terms
            ct = _mm_extract_fixed_effects(res, dv=dv, model=f"MixedLM_{used_formula}",
                                           keep_regex=r"^C\(dataset\)\[T\.unshuffled\](:|$)")
            if not ct.empty:
                coef_rows.append(ct)

            # TOST on dataset×yielding if available
            term = "C(dataset)[T.unshuffled]:yielding"
            if term in (ct["term"].tolist() if not ct.empty else []):
                row = ct[ct["term"] == term].iloc[0]
                sd = float(pd.to_numeric(work[dv], errors="coerce").std(skipna=True)) if dv in work.columns else np.nan
                delta = SESOI_SD_MULT * sd if (sd and not np.isnan(sd)) else np.nan
                tost = _mm_tost_equivalence(float(row["coef"]), float(row["se"]), float(delta))
                tost_rows.append({
                    "dv": dv,
                    "term": term,
                    "coef": float(row["coef"]),
                    "se": float(row["se"]),
                    "ci_lo": float(row["ci_lo"]),
                    "ci_hi": float(row["ci_hi"]),
                    "delta": tost["delta"],
                    "p_tost": tost["p_tost"],
                    "p_lower": tost["p_lower"],
                    "p_upper": tost["p_upper"],
                })

        if coef_rows:
            coef_df = pd.concat(coef_rows, ignore_index=True)
            coef_df["q_fdr"] = _mm_bh_fdr(coef_df["p"].to_numpy(dtype=float))
            out_path = os.path.join(OUTPUT_ROOT, "MM1_mixedlm_datasetX_terms.csv")
            coef_df.to_csv(out_path, index=False)
            _print_table(coef_df.sort_values("q_fdr").head(25),
                         title="=== [MM1] Top dataset interaction terms (BH-FDR) ===", max_rows=25)
            logger.info(f"[MM1] wrote: {out_path}")

            _mm_forest_plot(coef_df, title="Dataset interaction coefficients (unshuffled vs shuffled)",
                            name="MM1_forest_dataset_interactions", h=h)

        # Descriptive interaction mean plots
        for dv in dvs:
            _mm_means_plot(df, dv, "yielding", name=f"MM1_means_{dv}_dataset_yielding", h=h)
            _mm_means_plot(df, dv, "eHMIOn", name=f"MM1_means_{dv}_dataset_eHMIOn", h=h)

        if tost_rows:
            tost_df = pd.DataFrame(tost_rows)
            tost_df["q_fdr"] = _mm_bh_fdr(tost_df["p_tost"].to_numpy(dtype=float))
            out_path = os.path.join(OUTPUT_ROOT, "MM4_equivalence_tost_datasetXyielding.csv")
            tost_df.to_csv(out_path, index=False)
            _print_table(tost_df.sort_values("q_fdr").head(25),
                         title="=== [MM4] TOST equivalence on dataset×yielding ===", max_rows=25)
            logger.info(f"[MM4] wrote: {out_path}")

            if go is not None:
                for dv in tost_df["dv"].unique():
                    r = tost_df[tost_df["dv"] == dv].iloc[0]
                    fig = go.Figure()
                    fig.add_hrect(y0=-r["delta"], y1=r["delta"], opacity=0.15, line_width=0)
                    fig.add_trace(go.Scatter(
                        x=[0],
                        y=[r["coef"]],
                        mode="markers",
                        error_y=dict(type="data", array=[r["ci_hi"] - r["coef"]], arrayminus=[r["coef"] - r["ci_lo"]]),
                        hovertext=f"coef={r['coef']:.3g}<br>p_tost={r['p_tost']:.3g}<br>delta={r['delta']:.3g}",
                        hoverinfo="text",
                    ))
                    fig.add_hline(y=0)
                    fig.update_layout(title=f"Equivalence: dataset×yielding ({dv})", yaxis_title="Interaction coefficient")  # noqa: E501
                    _save_plot(h, fig, name=f"MM4_equivalence_{dv}")

    # ------------------------------------------------------------
    # 2) Learning model: dataset × trial_index
    # ------------------------------------------------------------
    logger.info("\n=== [MM2] dataset × trial_index (learning) ===")
    if "trial_index" not in df.columns or df["trial_index"].isna().all():
        logger.info("[MM2] trial_index missing; skipping.")
    elif not HAVE_SM:
        logger.info("[MM2] statsmodels not installed; skipping.")
    else:
        learn_rows = []
        for dv in dvs:
            dv_model = dv
            if dv == dv_unsafe:
                dv_model = "unsafe_logit"
            elif dv == dv_trans:
                dv_model = "trans_log1p"

            dfit = work[[dv_model, "dataset", "participant_id", "trial_index"]].dropna()
            if dfit.empty:
                continue

            formula = f"{dv_model} ~ C(dataset) * trial_index"
            logger.info(f"[MM2] DV={dv} formula: {formula}")
            res = _mm_fit_mixedlm(formula, dfit, group_col="participant_id", re_formula="1 + trial_index")
            if res is None:
                continue
            ct = _mm_extract_fixed_effects(res, dv=dv, model="MixedLM_learning",
                                           keep_regex=r"^C\(dataset\)\[T\.unshuffled\]:trial_index$")
            if not ct.empty:
                learn_rows.append(ct)

            # Plot mean curve by trial_index (descriptive)
            if px is not None and dv in df.columns:
                tmp = df[["dataset", "trial_index", dv]].dropna()
                if not tmp.empty:
                    grp = tmp.groupby(["dataset", "trial_index"])[dv].mean().reset_index()
                    grp["trial_num"] = _trial_num_display(grp["trial_index"])
                    fig = px.line(grp,
                                  x="trial_num",
                                  y=dv,
                                  color="dataset",
                                  color_discrete_map=DATASET_COLOR_MAP,
                                  category_orders={"dataset": ["shuffled", "unshuffled"]},
                                  markers=True)
                    fig.update_layout(title=f"{dv}: mean over trial number (dataset)", xaxis_title="Trial number")
                    _save_plot(h, fig, name=f"MM2_curve_{dv}_over_trial_index")

        if learn_rows:
            learn_df = pd.concat(learn_rows, ignore_index=True)
            learn_df["q_fdr"] = _mm_bh_fdr(learn_df["p"].to_numpy(dtype=float))
            out_path = os.path.join(OUTPUT_ROOT, "MM2_mixedlm_datasetXtrial_terms.csv")
            learn_df.to_csv(out_path, index=False)
            _print_table(learn_df.sort_values("q_fdr").head(25), title="=== [MM2] dataset×trial_index terms (BH-FDR) ===", max_rows=25)  # noqa:E501
            logger.info(f"[MM2] wrote: {out_path}")
            _mm_forest_plot(learn_df, title="Dataset × trial_index interaction", name="MM2_forest_dataset_trial", h=h)

    # ------------------------------------------------------------
    # 5) Exposure-based learning: dataset × prior exposure
    # ------------------------------------------------------------
    logger.info("\n=== [MM5] dataset × prior exposure (sensitivity) ===")
    if "trial_index" not in df.columns or df["trial_index"].isna().all():
        logger.info("[MM5] trial_index missing; skipping.")
    elif not HAVE_SM:
        logger.info("[MM5] statsmodels not installed; skipping.")
    else:
        exp_work = work.copy()
        exp_work = _mm_make_exposure_var(exp_work, "yielding", "exposure_yielding")
        exp_work = _mm_make_exposure_var(exp_work, "eHMIOn", "exposure_eHMI")

        mm5_targets = []
        if dv_trigger is not None:
            mm5_targets.append((dv_trigger, dv_trigger))
        if dv_q3 is not None:
            mm5_targets.append((dv_q3, dv_q3))
        if "dtrigger_sd" in exp_work.columns:
            mm5_targets.append(("dtrigger_sd", "dtrigger_sd"))

        exp_rows = []
        plot_done = set()
        for dv, dv_model in mm5_targets:
            for exposure_col in ["exposure_yielding", "exposure_eHMI"]:
                cols = [dv_model, "dataset", "participant_id", exposure_col]
                dfit = exp_work[cols].copy().dropna()
                if dfit.empty:
                    continue
                formula = f"{dv_model} ~ C(dataset) * {exposure_col}"
                logger.info(f"[MM5] DV={dv} formula: {formula}")
                res = _mm_fit_mixedlm(formula, dfit, group_col="participant_id", re_formula="1 + " + exposure_col)
                if res is None:
                    res = _mm_fit_mixedlm(formula, dfit, group_col="participant_id", re_formula="1")
                if res is None:
                    continue
                ct = _mm_extract_fixed_effects(
                    res, dv=f"{dv}|{exposure_col}", model="MixedLM_exposure",
                    keep_regex=rf"^C\(dataset\)\[T\.unshuffled\]:{exposure_col}$"
                )
                if not ct.empty:
                    ct["dv_base"] = dv
                    ct["exposure_family"] = exposure_col.replace("exposure_", "")
                    exp_rows.append(ct)

                if dv == "Q3":
                    if exposure_col == "exposure_yielding" and "MM5_curve_Q3_exposure_yielding" not in plot_done:
                        _mm_exposure_curve_plot(exp_work, dv="Q3", exposure_col=exposure_col,
                                                name="MM5_curve_Q3_exposure_yielding", h=h)
                        plot_done.add("MM5_curve_Q3_exposure_yielding")
                    elif exposure_col == "exposure_eHMI" and "MM5_curve_Q3_exposure_eHMI" not in plot_done:
                        _mm_exposure_curve_plot(exp_work, dv="Q3", exposure_col=exposure_col,
                                                name="MM5_curve_Q3_exposure_eHMI", h=h)
                        plot_done.add("MM5_curve_Q3_exposure_eHMI")

        if exp_rows:
            exp_df = pd.concat(exp_rows, ignore_index=True)
            exp_df["q_fdr"] = _mm_bh_fdr(exp_df["p"].to_numpy(dtype=float))
            # cleaner labels for the forest plot and export
            dv_label_map = {
                "trigger_mean": "Mean unsafety",
                "Q3": "Q3",
                "dtrigger_sd": "Unsafety volatility",
            }
            exposure_label_map = {
                "yielding": "Prior exposure to yielding",
                "eHMI": "Prior exposure to eHMI on",
            }
            exp_df["dv_clean"] = exp_df["dv_base"].map(dv_label_map).fillna(exp_df["dv_base"])
            exp_df["term_clean"] = exp_df["exposure_family"].map(exposure_label_map).fillna(exp_df["exposure_family"])
            out_path = os.path.join(OUTPUT_ROOT, "MM5_mixedlm_datasetXexposure_terms.csv")
            exp_df.to_csv(out_path, index=False)
            _print_table(exp_df.sort_values("q_fdr").head(25),
                         title="=== [MM5] Dataset×exposure interaction terms (BH-FDR) ===", max_rows=25)
            logger.info(f"[MM5] wrote: {out_path}")

            _mm_exposure_forest_plot(exp_df, name="MM5_forest_exposure_interactions", h=h)

    # ------------------------------------------------------------
    # 3) Sequential effects: DV_t ~ lag1 + switch + dataset interactions
    # ------------------------------------------------------------
    logger.info("\n=== [MM3] Sequential effects (lag/switch) ===")
    if not HAVE_SM:
        logger.info("[MM3] statsmodels not installed; skipping.")
    else:
        seq_rows = []
        for dv in dvs:
            dv_model = dv
            if dv == dv_unsafe:
                dv_model = "unsafe_logit"
            elif dv == dv_trans:
                dv_model = "trans_log1p"

            sdf = work[[dv_model, "dataset", "participant_id", "trial_index", "yielding",
                        "eHMIOn", "camera", "distPed"]].copy()
            sdf = sdf.dropna(subset=[dv_model, "dataset", "participant_id", "trial_index"])
            if sdf.empty:
                continue
            sdf = sdf.sort_values(["dataset", "participant_id", "trial_index"])
            sdf["lag1"] = sdf.groupby(["dataset", "participant_id"])[dv_model].shift(1)
            sdf["prev_yielding"] = sdf.groupby(["dataset", "participant_id"])["yielding"].shift(1)
            sdf["switch"] = (sdf["yielding"] != sdf["prev_yielding"]).astype(float) if "yielding" in sdf.columns else np.nan  # noqa: E501
            sdf["prev_eHMIOn"] = sdf.groupby(["dataset", "participant_id"])["eHMIOn"].shift(1) if "eHMIOn" in sdf.columns else np.nan  # noqa: E501
            sdf["prev_camera"] = sdf.groupby(["dataset", "participant_id"])["camera"].shift(1) if "camera" in sdf.columns else np.nan  # noqa: E501
            sdf["prev_distPed"] = sdf.groupby(["dataset", "participant_id"])["distPed"].shift(1) if "distPed" in sdf.columns else np.nan  # noqa: E501
            sdf = sdf.dropna(subset=["lag1"])
            if sdf.empty:
                continue

            rhs = ["C(dataset)", "lag1", "C(dataset):lag1"]
            if sdf["switch"].notna().any():
                rhs += ["switch", "C(dataset):switch"]
            for f in ["yielding", "eHMIOn", "camera", "distPed"]:
                if f in sdf.columns and sdf[f].notna().any():
                    rhs.append(f"C({f})" if f in ["camera", "distPed"] else f)
            for f in ["prev_yielding", "prev_eHMIOn", "prev_camera", "prev_distPed"]:
                if f in sdf.columns and sdf[f].notna().any():
                    rhs.append(f"C({f})" if f in ["prev_camera", "prev_distPed"] else f)

            formula = f"{dv_model} ~ " + " + ".join(rhs)
            logger.info(f"[MM3] DV={dv} formula: {formula}")
            res = _mm_fit_mixedlm(formula, sdf, group_col="participant_id", re_formula="1")
            if res is None:
                continue
            ct = _mm_extract_fixed_effects(res, dv=dv, model="MixedLM_sequential",
                                           keep_regex=r"^C\(dataset\)\[T\.unshuffled\]:(lag1|switch)$")
            if not ct.empty:
                seq_rows.append(ct)

            # Scatter lag plot (descriptive)
            if px is not None:
                tmp = sdf[["dataset", "lag1", dv_model]].dropna()
                if not tmp.empty:
                    fig = px.scatter(tmp,
                                     x="lag1",
                                     y=dv_model,
                                     color="dataset",
                                     color_discrete_map=DATASET_COLOR_MAP,
                                     category_orders={"dataset": ["shuffled", "unshuffled"]},
                                     trendline="ols")
                    fig.update_layout(title=f"{dv}: DV_t vs DV_(t-1) (by dataset)")
                    _save_plot(h, fig, name=f"MM3_scatter_lag_{dv}")

        if seq_rows:
            seq_df = pd.concat(seq_rows, ignore_index=True)
            seq_df["q_fdr"] = _mm_bh_fdr(seq_df["p"].to_numpy(dtype=float))
            out_path = os.path.join(OUTPUT_ROOT, "MM3_mixedlm_sequential_dataset_interactions.csv")
            seq_df.to_csv(out_path, index=False)
            _print_table(seq_df.sort_values("q_fdr").head(25),
                         title="=== [MM3] Sequential dataset interactions (BH-FDR) ===", max_rows=25)
            logger.info(f"[MM3] wrote: {out_path}")
            _mm_forest_plot(seq_df, title="Sequential dataset interactions (lag/switch)",
                            name="MM3_forest_sequential", h=h)

    logger.info(f"\n[MM] Done. Plots saved to: {_get_output_dir_for_logs()}")

    # -------------------------------------------------------------------
    # C: Between-subject dataset balance + sensitivity analyses
    # -------------------------------------------------------------------
    # NOTE: run_mixed_models_analysis() may be executed standalone (e.g., --mode mixed)
    # and therefore does NOT necessarily have the in-memory `merged` trial table.
    # We load it from disk if needed.
    try:
        merged_for_C = None
        # Prefer the most feature-complete file if present
        cand_paths = [
            os.path.join(OUTPUT_ROOT, "trigger_trial_features_all_with_yaw.csv"),
            os.path.join(OUTPUT_ROOT, "trigger_trial_features_all.csv"),
            os.path.join(OUTPUT_ROOT, "trial_features_all_with_yaw.csv"),
            os.path.join(OUTPUT_ROOT, "trial_features_all.csv"),
        ]
        for p in cand_paths:
            if os.path.exists(p):
                try:
                    merged_for_C = pd.read_csv(p)
                    break
                except Exception:
                    merged_for_C = None

        part_E_for_C = None
        # participant-level summary table (optional)
        cand_partE = [
            os.path.join(OUTPUT_ROOT, "participant_E_metrics.csv"),
            os.path.join(OUTPUT_ROOT, "participant_metrics_E.csv"),
        ]
        for p in cand_partE:
            if os.path.exists(p):
                try:
                    part_E_for_C = pd.read_csv(p)
                    break
                except Exception:
                    part_E_for_C = None

        if merged_for_C is None or merged_for_C.empty:
            logger.warning("[C] Between-subject block: could not load merged trial table from disk; skipping.")
        else:
            _between_subject_balance_and_sensitivity(
                merged=merged_for_C,
                part_E=part_E_for_C,
                out_root=OUTPUT_ROOT,
                h=h,
            )
    except Exception as e:
        logger.error(f"[C] Between-subject balance/sensitivity block failed (non-fatal): {e}")

    # Optional: open a single index page linking all plots
    # Set CSU_OPEN_PLOT_INDEX=1 if you want this behaviour
    if str(os.environ.get('CSU_OPEN_PLOT_INDEX', '')).strip().lower() in {'1', 'true', 'yes', 'y'}:
        _write_plot_index_and_open(h)


# ---------------------------------------------------------------------------
# Class-based pipeline wrapper
# ---------------------------------------------------------------------------

class MixedModelsPipeline:
    """OO wrapper around `run_mixed_models_analysis()`.

    Keeps OUTPUT_ROOT consistent with the entrypoint.
    """

    def __init__(self, output_root: str = "_compare_output") -> None:
        self.output_root = output_root

    def run(self, trial_df: Optional[pd.DataFrame] = None) -> None:
        global OUTPUT_ROOT
        OUTPUT_ROOT = self.output_root
        run_mixed_models_analysis(trial_df=trial_df)
