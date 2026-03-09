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
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

import plotly.express as px  # noqa:F401
import plotly.graph_objects as go  # noqa:F401


# Statsmodels is optional; used for mixed-effects / robustness models.
import statsmodels.formula.api as smf  # noqa:F401
import statsmodels.api as sm  # noqa:F401
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
    compare_participant_metrics,
    compare_shuffled_unshuffled,
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
            f"dv={r.dv}<br>term={r.term}<br>coef={r.coef:.4g}<br>p={r.p:.3g}<br>q={getattr(r,'q_fdr',np.nan):.3g}"
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
        fig = px.line(out, x=factor, y="mean", color="dataset", markers=True)
        fig.update_layout(title=f"{dv}: dataset × {factor} (means ±95% CI)", yaxis_title=dv)
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
    h: Optional[Any] = None,
) -> None:
    """Between-subject balance + sensitivity checks.

    Rationale:
      Shuffled vs unshuffled are *between-subject* datasets. Even with identical within-subject designs,
      dataset differences can be distorted by differences in participant mix (dropout, missingness, etc.).
    What we do:
      - Participant counts and trial completion per dataset
      - Per-participant missingness for core outcomes
      - Baseline (early-trial) outcome comparison as a balance check
      - Sensitivity: repeat baseline comparisons after filtering low-completion / high-missingness participants
    Outputs (written to out_root):
      - between_subject_participants.csv
      - between_subject_trials_per_participant.csv
      - between_subject_missingness_per_participant.csv
      - between_subject_baseline_metrics.csv
      - between_subject_baseline_comparison.csv
      - between_subject_sensitivity_baseline_comparison.csv
      - between_subject_balance_report.txt
    """
    logger.info("\n=== [C] Between-subject balance & sensitivity ===")
    _ensure_dir(out_root)

    if merged is None or merged.empty:
        logger.info("[C] merged trial table missing/empty; skipping between-subject checks.")
        return

    # --- core identifiers ---
    required = [c for c in ["dataset", "participant_id", "video_id"] if c in merged.columns]
    if len(required) < 2:
        logger.info(f"[C] merged missing required id cols (have {required}); skipping.")
        return

    # --- participant counts ---
    parts = merged[["dataset", "participant_id"]].drop_duplicates()
    part_counts = parts.groupby("dataset").size().reset_index(name="n_participants")
    part_counts_csv = os.path.join(out_root, "between_subject_participants.csv")
    part_counts.to_csv(part_counts_csv, index=False)
    logger.info(f"[C] wrote: {part_counts_csv} ({len(part_counts)} rows)")

    # --- trial completion per participant (unique video_id) ---
    main_mask = pd.Series(True, index=merged.index)
    if "trial_index" in merged.columns:
        # By convention mapping.csv has 2 practice trials first; treat trial_index<2 as practice
        main_mask = pd.to_numeric(merged["trial_index"], errors="coerce") >= 2

    trials_pp = (
        merged.loc[main_mask]
        .groupby(["dataset", "participant_id"])["video_id"]
        .nunique()
        .reset_index(name="n_trials_main")
    )
    # estimate expected trials using the per-dataset median (robust to a few dropouts)
    expected_main = trials_pp.groupby("dataset")["n_trials_main"].median().rename("expected_main_trials").reset_index()
    trials_pp = trials_pp.merge(expected_main, on="dataset", how="left")
    trials_pp["completion_frac"] = trials_pp["n_trials_main"] / trials_pp["expected_main_trials"].replace(0, np.nan)

    trials_csv = os.path.join(out_root, "between_subject_trials_per_participant.csv")
    trials_pp.to_csv(trials_csv, index=False)
    logger.info(f"[C] wrote: {trials_csv} ({len(trials_pp)} rows)")

    # --- per-participant missingness for key outcomes ---
    # pick candidate outcomes that should exist across datasets
    trig_mean = _pick_col(merged, ["trigger_mean", "avg_trigger", "mean_trigger"])
    unsafe = _pick_col(merged, ["frac_time_unsafe", "unsafe_time_frac", "frac_unsafe"])
    q3 = "Q3" if "Q3" in merged.columns else None
    yaw_sd = _pick_col(merged, ["yaw_sd", "yaw_sd_deg", "yaw_sd_deg_mean"])
    yaw_fwd = _pick_col(merged, ["yaw_forward_frac_15", "yaw_forward_frac_10"])

    outcome_cols = [c for c in [trig_mean, unsafe, q3, yaw_sd, yaw_fwd] if c is not None and c in merged.columns]
    miss_rows = []
    for (ds, pid), g in merged.loc[main_mask].groupby(["dataset", "participant_id"], dropna=False):
        rec = {"dataset": ds, "participant_id": pid, "n_trials_main": int(g["video_id"].nunique())}
        for oc in outcome_cols:
            rec[f"missing_frac_{oc}"] = float(g[oc].isna().mean())
        miss_rows.append(rec)
    miss_pp = pd.DataFrame(miss_rows) if miss_rows else pd.DataFrame(columns=["dataset", "participant_id"])
    miss_csv = os.path.join(out_root, "between_subject_missingness_per_participant.csv")
    miss_pp.to_csv(miss_csv, index=False)
    logger.info(f"[C] wrote: {miss_csv} ({len(miss_pp)} rows)")

    # --- baseline (early) participant means: first K main trials ---
    K = 8
    base = merged.loc[main_mask].copy()
    if "trial_index" in base.columns:
        base["trial_index_num"] = pd.to_numeric(base["trial_index"], errors="coerce")
        # early K main trials per participant in their presented order
        base = base.sort_values(["dataset", "participant_id", "trial_index_num"])
        base["_rank_trial"] = base.groupby(["dataset", "participant_id"]).cumcount() + 1
        base = base[base["_rank_trial"] <= K]
    else:
        # if no trial_index, just take first K rows per participant as they appear
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

    # baseline comparisons (balance check)
    base_metric_cols = [c for c in base_pp.columns if c.startswith("baseline_mean_") and c not in ("dataset",
                                                                                                   "participant_id")]
    if base_metric_cols:
        comp_base = compare_participant_metrics(base_pp, base_metric_cols, fdr=True)
        comp_base_csv = os.path.join(out_root, "between_subject_baseline_comparison.csv")
        comp_base.to_csv(comp_base_csv, index=False)
        logger.info(f"[C] wrote: {comp_base_csv} ({len(comp_base)} rows)")

    # --- sensitivity: filter low completion / high missingness then rerun baseline compare ---
    # thresholds based on robust dataset-wise summaries
    # completion: keep >= 80% of expected (dataset median)
    keep = trials_pp["completion_frac"].notna() & (trials_pp["completion_frac"] >= 0.80)
    keep_ids = trials_pp.loc[keep, ["dataset", "participant_id"]]

    base_filt = base_pp.merge(keep_ids, on=["dataset", "participant_id"], how="inner")
    # missingness filter: if we have trigger_mean, require <= 30% missing (baseline table is means; filter on miss_pp)
    if trig_mean is not None and (f"missing_frac_{trig_mean}" in miss_pp.columns):
        miss_keep = miss_pp[["dataset", "participant_id", f"missing_frac_{trig_mean}"]].copy()
        miss_keep = miss_keep[miss_keep[f"missing_frac_{trig_mean}"] <= 0.30]
        base_filt = base_filt.merge(miss_keep[["dataset",
                                               "participant_id"]], on=["dataset", "participant_id"], how="inner")

    sens_metric_cols = [c for c in base_filt.columns if c.startswith("baseline_mean_")]
    if sens_metric_cols and len(base_filt) >= 10:
        comp_sens = compare_participant_metrics(base_filt, sens_metric_cols, fdr=True)
        comp_sens_csv = os.path.join(out_root, "between_subject_sensitivity_baseline_comparison.csv")
        comp_sens.to_csv(comp_sens_csv, index=False)
        logger.info(f"[C] wrote: {comp_sens_csv} ({len(comp_sens)} rows)")

    # --- optional: incorporate participant E-metrics (learning/sequential) into balance report ---
    if part_E is not None and isinstance(part_E, pd.DataFrame) and (not part_E.empty):
        em_cols = [c for c in part_E.columns if c not in ["dataset", "participant_id"]]
        # quick check: do the participant E metrics show extreme group imbalance due to dropouts?
        if em_cols:
            comp_E = compare_participant_metrics(part_E, em_cols, fdr=True)
            comp_E_csv = os.path.join(out_root, "between_subject_participantE_comparison.csv")
            comp_E.to_csv(comp_E_csv, index=False)
            logger.info(f"[C] wrote: {comp_E_csv} ({len(comp_E)} rows)")

    # --- report text ---
    report_lines = []
    report_lines.append("Between-subject balance & sensitivity checks\n")
    report_lines.append("Participant counts by dataset:\n" + part_counts.to_string(index=False) + "\n")
    report_lines.append("\nTrial completion summary (main trials):\n")
    if not trials_pp.empty:
        summ = trials_pp.groupby("dataset")["n_trials_main"].agg(["count", "mean", "std", "min", "median", "max"])
        report_lines.append(summ.to_string() + "\n")
    if not miss_pp.empty and outcome_cols:
        report_lines.append("\nPer-participant missingness (means by dataset):\n")
        miss_cols = [c for c in miss_pp.columns if c.startswith("missing_frac_")]
        miss_summ = miss_pp.groupby("dataset")[miss_cols].mean()
        report_lines.append(miss_summ.to_string() + "\n")
    report_lines.append(f"\nBaseline window: first {K} main trials per participant.\n")
    report_path = os.path.join(out_root, "between_subject_balance_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    logger.info(f"[C] wrote: {report_path}")

    # --- plots (best-effort; do not fail the pipeline) ---
    try:
        if px is not None:
            # completion violin
            if not trials_pp.empty:
                fig = px.violin(trials_pp, x="dataset", color="dataset", y="n_trials_main",
                                box=True, points="all", hover_data=["participant_id"])
                fig.update_layout(title="Trial completion per participant (main trials)", yaxis_title="# main trials")
                _save_plot(h, fig, name="between_subject_trial_completion_violin")

                fig2 = px.violin(trials_pp, x="dataset", color="dataset", y="completion_frac",
                                 box=True, points="all", hover_data=["participant_id"])
                fig2.update_layout(title="Completion fraction per participant (vs dataset median)",
                                   yaxis_title="completion fraction")
                _save_plot(h, fig2, name="between_subject_completion_fraction_violin")

            # missingness violin for key outcomes
            if not miss_pp.empty:
                for oc in outcome_cols:
                    mc = f"missing_frac_{oc}"
                    if mc in miss_pp.columns:
                        figm = px.violin(miss_pp, x="dataset", color="dataset", y=mc, box=True,
                                         points="all", hover_data=["participant_id"])
                        figm.update_layout(title=f"Missingness per participant: {oc}", yaxis_title="missing fraction")
                        _save_plot(h, figm, name=f"between_subject_missingness_{oc}")

            # baseline outcome violin
            if not base_pp.empty:
                for oc in outcome_cols:
                    bc = f"baseline_mean_{oc}"
                    if bc in base_pp.columns:
                        figb = px.violin(base_pp, x="dataset", color="dataset", y=bc, box=True,
                                         points="all", hover_data=["participant_id"])
                        figb.update_layout(title=f"Baseline (first {K} main trials) mean: {oc}",
                                           yaxis_title=f"baseline mean {oc}")
                        _save_plot(h, figb, name=f"between_subject_baseline_mean_{oc}")
    except Exception as e:
        logger.info(f"[C] plotting failed (non-fatal): {e}")


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
                        hovertext=f"coef={r['coef']:.4g}<br>p_tost={r['p_tost']:.3g}<br>delta={r['delta']:.4g}",
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
                    fig = px.line(grp, x="trial_num", y=dv, color="dataset", markers=True)
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
                    fig = px.scatter(tmp, x="lag1", y=dv_model, color="dataset", trendline="ols")
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
