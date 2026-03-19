"""Compare shuffled vs unshuffled datasets.
Entrypoint for the refactored codebase.
1) Update the six paths below (3 shuffled + 3 unshuffled).
2) Run:
    python3 analysis.py
   Force a full rebuild even if a cache exists:
    python3 analysis.py --reanalyse
This script always runs:
- the A to F comparison pipeline
- the mixed models analysis
- compact paper oriented digests in the console and as text and CSV files
Modules
-------
- csu_core.py        : shared stats/CSV/plot helpers
- csu_features.py    : feature extraction helpers (trigger, yaw, Q1 to Q3)
- csu_pipeline.py    : A to F pipeline (reporting + plots)
- csu_mixed_models.py: mixed models / sensitivity analyses
"""
from __future__ import annotations
import argparse
import os
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import pandas as pd
import common
from custom_logger import CustomLogger
from logmod import logs
import csu_pipeline as pipeline
import csu_mixed_models as mm


logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)


MAPPING_CSV = common.get_configs("mapping")
OUTPUT_ROOT = common.get_configs("output")
PAPER_LOG_TOP_N = int(os.environ.get("CSU_PAPER_LOG_TOP_N", "8"))


def _read_csv_if_exists(path: str) -> Optional[pd.DataFrame]:
    if not path or not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        logger.error(f"[Paper] Failed reading {path}: {e}")
        return None


def _first_existing(df: pd.DataFrame, cols: List[str]) -> Optional[str]:
    for c in cols:
        if c in df.columns:
            return c
    return None


def _fmt_num(val: object, digits: int = 3) -> str:
    try:
        f = float(val)  # pyright: ignore[reportArgumentType]
    except Exception:
        return "n/a"
    if np.isnan(f):
        return "n/a"
    return f"{f:.{digits}f}"


def _sort_results(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    qcol = _first_existing(out, ["q_fdr", "q_value", "p_fdr"])
    pcol = _first_existing(out, ["p", "p_value"])
    sort_cols: List[str] = []
    asc: List[bool] = []
    if qcol is not None:
        sort_cols.append(qcol)
        asc.append(True)
    if pcol is not None and pcol not in sort_cols:
        sort_cols.append(pcol)
        asc.append(True)
    if sort_cols:
        out = out.sort_values(sort_cols, ascending=asc, na_position="last")
    return out


def _sig_counts(df: pd.DataFrame) -> Tuple[int, int, Optional[str], Optional[str]]:
    qcol = _first_existing(df, ["q_fdr", "q_value", "p_fdr"])
    pcol = _first_existing(df, ["p", "p_value", "p_tost"])
    qsig = 0
    psig = 0
    if qcol is not None:
        qsig = int(pd.to_numeric(df[qcol], errors="coerce").lt(0.05).sum())
    if pcol is not None:
        psig = int(pd.to_numeric(df[pcol], errors="coerce").lt(0.05).sum())
    return qsig, psig, qcol, pcol


def _extra_group_bits(row: pd.Series) -> str:
    skip = {
        "source", "unit", "metric", "dv", "model", "term",
        "n_shuffled", "n_unshuffled", "mean_shuffled", "mean_unshuffled",
        "sd_shuffled", "sd_unshuffled", "t", "t_stat", "p", "p_value",
        "p_fdr", "q_fdr", "q_value", "test", "note", "cohens_d",
        "coef", "se", "ci_lo", "ci_hi", "delta", "p_tost", "p_lower",
        "p_upper", "OR", "OR_lo", "OR_hi", "rank_within_source",
    }
    bits = []
    for col, val in row.items():
        if col in skip:
            continue
        if pd.isna(val):
            continue
        bits.append(f"{col}={val}")
    return "; ".join(bits)


def _render_compare_row(row: pd.Series) -> str:
    metric = str(row.get("metric", "metric"))
    context = _extra_group_bits(row)
    means = f"shuffled={_fmt_num(row.get('mean_shuffled'))}, unshuffled={_fmt_num(row.get('mean_unshuffled'))}"
    stats = f"d={_fmt_num(row.get('cohens_d'))}, p={_fmt_num(row.get('p_value', row.get('p')))}, q={_fmt_num(row.get('q_value', row.get('p_fdr')))}"  # noqa:E501
    if context:
        return f"- {metric} [{context}] | {means} | {stats}"
    return f"- {metric} | {means} | {stats}"


def _render_mixed_row(row: pd.Series) -> str:
    dv = str(row.get("dv", "dv"))
    term = str(row.get("term", "term"))
    model = str(row.get("model", ""))
    context = f"{dv} | {term}"
    if model and model != "nan":
        context += f" | {model}"
    return (
        f"- {context} | coef={_fmt_num(row.get('coef'))}, 95% CI [{_fmt_num(row.get('ci_lo'))}, {_fmt_num(row.get('ci_hi'))}]"  # noqa:E501
        f" | p={_fmt_num(row.get('p'))}, q={_fmt_num(row.get('q_fdr'))}"
    )


def _render_tost_row(row: pd.Series) -> str:
    dv = str(row.get("dv", "dv"))
    equivalent = pd.to_numeric(pd.Series([row.get("p_tost")]), errors="coerce").lt(0.05).iloc[0]
    verdict = "equivalent" if bool(equivalent) else "not equivalent"
    return (
        f"- {dv} | coef={_fmt_num(row.get('coef'))}, 95% CI [{_fmt_num(row.get('ci_lo'))}, {_fmt_num(row.get('ci_hi'))}]"  # noqa:E501
        f" | delta={_fmt_num(row.get('delta'))}, p_tost={_fmt_num(row.get('p_tost'))}, q={_fmt_num(row.get('q_fdr'))}"
        f" | {verdict}"
    )


def _log_and_collect(lines: List[str], msg: str) -> None:
    lines.append(msg)
    logger.info(msg)


def _append_table_preview(lines: List[str], label: str, df: Optional[pd.DataFrame],
                          renderer, top_n: int = PAPER_LOG_TOP_N) -> pd.DataFrame:
    if df is None or df.empty:
        _log_and_collect(lines, f"[Paper] {label}: no rows found")
        return pd.DataFrame()
    ordered = _sort_results(df)
    qsig, psig, qcol, pcol = _sig_counts(ordered)
    _log_and_collect(
        lines,
        f"[Paper] {label}: rows={len(ordered)}, q<.05={qsig}{f' ({qcol})' if qcol else ''}, p<.05={psig}{f' ({pcol})' if pcol else ''}",  # noqa:E501
    )
    top = ordered.head(top_n).copy()
    for _, row in top.iterrows():
        _log_and_collect(lines, renderer(row))
    return top


def _write_text_report(path: str, lines: List[str]) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        logger.info(f"[Paper] wrote: {path}")
    except Exception as e:
        logger.error(f"[Paper] Failed writing {path}: {e}")


def _write_csv_report(path: str, tables: List[pd.DataFrame]) -> None:
    tables = [t for t in tables if t is not None and not t.empty]
    if not tables:
        return
    try:
        out = pd.concat(tables, ignore_index=True, sort=False)
        out.to_csv(path, index=False)
        logger.info(f"[Paper] wrote: {path}")
    except Exception as e:
        logger.error(f"[Paper] Failed writing {path}: {e}")


def _add_trial_overview(lines: List[str], output_root: str) -> None:
    trial_path_candidates = [
        os.path.join(output_root, "trigger_trial_features_with_Q123_all.csv"),
        os.path.join(output_root, "trigger_trial_features_all.csv"),
    ]
    for path in trial_path_candidates:
        df = _read_csv_if_exists(path)
        if df is None or df.empty:
            continue
        if {"dataset", "participant_id"}.issubset(df.columns):
            counts = (
                df.groupby("dataset")
                .agg(
                    participant_count=("participant_id", pd.Series.nunique),
                    trial_rows=("participant_id", "size"),
                )
                .reset_index()
            )
            _log_and_collect(lines, "[Paper] trial table overview:")
            for row in counts.itertuples(index=False):
                _log_and_collect(
                    lines,
                    f"- {row.dataset}: participants={int(row.participant_count)}, trial_rows={int(row.trial_rows)}",  # type: ignore  # noqa: E501
                )
        break


def emit_compare_digest(output_root: str = OUTPUT_ROOT) -> None:
    lines: List[str] = []
    tables: List[pd.DataFrame] = []
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _log_and_collect(lines, "\n=== [Paper] Compare digest ===")
    _log_and_collect(lines, f"[Paper] generated at {stamp}")
    _add_trial_overview(lines, output_root)
    for label, filename in [
        ("overall participant level", "comparison_overall.csv"),
        ("overall trial level", "comparison_overall_trial_level.csv"),
        ("participant learning and sequential metrics", "comparison_participant_learning_sequential_metrics.csv"),
        ("condition level", "comparison_by_condition.csv"),
        ("factor level", "comparison_by_factors.csv"),
    ]:
        path = os.path.join(output_root, filename)
        df = _read_csv_if_exists(path)
        top = _append_table_preview(lines, label, df, _render_compare_row)
        if not top.empty:
            top = top.copy()
            top.insert(0, "source", label)
            tables.append(top)
    path = os.path.join(output_root, "between_subject_balance_report.txt")
    if os.path.exists(path):
        _log_and_collect(lines, f"[Paper] between subject balance report available: {path}")
    txt_path = os.path.join(output_root, "paper_compare_digest.txt")
    csv_path = os.path.join(output_root, "paper_compare_digest_top_rows.csv")
    _write_text_report(txt_path, lines)
    _write_csv_report(csv_path, tables)


def emit_mixed_digest(output_root: str = OUTPUT_ROOT) -> None:
    lines: List[str] = []
    tables: List[pd.DataFrame] = []
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _log_and_collect(lines, "\n=== [Paper] Mixed models digest ===")
    _log_and_collect(lines, f"[Paper] generated at {stamp}")
    for label, filename in [
        ("MM1 dataset by condition interactions", "MM1_mixedlm_datasetX_terms.csv"),
        ("MM2 dataset by trial index interactions", "MM2_mixedlm_datasetXtrial_terms.csv"),
        ("MM3 sequential dataset interactions", "MM3_mixedlm_sequential_dataset_interactions.csv"),
    ]:
        path = os.path.join(output_root, filename)
        df = _read_csv_if_exists(path)
        top = _append_table_preview(lines, label, df, _render_mixed_row)
        if not top.empty:
            top = top.copy()
            top.insert(0, "source", label)
            tables.append(top)
    tost_path = os.path.join(output_root, "MM4_equivalence_tost_datasetXyielding.csv")
    tost_df = _read_csv_if_exists(tost_path)
    top_tost = _append_table_preview(lines, "MM4 equivalence on dataset by yielding", tost_df, _render_tost_row)
    if not top_tost.empty:
        top_tost = top_tost.copy()
        top_tost.insert(0, "source", "MM4 equivalence on dataset by yielding")
        tables.append(top_tost)
    balance_csv = os.path.join(output_root, "between_subject_baseline_comparison.csv")
    if os.path.exists(balance_csv):
        _log_and_collect(lines, f"[Paper] baseline balance comparison available: {balance_csv}")
    txt_path = os.path.join(output_root, "paper_mixed_digest.txt")
    csv_path = os.path.join(output_root, "paper_mixed_digest_top_rows.csv")
    _write_text_report(txt_path, lines)
    _write_csv_report(csv_path, tables)


def emit_run_digest(output_root: str = OUTPUT_ROOT) -> None:
    lines: List[str] = []
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _log_and_collect(lines, "\n=== [Paper] Run artefacts ===")
    _log_and_collect(lines, f"[Paper] generated at {stamp}")
    for fn in [
        "paper_compare_digest.txt",
        "paper_compare_digest_top_rows.csv",
        "paper_mixed_digest.txt",
        "paper_mixed_digest_top_rows.csv",
        "between_subject_balance_report.txt",
    ]:
        path = os.path.join(output_root, fn)
        if os.path.exists(path):
            _log_and_collect(lines, f"[Paper] {fn}: {path}")
    _write_text_report(os.path.join(output_root, "paper_run_digest.txt"), lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare shuffled vs unshuffled datasets.")
    parser.add_argument(
        "--reanalyse",
        action="store_true",
        help="Ignore the cached pickle and rebuild the extracted analysis inputs.",
    )
    parser.add_argument(
        "--cache-path",
        default=None,
        help="Optional path for the cached analysis pickle. Defaults to <output_root>/analysis_cache.pkl.",
    )
    return parser.parse_args()


def run_compare(reanalyse: bool = False, cache_path: Optional[str] = None) -> Dict[str, Any]:
    """Run the A to F comparison pipeline."""
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
    return runner.run(reanalyse=reanalyse, cache_path=cache_path)


def run_mixed(trial_df: Optional[pd.DataFrame] = None) -> None:
    """Run the mixed models analysis."""
    runner = mm.MixedModelsPipeline(output_root=OUTPUT_ROOT)
    runner.run(trial_df=trial_df)


if __name__ == "__main__":
    args = parse_args()
    context = run_compare(reanalyse=args.reanalyse, cache_path=args.cache_path)
    emit_compare_digest()
    try:
        trial_df = context.get("merged") if isinstance(context, dict) else None
        run_mixed(trial_df=trial_df if isinstance(trial_df, pd.DataFrame) else None)
        emit_mixed_digest()
    except Exception as e:
        # Mixed models are best effort; keep the overall pipeline usable.
        logger.error(f"[MM] failed: {e}")
    finally:
        emit_run_digest()
