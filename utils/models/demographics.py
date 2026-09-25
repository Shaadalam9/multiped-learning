"""Link questionnaires and estimate demographic-adjusted comparisons."""

from __future__ import annotations

from typing import Any
from pathlib import Path
import math
import numpy as np
import pandas as pd
import re
import statsmodels.formula.api as smf
import scipy.stats as st
from utils.config import (
    AnalysisSettings,
    StudyConfig,
)
from utils.constants import (
    GROUP_FIXED,
    GROUP_RANDOMISED,
    INTAKE_AGE,
    INTAKE_GENDER,
    INTAKE_NATIONALITY,
    INTAKE_VR,
    LOGGER,
    OUTCOME_SPECS,
    PARTICIPANT_OUTCOMES,
)
from utils.models.helpers import (
    _add_global_holm,
    _average_prediction,
    _coefficient_table,
    _factorial_prediction_grid,
    _fit_grouped_binomial,
    _holm_within,
)
from utils.models.participants import (
    _hedges_g,
    _welch_difference,
)


def _normalise_participant_id(value: Any) -> str:
    digits = re.sub(r"\D+", "", str(value))
    if digits:
        return str(int(digits))
    return str(value).strip().lower()


def _infer_questionnaire_id_column(frame: pd.DataFrame) -> str | None:
    columns = list(frame.columns)
    for token in ["participant", "subject", "respondent"]:
        matches = [column for column in columns if token in str(column).lower()]
        if matches:
            return str(matches[0])
    exact = [column for column in columns if str(column).strip().lower() in {"id", "pid"}]
    return str(exact[0]) if exact else None


def _read_questionnaire(path: Path) -> pd.DataFrame:
    for separator in [",", ";"]:
        try:
            frame = pd.read_csv(path, sep=separator)
        except Exception:
            continue
        if len(frame.columns) > 1:
            return frame
    raise ValueError(f"Questionnaire could not be read: {path}")


def _normalise_gender(value: Any) -> str:
    if pd.isna(value) or not str(value).strip():
        return "Missing"
    text = str(value).strip().lower()
    if text in {"male", "man", "m"}:
        return "Male"
    if text in {"female", "woman", "f"}:
        return "Female"
    if "prefer" in text or "disclos" in text or text in {"other", "non-binary", "nonbinary"}:
        return "Other or not disclosed"
    return str(value).strip()


def _normalise_vr_experience(value: Any) -> str:
    if pd.isna(value) or not str(value).strip():
        return "Missing"
    text = str(value).strip().lower()
    if "never" in text or (("not" in text or "no " in text) and "month" in text):
        return "None in past month"
    if "less than once" in text or "<" in text:
        return "Less than weekly"
    if any(token in text for token in ["week", "daily", "regular", "often", "times"]):
        return "Regular"
    return "Other"


def load_demographics(
    config: StudyConfig,
    trials: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load prespecified baseline covariates and audit participant linkage."""

    records: list[pd.DataFrame] = []
    status_records: list[dict[str, Any]] = []
    for group, path in [
        (GROUP_RANDOMISED, config.randomised_intake),
        (GROUP_FIXED, config.fixed_intake),
    ]:
        if path is None or not path.is_file():
            status_records.append(
                {
                    "ordering_group": group,
                    "status": "unavailable",
                    "detail": f"Intake questionnaire not found: {path}",
                }
            )
            continue
        try:
            frame = _read_questionnaire(path)
            id_column = _infer_questionnaire_id_column(frame)
            if id_column is None:
                raise ValueError(
                    "No participant identifier column was found. Add participant or subject to its heading."
                )
            missing_fields = [
                field
                for field in [INTAKE_AGE, INTAKE_GENDER, INTAKE_VR]
                if field not in frame.columns
            ]
            if missing_fields:
                raise ValueError(f"Missing baseline fields: {missing_fields}")
            selected = pd.DataFrame(
                {
                    "ordering_group": group,
                    "participant_id": frame[id_column].map(_normalise_participant_id),
                    "age": pd.to_numeric(frame[INTAKE_AGE], errors="coerce"),
                    "gender": frame[INTAKE_GENDER].map(_normalise_gender),
                    "vr_experience": frame[INTAKE_VR].map(_normalise_vr_experience),
                    "nationality": frame[INTAKE_NATIONALITY].astype(str).str.strip()
                    if INTAKE_NATIONALITY in frame.columns
                    else "",
                }
            )
            selected["participant_uid"] = (
                selected["ordering_group"] + ":" + selected["participant_id"]
            )
            selected = selected.drop_duplicates("participant_uid", keep="first")
            records.append(selected)
            status_records.append(
                {
                    "ordering_group": group,
                    "status": "loaded",
                    "detail": f"{len(selected)} unique intake records; ID column: {id_column}",
                }
            )
        except Exception as exc:
            status_records.append(
                {"ordering_group": group, "status": "error", "detail": str(exc)}
            )

    demographics = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    analysed = set(trials["participant_uid"].unique())
    linked = set(demographics["participant_uid"].unique()) if not demographics.empty else set()
    for group in [GROUP_RANDOMISED, GROUP_FIXED]:
        group_analysed = {
            value for value in analysed if str(value).startswith(f"{group}:")
        }
        group_linked = group_analysed.intersection(linked)
        status_records.append(
            {
                "ordering_group": group,
                "status": "linkage",
                "detail": (
                    f"{len(group_linked)} of {len(group_analysed)} analysed participants linked; "
                    f"unlinked IDs: {sorted(group_analysed.difference(linked))}"
                ),
            }
        )
    return demographics, pd.DataFrame(status_records)


def baseline_demographic_tables(
    demographics: pd.DataFrame,
    trials: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Describe cohort composition and quantify baseline imbalance."""

    if demographics.empty:
        return pd.DataFrame(), pd.DataFrame()
    analysed = demographics[
        demographics["participant_uid"].isin(set(trials["participant_uid"]))
    ].copy()
    descriptive: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    for group in [GROUP_RANDOMISED, GROUP_FIXED]:
        current = analysed[analysed["ordering_group"] == group]
        age = current["age"].dropna()
        descriptive.append(
            {
                "variable": "age",
                "level": "continuous",
                "ordering_group": group,
                "n": int(age.size),
                "value": float(age.mean()) if not age.empty else np.nan,
                "standard_deviation": float(age.std(ddof=1)) if len(age) > 1 else np.nan,
            }
        )
        for variable in ["gender", "vr_experience", "nationality"]:
            for level, count in current[variable].fillna("Missing").value_counts().items():
                descriptive.append(
                    {
                        "variable": variable,
                        "level": level,
                        "ordering_group": group,
                        "n": int(count),
                        "value": float(count / len(current)) if len(current) else np.nan,
                        "standard_deviation": np.nan,
                    }
                )

    random_age = analysed.loc[
        analysed["ordering_group"] == GROUP_RANDOMISED, "age"
    ].dropna().to_numpy(float)
    fixed_age = analysed.loc[
        analysed["ordering_group"] == GROUP_FIXED, "age"
    ].dropna().to_numpy(float)
    if len(random_age) >= 2 and len(fixed_age) >= 2:
        age_result = _welch_difference(random_age, fixed_age)
        age_result.update(
            {
                "variable": "age",
                "test": "Welch two sample comparison",
                "effect_size": _hedges_g(random_age, fixed_age),
            }
        )
        comparisons.append(age_result)
    for variable in ["gender", "vr_experience"]:
        contingency = pd.crosstab(analysed["ordering_group"], analysed[variable])
        if contingency.shape[0] == 2 and contingency.shape[1] >= 2:
            chi_square, p_value, degrees, _ = st.chi2_contingency(contingency)
            total = contingency.to_numpy().sum()
            denominator = min(contingency.shape) - 1
            cramer_v = math.sqrt(chi_square / (total * denominator)) if denominator > 0 else np.nan
            comparisons.append(
                {
                    "variable": variable,
                    "test": "Pearson chi square",
                    "chi_square": chi_square,
                    "degrees_of_freedom": degrees,
                    "p_value": p_value,
                    "effect_size": cramer_v,
                }
            )
    return pd.DataFrame(descriptive), pd.DataFrame(comparisons)


def _standardised_demographic_marginal_contrast(
    result: Any,
    design_info: Any,
    demographics: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    factorial = _factorial_prediction_grid()
    demographic_values = demographics[
        ["age_centered", "age_missing", "gender", "vr_experience"]
    ].copy()
    estimates: dict[str, tuple[float, np.ndarray, float]] = {}
    for group in [GROUP_RANDOMISED, GROUP_FIXED]:
        grid = factorial.merge(demographic_values, how="cross")
        grid["ordering_group"] = group
        estimates[group] = _average_prediction(result, design_info, grid, "logit")
    fixed_estimate, fixed_gradient, _ = estimates[GROUP_FIXED]
    random_estimate, random_gradient, _ = estimates[GROUP_RANDOMISED]
    difference = fixed_estimate - random_estimate
    gradient = fixed_gradient - random_gradient
    variance = float(gradient @ np.asarray(result.cov_params()) @ gradient)
    standard_error = math.sqrt(max(0.0, variance))
    critical = st.norm.ppf(0.975)
    z_value = difference / standard_error if standard_error > 0 else np.nan
    return pd.DataFrame(
        [
            {
                "threshold": threshold,
                "contrast": "fixed sequence minus randomised order",
                "standardisation": "equal factorial cells and pooled analysed participant demographics",
                "difference_percentage_points": 100.0 * difference,
                "standard_error_percentage_points": 100.0 * standard_error,
                "ci_low": 100.0 * (difference - critical * standard_error),
                "ci_high": 100.0 * (difference + critical * standard_error),
                "z": z_value,
                "p_value": 2.0 * st.norm.sf(abs(z_value)) if np.isfinite(z_value) else np.nan,
            }
        ]
    )


def demographic_adjusted_sensitivity(
    trials: pd.DataFrame,
    demographics: pd.DataFrame,
    settings: AnalysisSettings,
) -> dict[str, pd.DataFrame]:
    """Adjust the primary group contrast for measured cohort composition."""

    empty = {
        "demographic_adjusted_primary_contrast": pd.DataFrame(),
        "demographic_adjusted_coefficients": pd.DataFrame(),
        "demographic_adjustment_status": pd.DataFrame(),
    }
    if demographics.empty:
        empty["demographic_adjustment_status"] = pd.DataFrame(
            [{"status": "not estimated", "reason": "No intake demographics were loaded"}]
        )
        return empty
    linked = demographics[
        demographics["participant_uid"].isin(set(trials["participant_uid"]))
    ].copy()
    counts = linked.groupby("ordering_group")["participant_uid"].nunique()
    if any(counts.get(group, 0) < 10 for group in [GROUP_RANDOMISED, GROUP_FIXED]):
        empty["demographic_adjustment_status"] = pd.DataFrame(
            [{"status": "not estimated", "reason": "Fewer than 10 linked participants in a group"}]
        )
        return empty
    linked["age_missing"] = linked["age"].isna().astype(int)
    pooled_age = float(linked["age"].median())
    linked["age_centered"] = linked["age"].fillna(pooled_age) - pooled_age
    linked["gender"] = linked["gender"].fillna("Missing")
    linked["vr_experience"] = linked["vr_experience"].fillna("Missing")
    model_frame = trials.merge(
        linked[
            [
                "participant_uid",
                "age_centered",
                "age_missing",
                "gender",
                "vr_experience",
            ]
        ],
        on="participant_uid",
        how="inner",
    )
    additional = "age_centered + C(gender) + C(vr_experience)"
    if linked["age_missing"].nunique() > 1:
        additional += " + age_missing"
    try:
        result, design_info, _, _ = _fit_grouped_binomial(
            model_frame,
            settings.primary_threshold,
            False,
            additional_rhs=additional,
        )
        contrast = _standardised_demographic_marginal_contrast(
            result,
            design_info,
            linked,
            settings.primary_threshold,
        )
        coefficients = _coefficient_table(
            result,
            "demographic_adjusted_mean_and_condition",
            settings.primary_threshold,
        )
        status = pd.DataFrame(
            [
                {
                    "status": "estimated",
                    "reason": "Sensitivity analysis only; measured covariates cannot remove run level confounding",
                    "participants": linked["participant_uid"].nunique(),
                }
            ]
        )
        return {
            "demographic_adjusted_primary_contrast": contrast,
            "demographic_adjusted_coefficients": coefficients,
            "demographic_adjustment_status": status,
        }
    except Exception as exc:
        empty["demographic_adjustment_status"] = pd.DataFrame(
            [{"status": "failed", "reason": str(exc)}]
        )
        return empty


def demographic_adjusted_participant_outcomes(
    participant_means: pd.DataFrame,
    demographics: pd.DataFrame,
) -> pd.DataFrame:
    """Independent-participant HC3 sensitivity models for all paper outcomes."""

    if demographics.empty or participant_means.empty:
        return pd.DataFrame()
    linked = demographics[
        demographics["participant_uid"].isin(set(participant_means["participant_uid"]))
    ].copy()
    counts = linked.groupby("ordering_group")["participant_uid"].nunique()
    if any(counts.get(group, 0) < 10 for group in [GROUP_RANDOMISED, GROUP_FIXED]):
        return pd.DataFrame()
    linked["age_missing"] = linked["age"].isna().astype(int)
    pooled_age = float(linked["age"].median())
    linked["age_centered"] = linked["age"].fillna(pooled_age) - pooled_age
    linked["gender"] = linked["gender"].fillna("Missing")
    linked["vr_experience"] = linked["vr_experience"].fillna("Missing")
    frame = participant_means.merge(
        linked[
            [
                "participant_uid",
                "age_centered",
                "age_missing",
                "gender",
                "vr_experience",
            ]
        ],
        on="participant_uid",
        how="inner",
    )
    group_term = (
        "C(ordering_group, Treatment(reference='randomised_order'))[T.fixed_sequence]"
    )
    rows: list[dict[str, Any]] = []
    outcomes = [outcome for outcome in PARTICIPANT_OUTCOMES if outcome in frame.columns]
    for outcome in outcomes:
        spec = OUTCOME_SPECS[outcome]
        model_frame = frame.dropna(subset=[outcome]).copy()
        if model_frame["ordering_group"].nunique() < 2 or len(model_frame) < 20:
            continue
        rhs = (
            "C(ordering_group, Treatment(reference='randomised_order')) "
            "+ age_centered + C(gender) + C(vr_experience)"
        )
        if model_frame["age_missing"].nunique() > 1:
            rhs += " + age_missing"
        try:
            result = smf.ols(f"{outcome} ~ {rhs}", data=model_frame).fit(cov_type="HC3")
            estimate = float(result.params[group_term])
            standard_error = float(result.bse[group_term])
            z_value = estimate / standard_error if standard_error > 0 else np.nan
            critical = st.norm.ppf(0.975)
            rows.append(
                {
                    "outcome": outcome,
                    "outcome_label": spec["label"],
                    "outcome_family": spec["family"],
                    "analysis_role": spec["role"],
                    "units": spec["units"],
                    "contrast": "fixed sequence minus randomised order",
                    "adjusted_difference": estimate,
                    "standard_error_hc3": standard_error,
                    "ci_low": estimate - critical * standard_error,
                    "ci_high": estimate + critical * standard_error,
                    "z": z_value,
                    "p_value": (
                        2.0 * st.norm.sf(abs(z_value))
                        if np.isfinite(z_value)
                        else np.nan
                    ),
                    "participants": int(len(model_frame)),
                    "adjusted_for": "age, age missingness, gender, and recent VR experience",
                    "interpretation": "sensitivity analysis; does not remove unmeasured run-level confounding",
                }
            )
        except Exception as exc:
            LOGGER.warning(
                "Participant-level demographic sensitivity failed for %s: %s",
                outcome,
                exc,
            )
    table = _holm_within(pd.DataFrame(rows), ["outcome_family"])
    if not table.empty:
        table["multiplicity_method"] = (
            "Holm across adjusted participant outcomes within outcome family"
        )
        table = _add_global_holm(table)
    return table
