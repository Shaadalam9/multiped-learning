"""Assemble participant trials, validate the sample and audit trial sequences."""

from __future__ import annotations

from typing import Any

from typing import Iterable
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from utils.config import (
    AnalysisSettings,
)
from utils.constants import (
    GROUP_FIXED,
    GROUP_LABELS,
    GROUP_RANDOMISED,
)
from utils.data_io import (
    _participant_number,
    attach_passage_timing,
    discover_participants,
    find_trial_file,
    load_mapping,
    read_participant_responses,
)
from utils.features import (
    extract_trial_features,
)


def _sequence_hash(video_ids: Iterable[str]) -> str:
    text = "|".join(video_ids)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _mapping_content_hash(mapping: pd.DataFrame) -> str:
    """Hash participant condition metadata independently of mapping row order."""

    stable = mapping.copy()
    stable.columns = stable.columns.astype(str)
    stable = stable.reindex(sorted(stable.columns), axis=1)
    stable = stable.sort_values("video_id").reset_index(drop=True)
    text = stable.to_csv(index=False, lineterminator="\n", na_rep="NA")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _participant_mapping_path(
    participant_dir: Path,
    participant_id: str,
    filename_template: str,
) -> Path:
    """Resolve the mapping saved inside one randomised participant folder."""

    rendered = filename_template.format(
        participant_id=participant_id,
        participant_folder=participant_dir.name,
    )
    path = participant_dir / rendered
    if not path.is_file():
        raise FileNotFoundError(
            f"Participant mapping file not found: {path}. Expected template: "
            f"{filename_template}"
        )
    return path


def extract_ordering_group(
    root: Path,
    ordering_group: str,
    settings: AnalysisSettings,
    *,
    shared_mapping: pd.DataFrame | None = None,
    shared_mapping_path: Path | None = None,
    participant_mapping_filename: str | None = None,
    timing_mapping: pd.DataFrame | None = None,
    timing_mapping_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Extract all trial rows and a transparent exclusion audit for one group."""

    participants = discover_participants(root)
    if (shared_mapping is None) == (participant_mapping_filename is None):
        raise ValueError(
            "Provide exactly one mapping strategy: a shared mapping or a "
            "participant mapping filename template"
        )
    rows: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    sequences: list[dict[str, Any]] = []

    for participant_dir in participants:
        participant_id = _participant_number(participant_dir)
        participant_uid = f"{ordering_group}:{participant_id}"
        if participant_mapping_filename is not None:
            try:
                mapping_path = _participant_mapping_path(
                    participant_dir,
                    participant_id,
                    participant_mapping_filename,
                )
                mapping = load_mapping(mapping_path)
            except Exception as exc:
                sequences.append(
                    {
                        "ordering_group": ordering_group,
                        "participant_id": participant_id,
                        "participant_uid": participant_uid,
                        "n_mapping_trials": 0,
                        "n_response_trials": 0,
                        "mapping_source": str(
                            participant_dir
                            / participant_mapping_filename.format(
                                participant_id=participant_id,
                                participant_folder=participant_dir.name,
                            )
                        ),
                        "mapping_content_hash": None,
                        "mapping_sequence_hash": None,
                        "mapping_sequence": "",
                        "mapping_order_matches_response": False,
                        "sequence_hash": _sequence_hash([]),
                        "sequence": "",
                    }
                )
                audit.append(
                    {
                        "ordering_group": ordering_group,
                        "participant_id": participant_id,
                        "participant_uid": participant_uid,
                        "video_id": None,
                        "included": False,
                        "reason": "participant_mapping_error",
                        "detail": str(exc),
                        "mapping_source": str(participant_dir),
                    }
                )
                continue
        else:
            assert shared_mapping is not None
            mapping = shared_mapping
            mapping_path = shared_mapping_path

        if timing_mapping is not None:
            try:
                mapping = attach_passage_timing(mapping, timing_mapping)
            except Exception as exc:
                audit.append(
                    {
                        "ordering_group": ordering_group,
                        "participant_id": participant_id,
                        "participant_uid": participant_uid,
                        "video_id": None,
                        "included": False,
                        "reason": "timing_mapping_error",
                        "detail": str(exc),
                        "mapping_source": str(mapping_path),
                        "timing_mapping_source": str(timing_mapping_path),
                    }
                )
                continue

        mapping_index = mapping.set_index("video_id", drop=False)
        mapping_video_ids = mapping["video_id"].astype(str).tolist()
        mapping_source = str(mapping_path) if mapping_path is not None else "shared_mapping"
        timing_mapping_source = (
            str(timing_mapping_path)
            if timing_mapping_path is not None
            else "mapping_contains_passage_timing"
        )
        mapping_hash = _mapping_content_hash(mapping)
        try:
            responses = read_participant_responses(participant_dir)
        except Exception as exc:
            audit.append(
                {
                    "ordering_group": ordering_group,
                    "participant_id": participant_id,
                    "participant_uid": participant_uid,
                    "video_id": None,
                    "included": False,
                    "reason": "response_file_error",
                    "detail": str(exc),
                    "mapping_source": mapping_source,
                    "timing_mapping_source": timing_mapping_source,
                }
            )
            continue

        experimental = responses[responses["video_id"].isin(mapping_index.index)].copy()
        response_video_ids = experimental["video_id"].astype(str).tolist()
        sequences.append(
            {
                "ordering_group": ordering_group,
                "participant_id": participant_id,
                "participant_uid": participant_uid,
                "n_mapping_trials": len(mapping_video_ids),
                "n_response_trials": len(experimental),
                "mapping_source": mapping_source,
                "timing_mapping_source": timing_mapping_source,
                "mapping_content_hash": mapping_hash,
                "mapping_sequence_hash": _sequence_hash(mapping_video_ids),
                "mapping_sequence": "|".join(mapping_video_ids),
                "mapping_order_matches_response": mapping_video_ids == response_video_ids,
                "sequence_hash": _sequence_hash(response_video_ids),
                "sequence": "|".join(response_video_ids),
            }
        )
        for response in experimental.itertuples(index=False):
            response_dict = response._asdict()
            video_id = str(response_dict["video_id"])
            record = {
                "ordering_group": ordering_group,
                "ordering_group_label": GROUP_LABELS[ordering_group],
                "participant_id": participant_id,
                "participant_uid": participant_uid,
                "participant_dir": str(participant_dir),
                "mapping_source": mapping_source,
                "timing_mapping_source": timing_mapping_source,
                "mapping_content_hash": mapping_hash,
                "video_id": video_id,
                "trial_number": int(response_dict["trial_number"]),
                "Q1": response_dict["Q1"],
                "Q2": response_dict["Q2"],
                "Q3": response_dict["Q3"],
            }
            mapping_row = mapping_index.loc[video_id]
            record.update(mapping_row.to_dict())
            try:
                trial_file = find_trial_file(participant_dir, video_id)
                record["trial_file"] = str(trial_file)
                features = extract_trial_features(trial_file, mapping_row, settings)
                record.update(features)
                if settings.require_complete_window and not features["window_complete"]:
                    raise ValueError(
                        f"Incomplete primary window: {features['valid_bins']} of "
                        f"{features['expected_bins']} bins"
                    )
                rows.append(record)
                audit.append(
                    {
                        "ordering_group": ordering_group,
                        "participant_id": participant_id,
                        "participant_uid": participant_uid,
                        "video_id": video_id,
                        "included": True,
                        "reason": "included",
                        "detail": "",
                        "mapping_source": mapping_source,
                        "timing_mapping_source": timing_mapping_source,
                    }
                )
            except Exception as exc:
                audit.append(
                    {
                        "ordering_group": ordering_group,
                        "participant_id": participant_id,
                        "participant_uid": participant_uid,
                        "video_id": video_id,
                        "included": False,
                        "reason": "trial_extraction_error",
                        "detail": str(exc),
                        "mapping_source": mapping_source,
                        "timing_mapping_source": timing_mapping_source,
                    }
                )

    return pd.DataFrame(rows), pd.DataFrame(audit), pd.DataFrame(sequences)


def validate_sample(
    trials: pd.DataFrame,
    audit: pd.DataFrame,
    sequences: pd.DataFrame,
    settings: AnalysisSettings,
) -> tuple[pd.DataFrame, list[str]]:
    """Apply participant level completeness rule and check order manipulation."""

    warnings_out: list[str] = []
    if trials.empty:
        if audit.empty:
            summary = "No extraction audit rows were produced."
        else:
            failure_counts = (
                audit.groupby(["ordering_group", "reason"], dropna=False)
                .size()
                .sort_values(ascending=False)
            )
            summary = "; ".join(
                f"{group}/{reason}: {int(count)}"
                for (group, reason), count in failure_counts.items()
            )
        raise RuntimeError(
            "No valid trial rows were extracted. Failure counts: "
            f"{summary}. See _output/exclusion_audit.csv for participant and "
            "trial level details."
        )
    counts = (
        trials.groupby(["ordering_group", "participant_uid"], observed=True)
        .size()
        .rename("valid_trials")
        .reset_index()
    )
    eligible = counts.loc[
        counts["valid_trials"] >= settings.minimum_valid_trials_per_participant,
        "participant_uid",
    ]
    excluded = counts.loc[
        counts["valid_trials"] < settings.minimum_valid_trials_per_participant
    ]
    if not excluded.empty:
        warnings_out.append(
            f"Excluded {len(excluded)} participant(s) with fewer than "
            f"{settings.minimum_valid_trials_per_participant} valid trials."
        )
    trials = trials[trials["participant_uid"].isin(set(eligible))].copy()

    sequence_summary = (
        sequences[sequences["participant_uid"].isin(set(eligible))]
        .groupby("ordering_group", observed=True)
        .agg(participants=("participant_uid", "nunique"), unique_sequences=("sequence_hash", "nunique"))
        .reset_index()
    )
    fixed = sequence_summary[sequence_summary["ordering_group"] == GROUP_FIXED]
    randomised = sequence_summary[sequence_summary["ordering_group"] == GROUP_RANDOMISED]
    if not fixed.empty and int(fixed.iloc[0]["unique_sequences"]) != 1:
        warnings_out.append(
            "The fixed sequence group contains more than one realised sequence; verify the manipulation."
        )
    if not randomised.empty and int(randomised.iloc[0]["participants"]) > 1:
        if int(randomised.iloc[0]["unique_sequences"]) <= 1:
            warnings_out.append(
                "The randomised order group contains only one realised sequence; verify the manipulation."
            )
        elif int(randomised.iloc[0]["unique_sequences"]) < int(
            randomised.iloc[0]["participants"]
        ):
            warnings_out.append(
                "At least two randomised-order participants have the same realised "
                "sequence. Verify the sequence-generation record and report the "
                "duplicate transparently."
            )

    eligible_sequences = sequences[
        sequences["participant_uid"].isin(set(eligible))
    ].copy()
    if "mapping_order_matches_response" in eligible_sequences:
        mapping_mismatches = eligible_sequences[
            ~eligible_sequences["mapping_order_matches_response"].fillna(False)
        ]
        if not mapping_mismatches.empty:
            warnings_out.append(
                f"The mapping row order differed from the recorded response order for "
                f"{len(mapping_mismatches)} participant(s). Actual trial position was "
                "taken from the response file; inspect mapping_audit.csv."
            )
    randomised_sequences = eligible_sequences[
        eligible_sequences["ordering_group"] == GROUP_RANDOMISED
    ]
    if not randomised_sequences.empty and (
        randomised_sequences["mapping_source"].nunique()
        != randomised_sequences["participant_uid"].nunique()
    ):
        warnings_out.append(
            "The randomised cohort did not resolve one unique mapping file per "
            "participant; inspect mapping_audit.csv."
        )
    fixed_sequences = eligible_sequences[
        eligible_sequences["ordering_group"] == GROUP_FIXED
    ]
    if not fixed_sequences.empty and fixed_sequences["mapping_source"].nunique() != 1:
        warnings_out.append(
            "The fixed sequence cohort used more than one mapping file; inspect "
            "mapping_audit.csv."
        )

    represented = set(trials["ordering_group"].unique())
    if represented != {GROUP_RANDOMISED, GROUP_FIXED}:
        raise RuntimeError(f"Both ordering groups are required; found {sorted(represented)}")
    if trials["participant_uid"].nunique() < 4:
        warnings_out.append("Fewer than four participants are available after exclusions.")
    return trials, warnings_out


def sample_flow_table(
    trials: pd.DataFrame,
    audit: pd.DataFrame,
    sequences: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group in [GROUP_RANDOMISED, GROUP_FIXED]:
        group_audit = audit[audit["ordering_group"] == group]
        group_trials = trials[trials["ordering_group"] == group]
        group_sequences = sequences[sequences["ordering_group"] == group]
        rows.append(
            {
                "ordering_group": group,
                "ordering_group_label": GROUP_LABELS[group],
                "participant_folders": int(group_sequences["participant_uid"].nunique()),
                "participants_analysed": int(group_trials["participant_uid"].nunique()),
                "trials_attempted": int(len(group_audit)),
                "trials_analysed": int(len(group_trials)),
                "trials_excluded": int((~group_audit["included"].fillna(False)).sum()),
                "unique_sequences": int(group_sequences["sequence_hash"].nunique()),
                "median_valid_bins": float(group_trials["valid_bins"].median())
                if not group_trials.empty
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def mapping_audit_table(sequences: pd.DataFrame) -> pd.DataFrame:
    """Compact provenance and order agreement report for every participant mapping."""

    columns = [
        "ordering_group",
        "participant_id",
        "participant_uid",
        "mapping_source",
        "timing_mapping_source",
        "mapping_content_hash",
        "n_mapping_trials",
        "mapping_sequence_hash",
        "n_response_trials",
        "sequence_hash",
        "mapping_order_matches_response",
    ]
    if sequences.empty:
        return pd.DataFrame(columns=columns)
    available = [column for column in columns if column in sequences.columns]
    return sequences[available].sort_values(
        ["ordering_group", "participant_id"]
    ).reset_index(drop=True)


def duplicate_sequence_audit(sequences: pd.DataFrame) -> pd.DataFrame:
    """List participants sharing a realised sequence within an ordering group."""

    duplicate = sequences[
        sequences.duplicated(["ordering_group", "sequence_hash"], keep=False)
    ].copy()
    if duplicate.empty:
        return pd.DataFrame(
            columns=[
                "ordering_group",
                "sequence_hash",
                "participants_sharing_sequence",
                "participant_id",
                "participant_uid",
                "sequence",
            ]
        )
    duplicate["participants_sharing_sequence"] = duplicate.groupby(
        ["ordering_group", "sequence_hash"], observed=True
    )["participant_uid"].transform("nunique")
    return duplicate[
        [
            "ordering_group",
            "sequence_hash",
            "participants_sharing_sequence",
            "participant_id",
            "participant_uid",
            "sequence",
        ]
    ].sort_values(["ordering_group", "sequence_hash", "participant_id"])
