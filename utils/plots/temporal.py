"""Temporal plots for the trial-order analysis."""

from __future__ import annotations

from utils.config import StudyConfig

from utils.constants import (
    EHMI_LEARNING_OUTCOMES,
    EXPOSURE_FACTORS,
    GROUP_FIXED,
    GROUP_LABELS,
    GROUP_RANDOMISED,
    OUTCOME_SPECS,
    SESSION_SEGMENTS,
)
from utils.plots.export import _save_plot
from plotly.subplots import make_subplots
import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_condition_adjusted_trial_profiles(
    temporal_predictions: pd.DataFrame,
    config: StudyConfig,
) -> None:
    """Condition adjusted trial profiles."""
    progression_outcomes = [
        outcome
        for outcome in [
            "unsafe_pct",
            "mean_trigger",
            "any_trigger_press",
            "Q3",
            "heading_at_pass_deg",
            "passage_change_deg",
        ]
        if outcome in set(temporal_predictions.get("outcome", pd.Series(dtype=str)))
    ]
    if progression_outcomes:
        subplot_titles = [OUTCOME_SPECS[outcome]["label"] for outcome in progression_outcomes]
        rows = int(math.ceil(len(progression_outcomes) / 2))
        fig = make_subplots(
            rows=rows,
            cols=2,
            subplot_titles=subplot_titles,
            vertical_spacing=0.11,
            horizontal_spacing=0.10,
        )
        colours = {GROUP_RANDOMISED: "#0072B2", GROUP_FIXED: "#D55E00"}
        fills = {GROUP_RANDOMISED: "rgba(0,114,178,0.15)", GROUP_FIXED: "rgba(213,94,0,0.15)"}
        for panel_index, outcome in enumerate(progression_outcomes):
            row = panel_index // 2 + 1
            column = panel_index % 2 + 1
            outcome_frame = temporal_predictions[temporal_predictions["outcome"] == outcome]
            for group in [GROUP_RANDOMISED, GROUP_FIXED]:
                subset = outcome_frame[outcome_frame["ordering_group"] == group].sort_values("trial_number")
                if subset.empty:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=pd.concat([subset["trial_number"], subset["trial_number"].iloc[::-1]]),
                        y=pd.concat([subset["ci_high"], subset["ci_low"].iloc[::-1]]),
                        fill="toself",
                        fillcolor=fills[group],
                        line={"color": "rgba(255,255,255,0)"},
                        hoverinfo="skip",
                        showlegend=False,
                        legendgroup=group,
                    ),
                    row=row,
                    col=column,
                )
                fig.add_trace(
                    go.Scatter(
                        x=subset["trial_number"],
                        y=subset["adjusted_estimate"],
                        mode="lines",
                        name=GROUP_LABELS[group],
                        legendgroup=group,
                        showlegend=panel_index == 0,
                        line={"color": colours[group], "width": 3},
                    ),
                    row=row,
                    col=column,
                )
            fig.add_vline(x=14.5, line_dash="dot", line_color="#777777", row=row, col=column)
            fig.add_vline(x=26.5, line_dash="dot", line_color="#777777", row=row, col=column)
            fig.update_xaxes(title_text="Trial position", row=row, col=column)
            fig.update_yaxes(title_text=OUTCOME_SPECS[outcome]["units"], row=row, col=column)
        fig.update_layout(
            template="plotly_white",
            font={"family": "Arial", "size": 17},
            legend_title_text="Trial ordering group",
            margin={"l": 110, "r": 30, "t": 90, "b": 80},
        )
        _save_plot(
            fig,
            config.figures / "figure_10_condition_adjusted_trial_profiles",
            width=1900,
            height=max(1000, 500 * rows),
        )


def plot_adjusted_temporal_effects(
    temporal_tests: pd.DataFrame,
    config: StudyConfig,
) -> None:
    """Adjusted temporal effects."""
    if not temporal_tests.empty:
        forest = temporal_tests[
            temporal_tests["test"].eq("group by linear trial position")
            & temporal_tests["standardised_estimate"].notna()
        ].copy()
        if not forest.empty:
            forest["error_plus"] = forest["standardised_ci_high"] - forest["standardised_estimate"]
            forest["error_minus"] = forest["standardised_estimate"] - forest["standardised_ci_low"]
            fig = px.scatter(
                forest,
                x="standardised_estimate",
                y="outcome_label",
                color="outcome_family",
                error_x="error_plus",
                error_x_minus="error_minus",
                hover_data={"p_value": ":.3g", "p_value_adjusted": ":.3g"},
                labels={
                    "standardised_estimate": "Fixed minus randomised linear change per 10 trials (SD units)",
                    "outcome_label": "Outcome",
                    "outcome_family": "Outcome family",
                },
                template="plotly_white",
            )
            fig.add_vline(x=0.0, line_dash="dash", line_color="black", line_width=1)
            fig.update_traces(marker={"size": 12})
            fig.update_layout(
                font={"family": "Arial", "size": 17},
                margin={"l": 310, "r": 40, "t": 50, "b": 100},
            )
            _save_plot(
                fig,
                config.figures / "figure_11_adjusted_temporal_effects",
                width=1800,
                height=1000,
            )


def plot_prior_exposure_interactions(
    exposure_tests: pd.DataFrame,
    config: StudyConfig,
) -> None:
    """Prior exposure interactions."""
    if not exposure_tests.empty:
        heat = exposure_tests[
            exposure_tests["exposure_factor"].ne("joint")
            & exposure_tests["standardised_estimate"].notna()
        ].copy()
        if not heat.empty:
            labels = dict(EXPOSURE_FACTORS)
            heat["factor_label"] = heat["exposure_factor"].map(labels)
            value_matrix = heat.pivot(
                index="outcome_label", columns="factor_label", values="standardised_estimate"
            )
            p_matrix = heat.pivot(
                index="outcome_label", columns="factor_label", values="p_value_adjusted"
            ).reindex(index=value_matrix.index, columns=value_matrix.columns)
            annotations = p_matrix.map(
                lambda value: "" if pd.isna(value) else f"Holm p={value:.3g}"
            )
            limit = float(np.nanmax(np.abs(value_matrix.to_numpy(float))))
            limit = max(limit, 0.10)
            fig = go.Figure(
                go.Heatmap(
                    z=value_matrix.to_numpy(float),
                    x=value_matrix.columns.tolist(),
                    y=value_matrix.index.tolist(),
                    text=annotations.to_numpy(str),
                    texttemplate="%{text}",
                    colorscale="RdBu_r",
                    zmid=0.0,
                    zmin=-limit,
                    zmax=limit,
                    colorbar={"title": "SD units<br>per 10 exposures"},
                    hovertemplate=(
                        "Outcome=%{y}<br>Prior exposure=%{x}<br>"
                        "Standardised interaction=%{z:.3f}<br>%{text}<extra></extra>"
                    ),
                )
            )
            fig.update_layout(
                template="plotly_white",
                font={"family": "Arial", "size": 16},
                xaxis_title="Prior exposure to the current factor level",
                yaxis_title="Outcome",
                margin={"l": 320, "r": 130, "t": 50, "b": 120},
            )
            _save_plot(
                fig,
                config.figures / "figure_12_prior_exposure_interactions",
                width=1900,
                height=1100,
            )


def plot_adjusted_session_segments(
    session_segment_predictions: pd.DataFrame,
    config: StudyConfig,
) -> None:
    """Adjusted session segments."""
    segment_outcomes = [
        outcome
        for outcome in [
            "unsafe_pct",
            "Q3",
            "any_trigger_press",
            "trigger_first_active_latency_s",
            "heading_yaw_activity_deg_s",
        ]
        if outcome
        in set(session_segment_predictions.get("outcome", pd.Series(dtype=str)))
    ]
    if segment_outcomes:
        rows = int(math.ceil(len(segment_outcomes) / 2))
        fig = make_subplots(
            rows=rows,
            cols=2,
            subplot_titles=[OUTCOME_SPECS[outcome]["label"] for outcome in segment_outcomes],
            vertical_spacing=0.12,
            horizontal_spacing=0.11,
        )
        colours = {GROUP_RANDOMISED: "#0072B2", GROUP_FIXED: "#D55E00"}
        segment_order = [label for _, label in SESSION_SEGMENTS]
        for panel_index, outcome in enumerate(segment_outcomes):
            row = panel_index // 2 + 1
            column = panel_index % 2 + 1
            outcome_frame = session_segment_predictions[
                session_segment_predictions["outcome"] == outcome
            ]
            for ordering_group in [GROUP_RANDOMISED, GROUP_FIXED]:
                subset = outcome_frame[
                    outcome_frame["ordering_group"] == ordering_group
                ].copy()
                subset["session_segment_label"] = pd.Categorical(
                    subset["session_segment_label"],
                    categories=segment_order,
                    ordered=True,
                )
                subset = subset.sort_values("session_segment_label")
                fig.add_trace(
                    go.Scatter(
                        x=subset["session_segment_label"],
                        y=subset["adjusted_estimate"],
                        mode="lines+markers",
                        name=GROUP_LABELS[ordering_group],
                        legendgroup=ordering_group,
                        showlegend=panel_index == 0,
                        line={"color": colours[ordering_group], "width": 3},
                        marker={"size": 10},
                        error_y={
                            "type": "data",
                            "symmetric": False,
                            "array": subset["ci_high"] - subset["adjusted_estimate"],
                            "arrayminus": subset["adjusted_estimate"] - subset["ci_low"],
                        },
                    ),
                    row=row,
                    col=column,
                )
            fig.update_xaxes(title_text="Session segment", row=row, col=column)
            fig.update_yaxes(
                title_text=OUTCOME_SPECS[outcome]["units"], row=row, col=column
            )
        fig.update_layout(
            template="plotly_white",
            font={"family": "Arial", "size": 16},
            legend_title_text="Trial ordering group",
            margin={"l": 110, "r": 30, "t": 90, "b": 100},
        )
        _save_plot(
            fig,
            config.figures / "figure_14_adjusted_session_segments",
            width=1900,
            height=max(900, 520 * rows),
        )


def plot_ehmi_cue_learning_contrasts(
    ehmi_learning_change_tests: pd.DataFrame,
    config: StudyConfig,
) -> None:
    """Ehmi cue learning contrasts."""
    cue_changes = ehmi_learning_change_tests[
        ehmi_learning_change_tests.get(
            "ordering_group", pd.Series(dtype=str)
        ).isin([GROUP_RANDOMISED, GROUP_FIXED])
    ].copy()
    if not cue_changes.empty:
        outcomes = [
            outcome
            for outcome in EHMI_LEARNING_OUTCOMES
            if outcome in set(cue_changes["outcome"])
        ]
        progression_panels = [
            (
                "trial_position",
                "Trial 40 minus trial 1",
            ),
            (
                "prior_yielding_ehmi_exposure",
                "Maximum minus zero prior yielding-eHMI encounters",
            ),
        ]
        titles = [
            f"{progression_label}: {OUTCOME_SPECS[outcome]['label']}"
            for _, progression_label in progression_panels
            for outcome in outcomes
        ]
        fig = make_subplots(
            rows=len(progression_panels),
            cols=len(outcomes),
            subplot_titles=titles,
            horizontal_spacing=0.16,
            vertical_spacing=0.20,
        )
        for row, (progression_metric, progression_label) in enumerate(
            progression_panels, start=1
        ):
            for column, outcome in enumerate(outcomes, start=1):
                subset = cue_changes[
                    (cue_changes["outcome"] == outcome)
                    & (cue_changes["progression_metric"] == progression_metric)
                ]
                for ordering_group, colour in [
                    (GROUP_RANDOMISED, "#0072B2"),
                    (GROUP_FIXED, "#D55E00"),
                ]:
                    row_data = subset[subset["ordering_group"] == ordering_group]
                    if row_data.empty:
                        continue
                    estimate = float(row_data["estimate"].iloc[0])
                    fig.add_trace(
                        go.Scatter(
                            x=[estimate],
                            y=[GROUP_LABELS[ordering_group]],
                            mode="markers",
                            marker={"size": 14, "color": colour},
                            error_x={
                                "type": "data",
                                "symmetric": False,
                                "array": [float(row_data["ci_high"].iloc[0]) - estimate],
                                "arrayminus": [estimate - float(row_data["ci_low"].iloc[0])],
                            },
                            showlegend=False,
                        ),
                        row=row,
                        col=column,
                    )
                fig.add_vline(
                    x=0.0,
                    line_dash="dash",
                    line_color="black",
                    line_width=1,
                    row=row,
                    col=column,
                )
                fig.update_xaxes(
                    title_text=(
                        "Change in yielding-trial eHMI effect "
                        f"({OUTCOME_SPECS[outcome]['units']})"
                    ),
                    row=row,
                    col=column,
                )
        fig.update_layout(
            template="plotly_white",
            font={"family": "Arial", "size": 17},
            margin={"l": 170, "r": 30, "t": 90, "b": 110},
        )
        _save_plot(
            fig,
            config.figures / "figure_15_ehmi_cue_learning_contrasts",
            width=1800,
            height=1200,
        )


def plot_spline_trial_trajectories(
    config: StudyConfig,
    spline_predictions: pd.DataFrame,
) -> None:
    """Spline trial trajectories."""
    if spline_predictions is not None and not spline_predictions.empty:
        spline_outcomes = [
            outcome
            for outcome in [
                "unsafe_pct",
                "Q3",
                "heading_common_window_sd_deg",
            ]
            if outcome in set(spline_predictions["outcome"])
        ]
        if spline_outcomes:
            fig = make_subplots(
                rows=1,
                cols=len(spline_outcomes),
                subplot_titles=[
                    OUTCOME_SPECS[outcome]["label"]
                    for outcome in spline_outcomes
                ],
                horizontal_spacing=0.09,
            )
            colours = {
                GROUP_RANDOMISED: "#0072B2",
                GROUP_FIXED: "#D55E00",
            }
            fills = {
                GROUP_RANDOMISED: "rgba(0,114,178,0.15)",
                GROUP_FIXED: "rgba(213,94,0,0.15)",
            }
            for column, outcome in enumerate(spline_outcomes, start=1):
                outcome_frame = spline_predictions[
                    spline_predictions["outcome"] == outcome
                ]
                for group in [GROUP_RANDOMISED, GROUP_FIXED]:
                    subset = outcome_frame[
                        outcome_frame["ordering_group"] == group
                    ].sort_values("trial_number")
                    if subset.empty:
                        continue
                    fig.add_trace(
                        go.Scatter(
                            x=pd.concat(
                                [
                                    subset["trial_number"],
                                    subset["trial_number"].iloc[::-1],
                                ]
                            ),
                            y=pd.concat(
                                [
                                    subset["ci_high"],
                                    subset["ci_low"].iloc[::-1],
                                ]
                            ),
                            fill="toself",
                            fillcolor=fills[group],
                            line={"color": "rgba(255,255,255,0)"},
                            hoverinfo="skip",
                            showlegend=False,
                            legendgroup=group,
                        ),
                        row=1,
                        col=column,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=subset["trial_number"],
                            y=subset["adjusted_estimate"],
                            mode="lines",
                            name=GROUP_LABELS[group],
                            legendgroup=group,
                            showlegend=column == 1,
                            line={"color": colours[group], "width": 3},
                        ),
                        row=1,
                        col=column,
                    )
                fig.add_vline(
                    x=14.5,
                    line_dash="dot",
                    line_color="#777777",
                    row=1,
                    col=column,
                )
                fig.add_vline(
                    x=26.5,
                    line_dash="dot",
                    line_color="#777777",
                    row=1,
                    col=column,
                )
                fig.update_xaxes(
                    title_text="Trial position",
                    row=1,
                    col=column,
                )
                fig.update_yaxes(
                    title_text=OUTCOME_SPECS[outcome]["units"],
                    row=1,
                    col=column,
                )
            fig.update_layout(
                template="plotly_white",
                font={"family": "Arial", "size": 17},
                legend_title_text="Trial ordering group",
                margin={"l": 110, "r": 30, "t": 90, "b": 90},
            )
            _save_plot(
                fig,
                config.figures / "figure_16_spline_trial_trajectories",
                width=2000,
                height=850,
            )
