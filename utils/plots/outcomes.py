"""Outcomes plots for the trial-order analysis."""

from __future__ import annotations

from utils.config import StudyConfig

from utils.constants import OUTCOME_SPECS
from utils.plots.export import _save_plot
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_primary_marginal_estimates(
    marginal_estimates: pd.DataFrame,
    config: StudyConfig,
) -> None:
    """Primary marginal estimates."""
    primary = marginal_estimates[
        np.isclose(marginal_estimates["threshold"], config.settings.primary_threshold)
    ].copy()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=primary["ordering_group_label"],
            y=primary["predicted_unsafe_pct"],
            mode="markers",
            marker={"size": 14, "color": ["#0072B2", "#D55E00"]},
            error_y={
                "type": "data",
                "symmetric": False,
                "array": primary["ci_high"] - primary["predicted_unsafe_pct"],
                "arrayminus": primary["predicted_unsafe_pct"] - primary["ci_low"],
            },
        )
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Trial ordering group",
        yaxis_title="Predicted unsafe bins (%)",
        showlegend=False,
        font={"family": "Arial", "size": 22},
    )
    _save_plot(fig, config.figures / "figure_1_primary_marginal_estimates")


def plot_participant_distributions(
    participant_means: pd.DataFrame,
    config: StudyConfig,
) -> None:
    """Participant distributions."""
    fig = px.box(
        participant_means,
        x="ordering_group_label",
        y="unsafe_pct",
        points="all",
        color="ordering_group_label",
        color_discrete_map={"Randomised order": "#0072B2", "Fixed sequence": "#D55E00"},
        labels={"ordering_group_label": "Trial ordering group", "unsafe_pct": "Participant mean unsafe bins (%)"},
        template="plotly_white",
    )
    fig.update_layout(showlegend=False, font={"family": "Arial", "size": 22})
    _save_plot(fig, config.figures / "figure_2_participant_distributions")


def plot_questionnaire_ratings(
    participant_means: pd.DataFrame,
    config: StudyConfig,
) -> None:
    """Questionnaire ratings."""
    rating_columns = [column for column in ["Q1", "Q2", "Q3"] if column in participant_means]
    if rating_columns:
        ratings = participant_means.melt(
            id_vars=["ordering_group", "ordering_group_label", "participant_uid"],
            value_vars=rating_columns,
            var_name="rating",
            value_name="participant_mean",
        ).dropna(subset=["participant_mean"])
        if not ratings.empty:
            fig = px.box(
                ratings,
                x="ordering_group_label",
                y="participant_mean",
                color="ordering_group_label",
                facet_col="rating",
                points="all",
                color_discrete_map={
                    "Randomised order": "#0072B2",
                    "Fixed sequence": "#D55E00",
                },
                labels={
                    "ordering_group_label": "Trial ordering group",
                    "participant_mean": "Participant mean rating (0 to 100)",
                },
                template="plotly_white",
            )
            fig.for_each_annotation(lambda annotation: annotation.update(text=annotation.text.split("=")[-1]))
            fig.update_yaxes(range=[0, 100])
            fig.update_layout(
                showlegend=False,
                font={"family": "Arial", "size": 18},
                margin={"l": 90, "r": 30, "t": 60, "b": 90},
            )
            _save_plot(
                fig,
                config.figures / "figure_6_questionnaire_ratings",
                width=1800,
                height=800,
            )


def plot_head_heading_at_passage(
    participant_means: pd.DataFrame,
    config: StudyConfig,
) -> None:
    """Head heading at passage."""
    heading_column = "heading_at_pass_deg"
    if heading_column in participant_means and participant_means[heading_column].notna().any():
        heading = participant_means.dropna(subset=[heading_column]).copy()
        fig = px.box(
            heading,
            x="ordering_group_label",
            y=heading_column,
            color="ordering_group_label",
            points="all",
            color_discrete_map={"Randomised order": "#0072B2", "Fixed sequence": "#D55E00"},
            labels={
                "ordering_group_label": "Trial ordering group",
                heading_column: "Baseline corrected heading at participant passage (degrees)",
            },
            template="plotly_white",
        )
        fig.add_hline(y=0.0, line_dash="dash", line_color="black", line_width=1)
        fig.update_layout(
            showlegend=False,
            font={"family": "Arial", "size": 20},
            margin={"l": 100, "r": 30, "t": 40, "b": 90},
        )
        _save_plot(fig, config.figures / "figure_7_head_heading_at_passage")


def plot_trigger_activation_and_magnitude(
    participant_means: pd.DataFrame,
    config: StudyConfig,
) -> None:
    """Trigger activation and magnitude."""
    trigger_columns = [
        column
        for column in ["mean_trigger", "peak_trigger", "any_trigger_press"]
        if column in participant_means
    ]
    if trigger_columns:
        trigger = participant_means[
            ["ordering_group_label", "participant_uid", *trigger_columns]
        ].copy()
        for column in ["mean_trigger", "peak_trigger"]:
            if column in trigger:
                trigger[column] *= 100.0
        trigger = trigger.melt(
            id_vars=["ordering_group_label", "participant_uid"],
            value_vars=trigger_columns,
            var_name="trigger_outcome",
            value_name="participant_mean_percent",
        ).dropna(subset=["participant_mean_percent"])
        trigger_labels = {
            "mean_trigger": "Mean trigger (% full scale)",
            "peak_trigger": "Peak trigger (% full scale)",
            "any_trigger_press": "Trials with activation (%)",
        }
        trigger["trigger_outcome"] = trigger["trigger_outcome"].map(trigger_labels)
        fig = px.box(
            trigger,
            x="ordering_group_label",
            y="participant_mean_percent",
            color="ordering_group_label",
            facet_col="trigger_outcome",
            points="all",
            color_discrete_map={
                "Randomised order": "#0072B2",
                "Fixed sequence": "#D55E00",
            },
            labels={
                "ordering_group_label": "Trial ordering group",
                "participant_mean_percent": "Participant mean (%)",
            },
            template="plotly_white",
        )
        fig.for_each_annotation(
            lambda annotation: annotation.update(text=annotation.text.split("=")[-1])
        )
        fig.update_yaxes(range=[0, 105])
        fig.update_layout(
            showlegend=False,
            font={"family": "Arial", "size": 17},
            margin={"l": 90, "r": 30, "t": 70, "b": 90},
        )
        _save_plot(
            fig,
            config.figures / "figure_8_trigger_activation_and_magnitude",
            width=1900,
            height=850,
        )


def plot_head_movement_outcomes(
    participant_means: pd.DataFrame,
    config: StudyConfig,
) -> None:
    """Head movement outcomes."""
    head_columns = [
        column
        for column in [
            "heading_at_pass_deg",
            "minimum_heading_deg",
            "passage_change_deg",
            "heading_common_window_sd_deg",
        ]
        if column in participant_means
    ]
    if head_columns:
        head = participant_means[
            ["ordering_group_label", "participant_uid", *head_columns]
        ].melt(
            id_vars=["ordering_group_label", "participant_uid"],
            value_vars=head_columns,
            var_name="head_outcome",
            value_name="participant_mean_deg",
        ).dropna(subset=["participant_mean_deg"])
        head_labels = {
            "heading_at_pass_deg": "Heading at passage",
            "minimum_heading_deg": "Minimum pre-passage heading",
            "passage_change_deg": "Change across passage",
            "heading_common_window_sd_deg": "Pre-passage variability",
        }
        head["head_outcome"] = head["head_outcome"].map(head_labels)
        fig = px.box(
            head,
            x="ordering_group_label",
            y="participant_mean_deg",
            color="ordering_group_label",
            facet_col="head_outcome",
            facet_col_wrap=2,
            points="all",
            color_discrete_map={
                "Randomised order": "#0072B2",
                "Fixed sequence": "#D55E00",
            },
            labels={
                "ordering_group_label": "Trial ordering group",
                "participant_mean_deg": "Participant mean (degrees)",
            },
            template="plotly_white",
        )
        fig.for_each_annotation(
            lambda annotation: annotation.update(text=annotation.text.split("=")[-1])
        )
        fig.add_hline(y=0.0, line_dash="dash", line_color="black", line_width=1)
        fig.update_layout(
            showlegend=False,
            font={"family": "Arial", "size": 17},
            margin={"l": 100, "r": 30, "t": 70, "b": 90},
        )
        _save_plot(
            fig,
            config.figures / "figure_9_head_movement_outcomes",
            width=1800,
            height=1200,
        )


def plot_event_defined_trigger_outcomes(
    secondary_marginal_estimates: pd.DataFrame,
    config: StudyConfig,
) -> None:
    """Event defined trigger outcomes."""
    event_outcomes = [
        outcome
        for outcome in [
            "trigger_first_active_latency_s",
            "trigger_return_to_safe",
            "trigger_first_return_latency_s",
        ]
        if outcome
        in set(secondary_marginal_estimates.get("outcome", pd.Series(dtype=str)))
    ]
    if event_outcomes:
        fig = make_subplots(
            rows=1,
            cols=len(event_outcomes),
            subplot_titles=[OUTCOME_SPECS[outcome]["label"] for outcome in event_outcomes],
            horizontal_spacing=0.10,
        )
        for column, outcome in enumerate(event_outcomes, start=1):
            subset = secondary_marginal_estimates[
                secondary_marginal_estimates["outcome"] == outcome
            ].copy()
            fig.add_trace(
                go.Scatter(
                    x=subset["ordering_group_label"],
                    y=subset["marginal_estimate"],
                    mode="markers",
                    marker={"size": 14, "color": ["#0072B2", "#D55E00"]},
                    error_y={
                        "type": "data",
                        "symmetric": False,
                        "array": subset["ci_high"] - subset["marginal_estimate"],
                        "arrayminus": subset["marginal_estimate"] - subset["ci_low"],
                    },
                    showlegend=False,
                    hovertemplate=(
                        "%{x}<br>Estimate=%{y:.2f}<extra></extra>"
                    ),
                ),
                row=1,
                col=column,
            )
            fig.update_xaxes(title_text="Trial ordering group", row=1, col=column)
            fig.update_yaxes(
                title_text=OUTCOME_SPECS[outcome]["units"], row=1, col=column
            )
        fig.update_layout(
            template="plotly_white",
            font={"family": "Arial", "size": 17},
            margin={"l": 100, "r": 30, "t": 80, "b": 100},
        )
        _save_plot(
            fig,
            config.figures / "figure_13_event_defined_trigger_outcomes",
            width=1900,
            height=800,
        )
