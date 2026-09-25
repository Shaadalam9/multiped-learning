"""Sequence plots for the trial-order analysis."""

from __future__ import annotations

from utils.config import StudyConfig

import pandas as pd

from utils.constants import GROUP_FIXED, GROUP_LABELS, GROUP_RANDOMISED
from utils.plots.export import _save_plot
from plotly.subplots import make_subplots
import numpy as np
import plotly.graph_objects as go


def plot_trial_position_profiles(
    trials: pd.DataFrame,
    config: StudyConfig,
) -> None:
    """Trial position profiles."""
    by_participant_trial = (
        trials.groupby(
            ["ordering_group", "ordering_group_label", "participant_uid", "trial_number"],
            observed=True,
        )["unsafe_pct"]
        .mean()
        .reset_index()
    )
    summary = (
        by_participant_trial.groupby(
            ["ordering_group", "ordering_group_label", "trial_number"], observed=True
        )["unsafe_pct"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary["se"] = summary["std"] / np.sqrt(summary["count"])
    fig = go.Figure()
    for group, colour in [(GROUP_RANDOMISED, "#0072B2"), (GROUP_FIXED, "#D55E00")]:
        subset = summary[summary["ordering_group"] == group]
        fig.add_trace(
            go.Scatter(
                x=subset["trial_number"],
                y=subset["mean"],
                mode="lines+markers",
                name=GROUP_LABELS[group],
                line={"color": colour},
                error_y={"type": "data", "array": 1.96 * subset["se"], "visible": True},
            )
        )
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Trial position",
        yaxis_title="Mean unsafe bins (%)",
        font={"family": "Arial", "size": 22},
        legend_title_text="Trial ordering group",
    )
    _save_plot(fig, config.figures / "figure_3_trial_position_profiles")


def plot_sequence_composition(
    trials: pd.DataFrame,
    config: StudyConfig,
) -> None:
    """Sequence composition."""
    composition = (
        trials.groupby(["ordering_group", "trial_number"], observed=True)[
            ["yielding", "eHMIOn", "camera", "distPed_m"]
        ]
        .mean()
        .reset_index()
    )
    panels = [
        ("yielding", "Proportion yielding"),
        ("eHMIOn", "Proportion eHMI active"),
        ("camera", "Proportion participant first"),
        ("distPed_m", "Mean spacing (m)"),
    ]
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.055)
    for row_number, (factor, y_label) in enumerate(panels, start=1):
        for group, colour in [(GROUP_RANDOMISED, "#0072B2"), (GROUP_FIXED, "#D55E00")]:
            subset = composition[composition["ordering_group"] == group]
            fig.add_trace(
                go.Scatter(
                    x=subset["trial_number"],
                    y=subset[factor],
                    mode="lines+markers",
                    name=GROUP_LABELS[group],
                    legendgroup=group,
                    showlegend=row_number == 1,
                    line={"color": colour, "width": 2},
                    marker={"size": 5},
                ),
                row=row_number,
                col=1,
            )
        fig.update_yaxes(title_text=y_label, row=row_number, col=1)
        if factor != "distPed_m":
            fig.update_yaxes(range=[-0.05, 1.05], row=row_number, col=1)
    fig.update_xaxes(title_text="Trial position", row=4, col=1)
    fig.update_layout(
        template="plotly_white",
        font={"family": "Arial", "size": 18},
        legend_title_text="Trial ordering group",
        margin={"l": 150, "r": 40, "t": 40, "b": 80},
    )
    _save_plot(
        fig,
        config.figures / "figure_5_sequence_composition",
        width=1600,
        height=1400,
    )
