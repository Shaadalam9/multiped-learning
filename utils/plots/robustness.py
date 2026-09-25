"""Robustness plots for the trial-order analysis."""

from __future__ import annotations

from utils.config import StudyConfig

import pandas as pd

from utils.constants import GROUP_FIXED, GROUP_RANDOMISED
from utils.plots.export import _save_plot
from plotly.subplots import make_subplots
import numpy as np
import plotly.graph_objects as go


def plot_primary_robustness(
    config: StudyConfig,
    cluster_bootstrap_replicates: pd.DataFrame,
    leave_one_out: pd.DataFrame,
    bayesian_bootstrap_draws: pd.DataFrame,
) -> None:
    """Primary robustness."""
    robustness_available = all(
        frame is not None and not frame.empty
        for frame in [
            cluster_bootstrap_replicates,
            leave_one_out,
            bayesian_bootstrap_draws,
        ]
    )
    if robustness_available:
        assert cluster_bootstrap_replicates is not None
        assert leave_one_out is not None
        assert bayesian_bootstrap_draws is not None
        ordered_leave_one_out = leave_one_out.sort_values(
            "difference_fixed_minus_randomised_percentage_points"
        ).reset_index(drop=True)
        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=[
                "Participant cluster bootstrap",
                "Leave one participant out",
                "Bayesian participant bootstrap",
            ],
            horizontal_spacing=0.10,
        )
        fig.add_trace(
            go.Histogram(
                x=cluster_bootstrap_replicates[
                    "difference_percentage_points"
                ],
                nbinsx=50,
                marker={"color": "#0072B2"},
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=np.arange(1, len(ordered_leave_one_out) + 1),
                y=ordered_leave_one_out[
                    "difference_fixed_minus_randomised_percentage_points"
                ],
                mode="markers",
                marker={
                    "color": ordered_leave_one_out[
                        "omitted_ordering_group"
                    ].map(
                        {
                            GROUP_RANDOMISED: "#0072B2",
                            GROUP_FIXED: "#D55E00",
                        }
                    ),
                    "size": 8,
                },
                text=ordered_leave_one_out["omitted_participant_uid"],
                hovertemplate=(
                    "Omitted=%{text}<br>Difference=%{y:.2f} percentage "
                    "points<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Histogram(
                x=bayesian_bootstrap_draws["difference_percentage_points"],
                nbinsx=50,
                marker={"color": "#009E73"},
                showlegend=False,
            ),
            row=1,
            col=3,
        )
        for column in [1, 2, 3]:
            if column == 2:
                fig.add_hline(
                    y=0.0,
                    line_dash="dash",
                    line_color="black",
                    row=1,
                    col=column,
                )
            else:
                fig.add_vline(
                    x=0.0,
                    line_dash="dash",
                    line_color="black",
                    row=1,
                    col=column,
                )
        fig.update_xaxes(
            title_text="Fixed minus randomised (percentage points)",
            row=1,
            col=1,
        )
        fig.update_xaxes(
            title_text="Ordered omission",
            row=1,
            col=2,
        )
        fig.update_xaxes(
            title_text="Fixed minus randomised (percentage points)",
            row=1,
            col=3,
        )
        fig.update_yaxes(title_text="Replicates", row=1, col=1)
        fig.update_yaxes(
            title_text="Primary difference (percentage points)",
            row=1,
            col=2,
        )
        fig.update_yaxes(title_text="Posterior draws", row=1, col=3)
        fig.update_layout(
            template="plotly_white",
            font={"family": "Arial", "size": 16},
            margin={"l": 100, "r": 30, "t": 90, "b": 100},
        )
        _save_plot(
            fig,
            config.figures / "figure_17_primary_robustness",
            width=2000,
            height=850,
        )
