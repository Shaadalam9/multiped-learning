"""Contrasts plots for the trial-order analysis."""

from __future__ import annotations

from utils.config import StudyConfig

from typing import Any

from utils.plots.export import _save_plot
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_condition_effect_differences(
    condition_effects: pd.DataFrame,
    config: StudyConfig,
) -> None:
    """Condition effect differences."""
    if not condition_effects.empty:
        forest = condition_effects.copy()
        forest["error_plus"] = (
            forest["difference_ci_high"]
            - forest["difference_of_effects_percentage_points"]
        )
        forest["error_minus"] = (
            forest["difference_of_effects_percentage_points"]
            - forest["difference_ci_low"]
        )
        family_colours = {
            "conditional eHMI": "#009E73",
            "relative pedestrian order": "#CC79A7",
            "AV behaviour": "#E69F00",
        }
        fig = px.scatter(
            forest,
            x="difference_of_effects_percentage_points",
            y="contrast",
            color="contrast_family",
            error_x="error_plus",
            error_x_minus="error_minus",
            color_discrete_map=family_colours,
            labels={
                "difference_of_effects_percentage_points": (
                    "Difference of condition effects: fixed minus randomised (percentage points)"
                ),
                "contrast": "Condition contrast",
                "contrast_family": "Contrast family",
            },
            hover_data={
                "p_value_holm": ":.3g",
                "randomised_effect_percentage_points": ":.2f",
                "fixed_effect_percentage_points": ":.2f",
            },
            template="plotly_white",
        )
        fig.add_vline(x=0.0, line_dash="dash", line_color="black", line_width=1)
        fig.update_traces(marker={"size": 11})
        fig.update_layout(
            font={"family": "Arial", "size": 18},
            yaxis={"categoryorder": "array", "categoryarray": forest["contrast"].tolist()[::-1]},
            margin={"l": 360, "r": 40, "t": 40, "b": 90},
        )
        _save_plot(
            fig,
            config.figures / "figure_4_condition_effect_differences",
            width=1800,
            height=1200,
        )


def plot_condition_cell_followup_heatmap(
    condition_cell_contrasts: pd.DataFrame,
    condition_cell_summary: pd.DataFrame,
    config: StudyConfig,
) -> None:
    """Condition cell followup heatmap."""
    if (
        not condition_cell_contrasts.empty
        and not condition_cell_summary.empty
    ):
        selected_summary = condition_cell_summary[
            condition_cell_summary[
                "omnibus_selected_for_condition_followup_figure"
            ].fillna(False)
        ].copy()
        selected_outcomes = selected_summary["outcome"].tolist()
        if selected_outcomes:
            selected_cells = condition_cell_contrasts[
                condition_cell_contrasts["outcome"].isin(selected_outcomes)
            ].copy()
            condition_order = (
                selected_cells[
                    [
                        "condition_id",
                        "condition_label",
                        "yielding",
                        "eHMIOn",
                        "camera",
                        "distPed_m",
                    ]
                ]
                .drop_duplicates()
                .sort_values(
                    ["yielding", "eHMIOn", "camera", "distPed_m"],
                    kind="stable",
                )
            )
            condition_ids = condition_order["condition_id"].tolist()
            condition_labels = condition_order["condition_label"].tolist()
            short_outcome_labels = {
                "unsafe_pct": "Unsafe bins",
                "Q1": "Q1",
                "Q2": "Q2",
                "Q3": "Q3",
                "mean_trigger": "Mean trigger",
                "peak_trigger": "Peak trigger",
                "any_trigger_press": "Any activation",
                "trigger_first_active_latency_s": "Activation latency",
                "trigger_return_to_safe": "Return to safe",
                "trigger_first_return_latency_s": "Return latency",
                "heading_at_pass_deg": "Heading at passage",
                "minimum_heading_deg": "Minimum heading",
                "passage_change_deg": "Passage change",
                "heading_common_window_sd_deg": "Heading variability",
                "heading_yaw_activity_deg_s": "Yaw activity",
            }
            x_labels = [
                short_outcome_labels.get(outcome, outcome)
                for outcome in selected_outcomes
            ]
            z_matrix = np.full(
                (len(condition_ids), len(selected_outcomes)),
                np.nan,
                dtype=float,
            )
            custom_data = np.empty(
                (len(condition_ids), len(selected_outcomes), 5),
                dtype=object,
            )
            custom_data[:] = np.nan
            significance_annotations: list[dict[str, Any]] = []
            for column, outcome in enumerate(selected_outcomes):
                outcome_cells = (
                    selected_cells[selected_cells["outcome"] == outcome]
                    .set_index("condition_id")
                    .reindex(condition_ids)
                )
                z_matrix[:, column] = pd.to_numeric(
                    outcome_cells["z"],
                    errors="coerce",
                ).to_numpy(float)
                custom_data[:, column, 0] = pd.to_numeric(
                    outcome_cells["difference_fixed_minus_randomised"],
                    errors="coerce",
                ).to_numpy(float)
                custom_data[:, column, 1] = pd.to_numeric(
                    outcome_cells["simultaneous_ci_low"],
                    errors="coerce",
                ).to_numpy(float)
                custom_data[:, column, 2] = pd.to_numeric(
                    outcome_cells["simultaneous_ci_high"],
                    errors="coerce",
                ).to_numpy(float)
                custom_data[:, column, 3] = pd.to_numeric(
                    outcome_cells["p_value_holm_within_outcome"],
                    errors="coerce",
                ).to_numpy(float)
                custom_data[:, column, 4] = outcome_cells["units"].to_numpy()
                significant = (
                    pd.to_numeric(
                        outcome_cells["p_value_holm_within_outcome"],
                        errors="coerce",
                    )
                    < 0.05
                ).fillna(False)
                for row_number in np.flatnonzero(significant.to_numpy()):
                    significance_annotations.append(
                        {
                            "x": x_labels[column],
                            "y": condition_labels[row_number],
                            "text": "●",
                            "showarrow": False,
                            "font": {"color": "black", "size": 15},
                        }
                    )

            finite_z = np.abs(z_matrix[np.isfinite(z_matrix)])
            colour_limit = (
                max(2.0, min(5.0, float(np.quantile(finite_z, 0.98))))
                if finite_z.size
                else 3.0
            )
            fig = go.Figure(
                data=[
                    go.Heatmap(
                        z=z_matrix,
                        x=x_labels,
                        y=condition_labels,
                        customdata=custom_data,
                        colorscale="RdBu",
                        reversescale=True,
                        zmid=0.0,
                        zmin=-colour_limit,
                        zmax=colour_limit,
                        colorbar={"title": "Robust z"},
                        hovertemplate=(
                            "%{y}<br>%{x}<br>"
                            "Difference=%{customdata[0]:.2f} %{customdata[4]}<br>"
                            "Simultaneous 95% CI=[%{customdata[1]:.2f}, "
                            "%{customdata[2]:.2f}]<br>"
                            "Holm p=%{customdata[3]:.3g}<extra></extra>"
                        ),
                    )
                ]
            )
            fig.update_layout(
                template="plotly_white",
                title=(
                    "Condition cell follow up after multiplicity controlled omnibus tests"
                    "<br><sup>Colour shows fixed minus randomised robust z; "
                    "dot denotes Holm p &lt; .05 within outcome</sup>"
                ),
                font={"family": "Arial", "size": 15},
                xaxis_title="Outcome",
                yaxis_title="Factorial condition",
                annotations=significance_annotations,
                margin={"l": 420, "r": 80, "t": 120, "b": 180},
            )
            fig.update_xaxes(tickangle=-35)
            _save_plot(
                fig,
                config.figures
                / "figure_18_condition_cell_followup_heatmap",
                width=1900,
                height=1550,
            )

        primary_cells = condition_cell_contrasts[
            condition_cell_contrasts["outcome"] == "unsafe_pct"
        ].copy()
        if not primary_cells.empty:
            primary_cells = primary_cells.sort_values(
                ["yielding", "eHMIOn", "camera", "distPed_m"],
                kind="stable",
            )
            category_order = primary_cells["condition_label"].tolist()
            fig = go.Figure()
            for yielding_value, colour in [(0, "#56B4E9"), (1, "#D55E00")]:
                subset = primary_cells[
                    primary_cells["yielding"] == yielding_value
                ]
                fig.add_trace(
                    go.Scatter(
                        x=subset["difference_fixed_minus_randomised"],
                        y=subset["condition_label"],
                        mode="markers",
                        name=(
                            "Yielding"
                            if yielding_value
                            else "Non yielding"
                        ),
                        marker={"size": 10, "color": colour},
                        error_x={
                            "type": "data",
                            "symmetric": False,
                            "array": (
                                subset["simultaneous_ci_high"]
                                - subset[
                                    "difference_fixed_minus_randomised"
                                ]
                            ),
                            "arrayminus": (
                                subset[
                                    "difference_fixed_minus_randomised"
                                ]
                                - subset["simultaneous_ci_low"]
                            ),
                        },
                        customdata=np.column_stack(
                            [
                                subset["randomised_estimate"],
                                subset["fixed_estimate"],
                                subset[
                                    "p_value_holm_within_outcome"
                                ],
                            ]
                        ),
                        hovertemplate=(
                            "%{y}<br>Randomised=%{customdata[0]:.2f}%<br>"
                            "Fixed=%{customdata[1]:.2f}%<br>"
                            "Difference=%{x:.2f} percentage points<br>"
                            "Holm p=%{customdata[2]:.3g}<extra></extra>"
                        ),
                    )
                )
            fig.add_vline(
                x=0.0,
                line_dash="dash",
                line_color="black",
                line_width=1,
            )
            fig.update_layout(
                template="plotly_white",
                title=(
                    "Primary outcome differences across factorial conditions"
                    "<br><sup>Fixed minus randomised with Bonferroni "
                    "simultaneous 95% confidence intervals</sup>"
                ),
                xaxis_title=(
                    "Difference in trigger active bins "
                    "(percentage points)"
                ),
                yaxis_title="Factorial condition",
                yaxis={
                    "categoryorder": "array",
                    "categoryarray": category_order[::-1],
                },
                legend_title_text="AV behaviour",
                font={"family": "Arial", "size": 15},
                margin={"l": 430, "r": 50, "t": 115, "b": 100},
            )
            _save_plot(
                fig,
                config.figures
                / "figure_19_primary_condition_cell_forest",
                width=1800,
                height=1650,
            )
