"""Trimmed helper utilities for the current CSU analysis pipeline.

This version keeps the helper methods that are exercised by the uploaded
project's live execution path and removes the orphaned legacy utilities that
were not referenced anywhere else in the uploaded codebase.

Scope kept here:
- plot export helpers
- keypress plotting helpers
- statistical marker helpers
- participant trigger and quaternion matrix export
- one high level trigger plotting helper (`plot_column`)

Removed from the original helper.py in this trimmed version:
- survey distribution plots
- nationality and gender plotting
- slider CSV aggregation
- legacy CSV averaging utilities
- unused heatmap and questionnaire post processing helpers
- unused yaw specific plotting utilities
"""

from __future__ import annotations

import os
import re
from typing import Optional

import numpy as np
import pandas as pd
import plotly as py
import plotly.graph_objects as go
import plotly.io as pio
from OneEuroFilter import OneEuroFilter
from scipy.stats import ttest_ind, ttest_rel
from tqdm import tqdm

import common
from custom_logger import CustomLogger

# This project is sometimes run with different repository layouts:
# some copies keep these modules under ``utils/`` while others keep them at the
# project root. Support both.
try:
    from utils.HMD_helper import HMD_yaw  # type: ignore
except Exception:  # pragma: no cover
    from HMD_helper import HMD_yaw  # type: ignore

try:
    from utils.tools import Tools  # type: ignore
except Exception:  # pragma: no cover
    from tools import Tools  # type: ignore

logger = CustomLogger(__name__)
HMD_class = HMD_yaw()
extra_class = Tools()


def _safe_get_config(key: str, default=None):
    """Return ``common.get_configs(key)`` if available, else ``default``."""
    try:
        val = common.get_configs(key)
    except Exception:
        return default
    return default if val is None else val


def _resolve_data_folder(dataset: Optional[str] = None) -> str:
    """Resolve the participant response folder from config values.

    Supports both the legacy single dataset key and the newer shuffled vs
    unshuffled split keys.
    """
    if dataset:
        v = _safe_get_config(f"{dataset}_data", None)
        if v:
            return v
    for key in ("data", "shuffled_data", "unshuffled_data"):
        v = _safe_get_config(key, None)
        if v:
            return v
    return ""


class HMD_helper:
    def __init__(self, *, dataset: Optional[str] = None, data_folder: Optional[str] = None,
                 output_folder: Optional[str] = None):
        self.template = common.get_configs('plotly_template')
        self.smoothen_signal = common.get_configs('smoothen_signal')
        self.folder_figures = common.get_configs('figures')  # subdirectory to save figures
        self.folder_stats = 'statistics'  # subdirectory to save statistical output
        self.data_folder = data_folder or _resolve_data_folder(dataset)  # participant data folder
        if not self.data_folder:
            logger.warning("No data folder configured (expected config key 'data' or 'shuffled_data'/'unshuffled_data').")  # noqa: E501
        self.output_folder = output_folder or common.get_configs("output")

    def smoothen_filter(self, signal, type_flter='OneEuroFilter'):
        """Smoothen list with a filter.

        Args:
            signal (list): input signal to smoothen
            type_flter (str, optional): type_flter of filter to use.

        Returns:
            list: list with smoothened data.
        """
        if type_flter == 'OneEuroFilter':
            filter_kp = OneEuroFilter(freq=common.get_configs('freq'),            # frequency
                                      mincutoff=common.get_configs('mincutoff'),  # minimum cutoff frequency
                                      beta=common.get_configs('beta'))            # beta value
            return [filter_kp(value) for value in signal]
        else:
            logger.error(f'Specified filter {type_flter} not implemented.')
            return -1

    def save_plotly(self, fig, name, remove_margins=False, width=1320, height=680, save_eps=True, save_png=True,
                    save_html=True, open_browser=True, save_mp4=False, save_final=False, strip_title=True,
                    strip_subplot_titles=True):
        """
        Helper function to save figure as html file.

        Args:
            fig (plotly figure): figure object.
            name (str): name of html file.
            path (str): folder for saving file.
            remove_margins (bool, optional): remove white margins around EPS figure.
            width (int, optional): width of figures to be saved.
            height (int, optional): height of figures to be saved.
            save_eps (bool, optional): save image as EPS file.
            save_png (bool, optional): save image as PNG file.
            save_html (bool, optional): save image as html file.
            open_browser (bool, optional): open figure in the browse.
            save_mp4 (bool, optional): save video as MP4 file.
            save_final (bool, optional): whether to save the "good" final figure.
        """
        # disable MathJax globally for Kaleido when available
        try:
            if getattr(pio, 'kaleido', None) is not None and getattr(pio.kaleido, 'scope', None) is not None:
                pio.kaleido.scope.mathjax = None
        except Exception:
            pass
        # build path
        path = os.path.join(common.get_configs("output"))
        if not os.path.exists(path):
            os.makedirs(path)

        # build path for final figure
        path_final = self.folder_figures
        if save_final and not os.path.exists(path_final):
            os.makedirs(path_final)

        # limit name to max 200 char (for Windows)
        if len(path) + len(name) > 195 or len(path_final) + len(name) > 195:
            name = name[:200 - len(path) - 5]

        # Remove titles from the figure itself (keeps the HTML file name and address intact).
        # This affects both the HTML and any exported static images.
        if strip_title:
            try:
                fig.update_layout(title_text=None)
            except Exception:
                pass

        # Remove subplot titles created via make_subplots(..., subplot_titles=[...]).
        # These are stored as layout.annotations near the top of the figure.
        if strip_subplot_titles:
            try:
                anns = list(getattr(fig.layout, 'annotations', []) or [])
                kept = []
                for a in anns:
                    y = getattr(a, 'y', None)
                    xref = getattr(a, 'xref', None)
                    yref = getattr(a, 'yref', None)
                    showarrow = getattr(a, 'showarrow', None)

                    # Heuristic: subplot titles are paper referenced, arrowless annotations at y around 1.
                    if (xref == 'paper' and yref == 'paper' and showarrow is False and isinstance(y, (int, float)) and y >= 0.98):  # noqa: E501
                        continue
                    kept.append(a)
                fig.update_layout(annotations=kept)
            except Exception:
                pass

        # save as html
        if save_html:
            if open_browser:
                # open in browser
                py.offline.plot(fig, filename=os.path.join(path, name + '.html'))
                # also save the final figure
                if save_final:
                    py.offline.plot(fig, filename=os.path.join(path_final, name + '.html'), auto_open=False)
            else:
                # do not open in browser
                py.offline.plot(fig, filename=os.path.join(path, name + '.html'), auto_open=False)
                # also save the final figure
                if save_final:
                    py.offline.plot(fig, filename=os.path.join(path_final, name + '.html'), auto_open=False)

        # remove white margins
        if remove_margins:
            fig.update_layout(margin=dict(l=100, r=2, t=20, b=12))

        # save as eps
        if save_eps:
            fig.write_image(os.path.join(path, name + '.eps'), width=width, height=height)

            # also save the final figure
            if save_final:
                fig.write_image(os.path.join(path_final, name + '.eps'), width=width, height=height)

        # save as png
        if save_png:
            fig.write_image(os.path.join(path, name + '.png'), width=width, height=height)

            # also save the final figure
            if save_final:
                fig.write_image(os.path.join(path_final, name + '.png'), width=width, height=height)

        # save as mp4
        if save_mp4:
            fig.write_image(os.path.join(path, name + '.mp4'), width=width, height=height)

    def plot_kp(self, df, y: list, y_legend_kp=None, x=None, events=None, events_width=1,
                events_dash='dot', events_colour='black', events_annotations_font_size=20,
                events_annotations_colour='black', xaxis_title='Time (s)',
                yaxis_title='Percentage of trials with response key pressed',
                xaxis_title_offset=0, yaxis_title_offset=0,
                xaxis_range=None, yaxis_range=None, stacked=False,
                pretty_text=False, orientation='v', show_text_labels=False,
                name_file='kp', save_file=False, save_final=False,
                fig_save_width=1320, fig_save_height=680, legend_x=0.7, legend_y=0.95, legend_columns=1,
                font_family=None, font_size=None, ttest_signals=None, ttest_marker='circle',
                ttest_marker_size=3, ttest_marker_colour='black', ttest_annotations_font_size=10,
                ttest_annotation_x=0, ttest_annotations_colour='black', anova_signals=None, anova_marker='cross',
                anova_marker_size=3, anova_marker_colour='black', anova_annotations_font_size=10,
                anova_annotations_colour='black', ttest_anova_row_height=0.5, xaxis_step=5,
                yaxis_step=5, y_legend_bar=None, line_width=1, bar_font_size=None,
                custom_line_colors=None, custom_line_dashes=None, flag_trigger=False, margin=None,
                cross_p1_times=None, cross_p1_marker='diamond',
                cross_p1_marker_size=10, cross_p1_marker_colour='black'):
        """
        Plots keypress (response) data from a dataframe using Plotly, with options for custom lines,
        annotations, t-test and ANOVA result overlays, event markers, per-line cross_p1 markers,
        and customisable styling and saving.
        """

        logger.info('Creating keypress figure.')
        # calculate times
        times = df['Timestamp'].values
        # plotly
        fig = go.Figure()

        # ensure yaxis_range is mutable if provided as a tuple
        if isinstance(yaxis_range, tuple):
            yaxis_range = list(yaxis_range)

        # track plotted values to compute min/max for ticks
        all_values = []

        # plot keypress data
        for row_number, key in enumerate(y):
            values = df[key]
            if y_legend_kp:
                name = y_legend_kp[row_number]
            else:
                name = key

            # smoothen signal
            if self.smoothen_signal:
                if isinstance(values, pd.Series):
                    # Replace NaNs with 0 before smoothing
                    values = values.fillna(0).tolist()
                    values = self.smoothen_filter(values)
            else:
                # If not smoothing, ensure no NaNs anyway
                if isinstance(values, pd.Series):
                    values = values.fillna(0).tolist()
                else:
                    values = [v if not pd.isna(v) else 0 for v in values]

            # convert to 0-100%
            if flag_trigger:
                values = [v * 100 for v in values]  # type: ignore
            else:
                values = [v for v in values]  # type: ignore

            # collect values for y-axis tick range
            all_values.extend(values)  # type: ignore

            name = y_legend_kp[row_number] if y_legend_kp else key

            # main line
            fig.add_trace(go.Scatter(
                y=values,
                mode='lines',
                x=times,
                line=dict(
                    width=line_width,
                    color=custom_line_colors[row_number] if custom_line_colors else None,
                    dash=custom_line_dashes[row_number] if custom_line_dashes else None,
                ),
                name=name
            ))

            # --- NEW: marker for cross_p1_time_s on this line ---
            if cross_p1_times and name in cross_p1_times:
                t_cross = cross_p1_times[name]

                # find nearest timestamp index (handles small timing mismatches)
                times_array = np.array(times, dtype=float)
                idx = int(np.abs(times_array - t_cross).argmin())

                x_marker = float(times_array[idx])
                y_marker = values[idx]

                fig.add_trace(go.Scatter(
                    x=[x_marker],
                    y=[y_marker],
                    mode='markers',
                    marker=dict(
                        symbol=cross_p1_marker,
                        size=cross_p1_marker_size,
                        color=cross_p1_marker_colour,
                    ),
                    name=f"{name} P1 cross",
                    showlegend=False
                ))

        # --- if no yaxis_range provided, derive it from the data so it's never None ---
        if yaxis_range is None:
            if all_values:  # safeguard against empty data
                actual_ymin = min(all_values)
                actual_ymax = max(all_values)
                yaxis_range = [actual_ymin, actual_ymax]
            else:
                # fallback range if for some reason there's no data
                yaxis_range = [0, 1]

        # draw events
        HMD_helper.draw_events(fig=fig,
                               yaxis_range=yaxis_range,
                               events=events,
                               events_width=events_width,
                               events_dash=events_dash,
                               events_colour=events_colour,
                               events_annotations_font_size=events_annotations_font_size,
                               events_annotations_colour=events_annotations_colour)

        # update x-axis
        if xaxis_step:
            fig.update_xaxes(title_text=xaxis_title,
                             range=xaxis_range,
                             dtick=xaxis_step,
                             title_font=dict(family=font_family,
                                             size=common.get_configs('font_size'))
                             )
        else:
            fig.update_xaxes(title_text=xaxis_title,
                             range=xaxis_range,
                             title_font=dict(family=font_family,
                                             size=common.get_configs('font_size')))

        # Find the actual y range across all series for tick generation only.
        if all_values:
            actual_ymin = float(min(all_values))
            actual_ymax = float(max(all_values))
        else:
            actual_ymin = float(yaxis_range[0])
            actual_ymax = float(yaxis_range[1])

        # Generate ticks from 0 up to actual_ymax.
        positive_ticks = np.arange(0, actual_ymax + yaxis_step, yaxis_step, dtype=float)

        # Generate ticks from 0 down to actual_ymin when needed.
        negative_ticks = np.arange(0, actual_ymin - yaxis_step, -yaxis_step, dtype=float)

        # Combine and sort ticks.
        visible_ticks = np.sort(np.unique(np.concatenate((negative_ticks, positive_ticks))))

        def _fmt_tick(value: float) -> str:
            return str(int(value)) if float(value).is_integer() else f"{value:.2f}"

        tick_labels = [_fmt_tick(float(t)) for t in visible_ticks]

        # Update y-axis with only relevant tick marks
        fig.update_yaxes(
            showgrid=True,
            range=yaxis_range,
            tickvals=visible_ticks,  # only show ticks for data range
            ticktext=tick_labels,
            automargin=True,
            title=dict(
                text="",
                font=dict(family=font_family,
                          size=common.get_configs('font_size')),
                standoff=0
            )
        )

        fig.add_annotation(
            text=yaxis_title,
            xref='paper',
            yref='paper',
            x=xaxis_title_offset,     # still left side
            y=0.5 + yaxis_title_offset,
            showarrow=False,
            textangle=-90,
            font=dict(family=font_family,
                      size=common.get_configs('font_size')),
            xanchor='center',
            yanchor='middle'
        )

        # prettify text
        if pretty_text:
            for variable in y:
                # check if column contains strings
                if isinstance(df.iloc[0][variable], str):
                    # replace underscores with spaces
                    df[variable] = df[variable].str.replace('_', ' ')
                    # capitalise
                    df[variable] = df[variable].str.capitalize()

        # use the dataframe index when no explicit x values are given
        if x is None:
            x = df.index

        # draw ttest and anova rows
        self.draw_ttest_anova(fig=fig,
                              times=times,
                              name_file=name_file,
                              yaxis_range=yaxis_range,
                              yaxis_step=yaxis_step,
                              ttest_signals=ttest_signals,
                              ttest_marker=ttest_marker,
                              ttest_marker_size=ttest_marker_size,
                              ttest_marker_colour=ttest_marker_colour,
                              ttest_annotations_font_size=ttest_annotations_font_size,
                              ttest_annotations_colour=ttest_annotations_colour,
                              anova_signals=anova_signals,
                              anova_marker=anova_marker,
                              anova_marker_size=anova_marker_size,
                              anova_marker_colour=anova_marker_colour,
                              anova_annotations_font_size=anova_annotations_font_size,
                              anova_annotations_colour=anova_annotations_colour,
                              ttest_anova_row_height=ttest_anova_row_height,
                              ttest_annotation_x=ttest_annotation_x,
                              flag_trigger=flag_trigger)

        # update template
        fig.update_layout(template=self.template)

        # format text labels
        if show_text_labels:
            fig.update_traces(texttemplate='%{text:.2f}')

        # stacked bar chart
        if stacked:
            fig.update_layout(barmode='stack')

        # legend
        if legend_columns == 1:  # single column
            fig.update_layout(legend=dict(x=legend_x,
                                          y=legend_y,
                                          bgcolor='rgba(0,0,0,0)',
                                          font=dict(family=font_family,
                                                    size=common.get_configs('font_size') - 6)))

        # multiple columns
        elif legend_columns == 2:
            fig.update_layout(
                legend=dict(
                    x=legend_x,
                    y=legend_y,
                    bgcolor='rgba(0,0,0,0)',
                    font=dict(size=common.get_configs('font_size')),
                    orientation='h',
                    traceorder='normal',
                    itemwidth=30,
                    itemsizing='constant'
                ),
                legend_title_text='',
                legend_tracegroupgap=5,
                legend_groupclick='toggleitem',
                legend_itemclick='toggleothers',
                legend_itemdoubleclick='toggle',
            )

        # adjust margins because of hardcoded ylim axis
        if margin:
            fig.update_layout(margin=margin)

        # update font family
        fig.update_layout(font=dict(family=common.get_configs('font_family')))

        # update font size
        fig.update_layout(font=dict(size=common.get_configs('font_size')))

        # save file to local output folder
        if save_file:
            self.save_plotly(fig=fig,
                             name=name_file,
                             remove_margins=False,
                             width=fig_save_width,
                             height=fig_save_height,
                             save_final=save_final)
        else:
            fig.show()

    def ttest(self, signal_1, signal_2, type='two-sided', paired=True):
        """
        Perform a t-test on two signals, computing p-values and significance.

        Args:
            signal_1 (list): First signal, a list of numeric values.
            signal_2 (list): Second signal, a list of numeric values.
            type (str, optional): Type of t-test to perform. Options are "two-sided",
                                  "greater", or "less". Defaults to "two-sided".
            paired (bool, optional): Indicates whether to perform a paired t-test
                                     (ttest_rel) or an independent t-test (ttest_ind).
                                     Defaults to True (paired).

        Returns:
            list: A list containing two elements:
                  - p_values (list): Raw p-values for each bin.
                  - significance (list): Binary flags (0 or 1) indicating whether
                    the p-value for each bin is below the threshold configured in
                    tr.common.get_configs('p_value').
        """
        # Check if the lengths of the two signals are the same
        if len(signal_1) != len(signal_2):
            logger.error('The lengths of signal_1 and signal_2 must be the same.')
            return -1

        p_values = []
        significance = []
        threshold = common.get_configs("p_value")

        for i in range(len(signal_1)):
            data1 = signal_1[i]
            data2 = signal_2[i]

            # Skip if data is empty
            if not data1 or not data2 or (paired and len(data1) != len(data2)):
                p_values.append(1.0)
                significance.append(0)
                continue

            try:
                if paired:
                    t_stat, p_val = ttest_rel(data1, data2, alternative=type)  # type: ignore
                else:
                    t_stat, p_val = ttest_ind(data1, data2, equal_var=False, alternative=type)  # type: ignore

                # Handles the nan cases
                if np.isnan(p_val):  # type: ignore
                    p_val = 1.0
            except Exception as e:
                logger.warning(f"Skipping t-test at time index {i} due to error: {e}")
                p_val = 1.0

            p_values.append(p_val)
            significance.append(int(p_val < threshold))

        return [p_values, significance]

    def avg_csv_files(self, data_folder, mapping):
        """
        Averages multiple CSV files corresponding to the same video ID. Each file is expected to contain
        time-series data, including quaternion rotations and potentially other columns. The output is a
        CSV file with averaged values for each timestamp across the files.

        Parameters:
            data_folder (str): Path to the folder containing input CSV files.
            mapping (pd.DataFrame): A DataFrame containing metadata, including 'video_id' and 'video_length'.

        Outputs:
            For each video_id, saves an averaged DataFrame as a CSV in the output directory.
            The output CSV is named as "<video_id>_avg_df.csv".
        """

        # Group file paths by video_id using a helper function
        grouped_data = HMD_class.group_files_by_video_id(data_folder, mapping)

        # calculate resolution based on the param in
        resolution = common.get_configs("yaw_resolution") / 1000.0

        # Process each video ID and its associated files
        logger.info("Exporting CSV files.")
        for video_id, file_locations in tqdm(grouped_data.items()):
            all_dfs = []

            # Retrieve the video length from the mapping DataFrame
            video_length_row = mapping.loc[mapping["video_id"] == video_id, "video_length"]
            if video_length_row.empty:
                logger.info(f"Video length not found for video_id: {video_id}")
                continue

            video_length = video_length_row.values[0] / 1000  # Convert milliseconds to seconds

            # Read and process each file associated with the video ID
            for file_location in file_locations:
                df = pd.read_csv(file_location)

                # Filter the DataFrame to only include rows where Timestamp >= 0 and <= video_length
                # todo: 0.01 hardcoded value does not work?
                df = df[(df["Timestamp"] >= 0) & (df["Timestamp"] <= video_length + 0.01)]

                # Round the Timestamp to the nearest multiple of resolution
                df["Timestamp"] = ((df["Timestamp"] / resolution).round() * resolution).astype(float)

                all_dfs.append(df)

            # Skip if no dataframes were collected
            if not all_dfs:
                continue

            # Concatenate all DataFrames row-wise
            combined_df = pd.concat(all_dfs, ignore_index=True)

            # Group by 'Timestamp'
            grouped = combined_df.groupby('Timestamp')

            avg_rows = []
            for timestamp, group in grouped:
                row = {'Timestamp': timestamp}

                # Perform SLERP-based quaternion averaging if quaternion columns are present
                if {"HMDRotationW", "HMDRotationX", "HMDRotationY", "HMDRotationZ"}.issubset(group.columns):
                    quats = group[["HMDRotationW", "HMDRotationX", "HMDRotationY", "HMDRotationZ"]].values.tolist()
                    avg_quat = HMD_class.average_quaternions_eigen(quats)
                    row.update({
                        "HMDRotationW": avg_quat[0],
                        "HMDRotationX": avg_quat[1],
                        "HMDRotationY": avg_quat[2],
                        "HMDRotationZ": avg_quat[3],
                    })

                # Average all remaining columns (excluding Timestamp and quaternion cols)
                other_cols = [col for col in group.columns if col not in ["Timestamp",
                                                                          "HMDRotationW",
                                                                          "HMDRotationX",
                                                                          "HMDRotationY",
                                                                          "HMDRotationZ"]]
                for col in other_cols:
                    row[col] = group[col].mean()

                avg_rows.append(row)

            # Create a new DataFrame from the averaged rows
            avg_df = pd.DataFrame(avg_rows)

            # Save dataframe in the output folder
            avg_df.to_csv(os.path.join(common.get_configs("output"), f"{video_id}_avg_df.csv"), index=False)

    def draw_ttest_anova(self, fig, times, name_file, yaxis_range, yaxis_step, ttest_signals, ttest_marker,
                         ttest_marker_size, ttest_marker_colour, ttest_annotations_font_size, ttest_annotations_colour,
                         anova_signals, anova_marker, anova_marker_size, anova_marker_colour,
                         anova_annotations_font_size, anova_annotations_colour, ttest_anova_row_height,
                         ttest_annotation_x, flag_trigger=False):
        """Draw ttest and anova test rows.

        Args:
            fig (figure): figure object.
            name_file (str): name of file to save.
            yaxis_range (list): range of y axis in format [min, max] for the keypress plot.
            yaxis_step (int): step between ticks on y axis.
            ttest_signals (list): signals to compare with ttest. None = do not compare.
            ttest_marker (str): symbol of markers for the ttest.
            ttest_marker_size (int): size of markers for the ttest.
            ttest_marker_colour (str): colour of markers for the ttest.
            ttest_annotations_font_size (int): font size of annotations for ttest.
            ttest_annotations_colour (str): colour of annotations for ttest.
            anova_signals (dict): signals to compare with ANOVA. None = do not compare.
            anova_marker (str): symbol of markers for the ANOVA.
            anova_marker_size (int): size of markers for the ANOVA.
            anova_marker_colour (str): colour of markers for the ANOVA.
            anova_annotations_font_size (int): font size of annotations for ANOVA.
            anova_annotations_colour (str): colour of annotations for ANOVA.
            ttest_anova_row_height (float): height of row of ttest/anova markers in y units.
        """
        # Save original axis limits (bottom/top of the main data area)
        original_min, original_max = yaxis_range
        # Counters for marker rows
        counter_ttest = 0
        counter_anova = 0

        # calculate resolution based on the param
        if flag_trigger:
            resolution = common.get_configs("kp_resolution") / 1000.0
        else:
            resolution = common.get_configs("yaw_resolution") / 1000.0

        # --- t-test markers ---
        if ttest_signals:
            for comp in ttest_signals:
                p_vals, sig = self.ttest(
                    signal_1=comp['signal_1'], signal_2=comp['signal_2'], paired=comp['paired']
                )  # type: ignore

                # Save csv
                # TODO: rounding to 2 is hardcoded and wrong?
                times_csv = [round(i * resolution, 2) for i in range(len(comp['signal_1']))]
                self.save_stats_csv(t=times_csv,
                                    p_values=p_vals,
                                    name_file=f"{comp['label']}_{name_file}.csv")

                if any(sig):
                    xs, ys = [], []

                    # Place this row below the curves, one row further down per comparison
                    # (same logic for kp/yaw; ttest_anova_row_height is in the same units as y)
                    y_offset = original_min - ttest_anova_row_height * (counter_ttest + 1)

                    for i, s in enumerate(sig):
                        if s:
                            xs.append(times[i])
                            ys.append(y_offset)

                    # plot markers
                    for x, y, p_val in zip(xs, ys, p_vals):
                        fig.add_annotation(
                            x=x,
                            y=y,
                            text='*',  # TODO: use ttest_marker symbol if desired
                            showarrow=False,
                            yanchor='middle',
                            font=dict(family=common.get_configs("font_family"),
                                      size=ttest_marker_size,
                                      color=ttest_marker_colour),
                            hovertext=f"{comp['label']}: time={x}, p={p_val}",
                            hoverlabel=dict(bgcolor="white"),
                        )

                    # label row
                    fig.add_annotation(x=ttest_annotation_x,
                                       y=y_offset,
                                       text=comp['label'],
                                       xanchor='right',
                                       showarrow=False,
                                       font=dict(family=common.get_configs("font_family"),
                                                 size=ttest_annotations_font_size,
                                                 color=ttest_annotations_colour))
                    counter_ttest += 1

        # TODO: ANOVA support is currently broken in original code; left untouched other than counting.
        # If you later add ANOVA rows, increment `counter_anova` similarly and compute their y_offset.

        # --- Adjust axis to include marker rows ---
        if counter_ttest or counter_anova:
            n_rows = max(counter_ttest, counter_anova)
            # Extend the axis downward enough to include all rows, plus one extra row of padding
            min_y = original_min - ttest_anova_row_height * (n_rows + 1)

            fig.update_layout(yaxis=dict(
                range=[min_y, original_max],
                dtick=yaxis_step,
                tickformat='.2f'
            ))

    def save_stats_csv(self, t, p_values, name_file):
        """Save results of statistical test in csv.

        Args:
            t (list): list of time slices.
            p_values (list): list of p values.
            name_file (str): name of file.
        """
        path = os.path.join(common.get_configs("output"), self.folder_stats)  # where to save csv
        # build path
        if not os.path.exists(path):
            os.makedirs(path)
        df = pd.DataFrame(columns=['t', 'p-value'])  # dataframe to save to csv
        df['t'] = t
        df['p-value'] = p_values
        df.to_csv(os.path.join(path, name_file))

    @staticmethod
    def draw_events(fig, yaxis_range, events, events_width, events_dash, events_colour,
                    events_annotations_font_size, events_annotations_colour):
        """Draw vertical lines and text labels for events (no arrows), with grouping by 'id'.

        - Events with the same 'id' share a horizontal row near the top of the plot.
        - Row with id == 1 (e.g. 'Car decelerates', 'Car stops', 'Car accelerates')
          is placed very close to the top.
        - Labels are horizontally centered on their own vertical line (x = start).
        """

        if not events:
            return

        y_min, y_max = yaxis_range
        height = max(y_max - y_min, 1e-6)  # avoid zero height

        # Group events by 'id'. Events without an id get their own group.
        groups = {}
        for idx, ev in enumerate(events):
            key = ev.get("id")
            if key is None:
                key = f"_noid_{idx}"
            groups.setdefault(key, []).append(idx)

        # Base spacing between row bands (for ids other than 1)
        row_height_frac = 0.10    # fraction of plot height between rows
        base_offset_frac = 0.05   # offset below top of plot for non-id-1 rows

        # Iterate bands in insertion order (id=1 first, then id=2, etc.)
        for row_index, (group_key, idx_list) in enumerate(groups.items()):
            if not idx_list:
                continue

            # Fractional vertical position for this row
            if str(group_key) == "1":
                # put id=1 row very close to the top
                frac = 0.01
            else:
                frac = base_offset_frac + row_index * row_height_frac

            # keep inside the plot
            frac = min(max(frac, 0.0), 0.95)

            label_y = y_max - frac * height

            for event_index in idx_list:
                ev = events[event_index]
                start = float(ev["start"])
                end = float(ev["end"])
                label = ev.get("annotation", "")

                # Center label horizontally on its own line
                label_x = 0.5 * (start + end) if start != end else start

                # --- Vertical line(s) ---
                fig.add_shape(
                    type="line",
                    x0=start,
                    y0=y_min,
                    x1=start,
                    y1=y_max,
                    line=dict(
                        color=events_colour,
                        dash=events_dash,
                        width=events_width,
                    ),
                )

                if start != end:
                    fig.add_shape(
                        type="line",
                        x0=end,
                        y0=y_min,
                        x1=end,
                        y1=y_max,
                        line=dict(
                            color=events_colour,
                            dash=events_dash,
                            width=events_width,
                        ),
                    )

                # --- Text label ---
                fig.add_annotation(
                    text=label,
                    x=label_x,
                    y=label_y,
                    xanchor="center",
                    yanchor="bottom",
                    showarrow=False,
                    font=dict(
                        family=common.get_configs("font_family"),
                        size=int(events_annotations_font_size * 3.3),
                        color=events_annotations_colour,
                    ),
                )

    def export_participant_trigger_matrix(self, data_folder, video_id, output_file, column_name, mapping):
        """
        Export a matrix of trigger (or other column) values per participant for a given video.

        Each cell contains a list of values (one per frame or timepoint) for that participant and timestamp.
        Missing data is left as NaN, not zero.

        Args:
            data_folder (str): Path to folder containing participant subfolders with CSVs.
            video_id (str): Target video identifier (e.g. '002', 'test', etc.).
            output_file (str): Path to output CSV file (e.g. '_output/participant_trigger_002.csv').
            column_name (str): Name of the column to export (e.g. 'TriggerValueRight').
            mapping (pd.DataFrame): Mapping DataFrame containing at least 'video_id' and 'video_length'.
        """

        participant_matrix = {}    # Store trigger value lists for each participant, keyed by timestamp
        all_timestamps = set()     # Collect all observed timestamps for alignment

        # Calculate time bin resolution (in seconds) from config
        resolution = common.get_configs("kp_resolution") / 1000.0

        # Iterate over participant folders
        for folder in sorted(os.listdir(data_folder)):
            folder_path = os.path.join(data_folder, folder)
            if not os.path.isdir(folder_path):
                continue  # Ignore files, only process directories

            # Extract participant ID from folder name (expecting "Participant_###_...")
            match = re.match(r'Participant_(\d+)', folder)
            if not match:
                continue
            participant_id = int(match.group(1))

            # Search for this participant's file matching the video ID
            for file in os.listdir(folder_path):
                if f"{video_id}.csv" in file:
                    file_path = os.path.join(folder_path, file)
                    df = pd.read_csv(file_path)

                    # Check required columns
                    if "Timestamp" not in df or column_name not in df:
                        continue

                    # Bin timestamps to specified resolution
                    df["Timestamp"] = ((df["Timestamp"] / resolution).round() * resolution).round(2)

                    # Group by timestamp, collect all values in a list per bin
                    grouped = df.groupby("Timestamp", as_index=True)[column_name].apply(list)

                    # Store the resulting dict: timestamp -> list of values
                    participant_matrix[f"P{participant_id}"] = grouped.to_dict()
                    all_timestamps.update(grouped.index)
                    break  # Only process the first matching file for this participant

        # Get the expected timeline from mapping for alignment (using video_length)
        video_length_row = mapping.loc[mapping["video_id"] == video_id, "video_length"]
        if not video_length_row.empty:
            video_length_sec = video_length_row.values[0] / 1000  # Convert ms to seconds
            all_timestamps = np.round(np.arange(0.0, video_length_sec + resolution, resolution), 2).tolist()
        else:
            logger.warning(f"Video length not found in mapping for video_id {video_id}")

        # Build DataFrame with one row per timestamp
        combined_df = pd.DataFrame({"Timestamp": all_timestamps})

        # For each participant, add a column: each entry is a list or NaN (if no data for that timestamp)
        for participant, values in participant_matrix.items():
            combined_df[participant] = combined_df["Timestamp"].map(values)

        # Save matrix to CSV (do NOT fill missing with zero; keep NaN for clarity)
        combined_df.to_csv(output_file, index=False)

    def export_participant_quaternion_matrix(self, data_folder, video_id, output_file, mapping, overwrite=False):
        """
        Export a matrix of raw HMD quaternions per participant per timestamp for a given video.
        If overwrite=False and output_file exists, it is reused.
        """

        # short-circuit if already exists
        if not overwrite and os.path.isfile(output_file):
            return

        participant_matrix = {}
        all_timestamps = set()

        resolution = common.get_configs("yaw_resolution") / 1000.0

        for folder in sorted(os.listdir(data_folder)):
            folder_path = os.path.join(data_folder, folder)
            if not os.path.isdir(folder_path):
                continue

            match = re.match(r"Participant_(\d+)$", folder, re.IGNORECASE)
            if not match:
                continue

            participant_id = int(match.group(1))

            for file in os.listdir(folder_path):
                if file == f"{video_id}.csv":
                    file_path = os.path.join(folder_path, file)
                    df = pd.read_csv(file_path)

                    required_cols = {
                        "Timestamp",
                        "HMDRotationW",
                        "HMDRotationX",
                        "HMDRotationY",
                        "HMDRotationZ",
                    }
                    if not required_cols.issubset(df.columns):
                        continue

                    df["Timestamp"] = (
                        (df["Timestamp"] / resolution).round() * resolution
                    ).round(2)

                    quats_by_time = (
                        df.groupby("Timestamp")[
                            ["HMDRotationW", "HMDRotationX", "HMDRotationY", "HMDRotationZ"]
                        ]
                        .apply(lambda g: g.values.tolist())
                        .to_dict()
                    )

                    participant_matrix[f"P{participant_id}"] = quats_by_time
                    all_timestamps.update(quats_by_time.keys())
                    break

        video_length_row = mapping.loc[mapping["video_id"] == video_id, "video_length"]
        if not video_length_row.empty:
            video_length_sec = video_length_row.values[0] / 1000.0
            all_timestamps = (
                np.round(
                    np.arange(0, video_length_sec + resolution, resolution), 2
                ).tolist()
            )
        else:
            logger.warning(f"Video length not found in mapping for video_id {video_id}")
            all_timestamps = sorted(all_timestamps)

        combined_df = pd.DataFrame({"Timestamp": all_timestamps})
        for participant, values in participant_matrix.items():
            combined_df[participant] = combined_df["Timestamp"].map(
                lambda ts: str(values.get(ts, []))
            )

        combined_df.to_csv(output_file, index=False)

    def plot_column(self, mapping, column_name="TriggerValueRight", parameter=None, parameter_value=None,
                    additional_parameter=None, additional_parameter_value=None,
                    compare_trial="video_1", xaxis_title=None, yaxis_title=None, xaxis_range=None,
                    yaxis_range=[0, 100], margin=None, name=None):
        """
        Generate a comparison plot of keypress data (or other time-series columns) and subjective slider ratings
        across multiple video trials relative to a test/reference condition.

        This function processes participant trigger matrices for each trial,
        aligns timestamps, attaches slider-based subjective ratings (annoyance,
        informativeness, noticeability), and prepares data for visualisation,
        including significance testing (paired t-tests) between the test trial and each comparison trial.

        Args:
            mapping (pd.DataFrame): DataFrame containing video metadata, including
                'video_id', 'sound_clip_name', 'display_name', and 'colour'.
            column_name (str): The column to extract for plotting (e.g., 'TriggerValueRight').
            xaxis_title (str, optional): Custom label for the x-axis.
            yaxis_title (str, optional): Custom label for the y-axis.
            xaxis_range (list, optional): x-axis [min, max] limits for the plot.
            yaxis_range (list, optional): y-axis [min, max] limits for the plot.
            margin (dict, optional): Custom plot margin dictionary.
        """

        # make yaxis_range mutable if it's a tuple
        if isinstance(yaxis_range, tuple):
            yaxis_range = list(yaxis_range)

        # === Filter mapping to same video_length as reference trial ===
        lens = mapping.loc[mapping["video_id"].eq(compare_trial), "video_length"].unique()

        if len(lens) == 0:
            raise ValueError(f"No rows found for video_id='{compare_trial}'")
        elif len(lens) > 1:
            # same video_id appears with different lengths; keep all those lengths
            mapping_filtered = mapping[mapping["video_length"].isin(lens)].copy()
        else:
            mapping_filtered = mapping[mapping["video_length"].eq(lens[0])].copy()

        if parameter is not None:
            mapping_filtered = mapping_filtered[mapping_filtered[parameter] == parameter_value]

        if additional_parameter is not None:
            mapping_filtered = mapping_filtered[mapping_filtered[additional_parameter] == additional_parameter_value]

        # Filter out control/test video IDs for comparison
        mapping_filtered = mapping_filtered[~mapping_filtered["video_id"].isin(["baseline_1", "baseline_2"])]
        plot_videos = mapping_filtered["video_id"]

        # Prepare containers for results and stats
        all_dfs = []        # averaged time-series for each trial
        all_labels = []     # display names for legend
        all_video_ids = []  # original video ids in plotting order
        ttest_signals = []  # for significance testing

        # === Export trigger matrix for test (reference) trial ===
        test_output_csv = os.path.join(
            common.get_configs("output"),
            f"participant_{column_name}_{compare_trial}.csv"
        )

        self.export_participant_trigger_matrix(
            data_folder=self.data_folder,
            video_id=compare_trial,
            output_file=test_output_csv,
            column_name=column_name,
            mapping=mapping_filtered
        )

        # Read matrix and extract time-series for the test trial
        test_raw_df = pd.read_csv(test_output_csv)
        test_matrix = extra_class.extract_time_series_values(test_raw_df)

        # === Loop through each trial (including reference) ===
        for video in plot_videos:
            # Get human-readable display name for this trial
            if "display_name" in mapping_filtered.columns:
                display_name_series = mapping_filtered.loc[
                    mapping_filtered["video_id"] == video, "display_name"
                ].dropna()
                display_name = display_name_series.iloc[0] if not display_name_series.empty else video
            else:
                display_name = video

            trial_output_csv = os.path.join(
                common.get_configs("output"),
                f"participant_{column_name}_{video}.csv"
            )

            # Export trigger matrix for this video
            self.export_participant_trigger_matrix(
                data_folder=self.data_folder,
                video_id=video,
                output_file=trial_output_csv,
                column_name=column_name,
                mapping=mapping_filtered
            )

            # Read and process the trigger matrix to extract time series for this trial
            trial_raw_df = pd.read_csv(trial_output_csv)
            trial_matrix = extra_class.extract_time_series_values(trial_raw_df)

            # Compute participant-averaged time series (by timestamp) for this trial
            avg_df = extra_class.average_dataframe_vectors_with_timestamp(
                trial_raw_df,
                column_name=f"{column_name}"
            )

            all_dfs.append(avg_df)
            all_labels.append(display_name)
            all_video_ids.append(video)

            # Prepare paired t-test between reference trial and each comparison trial
            if video != compare_trial:
                ttest_signals.append({
                    "signal_1": test_matrix,
                    "signal_2": trial_matrix,
                    "paired": True,
                    "label": f"{display_name}"
                })

        # === Combine all trial DataFrames for multi-trial plotting ===
        if not all_dfs:
            raise RuntimeError("No data frames found to plot.")

        combined_df = pd.DataFrame()
        combined_df["Timestamp"] = all_dfs[0]["Timestamp"]

        for df, label in zip(all_dfs, all_labels):
            combined_df[label] = df[column_name]

        # === Helper for event times (ignore ±0.02 s by rounding + mode) ===
        def _get_mode_time(df, col, round_decimals=2):
            """Return the mode of a time column, ignoring NaNs and
            small numeric differences by rounding first."""
            if col not in df.columns:
                return None

            series = df[col].dropna()
            if series.empty:
                return None

            rounded = series.round(round_decimals)
            mode_vals = rounded.mode()
            if mode_vals.empty:
                return None

            return float(mode_vals.iloc[0])

        # === Build events from mapping_filtered timing columns ===
        events = []

        # Row 1: all main car events at same height
        first_row_events = [
            ("yield_start_time_s", "Car decelerates"),
            ("yield_stop_time_s",  "Car stops"),
            ("yield_resume_time_s", "Car accelerates"),
        ]
        for col_name, label in first_row_events:
            t = _get_mode_time(mapping_filtered, col_name)
            if t is not None and not np.isnan(t):
                events.append({
                    "id": 1,
                    "start": t,
                    "end": t,
                    "annotation": label
                })

        # Row 2: crossing event on its own line below
        second_row_events = [
            ("cross_p2_time_s", "Car crosses the 1st pedestrian"),
        ]
        for col_name, label in second_row_events:
            t = _get_mode_time(mapping_filtered, col_name)
            if t is not None and not np.isnan(t):
                events.append({
                    "id": 2,
                    "start": t,
                    "end": t,
                    "annotation": label
                })

        has_top_row = any(ev.get("id") == 1 for ev in events)
        if not has_top_row:
            for ev in events:
                if ev.get("id") is not None:
                    ev["id"] = 1

        # === cross_p1_time_s: per-line marker time for each video ===
        cross_p1_times = {}
        if "cross_p1_time_s" in mapping_filtered.columns:
            for video, label in zip(plot_videos, all_labels):
                series = mapping_filtered.loc[
                    mapping_filtered["video_id"] == video, "cross_p1_time_s"
                ].dropna()
                if not series.empty:
                    cross_p1_times[label] = float(series.iloc[0])

        # === Set line style: dashed for reference (compare_trial), solid for others ===
        custom_line_dashes = []
        for vid in all_video_ids:
            if vid == compare_trial:
                custom_line_dashes.append("dot")
            else:
                custom_line_dashes.append("solid")

        # === Generate the main plot (delegated to plot_kp helper) ===
        self.plot_kp(
            df=combined_df,
            y=all_labels,
            y_legend_kp=all_labels,
            yaxis_range=yaxis_range,
            xaxis_range=xaxis_range,
            xaxis_title=xaxis_title,  # type: ignore
            yaxis_title=yaxis_title,  # type: ignore
            xaxis_title_offset=-0.04,  # type: ignore
            yaxis_title_offset=0.18,   # type: ignore
            name_file=f"all_videos_kp_slider_plot_{name}",
            show_text_labels=True,
            pretty_text=True,
            events=events,
            events_width=2,
            events_annotations_font_size=common.get_configs("font_size") - 8,
            stacked=False,
            ttest_signals=ttest_signals,
            ttest_anova_row_height=6,
            ttest_annotations_font_size=common.get_configs("font_size") - 8,
            ttest_annotation_x=0.001,  # type: ignore
            ttest_marker="circle",
            ttest_marker_size=common.get_configs("font_size")-6,
            legend_x=0,
            legend_y=1.225,
            legend_columns=2,
            xaxis_step=1,
            yaxis_step=20,  # type: ignore
            line_width=3,
            font_size=common.get_configs("font_size"),
            fig_save_width=1470,
            fig_save_height=850,
            save_file=True,
            save_final=True,
            custom_line_dashes=custom_line_dashes,
            flag_trigger=True,
            margin=margin,
            cross_p1_times=cross_p1_times,
        )
