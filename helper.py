import pandas as pd
import os
import glob
import plotly.graph_objects as go
import plotly as py
import plotly.io as pio
import plotly.express as px
from plotly.subplots import make_subplots
# For OneEuroFilter, see https://github.com/casiez/OneEuroFilter
from OneEuroFilter import OneEuroFilter
import common
from custom_logger import CustomLogger
import re
import numpy as np
from scipy.stats import ttest_rel, ttest_ind
# This project is sometimes run with different repository layouts:
# some copies keep these modules under ``utils/`` while others keep them at the
# project root. Support both.
try:
    from utils.HMD_helper import HMD_yaw  # type: ignore
except Exception:  # pragma: no cover
    from utils.HMD_helper import HMD_yaw  # type: ignore

try:
    from utils.tools import Tools  # type: ignore
except Exception:  # pragma: no cover
    from utils.tools import Tools  # type: ignore
from tqdm import tqdm
from datetime import datetime
import ast
from typing import Optional

import math

# ---------------------------------------------------------------------------
# Config helpers (support both legacy single-dataset keys and new shuffled/unshuffled keys)
# ---------------------------------------------------------------------------

def _safe_get_config(key: str, default=None):
    """Return common.get_configs(key) if available, else default."""
    try:
        val = common.get_configs(key)
    except Exception:
        return default
    return default if val is None else val


def _resolve_data_folder(dataset: Optional[str] = None) -> str:
    """Resolve participant_response folder from configs.

    Supports:
    - legacy:  data
    - new:     shuffled_data / unshuffled_data (when dataset is provided)
    - fallback: shuffled_data then unshuffled_data
    """
    if dataset:
        v = _safe_get_config(f"{dataset}_data", None)
        if v:
            return v
    v = _safe_get_config("data", None)
    if v:
        return v
    v = _safe_get_config("shuffled_data", None)
    if v:
        return v
    v = _safe_get_config("unshuffled_data", None)
    if v:
        return v
    # Last resort: keep old behavior but make it explicit.
    return ""



logger = CustomLogger(__name__)  # use custom logger

HMD_class = HMD_yaw()
extra_class = Tools()

# Consts
SAVE_PNG = True
SAVE_EPS = True
output = common.get_configs("output")
plotly_template = common.get_configs("plotly_template")
font_size = common.get_configs("font_size")
font_family = common.get_configs("font_family")


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
            logger.error('Specified filter {} not implemented.', type_flter)
            return -1

    def plot_column_distribution(self, df, columns, output_folder, save_file=True, tag=None):
        """
        Plots and prints distributions of specified survey columns.

        Parameters:
            df (DataFrame or str): DataFrame or path to CSV.
            columns (list): List of column names to analyse.
            output_folder (str): Folder where plots will be saved.
            save_file (bool): Whether to save plots or just show them.
        """
        if isinstance(df, str):
            df = pd.read_csv(df)

        for column in columns:
            if column not in df.columns:
                logger.error(f"Column not found: {column}")
                continue

            logger.info(f"Distribution for: '{column}'")
            # Drop missing
            data = df[column].dropna().astype(str).str.strip()
            value_counts = data.value_counts()

            # Print counts
            for value, count in value_counts.items():
                logger.info(f"{value}: {count}")

            # Create pie chart
            fig = go.Figure(data=[
                go.Pie(labels=value_counts.index, values=value_counts.values, hole=0.0)
            ])

            fig.update_layout(
                margin=dict(l=10, r=10, t=40, b=10)
            )

            # Save or display
            if save_file:
                # Replace spaces with underscores, remove question marks, strip final periods
                filename = column.strip()  # remove leading/trailing whitespace
                filename = re.sub(r"[^\w\s-]", "", filename)  # remove punctuation except underscores/hyphens
                filename = filename.replace(" ", "_").lower()
                if tag:
                    filename = f"{filename}_{tag}"
                self.save_plotly(fig, filename, save_final=True)
            else:
                fig.show()

    def distribution_plots(self, df, column_names, output_folder, save_file=True):

        if isinstance(df, str):
            df = pd.read_csv(df)

        current_year = datetime.now().year

        for column_name in column_names:
            if column_name not in df.columns:
                logger.warning(f"Column not found: {column_name}")
                continue

            # Try numeric conversion
            temp_series = pd.to_numeric(df[column_name], errors='coerce')
            is_numeric = pd.api.types.is_numeric_dtype(temp_series)

            # Drop NaNs
            df_clean = df.dropna(subset=[column_name]).copy()

            if df_clean.empty:
                logger.warning(f"No valid data in column: {column_name}")
                continue

            if is_numeric:
                # Numeric column processing
                df_clean[column_name] = pd.to_numeric(df_clean[column_name], errors='coerce')

                # Special cleaning for age column
                if column_name.strip().lower() == "what is your age (in years)?".lower():
                    cleaned_ages = []
                    for val in df_clean[column_name]:
                        if 18 <= val <= 99:  # Valid age
                            cleaned_ages.append(val)
                        elif 1900 <= val <= current_year:  # Looks like year of birth
                            age = current_year - val
                            if 18 <= age <= 99:
                                cleaned_ages.append(age)
                        # Else: ignore nonsensical values
                    df_clean[column_name] = cleaned_ages

                mean_val = df_clean[column_name].mean()
                std_val = df_clean[column_name].std()
                logger.info(f"{column_name} - Mean: {mean_val:.2f}, Std Dev: {std_val:.2f}")

                value_counts = df_clean[column_name].round().value_counts().sort_index()
                labels = [f"{int(v)}" for v in value_counts.index]
                values = value_counts.values
            else:
                # Categorical column processing
                df_clean[column_name] = df_clean[column_name].astype(str).str.strip()
                value_counts = df_clean[column_name].value_counts()
                labels = value_counts.index.tolist()
                values = value_counts.values.tolist()
                logger.info(f"{column_name} - Response counts: {dict(zip(labels, values))}")

            # Plotting
            fig = go.Figure(data=[
                go.Pie(labels=labels, values=values, hole=0.0, showlegend=True, sort=False)
            ])

            fig.update_layout(
                margin=dict(l=10, r=10, t=40, b=10)
            )

            # Save or display
            if save_file:
                # Replace spaces with underscores, remove question marks, strip final periods
                filename = column_name.strip()  # remove leading/trailing whitespace
                filename = re.sub(r"[^\w\s-]", "", filename)  # remove punctuation except underscores/hyphens
                filename = filename.replace(" ", "_").lower()
                self.save_plotly(fig, filename, save_final=True)
            else:
                fig.show()

    def plot_gender_by_nationality(self, csv_path,
                                   gender_col="What is your gender?",
                                   nationality_col="Nationality"):
        """
        Reads a CSV file and generates an interactive Plotly bar chart
        showing gender distribution for each nationality.
        """

        df = pd.read_csv(csv_path)

        nationality_map = {
            "Pakistani": "Pakistan",
            "Yemini": "Yemen",
            "Yemeni": "Yemen",
            "Nepalese": "Nepal",
            "Chinese": "China",
            "chinese": "China",
            " Chinese": "China",
            "Polish": "Poland",
            "Indian ": "India",
            "Dutch ": "Netherlands",
            "Dutch": "Netherlands",
            "Iranian": "Iran",
            "Romanian": "Romania",
            "Spanish": "Spain",
            "Colombian": "Colombia",
            "portuguese": "Portugal",
            "Taiwanese": "Taiwan",
            "German": "Germany",
            "Indian": "India",
            "dutch": "Netherlands"
        }

        # Apply mapping
        df[nationality_col] = df[nationality_col].map(nationality_map).fillna(
            df[nationality_col].str.capitalize()
        )

        # --- Group data ---
        s = df.groupby([nationality_col, gender_col]).size()
        s.name = "Count"
        grouped = s.reset_index()

        # --- Plot ---
        fig = px.bar(
            grouped,
            x=nationality_col,
            y="Count",
            color=gender_col,
            barmode="group",
            title=""
        )

        fig.show()

    @staticmethod
    def read_slider_data(data_folder, mapping, output_folder):
        """
        Reads participant slider CSVs from all participant folders, aggregates the
        ratings (noticeability, informativeness, annoyance) for all trials,
        and saves a summary CSV per slider to the output folder.

        Args:
            data_folder (str): Path to the folder containing participant subfolders.
            mapping (pd.DataFrame): Mapping DataFrame with 'video_id' and 'sound_clip_name'.
            output_folder (str): Directory to save aggregated CSVs for each slider.
        """
        participant_data = {}  # Store per-participant DataFrames
        all_trials = set()  # Collect all unique trial IDs

        # Iterate over each participant's folder
        for folder in sorted(os.listdir(data_folder)):
            folder_path = os.path.join(data_folder, folder)
            if not os.path.isdir(folder_path):
                continue

            # Parse participant ID from folder name
            match = re.match(r'Participant_(\d+)', folder)
            if not match:
                continue
            participant_id = int(match.group(1))

            # Find the CSV with slider data for this participant
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                # Expected pattern: Participant_[id]_[number]_[number].csv
                if re.match(rf'Participant_{participant_id}_\d+_\d+\.csv', file):
                    # Assume no header: columns are trial, noticeability, info, annoyance
                    df = pd.read_csv(file_path,
                                     header=None,
                                     names=["trial", "behaviour", "distance", "intention"])
                    df.set_index("trial", inplace=True)
                    participant_data[participant_id] = df
                    all_trials.update(df.index)
                    break  # Stop at first valid slider CSV

        # Build a sorted trial list (with 'test' first if present)
        all_trials = sorted([t for t in all_trials if t != "test"],
                            key=lambda x: int(re.search(r'\d+', x).group()))  # type: ignore
        all_trials.insert(0, "test") if "test" in all_trials else None

        # Prepare dict to aggregate each slider rating across all participants
        slider_data = {"behaviour": [], "distance": [], "intention": []}

        # For each participant, gather ratings for all trials, filling missing with None
        for participant_id, df in sorted(participant_data.items()):
            row = {"participant_id": participant_id}
            for trial in all_trials:
                if trial in df.index:
                    row[trial] = df.loc[trial].to_list()
                else:
                    row[trial] = [None, None, None]

            # Split values for each slider
            slider_data["behaviour"].append([participant_id] + [vals[0] for vals in row.values() if isinstance(vals, list)])  # noqa: E501
            slider_data["distance"].append([participant_id] + [vals[1] for vals in row.values() if isinstance(vals, list)])  # noqa: E501
            slider_data["intention"].append([participant_id] + [vals[2] for vals in row.values() if isinstance(vals, list)])  # noqa: E501

        # Convert lists to DataFrames, rename columns, and add average row
        for slider, data in slider_data.items():
            df = pd.DataFrame(data, columns=["participant_id"] + all_trials)
            # Rename trial columns using mapping (video_id to sound_clip_name)
            # df.rename(columns={trial: mapping_dict.get(trial, trial) for trial in all_trials}, inplace=True)

            # Add average row at the end (ignoring participant_id)
            avg_values = df.iloc[:, 1:].mean(skipna=True)
            avg_row = pd.DataFrame([["average"] + avg_values.tolist()], columns=df.columns)
            df = pd.concat([df, avg_row], ignore_index=True)

            # Save the aggregated slider data to CSV
            output_path = os.path.join(output_folder, f"slider_input_{slider}.csv")
            df.to_csv(output_path, index=False)

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
        # disable mathjax globally for Kaleido
        pio.kaleido.scope.mathjax = None
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
                    if (xref == 'paper' and yref == 'paper' and showarrow is False and isinstance(y, (int, float)) and y >= 0.98):
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
                                             size=font_size or common.get_configs('font_size'))
                             )
        else:
            fig.update_xaxes(title_text=xaxis_title,
                             range=xaxis_range,
                             title_font=dict(family=font_family,
                                             size=font_size or common.get_configs('font_size')))

        # Find actual y range across all series (for tick generation only)
        actual_ymin = min(all_values)
        actual_ymax = max(all_values)

        # Generate ticks from 0 up to actual_ymax
        positive_ticks = np.arange(0, actual_ymax + yaxis_step, yaxis_step)
        formatted_positive_ticks = [int(tick) if tick.is_integer() else tick for tick in positive_ticks]

        # Generate ticks from 0 down to actual_ymin (note: ymin is negative)
        negative_ticks = np.arange(0, actual_ymin - yaxis_step, -yaxis_step)
        formatted_negative_ticks = [int(tick) if tick.is_integer() else tick for tick in negative_ticks]

        # Combine and sort ticks
        visible_ticks = np.sort(np.unique(
            np.concatenate((formatted_negative_ticks, formatted_positive_ticks))
        ))

        tick_labels = [str(int(t)) if t.is_integer() else f"{t:.2f}" for t in visible_ticks]

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
                          size=font_size or common.get_configs('font_size')),
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
                      size=font_size or common.get_configs('font_size')),
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

        # use index of df if none is given
        if not x:
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
                                                    size=font_size or common.get_configs('font_size') - 6)))

        # multiple columns
        elif legend_columns == 2:
            fig.update_layout(
                legend=dict(
                    x=legend_x,
                    y=legend_y,
                    bgcolor='rgba(0,0,0,0)',
                    font=dict(size=font_size or common.get_configs('font_size')),
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
        if font_family:
            fig.update_layout(font=dict(family=font_family))
        else:
            fig.update_layout(font=dict(family=common.get_configs('font_family')))

        # update font size
        if font_size:
            fig.update_layout(font=dict(size=font_size))
        else:
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
                    t_stat, p_val = ttest_rel(data1, data2, alternative=type)
                else:
                    t_stat, p_val = ttest_ind(data1, data2, equal_var=False, alternative=type)

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
            display_name = mapping_filtered.loc[mapping_filtered["video_id"] == video, "video_id"].values[0]

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
        for label in all_labels:
            vid = mapping_filtered.loc[mapping_filtered["video_id"] == label, "video_id"].values[0]
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

    def heat_plot(self, folder_path: str, mapping_df: pd.DataFrame, relation: str = "ratio",
                  colorscale: str = "Viridis", summary_func=np.mean):
        """
        Compute summary values per video CSV, rename axes using `mapping_df`,
        and show a Plotly heatmap of pairwise relations (no numbers, no colorbar).

        Additionally, this function writes a 'trigger_summary.csv' file into
        `folder_path` with columns:
            ['label', 'avg_trigger', 'sd_trigger', 'n_samples']

        Parameters
        ----------
        folder_path : str
            Folder with 'participant_TriggerValueRight_video_*.csv'.
        mapping_df : pd.DataFrame
            Must include columns: 'video_id', 'condition_name', 'camera',
            'cross_p1_time_s', 'cross_p2_time_s'.
            The cutoff time used per video is cross_p1_time_s if camera==0,
            otherwise cross_p2_time_s. Only rows with Timestamp <= cutoff are used.
        relation : {'ratio','diff'}
            'ratio' uses vb/va; 'diff' uses vb - va.
        colorscale : str
            Plotly colorscale name.
        summary_func : callable
            Aggregator applied to all numeric values per condition (default: np.mean).
            (Used for `avg_trigger`.)

        Returns
        -------
        averages : dict
            {mapped_label: summary_value}  (means per condition; for compatibility)
        relation_df : pd.DataFrame
            Pairwise matrix (mapped_label × mapped_label)
        fig : plotly.graph_objects.Figure
            The heatmap figure.
        """
        # ---- Validate mapping_df ----
        required_cols = {"video_id", "condition_name", "camera", "cross_p1_time_s", "cross_p2_time_s"}
        if not required_cols.issubset(set(mapping_df.columns)):
            raise ValueError(f"mapping_df must have columns {required_cols}, got {list(mapping_df.columns)}")

        # Normalize dtypes
        md = mapping_df.copy()
        md["video_id"] = md["video_id"].astype(str)

        # Force numerics for times; invalids -> NaN
        md["cross_p1_time_s"] = pd.to_numeric(md["cross_p1_time_s"], errors="coerce")
        md["cross_p2_time_s"] = pd.to_numeric(md["cross_p2_time_s"], errors="coerce")
        md["camera"] = pd.to_numeric(md["camera"], errors="coerce").astype("Int64")

        # Build lookup: video_id -> {label, camera, p1, p2}
        mapping_info = {
            row["video_id"]: {
                "label": str(row["condition_name"]),
                "camera": int(row["camera"]) if pd.notna(row["camera"]) else None,
                "p1": float(row["cross_p1_time_s"]) if pd.notna(row["cross_p1_time_s"]) else None,
                "p2": float(row["cross_p2_time_s"]) if pd.notna(row["cross_p2_time_s"]) else None,
            }
            for _, row in md.iterrows()
        }

        # ---- Find files ----
        pattern = os.path.join(folder_path, "participant_TriggerValueRight_video_*.csv")
        file_list = glob.glob(pattern)
        if not file_list:
            raise FileNotFoundError(f"No files matched pattern: {pattern}")

        # Sort by numeric video index if present (video_123), else alphabetically
        def _video_key(fn):
            m = re.search(r"video_(\d+)\.csv$", os.path.basename(fn), flags=re.IGNORECASE)
            return (0, int(m.group(1))) if m else (1, os.path.basename(fn).lower())

        file_list = sorted(file_list, key=_video_key)

        # ---- Helper: extract video_id from filename ----
        def _extract_video_id(filename: str) -> str:
            base = os.path.basename(filename)
            m = re.search(r"(video_\d+)\.csv$", base, flags=re.IGNORECASE)
            return m.group(1) if m else base

        # ---- Collect all trigger values per condition label ----
        # per_label_values[label] = np.array([... trigger values from all files ...])
        per_label_values: dict[str, np.ndarray] = {}

        for file_path in file_list:
            video_id = _extract_video_id(file_path)
            info = mapping_info.get(video_id)

            # Determine cutoff (if mapping missing or incomplete, we'll fall back to no cutoff)
            cutoff = None
            label = video_id
            if info:
                label = info["label"]
                cam = info["camera"]
                if cam == 0:
                    cutoff = info["p1"]
                elif cam == 1:
                    cutoff = info["p2"]
                else:
                    logger.warning(f"⚠️ camera not 0/1 for {video_id}; using full data (no cutoff).")

            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                logger.error(f"⚠️ Error reading {file_path}: {e}")
                continue

            if "Timestamp" not in df.columns:
                logger.warning(f"⚠️ 'Timestamp' column missing in {file_path}; using full data.")
                ts_filtered = df
            else:
                # Coerce Timestamp to numeric seconds, drop rows with invalid timestamps
                ts = pd.to_numeric(df["Timestamp"], errors="coerce")
                valid = ts.notna()
                df = df.loc[valid].copy()
                ts = ts.loc[valid]

                if cutoff is not None and np.isfinite(cutoff):
                    mask = ts <= float(cutoff)
                    ts_filtered = df.loc[mask].copy()
                else:
                    ts_filtered = df

            # If nothing remains after filtering, skip
            if ts_filtered.empty:
                logger.warning(f"⚠️ No rows after time filtering for {file_path}; skipping.")
                continue

            # Drop Timestamp before aggregating values
            ts_filtered = ts_filtered.drop(columns=["Timestamp"], errors="ignore")

            # Parse list-like cells and collect finite numerics
            all_values = []
            for col in ts_filtered.columns:
                for val in ts_filtered[col]:
                    try:
                        nums = ast.literal_eval(val) if isinstance(val, str) else val
                        if isinstance(nums, list):
                            all_values.extend(
                                float(x) for x in nums
                                if isinstance(x, (int, float)) and np.isfinite(x)
                            )
                    except Exception:
                        # skip unparseable cells
                        continue

            if not all_values:
                logger.warning(f"⚠️ No numeric values found (after cutoff) in {file_path}; skipping.")
            else:
                vals = np.array(all_values, dtype=float)
                if label not in per_label_values:
                    per_label_values[label] = vals
                else:
                    per_label_values[label] = np.concatenate([per_label_values[label], vals])

        if not per_label_values:
            raise ValueError("No valid numeric data found in videos (after applying time cutoffs).")

        # ---- Build trigger summary (mean + SD + n) ----
        trigger_summary_df = (
            pd.DataFrame(
                [
                    {
                        "label": label,
                        "avg_trigger": float(summary_func(vals)),
                        "sd_trigger": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
                        "n_samples": int(len(vals)),
                    }
                    for label, vals in per_label_values.items()
                ]
            )
            .sort_values("label")
            .reset_index(drop=True)
        )

        # Backwards-compatible dict of means only (for relation matrix)
        averages = dict(
            zip(trigger_summary_df["label"], trigger_summary_df["avg_trigger"])
        )

        # ---- Save trigger_summary.csv ----
        trigger_summary_path = os.path.join(folder_path, "trigger_summary.csv")
        trigger_summary_df.to_csv(trigger_summary_path, index=False)
        logger.info(f"Saved trigger summary (mean + SD) to: {trigger_summary_path}")

        # ---- Build pairwise relation matrix ----
        labels = list(averages.keys())
        n = len(labels)
        mat = np.full((n, n), np.nan, dtype=float)

        for i, a in enumerate(labels):
            va = averages[a]
            for j, b in enumerate(labels):
                vb = averages[b]
                if relation == "ratio":
                    mat[i, j] = (vb / va) if va not in (0, None) else np.nan
                elif relation == "diff":
                    mat[i, j] = vb - va
                else:
                    raise ValueError("relation must be 'ratio' or 'diff'")

        relation_df = pd.DataFrame(mat, index=labels, columns=labels)

        # ---- Add formatted decimal text for cells ----
        text_values = np.where(
            np.isnan(relation_df.values),
            "",
            np.round(relation_df.values, 2).astype(str)
        )

        # ---- Plotly heatmap (no cell text, no colorbar) ----
        fig = go.Figure(
            data=go.Heatmap(
                z=relation_df.values,
                x=relation_df.columns,
                y=relation_df.index,
                text=text_values,           # show decimals in cells
                texttemplate="%{text}",     # ensures text is rendered
                colorscale=colorscale,
                showscale=False,  # remove colorbar
                hovertemplate=(
                    "<b>%{y}</b> → <b>%{x}</b><br>"
                    f"{relation}: %{{z:.6f}}<extra></extra>"
                )
            )
        )

        fig.update_layout(
            title="",
            xaxis=dict(
                title="",
                tickangle=45,
                tickfont=dict(size=18, color="black"),
                titlefont=dict(size=16, color="black")
            ),
            yaxis=dict(
                title="",
                autorange="reversed",
                tickfont=dict(size=18, color="black"),
                titlefont=dict(size=16, color="black")
            ),
            margin=dict(l=80, r=40, t=60, b=90),
            width=2400,
            height=2400,
            plot_bgcolor="white",
            paper_bgcolor="white"
        )

        self.save_plotly(fig, 'heatmap', height=2400, width=2400, save_final=True)

        # Optional: also save the pairwise relation matrix
        relation_path = os.path.join(folder_path, "trigger_relation_matrix.csv")
        relation_df.to_csv(relation_path)
        logger.info(f"Saved trigger relation matrix to: {relation_path}")

        return averages, relation_df, fig

    def load_and_average_Q2(
        self,
        trigger_summary_csv: str,
        responses_root: str,
        mapping_df: pd.DataFrame,
        n_participants: int = 50,
        response_col_index: int = 2,  # column index of Q2 (middle)
        save_combined: bool = True,
    ):
        """
        Combine per-condition trigger averages with averaged Q1, Q2, Q3
        per condition.

        Parameters
        ----------
        trigger_summary_csv : str
            Path to 'trigger_summary.csv' saved by `heat_plot`.
            Must have columns: ['label', 'avg_trigger', 'sd_trigger'] where 'label'
            matches mapping_df['condition_name'] (or video_id if you used that as label).
        responses_root : str
            Root folder containing Participant_1, Participant_2, ... subfolders.
        mapping_df : pd.DataFrame
            Must contain at least ['video_id', 'condition_name'].
        n_participants : int
            Total number of participants (default 50).
        response_col_index : int
            Zero-based index of the column containing the distance question Q2.
            Q1 is assumed to be at response_col_index - 1,
            Q3 at response_col_index + 1.
            With lines like 'video_25,0,41,71', index 2 corresponds to Q2=41
            and we also take Q1=0 (col 1) and Q3=71 (col 3).
        save_combined : bool
            If True, save condition-level summary to
            responses_root/condition_level_trigger_Q123.csv

        Returns
        -------
        trial_df : pd.DataFrame
            Trial-level data with columns:
            ['participant', 'video_id', 'condition_name',
             'Q1', 'Q2', 'Q3',
             'avg_trigger', 'sd_trigger']

        condition_df : pd.DataFrame
            Condition-level summary with columns:
            ['condition_name',
             'avg_trigger', 'std_trigger',
             'mean_Q1', 'std_Q1',
             'mean_Q2', 'std_Q2',
             'mean_Q3', 'std_Q3',
             'n_trials']
        """
        # --- Load trigger summary (condition -> avg_trigger, sd_trigger) --- #
        trigger_df = pd.read_csv(trigger_summary_csv)

        if "label" not in trigger_df.columns or "avg_trigger" not in trigger_df.columns:
            raise ValueError(
                f"trigger_summary_csv must have columns at least ['label', 'avg_trigger'], "
                f"got {trigger_df.columns.tolist()}"
            )

        # If sd_trigger is missing (e.g., older files), create it as NaN but warn
        if "sd_trigger" not in trigger_df.columns:
            logger.warning(
                "Column 'sd_trigger' not found in trigger_summary.csv; "
                "filling with NaN. Consider re-running `heat_plot`."
            )
            trigger_df["sd_trigger"] = np.nan

        # Rename 'label' to 'condition_name' to match mapping_df
        trigger_df = trigger_df.rename(columns={"label": "condition_name"})
        trigger_df["condition_name"] = trigger_df["condition_name"].astype(str)

        # --- Prepare mapping: video_id -> condition_name --- #
        map_df = mapping_df[["video_id", "condition_name"]].copy()
        map_df["video_id"] = map_df["video_id"].astype(str)
        map_df["condition_name"] = map_df["condition_name"].astype(str)

        # --- Indices for Q1/Q2/Q3 --- #
        q1_idx = response_col_index - 1
        q2_idx = response_col_index
        q3_idx = response_col_index + 1

        if q1_idx < 1:
            raise ValueError(
                f"response_col_index={response_col_index} is too small to infer Q1 at index {q1_idx}."
            )

        all_records = []

        # --- Read all participant response CSVs (no headers) --- #
        for pid in range(1, n_participants + 1):
            participant_folder = os.path.join(responses_root, f"Participant_{pid}")
            if not os.path.isdir(participant_folder):
                logger.warning(f"Participant folder not found: {participant_folder}")
                continue

            pattern = os.path.join(participant_folder, f"Participant_{pid}_*.csv")
            files = glob.glob(pattern)
            if not files:
                logger.warning(f"No response CSVs for Participant {pid} with pattern {pattern}")
                continue

            for fp in files:
                df = pd.read_csv(fp, header=None)

                # Need at least: trial_name (col 0) + Q1/Q2/Q3 columns
                max_needed_idx = q3_idx
                if df.shape[1] <= max_needed_idx:
                    logger.warning(
                        f"File {fp} has only {df.shape[1]} columns, "
                        f"cannot take Q1/Q2/Q3 at indices {q1_idx}, {q2_idx}, {q3_idx}."
                    )
                    continue

                tmp = df[[0, q1_idx, q2_idx, q3_idx]].copy()
                tmp.columns = ["video_id", "Q1", "Q2", "Q3"]
                tmp["participant"] = pid

                # Keep only actual trials (video_*) and drop baselines
                mask = tmp["video_id"].astype(str).str.startswith("video_")
                tmp = tmp.loc[mask]

                all_records.append(tmp)

        if not all_records:
            raise ValueError("No participant response data found. Check paths / patterns.")

        trial_df = pd.concat(all_records, ignore_index=True)

        # ensure numeric Qs
        for q_col in ["Q1", "Q2", "Q3"]:
            trial_df[q_col] = pd.to_numeric(trial_df[q_col], errors="coerce")

        # --- Attach condition_name via mapping_df (video_id -> condition_name) --- #
        trial_df["video_id"] = trial_df["video_id"].astype(str)
        trial_df = trial_df.merge(
            map_df,
            on="video_id",
            how="left",
        )

        # --- Attach avg_trigger and sd_trigger per condition_name --- #
        trial_df = trial_df.merge(
            trigger_df[["condition_name", "avg_trigger", "sd_trigger"]],
            on="condition_name",
            how="left",
        )

        # --- Average Q1/Q2/Q3 per condition, keep trigger mean/SD --- #
        condition_df = (
            trial_df
            .groupby("condition_name", as_index=False)
            .agg(
                avg_trigger=("avg_trigger", "first"),   # one value per condition
                std_trigger=("sd_trigger", "first"),    # one value per condition
                mean_Q1=("Q1", "mean"),
                std_Q1=("Q1", "std"),
                mean_Q2=("Q2", "mean"),
                std_Q2=("Q2", "std"),
                mean_Q3=("Q3", "mean"),
                std_Q3=("Q3", "std"),
                n_trials=("Q2", "size"),
            )
            .sort_values("condition_name")
        )

        if save_combined:
            os.makedirs(output, exist_ok=True)  # optional, ensures directory exists
            out_path = os.path.join(output, "condition_level_trigger_Q123.csv")
            condition_df.to_csv(out_path, index=False)
            logger.info(f"Saved condition-level trigger + Q1/Q2/Q3 data to: {out_path}")

    def analyze_and_plot_distance_effect_plotly(self, condition_df, mapping_df, out_dir=None):
        """
        Merge condition-level averages with distance, yielding, eHMI, and camera,
        compute full-factorial summaries (means + SDs), and create Plotly figures.

        Assumes
        -------
        condition_df has at least:
            ['condition_name', 'avg_trigger', 'std_trigger',
             'mean_Q1', 'std_Q1',
             'mean_Q2', 'std_Q2',
             'mean_Q3', 'std_Q3']
            - avg_trigger: mean proportion of time trigger was held (unsafe) per condition.
            - std_trigger: SD of that trigger-based measure per condition.
            - mean_Q1/Q2/Q3: mean responses per condition (0–100).
            - std_Q1/Q2/Q3: SD of Q1/Q2/Q3 per condition.

        mapping_df has at least:
            ['condition_name', 'distPed', 'yielding', 'eHMIOn', 'camera']
        """

        # Helper: turn "var=value" facet titles into just "value"
        facet_font = dict(size=font_size + 8, family=font_family)

        def _strip_facet_equals(fig):
            fig.for_each_annotation(
                lambda a: a.update(
                    text=a.text.split("=", 1)[-1].strip(),
                    font=facet_font,
                )
            )

        # --- REQUIRED columns ---
        required_cols = ["condition_name", "distPed", "yielding", "eHMIOn", "camera"]
        missing = [c for c in required_cols if c not in mapping_df.columns]
        if missing:
            raise ValueError(f"mapping_df missing: {missing}")

        needed_cond_cols = [
            "condition_name",
            "avg_trigger", "std_trigger",
            "mean_Q1", "std_Q1",
            "mean_Q2", "std_Q2",
            "mean_Q3", "std_Q3",
        ]
        missing2 = [c for c in needed_cond_cols if c not in condition_df.columns]
        if missing2:
            raise ValueError(f"condition_df missing: {missing2}")

        # --- Prepare mapping info ---
        mapping_df = mapping_df.copy()
        mapping_df["condition_name"] = mapping_df["condition_name"].astype(str)

        cond_plot_df = condition_df.copy()
        cond_plot_df["condition_name"] = cond_plot_df["condition_name"].astype(str)

        # Merge mapping info onto condition-level data
        cond_plot_df = cond_plot_df.merge(
            mapping_df[required_cols],
            on="condition_name",
            how="left",
        )

        # Convert coded distance values to actual meters (1→2 m, 2→4 m, ...)
        cond_plot_df["distPed_m"] = cond_plot_df["distPed"] * 2

        # Scale trigger to 0–100: "Mean perceived crossing risk (0–100)"
        cond_plot_df["crossing_risk"] = cond_plot_df["avg_trigger"] * 100.0
        cond_plot_df["crossing_risk_sd"] = cond_plot_df["std_trigger"] * 100.0

        # Drop if some factors missing
        cond_plot_df = cond_plot_df.dropna(
            subset=["distPed", "yielding", "eHMIOn", "camera"]
        )

        # Label maps for binary factors (0/1 → text)
        label_map_yield = {0: "Not yielding", 1: "Yielding"}
        label_map_ehmi = {0: "eHMI off", 1: "eHMI on"}
        label_map_cam = {0: "Can see other person", 1: "Cannot see other person"}

        cond_plot_df["yielding_label"] = cond_plot_df["yielding"].map(label_map_yield)
        cond_plot_df["eHMI_label"] = cond_plot_df["eHMIOn"].map(label_map_ehmi)
        cond_plot_df["camera_label"] = cond_plot_df["camera"].map(label_map_cam)

        # ============================
        # Full-factorial summary (means + SD from condition_df)
        # ============================
        group_cols = ["distPed_m", "yielding", "eHMIOn", "camera"]

        # If there are multiple rows per combination (e.g., multiple videos),
        # they should have identical stats; we take the mean as a safe aggregator.
        by_cond = (
            cond_plot_df
            .groupby(group_cols, as_index=False)
            .agg(
                mean_crossing_risk=("crossing_risk", "mean"),
                sd_crossing_risk=("crossing_risk_sd", "mean"),

                Q1_mean=("mean_Q1", "mean"),
                Q1_sd=("std_Q1", "mean"),

                Q2_mean=("mean_Q2", "mean"),
                Q2_sd=("std_Q2", "mean"),

                Q3_mean=("mean_Q3", "mean"),
                Q3_sd=("std_Q3", "mean"),
            )
            .sort_values(group_cols)
        )

        # Add label columns for plotting facets
        by_cond["yielding_label"] = by_cond["yielding"].map(label_map_yield)
        by_cond["eHMI_label"] = by_cond["eHMIOn"].map(label_map_ehmi)
        by_cond["camera_label"] = by_cond["camera"].map(label_map_cam)

        logger.info("\n=== Full-factorial condition table (MEAN + SD, 0–100 scales) ===")
        logger.info(
            "\n=== Full-factorial condition table (MEAN + SD, 0–100 scales) ===\n{}",
            by_cond.to_string(index=False),
        )
        logger.info("===========================================================\n")

        # Common label mapping for all figures
        base_labels = {
            "distPed_m": "Distance between pedestrians (m)",
            "crossing_risk": "Mean perceived crossing<br>risk (0–100)",
            "mean_crossing_risk": "Mean perceived crossing<br>risk (0–100)",
            "sd_crossing_risk": "SD of perceived crossing<br>risk (0–100)",

            "Q1_mean": "Q1 (0–100)",
            "Q1_sd": "SD of Q1 (0–100)",
            "Q2_mean": "Q2 (0–100)",
            "Q2_sd": "SD of Q2 (0–100)",
            "Q3_mean": "Q3 (0–100)",
            "Q3_sd": "SD of Q3 (0–100)",

            "camera_label": "Camera",
            "yielding_label": "Yielding",
            "eHMI_label": "eHMI",
            "context": "Context (Y = yielding, H = eHMI, C = camera)",
            "delta": "Near–far difference (0–100)",
            "measure": "Measure",
        }

        # Category ordering for cleaner facets / legends
        category_orders = {
            "eHMI_label": ["eHMI off", "eHMI on"],
            "yielding_label": ["Not yielding", "Yielding"],
            "camera_label": ["Can see other person", "Cannot see other person"],
        }
        axis_title_font = dict(size=font_size + 8, family=font_family)

        # ============================
        # Figure 1 — Mean crossing risk vs Distance (legend = camera)
        # ============================
        fig_beh = px.line(
            by_cond,
            x="distPed_m",
            y="mean_crossing_risk",
            color="camera_label",
            facet_col="eHMI_label",
            facet_row="yielding_label",
            markers=True,
            labels=base_labels,
            category_orders=category_orders,
            title="",
        )

        fig_beh.update_layout(
            template=plotly_template,
            legend=dict(
                x=0.5,
                y=0.5,
                xanchor="center",
                yanchor="bottom",
                font=axis_title_font,
                title_text="",
            ),
        )

        tick_font = dict(size=font_size + 8, family=font_family)

        fig_beh.update_xaxes(title_font=axis_title_font, tickfont=tick_font)
        fig_beh.update_yaxes(title_font=axis_title_font, tickfont=tick_font)

        fig_beh.update_xaxes(title_font=axis_title_font)
        fig_beh.update_yaxes(title_font=axis_title_font)
        _strip_facet_equals(fig_beh)

        # ============================
        # Figure 2 — Q2 vs Distance (legend = camera)
        # ============================
        fig_q2 = px.line(
            by_cond,
            x="distPed_m",
            y="Q2_mean",
            color="camera_label",
            facet_col="eHMI_label",
            facet_row="yielding_label",
            markers=True,
            labels=base_labels,
            category_orders=category_orders,
            title="",
        )

        fig_q2.update_layout(
            template=plotly_template,
            legend=dict(
                x=0.5,
                y=0.9,
                xanchor="center",
                yanchor="bottom",
                font=axis_title_font,
                title_text="",
            ),
        )
        fig_q2.update_xaxes(title_font=axis_title_font)
        fig_q2.update_yaxes(title_font=axis_title_font)
        _strip_facet_equals(fig_q2)

        # ============================
        # EXTRA Figure A — Mean crossing risk vs distance, legend = yielding
        # ============================
        fig_beh_yield = px.line(
            by_cond,
            x="distPed_m",
            y="mean_crossing_risk",
            color="yielding_label",
            facet_col="eHMI_label",
            facet_row="camera_label",
            markers=True,
            labels=base_labels,
            category_orders=category_orders,
            title="",
        )

        fig_beh_yield.update_layout(
            template=plotly_template,
            legend=dict(
                x=0.5,
                y=0.5,
                xanchor="center",
                yanchor="bottom",
                font=axis_title_font,
                title_text="",
            ),
        )
        fig_beh_yield.update_xaxes(title_font=axis_title_font)
        fig_beh_yield.update_yaxes(title_font=axis_title_font)
        _strip_facet_equals(fig_beh_yield)

        # ============================
        # EXTRA Figure B — Mean crossing risk vs distance, legend = eHMI
        # ============================
        fig_beh_ehmi = px.line(
            by_cond,
            x="distPed_m",
            y="mean_crossing_risk",
            color="eHMI_label",
            facet_col="yielding_label",
            facet_row="camera_label",
            markers=True,
            labels=base_labels,
            category_orders=category_orders,
            title="",
        )

        fig_beh_ehmi.update_layout(
            template=plotly_template,
            legend=dict(
                x=0.5,
                y=0.5,
                xanchor="center",
                yanchor="bottom",
                font=axis_title_font,
                title_text="",
            ),
        )
        fig_beh_ehmi.update_xaxes(title_font=axis_title_font)
        fig_beh_ehmi.update_yaxes(title_font=axis_title_font)
        _strip_facet_equals(fig_beh_ehmi)

        # ============================
        # EXTRA Figure C — Q2 vs distance, legend = yielding
        # ============================
        fig_q2_yield = px.line(
            by_cond,
            x="distPed_m",
            y="Q2_mean",
            color="yielding_label",
            facet_col="eHMI_label",
            facet_row="camera_label",
            markers=True,
            labels=base_labels,
            category_orders=category_orders,
            title="",
        )

        fig_q2_yield.update_layout(
            template=plotly_template,
            legend=dict(
                x=0.5,
                y=0.5,
                xanchor="center",
                yanchor="bottom",
                font=axis_title_font,
                title_text="",
            ),
        )
        fig_q2_yield.update_xaxes(title_font=axis_title_font)
        fig_q2_yield.update_yaxes(title_font=axis_title_font)
        _strip_facet_equals(fig_q2_yield)

        # ============================
        # EXTRA Figure D — Q2 vs distance, legend = eHMI
        # ============================
        fig_q2_ehmi = px.line(
            by_cond,
            x="distPed_m",
            y="Q2_mean",
            color="eHMI_label",
            facet_col="yielding_label",
            facet_row="camera_label",
            markers=True,
            labels=base_labels,
            category_orders=category_orders,
            title="",
        )

        fig_q2_ehmi.update_layout(
            template=plotly_template,
            legend=dict(
                x=0.5,
                y=0.5,
                xanchor="center",
                yanchor="bottom",
                font=axis_title_font,
                title_text="",
            ),
        )
        fig_q2_ehmi.update_xaxes(title_font=axis_title_font)
        fig_q2_ehmi.update_yaxes(title_font=axis_title_font)
        _strip_facet_equals(fig_q2_ehmi)

        # ============================
        # Figure 3 — Mean crossing risk vs Q2 scatter (condition-level)
        # ============================
        fig_scatter = px.scatter(
            cond_plot_df,
            x="crossing_risk",
            y="mean_Q2",
            color="distPed_m",
            labels=base_labels,
            title="",
        )

        x_vals = cond_plot_df["crossing_risk"].values
        y_vals = cond_plot_df["mean_Q2"].values
        if len(x_vals) >= 2 and np.isfinite(x_vals).all() and np.isfinite(y_vals).all():
            b1, b0 = np.polyfit(x_vals, y_vals, 1)
            xs = np.linspace(x_vals.min(), x_vals.max(), 100)
            ys = b0 + b1 * xs
            fig_scatter.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    name="",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        fig_scatter.update_layout(
            template=plotly_template,
            legend=dict(
                x=0.5,
                y=0.5,
                xanchor="center",
                yanchor="bottom",
                font=axis_title_font,
            ),
        )

        # ============================
        # Figure 4 — NEAR (1–2 m) minus FAR (4–5 m) per context
        # ============================
        ctx_cols = ["yielding", "eHMIOn", "camera"]

        near = (
            by_cond[by_cond["distPed_m"].isin([2, 4])]
            .groupby(ctx_cols, as_index=False)
            .agg(
                crossing_risk_near=("mean_crossing_risk", "mean"),
                Q1_near=("Q1_mean", "mean"),
                Q2_near=("Q2_mean", "mean"),
                Q3_near=("Q3_mean", "mean"),
            )
        )
        far = (
            by_cond[by_cond["distPed_m"].isin([8, 10])]
            .groupby(ctx_cols, as_index=False)
            .agg(
                crossing_risk_far=("mean_crossing_risk", "mean"),
                Q1_far=("Q1_mean", "mean"),
                Q2_far=("Q2_mean", "mean"),
                Q3_far=("Q3_mean", "mean"),
            )
        )

        diff_df = near.merge(far, on=ctx_cols, how="inner")
        diff_df["delta_crossing_risk"] = (
            diff_df["crossing_risk_near"] - diff_df["crossing_risk_far"]
        )
        diff_df["delta_Q1"] = diff_df["Q1_near"] - diff_df["Q1_far"]
        diff_df["delta_Q2"] = diff_df["Q2_near"] - diff_df["Q2_far"]
        diff_df["delta_Q3"] = diff_df["Q3_near"] - diff_df["Q3_far"]

        # Context string with on/off text instead of 0/1
        diff_df["context"] = diff_df.apply(
            lambda r: (
                f"Y{'on' if r['yielding'] == 1 else 'off'}_"
                f"H{'on' if r['eHMIOn'] == 1 else 'off'}_"
                f"C{int(r['camera'])}"
            ),
            axis=1,
        )

        logger.info("\n=== NEAR–FAR differences per context ===")
        logger.info(
            "\n=== NEAR–FAR differences per context ===\n{}",
            diff_df[
                ["context", "delta_crossing_risk", "delta_Q1", "delta_Q2", "delta_Q3"]
            ].to_string(index=False),
        )
        logger.info("========================================\n")

        long_diff = diff_df.melt(
            id_vars=["context"],
            value_vars=["delta_crossing_risk", "delta_Q1", "delta_Q2", "delta_Q3"],
            var_name="measure",
            value_name="delta",
        )

        long_diff["measure"] = long_diff["measure"].map({
            "delta_crossing_risk": "Mean perceived crossing risk (0–100)",
            "delta_Q1": "Q1 (0–100)",
            "delta_Q2": "Q2 (0–100)",
            "delta_Q3": "Q3 (0–100)",
        })

        fig_diff = px.bar(
            long_diff,
            x="context",
            y="delta",
            color="measure",
            barmode="group",
            labels={**base_labels, "delta": "Near–far difference (0–100)"},
            title="",
        )

        fig_diff.update_traces(
            texttemplate="%{y:.1f}",
            textposition="outside",
            textfont=axis_title_font,
        )

        fig_diff.update_layout(
            template=plotly_template,
            legend=dict(
                x=0.88,
                y=0.85,
                xanchor="center",
                yanchor="bottom",
                font=axis_title_font,
            ),
        )
        fig_diff.update_xaxes(title_font=axis_title_font, tickfont=axis_title_font)
        fig_diff.update_yaxes(title_font=axis_title_font, tickfont=axis_title_font)
        fig_diff.add_hline(y=0, line_dash="dash", line_color="black")

        # ============================
        # Stats summary
        # ============================
        # Correlations with crossing risk
        for q_label, col in [("Q1", "mean_Q1"), ("Q2", "mean_Q2"), ("Q3", "mean_Q3")]:
            if col in cond_plot_df.columns:
                r = cond_plot_df["crossing_risk"].corr(cond_plot_df[col])
                logger.info(f"Correlation (Mean perceived crossing risk, {q_label}): r = {r:.3f}")

        xd = by_cond["distPed_m"].values

        # Slopes vs distance for crossing risk and Q1–Q3
        yd_risk = by_cond["mean_crossing_risk"].values
        slope_risk, intercept_risk = np.polyfit(xd, yd_risk, 1)
        logger.info(
            "Overall Mean perceived crossing risk vs distance: "
            f"slope = {slope_risk:.4f} (risk units per 1 m)"
        )

        for q_label, col in [("Q1", "Q1_mean"), ("Q2", "Q2_mean"), ("Q3", "Q3_mean")]:
            y = by_cond[col].values
            slope_q, intercept_q = np.polyfit(xd, y, 1)
            logger.info(
                f"Overall {q_label} vs distance: "
                f"slope = {slope_q:.4f} ({q_label} units per 1 m)"
            )

        # ============================
        # Save figures
        # ============================
        self.save_plotly(fig_beh, "crossing_risk_full_factorial", save_final=True)
        self.save_plotly(fig_q2, "Q2_full_factorial", save_final=True)
        self.save_plotly(fig_scatter, "crossing_risk_vs_Q2_scatter", save_final=True)
        self.save_plotly(fig_diff, "near_minus_far_crossing_risk_vs_Q123", save_final=True)
        self.save_plotly(fig_beh_yield, "crossing_risk_full_factorial_legend_yielding", save_final=True)
        self.save_plotly(fig_beh_ehmi, "crossing_risk_full_factorial_legend_eHMI", save_final=True)
        self.save_plotly(fig_q2_yield, "Q2_full_factorial_legend_yielding", save_final=True)
        self.save_plotly(fig_q2_ehmi, "Q2_full_factorial_legend_eHMI", save_final=True)

    def plot_2x4_violins(self, responses_csv: str, mapping, name, x_ann, y_ann):
        """
        Create a 2x4 grid (8 subplots) of violin plots for all combinations
        of (yielding x eHMIOn x camera) defined in the mapping file.
        """

        # Font sizes
        BASE_FONT = 14              # body-ish text (stats labels, general)
        SUBPLOT_TITLE_FONT = 22     # header on each subplot
        Y_TICK_FONT = 22            # y tick labels
        Y_TITLE_FONT = 24           # y-axis title "Rating"

        # Force 3 y-ticks for all subplots
        Y_TICKVALS = [0, 50, 100]

        # 1. Load data
        responses = pd.read_csv(responses_csv)

        # 2. Drop baselines and reshape to long format
        responses = responses.drop(columns=["baseline_1", "baseline_2"], errors="ignore")
        video_cols = [c for c in responses.columns if c.startswith("video_")]

        long_df = responses.melt(
            id_vars=["participant_id"],
            value_vars=video_cols,
            var_name="video_id",
            value_name="rating"
        )

        long_df["video_id"] = long_df["video_id"].astype(str)
        mapping["video_id"] = mapping["video_id"].astype(str)

        # 3. Merge with mapping
        mapping_cond = mapping[["video_id", "yielding", "eHMIOn", "camera"]].drop_duplicates()
        long_cond = long_df.merge(mapping_cond, on="video_id", how="left")

        long_cond["rating"] = pd.to_numeric(long_cond["rating"], errors="coerce")
        long_cond = long_cond.dropna(subset=["rating", "yielding", "eHMIOn", "camera"])

        # 4. Unique condition combinations (should be 8)
        conds = (
            long_cond[["yielding", "eHMIOn", "camera"]]
            .drop_duplicates()
            .sort_values(["yielding", "eHMIOn", "camera"])
            .reset_index(drop=True)
        )

        max_plots = 8
        if len(conds) > max_plots:
            conds = conds.iloc[:max_plots]

        def camera_label(cam):
            return "Can see other person" if cam == 0 else "Cannot see other person"

        def case_title(row):
            line1 = (
                f"{'Yielding' if row['yielding'] == 1 else 'Non-yielding'}, "
                f"eHMI {'on' if row['eHMIOn'] == 1 else 'off'}"
            )
            line2 = camera_label(int(row["camera"]))
            return f"{line1}<br>{line2}"

        def case_name(row):
            return (
                f"{'Yielding' if row['yielding'] == 1 else 'Non-yielding'}, "
                f"eHMI {'on' if row['eHMIOn'] == 1 else 'off'}, "
                f"{camera_label(int(row['camera']))}"
            )

        def fmt_stat(x):
            try:
                return "NA" if pd.isna(x) else f"{x:.2f}"
            except Exception:
                return "NA"

        titles = [case_title(row) for _, row in conds.iterrows()]
        names = [case_name(row) for _, row in conds.iterrows()]

        # 5. Create 2x4 subplot figure and add violins
        fig = make_subplots(
            rows=2,
            cols=4,
            subplot_titles=titles
        )

        for i, cond_row in conds.iterrows():
            sub = long_cond[
                (long_cond["yielding"] == cond_row["yielding"]) &
                (long_cond["eHMIOn"] == cond_row["eHMIOn"]) &
                (long_cond["camera"] == cond_row["camera"])
            ]

            r = i // 4 + 1
            c = i % 4 + 1

            fig.add_trace(
                go.Violin(
                    y=sub["rating"],
                    box_visible=True,
                    meanline_visible=True,
                    points="all",
                    name=names[i],
                    showlegend=False
                ),
                row=r,
                col=c
            )

            # --- M / SD / Median labels (right side) ---
            m = sub["rating"].mean()
            sd = sub["rating"].std(ddof=1)
            med = sub["rating"].median()

            axis_num = i + 1
            xref = "x domain" if axis_num == 1 else f"x{axis_num} domain"
            yref = "y domain" if axis_num == 1 else f"y{axis_num} domain"

            fig.add_annotation(
                xref=xref,
                yref=yref,
                x=x_ann,
                y=y_ann,
                xanchor="right",
                yanchor="top",
                align="right",
                showarrow=False,
                text=f"M={fmt_stat(m)}<br>SD={fmt_stat(sd)}<br>Median={fmt_stat(med)}",
                font=dict(size=BASE_FONT)
            )

        # 6. Layout
        fig.update_layout(
            height=900,
            width=1600,
            title_text="",
            template=plotly_template,
            font=dict(size=BASE_FONT)
        )

        # Increase ONLY the subplot header size
        if getattr(fig.layout, "annotations", None):
            for ann in fig.layout.annotations:
                if getattr(ann, "text", None) in set(titles):
                    ann.font = dict(size=SUBPLOT_TITLE_FONT)

        # Force the same 3 y-ticks everywhere + bigger fonts
        for r in range(1, 3):
            for c in range(1, 5):
                fig.update_yaxes(
                    tickfont=dict(size=Y_TICK_FONT),
                    tickmode="array",
                    tickvals=Y_TICKVALS,
                    row=r, col=c
                )

            fig.update_yaxes(
                title_text="Rating",
                title_font=dict(size=Y_TITLE_FONT),
                row=r, col=1
            )

        # Hide x tick labels
        for r in range(1, 3):
            for c in range(1, 5):
                fig.update_xaxes(showticklabels=False, row=r, col=c)

        self.save_plotly(fig, name, save_final=True)

    def plot_yaw(self, mapping, column_name="Yaw", parameter=None, parameter_value=None,
                 additional_parameter=None, additional_parameter_value=None, compare_trial="video_1",
                 xaxis_title=None, yaxis_title=None, xaxis_range=None, yaxis_range=None,
                 margin=None, name=None, recompute=False):
        """
        Generate a comparison plot of keypress yaw data and subjective slider ratings
        for multiple video trials relative to a test condition.

        The function processes trigger matrices for each participant and trial,
        aligns time series data, attaches subjective slider-based ratings (annoyance,
        informativeness, noticeability), and prepares the data for visualization.
        Significance testing (paired t-tests) is performed between the test condition
        and each other trial.

        Args:
            mapping (pd.DataFrame): DataFrame with video metadata, including
                'video_id', 'sound_clip_name', 'display_name', and 'colour'.
            column_name (str, optional): The matrix column to process (default "Yaw").
            parameter / parameter_value (optional): Filter `mapping` by column == value.
            additional_parameter / additional_parameter_value (optional): Second filter.
            compare_trial (str, optional): Reference trial video_id.
            xaxis_title (str, optional): Custom label for the x-axis.
            yaxis_title (str, optional): Custom label for the y-axis.
            xaxis_range (list, optional): x-axis [min, max] limits for the plot.
            yaxis_range (list, optional): y-axis [min, max] limits for the plot.
            margin (dict, optional): Custom plot margin dictionary.
            name (str, optional): (currently unused).
            recompute (bool, optional): If True, regenerate CSVs/TXT even if they exist.
                                        If False, reuse existing files when present.
        """
        # ensure yaxis_range is mutable (plot_kp modifies it in-place)
        if isinstance(yaxis_range, tuple):
            yaxis_range = list(yaxis_range)

        # Find the video_length for the given video_id
        lens = mapping.loc[mapping["video_id"].eq(compare_trial), "video_length"].unique()

        if len(lens) == 0:
            raise ValueError(f"No rows found for video_id='{compare_trial}'")
        elif len(lens) > 1:
            # If the same video_id appears with different lengths, keep all matching lengths
            mapping_filtered = mapping[mapping["video_length"].isin(lens)].copy()
        else:
            # Typical case: one length
            mapping_filtered = mapping[mapping["video_length"].eq(lens[0])].copy()

        if parameter is not None:
            mapping_filtered = mapping_filtered[mapping_filtered[parameter] == parameter_value]

        if additional_parameter is not None:
            mapping_filtered = mapping_filtered[mapping_filtered[additional_parameter] == additional_parameter_value]

        # Filter out control/test video IDs for comparison
        mapping_filtered = mapping_filtered[~mapping_filtered["video_id"].isin(["baseline_1", "baseline_2"])]
        plot_videos = mapping_filtered["video_id"]

        all_dfs = []          # List to collect DataFrames for each trial
        all_labels = []       # Corresponding list of human-friendly trial labels
        ttest_signals = []    # Store t-test pairs for stats annotations

        data_folder = self.data_folder  # Get path to raw data

        # === Reference (test) trial: export yaw matrix and compute average yaw per timestamp ===
        test_participant_csv = os.path.join(
            self.output_folder,
            f"participant_{column_name}_{compare_trial}.csv"
        )

        # Export participant quaternion matrix for reference trial (only if needed)
        if recompute or not os.path.exists(test_participant_csv):
            self.export_participant_quaternion_matrix(
                data_folder=self.data_folder,   # keep original behaviour for reference
                video_id=compare_trial,
                output_file=test_participant_csv,
                mapping=mapping
            )

        # Compute average yaw for the reference (test) trial and save
        test_yaw_csv = os.path.join(
            self.output_folder,
            f"yaw_avg_{compare_trial}.csv"     # IMPORTANT: separate file from participant_*.csv
        )

        if recompute or not os.path.exists(test_yaw_csv):
            HMD_class.compute_avg_yaw_from_matrix_csv(
                input_csv=test_participant_csv,
                output_csv=test_yaw_csv
            )

        # Matrix for t-tests: must use participant matrix, not averaged yaw CSV
        test_matrix = extra_class.all_yaws_per_bin(
            input_csv=test_participant_csv
        )

        # === Iterate through each video trial (excluding control/test) ===
        for video in plot_videos:
            # Get display name for current trial
            display_name = mapping.loc[mapping["video_id"] == video, "video_id"].values[0]
            participant_csv = os.path.join(
                self.output_folder,
                f"participant_{column_name}_{video}.csv"
            )

            # Export quaternion/yaw matrix for this trial (if needed)
            if recompute or not os.path.exists(participant_csv):
                self.export_participant_quaternion_matrix(
                    data_folder=data_folder,
                    video_id=video,
                    output_file=participant_csv,
                    mapping=mapping
                )

            # Compute avg yaw for this trial (if needed)
            yaw_csv = os.path.join(self.output_folder, f"yaw_avg_{video}.csv")
            if recompute or not os.path.exists(yaw_csv):
                HMD_class.compute_avg_yaw_from_matrix_csv(
                    input_csv=participant_csv,
                    output_csv=yaw_csv
                )

            df = pd.read_csv(yaw_csv)
            all_dfs.append(df)
            all_labels.append(display_name)

            # Extract all per-bin yaw values (for saving and t-test)
            trial_matrix = extra_class.all_yaws_per_bin(
                input_csv=participant_csv
            )

            yaw_values = extra_class.flatten_trial_matrix(trial_matrix)
            yaw_values = yaw_values[~np.isnan(yaw_values)]  # Remove NaNs if present

            trial_txt_path = os.path.join(self.output_folder, f"yaw_values_{video}.txt")
            if recompute or not os.path.exists(trial_txt_path):
                np.savetxt(trial_txt_path, yaw_values)

            # Prepare for t-test: compare each trial vs. test reference (exclude self-comparison)
            if video != compare_trial:
                ttest_signals.append({
                    "signal_1": test_matrix,
                    "signal_2": trial_matrix,
                    "paired": True,
                    "label": f"{display_name}"
                })

        # === Combine all trial DataFrames into a single one for plotting ===
        if not all_dfs:
            raise RuntimeError("No data frames found to plot.")

        combined_df = pd.DataFrame()
        combined_df["Timestamp"] = all_dfs[0]["Timestamp"]

        # Add trial average yaw series as columns
        for df, label in zip(all_dfs, all_labels):
            combined_df[label] = df["AvgYaw"]

        # For interpretability (and consistency with the paper), plot yaw in degrees.
        # The underlying saved CSVs remain in radians.
        for label in all_labels:
            try:
                combined_df[label] = np.degrees(pd.to_numeric(combined_df[label], errors="coerce"))
            except Exception:
                pass

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

        # Row group 1: all main car events at the same (top) height (id=1)
        first_row_events = [
            ("yield_start_time_s",   "Car decelerates"),
            ("yield_stop_time_s",    "Car stops"),
            ("yield_resume_time_s",  "Car accelerates"),
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

        # Row group 2: crossing event on its own lower row (id=2)
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

        # === cross_p1_time_s: per-line time (one marker per plotted line) ===
        cross_p1_times = {}
        if "cross_p1_time_s" in mapping_filtered.columns:
            for video, label in zip(plot_videos, all_labels):
                series = mapping_filtered.loc[
                    mapping_filtered["video_id"] == video, "cross_p1_time_s"
                ].dropna()
                if not series.empty:
                    # take first non-NaN for that video/line
                    cross_p1_times[label] = float(series.iloc[0])

        # Choose line style: dashed for test trial, solid for others
        custom_line_dashes = []
        for label in all_labels:
            vid = mapping.loc[mapping["video_id"] == label, "video_id"].values[0]
            if vid == compare_trial:
                custom_line_dashes.append("dot")
            else:
                custom_line_dashes.append("solid")

        # === Call central plotting function with all visualization & stats options ===
        self.plot_kp(
            df=combined_df,
            y=all_labels,
            y_legend_kp=all_labels,
            xaxis_range=xaxis_range,
            yaxis_range=yaxis_range,
            xaxis_title=xaxis_title,  # type: ignore
            # plot in degrees (the combined_df was converted above)
            yaxis_title="Yaw angle, [deg]",
            xaxis_title_offset=-0.047,  # type: ignore
            name_file=f"{name}",
            show_text_labels=True,
            pretty_text=True,
            events=events,
            events_width=2,
            events_annotations_font_size=common.get_configs("font_size") - 8,
            stacked=False,
            ttest_signals=ttest_signals,
            ttest_anova_row_height=0.006,
            ttest_annotations_font_size=common.get_configs("font_size") - 8,
            ttest_annotation_x=0.001,  # type: ignore
            ttest_marker_size=common.get_configs("font_size") - 6,
            xaxis_step=1,
            yaxis_step=2,  # ~2° ticks
            legend_x=0,
            legend_y=1.225,
            legend_columns=2,
            line_width=3,
            fig_save_width=1470,
            fig_save_height=850,
            font_size=common.get_configs("font_size"),
            save_file=True,
            save_final=True,
            custom_line_dashes=custom_line_dashes,
            flag_trigger=False,
            margin=margin,
            cross_p1_times=cross_p1_times,
        )

    def plot_yaw_frequencies_by_condition(self, mapping, yaw_files_dir):
        yaw_files_dir = os.path.abspath(yaw_files_dir)

        # Remove baseline videos and copy to avoid SettingWithCopyWarning
        mapping = mapping.loc[~mapping["video_id"].isin(["baseline_1", "baseline_2"])].copy()

        # Keys for grouping
        # case_key: (yielding, eHMIOn)  -> for colour mapping

        mapping["case_key"] = (
            "y" + mapping["yielding"].astype(str) +
            "_e" + mapping["eHMIOn"].astype(str)
        )
        # condition_key: (yielding, eHMIOn, camera)  -> for the actual curves
        mapping["condition_key"] = mapping["case_key"] + "_c" + mapping["camera"].astype(str)
        # Distances (for the 10 subplots)
        dist_values = sorted(mapping["distPed"].unique())

        if len(dist_values) != 5:
            logger.warning(
                f"Expected 5 distinct distPed values, found {len(dist_values)}: {dist_values}"
            )
        # G10 colours mapped by case_key so the same case shares colour across all figures
        colors = px.colors.qualitative.G10

        case_keys = sorted(mapping["case_key"].unique())
        color_map = {ck: colors[i % len(colors)] for i, ck in enumerate(case_keys)}
        line_width = 6

        # Histogram settings (no smoothing) in [-90, 90]
        bins = np.linspace(-90, 90, 181)   # 1° bins from -90 to 90

        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        # lists to collect curve metrics
        metrics_cam0 = []

        metrics_cam1 = []
        metrics_grid = []   # metrics for the 10 (camera, distPed) plots

        def _compute_freq_for_subset(sub_subset, cam_value, metrics_list=None, dist_value=None):
            """

            Given a subset of mapping (already filtered by camera and possibly distPed),
            compute frequencies per (yielding, eHMIOn) case, and optionally store metrics.
            Returns
            -------

            curves : list of (case_key, cond_label, color, freq_array)
            local_max : float
                Maximum frequency value among curves for this subset.
            """
            curves = []
            local_max = 0.0
            # Group by (yielding, eHMIOn, camera) via condition_key
            for cond_key, sub in sub_subset.groupby("condition_key", sort=True):

                case_key = sub["case_key"].iloc[0]
                color = color_map[case_key]
                # --- Legend label based ONLY on yielding & eHMI ---
                y_val = int(sub["yielding"].iloc[0])

                e_val = int(sub["eHMIOn"].iloc[0])
                yielding_label = "No yielding" if y_val == 0 else "Yes yielding"
                ehmi_label = "eHMI off" if e_val == 0 else "eHMI on"

                cond_label = f"{yielding_label}, {ehmi_label}"

                all_yaws_deg = []

                for video_id in sub["video_id"].unique():
                    yaw_path = os.path.join(yaw_files_dir, f"yaw_values_{video_id}.txt")

                    if not os.path.exists(yaw_path):
                        logger.warning(f"Warning: {yaw_path} not found, skipping this video.")
                        continue
                    yaws_rad = np.loadtxt(yaw_path)
                    yaws_rad = np.atleast_1d(yaws_rad)

                    # radians → degrees, wrap to [-180, 180]
                    yaws_deg = np.degrees(yaws_rad)

                    yaws_deg = (yaws_deg + 180) % 360 - 180
                    # keep only [-90, 90] observations
                    mask = (yaws_deg >= -90) & (yaws_deg <= 90)

                    yaws_deg = yaws_deg[mask]
                    if yaws_deg.size == 0:
                        continue

                    all_yaws_deg.append(yaws_deg)

                if not all_yaws_deg:
                    logger.error(f"Warning: no yaw samples for condition '{cond_label}'.")

                    continue
                all_yaws_deg = np.concatenate(all_yaws_deg)

                counts, _ = np.histogram(all_yaws_deg, bins=bins)
                if counts.sum() == 0:

                    continue
                # Frequency (%) – NO smoothing
                freq = counts / counts.sum() * 100.0

                local_max = max(local_max, freq.max())
                # Metrics (if a metrics_list is provided)
                if metrics_list is not None:

                    area_total = freq.sum()  # should be ~100
                    left_mask = bin_centers < 0
                    right_mask = bin_centers > 0

                    central_mask = np.abs(bin_centers) <= 15
                    area_left = freq[left_mask].sum()
                    area_right = freq[right_mask].sum()

                    area_central = freq[central_mask].sum()
                    peak_idx = np.argmax(freq)
                    peak_yaw = bin_centers[peak_idx]

                    peak_freq = freq[peak_idx]
                    mean_yaw = np.sum(bin_centers * freq) / area_total
                    var_yaw = np.sum(((bin_centers - mean_yaw) ** 2) * freq) / area_total

                    std_yaw = np.sqrt(var_yaw)
                    metrics_list.append(
                        {
                            "camera": cam_value,
                            "distPed": dist_value,
                            "condition": cond_label,
                            "case_key": case_key,
                            "area_total_pct": area_total,
                            "area_left_pct": area_left,
                            "area_right_pct": area_right,
                            "area_central_±15_pct": area_central,
                            "peak_yaw_deg": peak_yaw,
                            "peak_freq_pct": peak_freq,
                            "mean_yaw_deg": mean_yaw,
                            "std_yaw_deg": std_yaw,
                        }
                    )
                curves.append((case_key, cond_label, color, freq))

            return curves, local_max

        def build_figure_for_camera(cam_value, metrics_list):
            sub_cam = mapping[mapping["camera"] == cam_value]

            if sub_cam.empty:
                return None, 0.0
            fig = go.Figure()
            curves, max_y = _compute_freq_for_subset(

                sub_cam,
                cam_value=cam_value,
                metrics_list=metrics_list,
                dist_value=None,  # aggregated across distances
            )
            for case_key, cond_label, color, freq in curves:
                fig.add_trace(

                    go.Scatter(
                        x=bin_centers,
                        y=freq,
                        mode="lines",
                        name=cond_label,
                        line=dict(width=line_width, color=color, dash="solid"),
                    )
                )
            return fig, max_y

        # === 1) Build the two main camera-level figures (camera 0, camera 1) ===
        fig_cam0, max0 = build_figure_for_camera(0, metrics_cam0)

        fig_cam1, max1 = build_figure_for_camera(1, metrics_cam1)
        if fig_cam0 is None and fig_cam1 is None:
            raise ValueError("No yaw samples found for any camera.")

        global_max = max(max0, max1)
        if global_max <= 0:

            raise ValueError("Non-positive maximum frequency encountered.")
        # Common y-axis range & ticks for main figs: 1, 2, 3, ...
        ymax_axis = math.ceil(global_max)

        if ymax_axis < 1:
            ymax_axis = 1  # at least one tick
        # Log metrics as tables for your report (camera-level)
        metrics_cam0_df = pd.DataFrame(metrics_cam0)

        metrics_cam1_df = pd.DataFrame(metrics_cam1)
        if not metrics_cam0_df.empty:
            logger.info(f"Camera 0 metrics:\n{metrics_cam0_df.to_string(index=False)}")
        if not metrics_cam1_df.empty:
            logger.info(f"Camera 1 metrics:\n{metrics_cam1_df.to_string(index=False)}")

        # Persist metrics tables for reporting/reproducibility
        try:
            if not metrics_cam0_df.empty:
                metrics_cam0_df.to_csv(
                    os.path.join(yaw_files_dir, "yaw_freq_metrics_camera0.csv"),
                    index=False,
                )
            if not metrics_cam1_df.empty:
                metrics_cam1_df.to_csv(
                    os.path.join(yaw_files_dir, "yaw_freq_metrics_camera1.csv"),
                    index=False,
                )
            # also write a human-readable log
            with open(os.path.join(yaw_files_dir, "yaw_freq_metrics_camera_level.txt"), "w") as f:
                if not metrics_cam0_df.empty:
                    f.write("Camera 0 metrics\n")
                    f.write(metrics_cam0_df.to_string(index=False))
                    f.write("\n\n")
                if not metrics_cam1_df.empty:
                    f.write("Camera 1 metrics\n")
                    f.write(metrics_cam1_df.to_string(index=False))
                    f.write("\n")
        except Exception as e:
            logger.warning(f"Could not write yaw frequency metrics tables: {e}")

        def finalize_figure(fig, cam_value):
            if fig is None:

                return None
            # Vertical reference line at 0° (thin, grey, dotted)
            fig.add_shape(

                type="line",
                x0=0, x1=0,
                y0=0, y1=ymax_axis,
                line=dict(color="gray", dash="dot", width=3.0),
            )
            # White background, grids, solid axes, legend inside
            fig.update_layout(
                template="none",
                paper_bgcolor="white",
                plot_bgcolor="white",
                title="",
                xaxis_title="</b>Gaze yaw angle (deg)</b>",
                yaxis_title="</b>Frequency</b>",
                legend_title=None,
                legend=dict(
                    x=0.02,
                    y=0.98,
                    xanchor="left",
                    yanchor="top",
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="black",
                    borderwidth=1,
                    font=dict(
                        family=font_family, size=font_size+20
                    ),
                ),
                xaxis_title_font=dict(family=font_family, size=font_size+20),
                yaxis_title_font=dict(family=font_family, size=font_size+20),
            )
            fig.update_xaxes(
                range=[-90, 90],
                showgrid=True,
                gridcolor="lightgray",
                tickmode="array",
                tickvals=[-90, -60, -30, 0, 30, 60, 90],  # every 30°, excluding 0
                zeroline=False,                        # no built-in zero line
                showline=True,
                linecolor="black",
                tickfont=dict(family=font_family, size=font_size+20),
                ticks="outside",
                ticklen=10,
                tickwidth=3,
                title=dict(
                    text="Gaze yaw angle (deg)",
                    font=dict(family=font_family, size=font_size+20),
                    standoff=0,
                ),
                automargin=True,   # lets Plotly expand margins for the title
                            )

            fig.update_yaxes(
                range=[0, ymax_axis],
                tick0=1,
                dtick=1,                               # ticks & grid at 1,2,3,...
                showgrid=True,
                gridcolor="lightgray",
                zeroline=True,
                zerolinecolor="black",
                showline=True,
                linecolor="black",
                tickfont=dict(family=font_family, size=font_size+20),
                ticks="outside",
                ticklen=10,
                tickwidth=3,
            )
            return fig

        fig_cam0 = finalize_figure(fig_cam0, 0)
        fig_cam1 = finalize_figure(fig_cam1, 1)

        # Save main camera-level figs
        if fig_cam0 is not None:
            self.save_plotly(
                fig_cam0,
                "yaw_hist_can_see",
                width=1600,
                height=900,
                save_final=True,
            )

        if fig_cam1 is not None:
            self.save_plotly(
                fig_cam1,
                "yaw_hist_cannot_see",
                width=1600,
                height=900,
                save_final=True,
            )

        # === 2) Build the additional 10 plots: camera × distPed in a 2×5 grid ===

        # Subplot titles: only show distance on top row, blank on bottom row
        subplot_titles = []
        for r in range(2):
            for dist in dist_values[:5]:
                if r == 0:
                    subplot_titles.append(f"Distance = {dist}")
                else:
                    subplot_titles.append("")  # no title in second row

        fig_grid = make_subplots(
            rows=2,
            cols=5,
            shared_xaxes=True,
            shared_yaxes=True,
            horizontal_spacing=0.03,
            vertical_spacing=0.10,
            subplot_titles=subplot_titles,
        )

        # For grid: fix y_max (you can use 8 if you want stricter cap)
        grid_ymax = 9

        for r, cam in enumerate([0, 1], start=1):
            for c, dist in enumerate(dist_values[:5], start=1):
                sub_camdist = mapping[
                    (mapping["camera"] == cam) &
                    (mapping["distPed"] == dist)
                ]
                if sub_camdist.empty:
                    continue

                curves, local_max = _compute_freq_for_subset(
                    sub_camdist,
                    cam_value=cam,
                    metrics_list=metrics_grid,   # collect metrics for grid as well
                    dist_value=dist,
                )

                for case_key, cond_label, color, freq in curves:
                    # Only show legend entry once (top-left subplot),
                    # but link ALL traces of the same case via legendgroup
                    show_legend = (r == 1 and c == 1)

                    fig_grid.add_trace(
                        go.Scatter(
                            x=bin_centers,
                            y=freq,
                            mode="lines",
                            name=cond_label,
                            legendgroup=case_key,
                            line=dict(width=line_width-3, color=color, dash="solid"),
                            showlegend=show_legend,
                        ),
                        row=r,
                        col=c,
                    )

                # Vertical line at 0° for this subplot
                fig_grid.add_vline(
                    x=0,
                    line_dash="dot",
                    line_color="gray",
                    line_width=1.5,
                    row=r,  # type: ignore
                    col=c,  # type: ignore
                )

        # Log metrics for the 10 camera×distPed plots
        metrics_grid_df = pd.DataFrame(metrics_grid)
        if not metrics_grid_df.empty:
            logger.info(f"Grid (camera × distPed) metrics:\n{metrics_grid_df.to_string(index=False)}")

        # Persist grid metrics
        try:
            if not metrics_grid_df.empty:
                metrics_grid_df.to_csv(
                    os.path.join(yaw_files_dir, "yaw_freq_metrics_camera_by_distPed.csv"),
                    index=False,
                )
                with open(os.path.join(yaw_files_dir, "yaw_freq_metrics_camera_by_distPed.txt"), "w") as f:
                    f.write(metrics_grid_df.to_string(index=False))
                    f.write("\n")
        except Exception as e:
            logger.warning(f"Could not write yaw grid metrics tables: {e}")

        # Common layout for the grid figure
        fig_grid.update_layout(
            template="none",
            paper_bgcolor="white",
            plot_bgcolor="white",
            title="",
            legend_title=None,
            legend=dict(
                x=0.87,
                y=0.99,
                xanchor="right",
                yanchor="top",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="black",
                borderwidth=1,
                font=dict(family=font_family, size=font_size),
            ),
        )

        # Bigger tick labels for grid
        fig_grid.update_xaxes(
            range=[-90, 90],
            showgrid=True,
            gridcolor="lightgray",
            tickmode="array",
            tickvals=[-90, -60, -30, 0, 30, 60, 90],
            zeroline=False,
            showline=True,
            linecolor="black",
            tickfont=dict(family=font_family, size=font_size),
        )
        fig_grid.update_yaxes(
            range=[0, grid_ymax],
            tick0=1,
            dtick=1,
            showgrid=True,
            gridcolor="lightgray",
            zeroline=True,
            zerolinecolor="black",
            showline=True,
            linecolor="black",
            tickfont=dict(family=font_family, size=font_size),
        )

        # Axis labels for the grid
        fig_grid.update_yaxes(title_text="Frequency", row=1, col=1)
        fig_grid.update_yaxes(title_text="Frequency", row=2, col=1)
        fig_grid.update_xaxes(title_text="Gaze yaw angle (deg)", row=2, col=3)

        # Save grid figure
        self.save_plotly(
            fig_grid,
            "yaw_hist_cam_dist",
            remove_margins=True,
            width=1600,
            height=900,
            save_final=True,
        )
