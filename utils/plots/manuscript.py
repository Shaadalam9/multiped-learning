"""Generate manuscript figures from saved analysis tables, without refitting models.

Called automatically by utils.plots.regenerate_all_figures.
Uses the project's pinned pandas, Plotly and Kaleido dependencies.
All plotted values come from _output; a SHA-256 manifest records their provenance.
"""
import pickle
import hashlib
import json
from custom_logger import logger
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
import plotly.io as pio
from plotly.subplots import make_subplots
from utils.plots.setup import experimental_setup

# No mathematical labels are used; disable remote MathJax loading in exports.
pio.kaleido.scope.mathjax = None

COLOURS = {'randomised_order': '#0072B2', 'fixed_sequence': '#D55E00'}
LABELS = {'randomised_order': 'Randomised order', 'fixed_sequence': 'Fixed order'}
FIGURE_LABELS = {
    'experimental_setup': 'fig:setup',
    'trial_sequence': 'fig:sequence',
    'distance_ratings': 'fig:q2',
    'session_changes': 'fig:segments',
    'ehmi_change': 'fig:ehmi',
    'trial_profile': 'fig:profile',
}


def generate(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    files = ['trial_level_common_window.csv', 'condition_cell_contrasts.csv',
             'session_segment_predictions.csv', 'ehmi_learning_trajectory.csv']
    tables = {name: pd.read_csv(source / name) for name in files}
    trials = tables[files[0]]
    if len(trials) != 4000 or trials.participant_uid.nunique() != 100:
        raise ValueError('Expected 100 participants and 4,000 trials; review the manuscript before plotting.')
    for _, group in trials.groupby('participant_uid'):
        if sorted(group.trial_number) != list(range(1, 41)):
            raise ValueError('Each participant must have 40 distinct trial positions.')
    fixed = trials[trials.ordering_group == 'fixed_sequence']
    factors = ['yielding', 'eHMIOn', 'camera', 'distPed_m']
    if (fixed.groupby('trial_number')[factors].nunique() != 1).any().any():
        raise ValueError('Fixed group does not share one sequence.')
    manifest = {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in files}

    def save(fig, name, width, height):
        fig.update_layout(template='plotly_white', font=dict(family='Arial', size=15, color='#222'),
                          width=width, height=height, margin=dict(l=70, r=25, t=60, b=75),
                          legend=dict(orientation='h', x=0, y=-.17), paper_bgcolor='white')
        fig.update_xaxes(showline=True, linecolor='#888', zeroline=False)
        fig.update_yaxes(showline=True, linecolor='#888', gridcolor='#e6e6e6', zeroline=False)
        with (destination / f'{name}.pickle').open('wb') as handle:
            pickle.dump(fig, handle, protocol=pickle.HIGHEST_PROTOCOL)
        fig.write_html(str(destination / f'{name}.html'), include_plotlyjs='cdn')
        fig.write_image(str(destination / f'{name}.pdf'))
        fig.write_image(str(destination / f'{name}.png'), scale=2)
        logger.info('%s -> %s (PDF, PNG, HTML, pickle)',
                    FIGURE_LABELS[name], destination / f'{name}.pdf')

    save(experimental_setup(), 'experimental_setup', 1100, 680)

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=.05)
    for row, (factor, label) in enumerate(
            zip(factors, ['Yielding (%)', 'eHMI on (%)', 'Participant first (%)', 'Distance (m)']), 1):
        for group in LABELS:
            values = trials[trials.ordering_group == group].groupby('trial_number')[factor].mean()
            if factor != 'distPed_m':
                values = 100 * values
            fig.add_trace(go.Scatter(x=values.index, y=values, mode='lines', name=LABELS[group],
                                     legendgroup=group, showlegend=row == 1,
                                     line=dict(color=COLOURS[group], width=2,
                                               dash='solid' if group == 'randomised_order' else 'dot')),
                          row=row, col=1)
        fig.update_yaxes(title_text=label, row=row, col=1, range=[-5, 105] if row < 4 else [1, 11])
    fig.update_xaxes(title_text='Trial position', row=4, col=1, range=[1, 40])
    save(fig, 'trial_sequence', 1000, 820)

    cells = tables[files[1]]
    cells = cells[(cells.outcome == 'Q2') & (cells.yielding == 1) &
                  (cells.eHMIOn == 0) & (cells.camera == 0)].sort_values('distPed_m')
    if len(cells) != 5:
        raise ValueError('Expected five Q2 distance contrasts.')
    subset = trials[(trials.yielding == 1) & (trials.eHMIOn == 0) &
                    (trials.camera == 0) & (trials.distPed_m == 2)]
    if set(subset[subset.ordering_group == 'fixed_sequence'].trial_number) != {1}:
        raise ValueError('The highlighted condition is no longer fixed trial 1.')
    positions = subset[subset.ordering_group ==
                       'randomised_order'].trial_number.value_counts().reindex(range(1, 41), fill_value=0)
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=.14,
                        subplot_titles=['A. Difference in distance ratings', 'B. Position of the 2 m condition'])
    diff = cells.difference_fixed_minus_randomised
    fig.add_trace(go.Scatter(x=diff, y=cells.distPed_m, mode='markers', marker=dict(color='#333', size=9),
                             error_x=dict(type='data', symmetric=False,
                                          array=cells.simultaneous_ci_high - diff,
                                          arrayminus=diff - cells.simultaneous_ci_low), showlegend=False),
                  row=1, col=1)
    fig.add_vline(x=0, line_dash='dot', line_color='#999', row=1, col=1)
    fig.update_xaxes(title_text='Fixed − randomised (rating points)', row=1, col=1)
    fig.update_yaxes(title_text='Distance between pedestrians (m)', tickvals=[2, 4, 6, 8, 10], row=1, col=1)
    fig.add_trace(go.Bar(x=positions.index, y=positions.values,
                  marker_color=COLOURS['randomised_order'], showlegend=False), row=1, col=2)
    fig.add_vline(x=1, line_color=COLOURS['fixed_sequence'], line_dash='dash', row=1, col=2)
    fig.add_annotation(x=1, y=1, xref='x2', yref='paper', text='Fixed order: trial 1', showarrow=False,
                       xanchor='left', yshift=10, font=dict(color=COLOURS['fixed_sequence'], size=13))
    fig.update_xaxes(title_text='Trial position (randomised order)', range=[0, 41], row=1, col=2)
    fig.update_yaxes(title_text='Number of participants', dtick=1, row=1, col=2)
    save(fig, 'distance_ratings', 1100, 460)

    segments = tables[files[2]]
    outcomes = [('unsafe_pct', 'A. Crossing judged unsafe (%)'),
                ('Q3', 'B. Understanding of vehicle intention (Q3)'),
                ('heading_yaw_activity_deg_s', 'C. Head yaw rate (degrees/s)')]
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=.10, subplot_titles=[title for _, title in outcomes])
    order = ['before_break_14', 'after_break_14', 'after_break_26']
    for col, (outcome, _) in enumerate(outcomes, 1):
        for i, group in enumerate(LABELS):
            rows = segments[(segments.outcome == outcome) & (segments.ordering_group == group)
                            ].set_index('session_segment').reindex(order)
            if rows.adjusted_estimate.isna().any():
                raise ValueError(f'Missing segment estimates for {outcome}, {group}')
            y = rows.adjusted_estimate
            fig.add_trace(go.Scatter(x=[v + (i - .5) * .09 for v in range(3)], y=y, mode='markers+lines',
                                     name=LABELS[group], legendgroup=group, showlegend=col == 1,
                                     line=dict(color=COLOURS[group]),
                                     marker=dict(symbol='circle' if i == 0 else 'square', size=7),
                                     error_y=dict(type='data', symmetric=False, array=rows.ci_high - y,
                                                  arrayminus=y - rows.ci_low)),
                          row=1, col=col)
        fig.update_xaxes(tickvals=[0, 1, 2], ticktext=['1–14', '15–26', '27–40'], title_text='Trials', row=1, col=col)
    save(fig, 'session_changes', 1200, 430)

    trajectory = tables[files[3]]
    outcomes = [('unsafe_pct', 'A. Crossing judged unsafe', 'eHMI on − off (percentage points)'),
                ('Q3', 'B. Understanding of vehicle intention', 'eHMI on − off (rating points)')]
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=.12, subplot_titles=[title for _, title, _ in outcomes])
    for col, (outcome, _, axis_title) in enumerate(outcomes, 1):
        for group in LABELS:
            rows = trajectory[(trajectory.outcome == outcome) & (trajectory.ordering_group == group)
                              ].sort_values('trial_number')
            if list(rows.trial_number) != list(range(1, 41)):
                raise ValueError(f'Missing eHMI trajectory for {outcome}, {group}')
            band = 'rgba(0,114,178,.15)' if group == 'randomised_order' else 'rgba(213,94,0,.15)'
            fig.add_trace(go.Scatter(x=list(rows.trial_number) + list(rows.trial_number[::-1]),
                                     y=list(rows.ci_high) + list(rows.ci_low[::-1]), fill='toself',
                                     fillcolor=band, line=dict(width=0), hoverinfo='skip',
                                     legendgroup=group, showlegend=False), row=1, col=col)
            fig.add_trace(go.Scatter(x=rows.trial_number, y=rows.estimate, mode='lines', name=LABELS[group],
                                     legendgroup=group, showlegend=col == 1,
                                     line=dict(color=COLOURS[group], width=2.5,
                                               dash='solid' if group == 'randomised_order' else 'dot')),
                          row=1, col=col)
        fig.add_hline(y=0, line_color='#999', line_width=1, row=1, col=col)
        for boundary in (14.5, 26.5):
            fig.add_vline(x=boundary, line_dash='dot', line_color='#bbb', row=1, col=col)
        fig.update_xaxes(title_text='Trial position', range=[1, 40], row=1, col=col)
        fig.update_yaxes(title_text=axis_title, row=1, col=col)
    save(fig, 'ehmi_change', 1100, 460)

    # Unadjusted perceived crossing safety at each trial position. In the fixed
    # group each position is one condition, so the line follows the sequence.
    fixed_yielding = fixed.drop_duplicates('trial_number').set_index('trial_number').sort_index().yielding
    runs, start = [], None
    for trial, value in fixed_yielding.items():
        if value == 0 and start is None:
            start = trial
        if value != 0 and start is not None:
            runs.append((start, trial - 1))
            start = None
    if start is not None:
        runs.append((start, int(fixed_yielding.index.max())))
    run_start, run_end = max(runs, key=lambda run: run[1] - run[0])
    fig = go.Figure()
    fig.add_vrect(x0=run_start - .5, x1=run_end + .5, fillcolor='#eeeeee', line_width=0, layer='below')
    fig.add_annotation(x=(run_start + run_end) / 2, y=1, yref='paper', yshift=8, showarrow=False,
                       text=f'Fixed order: non-yielding vehicle in trials {run_start}–{run_end}',
                       font=dict(size=13, color='#555'))
    for boundary in (14.5, 26.5):
        fig.add_vline(x=boundary, line_dash='dot', line_color='#bbb')
    for group in LABELS:
        rows = trials[trials.ordering_group == group].groupby('trial_number').unsafe_pct
        mean, sem, count = rows.mean(), rows.sem(), rows.count()
        half = sem * stats.t.ppf(.975, count - 1)
        band = 'rgba(0,114,178,.15)' if group == 'randomised_order' else 'rgba(213,94,0,.15)'
        fig.add_trace(go.Scatter(x=list(mean.index) + list(mean.index[::-1]),
                                 y=list(mean + half) + list((mean - half)[::-1]), fill='toself',
                                 fillcolor=band, line=dict(width=0), hoverinfo='skip',
                                 legendgroup=group, showlegend=False))
        fig.add_trace(go.Scatter(x=mean.index, y=mean, mode='lines+markers', name=LABELS[group],
                                 legendgroup=group, marker=dict(size=5),
                                 line=dict(color=COLOURS[group], width=2,
                                           dash='solid' if group == 'randomised_order' else 'dot')))
    fig.update_xaxes(title_text='Trial position', range=[.5, 40.5], dtick=5)
    fig.update_yaxes(title_text='Crossing judged unsafe (%)', range=[0, 100])
    save(fig, 'trial_profile', 1100, 480)
    (destination / 'source_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    index = {label: {extension: f'{name}.{extension}' for extension in ('pdf', 'png', 'html', 'pickle')}
             for name, label in FIGURE_LABELS.items()}
    (destination / 'figure_index.json').write_text(json.dumps(index, indent=2) + '\n')
    logger.info('Generated all %d manuscript figures in %s', len(FIGURE_LABELS), destination)
