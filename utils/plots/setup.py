"""Draw the experimental setup as a reproducible, not-to-scale schematic."""

import plotly.graph_objects as go


def experimental_setup() -> go.Figure:
    """Show both pedestrian orders and the yielding vehicle's stopping position."""
    fig = go.Figure()
    for offset, title, first, second in [
        (4.3, '(a) Participant first', 'Participant', 'Virtual pedestrian'),
        (0, '(b) Participant second', 'Virtual pedestrian', 'Participant'),
    ]:
        def label(x, y, text, **kwargs):
            fig.add_annotation(x=x, y=y + offset, text=text, showarrow=False, **kwargs)

        fig.add_shape(type='rect', x0=0, x1=14, y0=offset, y1=offset + 1.4,
                      fillcolor='#eeeeee', line_width=0, layer='below')
        fig.add_shape(type='rect', x0=0, x1=14, y0=offset + 1.4, y1=offset + 3.25,
                      fillcolor='#fafafa', line_width=0, layer='below')
        fig.add_shape(type='line', x0=0, x1=14, y0=offset + 1.4, y1=offset + 1.4,
                      line=dict(color='black', width=1))
        label(0, 3.65, '<b>' + title + '</b>', xanchor='left', font=dict(size=20))
        label(.2, 2.9, 'Pavement', xanchor='left')
        label(.2, 1.65, 'Kerb', xanchor='left')
        fig.add_shape(type='rect', x0=1.7, x1=4, y0=offset + .55, y1=offset + 1.18,
                      fillcolor='white', line=dict(color='black', width=2))
        label(2.85, .865, 'Vehicle')
        fig.add_shape(type='line', x0=4, x1=4, y0=offset + .65, y1=offset + 1.08,
                      line=dict(color='#008b9a', width=4))
        fig.add_annotation(x=4, y=offset + .25, ax=1.7, ay=offset + .25,
                           axref='x', ayref='y', text='', showarrow=True,
                           arrowhead=2, arrowwidth=1.5)
        label(4.25, .25, 'Direction of travel', xanchor='left')
        for x, top in [(4, 2.55), (6, 3.02), (11, 3.02)]:
            fig.add_shape(type='line', x0=x, x1=x, y0=offset + 1.4, y1=offset + top,
                          line=dict(color='#777777', width=1, dash='dash'))
        for left, right, y, text in [(4, 6, 2.55, '3 m'), (6, 11, 2.9, '2, 4, 6, 8 or 10 m')]:
            fig.add_annotation(x=right, y=offset + y, ax=left, ay=offset + y,
                               axref='x', ayref='y', text='', showarrow=True,
                               arrowside='end+start', arrowhead=2, startarrowhead=2, arrowwidth=1)
            label((left + right) / 2, y, text, bgcolor='white', borderpad=2)
        for x, name in [(6, first), (11, second)]:
            label(x, 2, name)
            fig.add_trace(go.Scatter(x=[x], y=[offset + 1.7], mode='markers',
                                     marker=dict(symbol='circle' if name == 'Participant' else 'square',
                                                 color='black' if name == 'Participant' else 'white',
                                                 size=12, line=dict(color='black', width=1)),
                                     showlegend=False, hoverinfo='skip'))
    fig.update_xaxes(range=[-.1, 14.1], visible=False, fixedrange=True)
    fig.update_yaxes(range=[-.1, 8.3], visible=False, fixedrange=True)
    fig.update_layout(showlegend=False)
    return fig
