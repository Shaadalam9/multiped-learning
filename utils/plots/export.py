"""Export analysis figures in the supported formats."""

from pathlib import Path
from typing import Any
import pickle
from utils.constants import LOGGER


def _save_plot(figure: Any, stem: Path, width: int = 1500, height: int = 900) -> None:
    """Save editable pickle, HTML, vector PDF, PNG and optional EPS; fail on missing core formats."""

    import plotly.io as pio

    # Match FigureExportMixin.save_plotly in the supplied human_analysis code.
    # Disabling MathJax prevents Kaleido from loading an external renderer.
    pio.kaleido.scope.mathjax = None
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(stem.with_suffix(".html")), include_plotlyjs="cdn")
    # Save the editable Plotly object as well as portable figure formats.
    with stem.with_suffix(".pickle").open("wb") as handle:
        pickle.dump(figure, handle, protocol=pickle.HIGHEST_PROTOCOL)
    figure.write_image(str(stem.with_suffix(".pdf")), width=width, height=height)
    figure.write_image(str(stem.with_suffix(".png")), width=width, height=height, scale=2)
    try:
        figure.write_image(str(stem.with_suffix(".eps")), width=width, height=height)
    except Exception as exc:
        LOGGER.warning("Optional EPS export failed for %s: %s", stem.name, exc)
