"""
Analysis sub-package for radar performance visualization.

Contains:
    - AmbiguityPlot: Range/Velocity ambiguity trade-off analysis
    - ROCCurveWidget: Detection vs False Alarm trade-off
    - SNRHistogramWidget: SNR distribution visualization
"""

from .ambiguity_plot import AmbiguityPlot
from .roc_curve import ROCCurveWidget
from .snr_histogram import SNRHistogramWidget

__all__ = ['AmbiguityPlot', 'ROCCurveWidget', 'SNRHistogramWidget']
