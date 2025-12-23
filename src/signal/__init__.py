"""
Signals Package

Radar signal processing modules including waveform generation,
matched filtering, and CFAR detection.

References:
    - Richards, "Fundamentals of Radar Signal Processing"
    - Rohling, IEEE Trans. AES, 1983
"""

from .waveforms import RadarWaveforms, WaveformType, BARKER_CODES
from .cfar import CFARDetector, CFARType

__all__ = [
    'RadarWaveforms',
    'WaveformType',
    'BARKER_CODES',
    'CFARDetector',
    'CFARType',
]
