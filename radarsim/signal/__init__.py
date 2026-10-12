# Developed by Mehmet Gümüş (@SpaceEngineerSS) - RadarSim v2.x
"""
Signals Package

Radar signal processing modules including waveform generation,
matched filtering, and CFAR detection.

References:
    - Richards, "Fundamentals of Radar Signal Processing"
    - Rohling, IEEE Trans. AES, 1983
"""

from .cfar import CFARDetector, CFARType
from .waveforms import BARKER_CODES, RadarWaveforms, WaveformType

__all__ = [
    "RadarWaveforms",
    "WaveformType",
    "BARKER_CODES",
    "CFARDetector",
    "CFARType",
]
