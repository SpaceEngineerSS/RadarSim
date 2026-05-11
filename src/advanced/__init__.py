# Developed by Mehmet Gümüş (@SpaceEngineerSS) - RadarSim v2.x
"""
Advanced Radar Modules

LPI radar, SAR/ISAR, Sensor Fusion, and advanced signal processing.
"""

# Export advanced module classes for easy import
from .lpi_radar import AdvancedLPIRadar
from .sar_isar import AdvancedSARISAR
from .sensor_fusion import AdvancedSensorFusion, SensorMeasurement
from .signal_processing import AdvancedSignalProcessor

__all__ = [
    "AdvancedLPIRadar",
    "AdvancedSARISAR",
    "AdvancedSensorFusion",
    "SensorMeasurement",
    "AdvancedSignalProcessor",
    # Phase 28: ECCM
    "ECCMController",
    "FrequencyAgility",
    "PRFStagger",
    # Phase 30: SAR/ISAR & AI Director
    "ISARProcessor",
    "SARImageResult",
    "rda_vectorized",
    "AIDirector",
    "Difficulty",
]

# Phase 28: ECCM (conditional for backward compatibility)
try:
    from .eccm import ECCMController, FrequencyAgility, PRFStagger
except ImportError:
    pass

# Phase 30: SAR/ISAR & AI Director (conditional)
try:
    from .sar_isar import ISARProcessor, SARImageResult, rda_vectorized
except ImportError:
    pass

try:
    from .ai_director import AIDirector, Difficulty
except ImportError:
    pass
