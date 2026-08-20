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
    "ECCMController",
    "FrequencyAgility",
    "PRFStagger",
    # SAR/ISAR imaging
    "ISARProcessor",
    "SARImageResult",
    "rda_vectorized",
]

# Optional import retained for installations using the ECCM controller.
try:
    from .eccm import ECCMController, FrequencyAgility, PRFStagger
except ImportError:
    pass

# SAR/ISAR imaging (conditional)
try:
    from .sar_isar import ISARProcessor, SARImageResult, rda_vectorized
except ImportError:
    pass
