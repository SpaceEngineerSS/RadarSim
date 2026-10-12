"""
Simulation Engine

Main time-stepping loop for radar simulation execution.

Features:
    - Discrete time evolution
    - Physics integration (radar equation, atmospheric loss)
    - Detection logic with probability of detection
    - Truth vs Measured data logging

Reference: Skolnik, "Radar Handbook", 3rd Ed., Chapter 2
"""

import collections
import copy
import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.physics.atmospheric import ITU_R_P676
from src.physics.geometry import azimuth_from_enu, enu_from_polar, wrap_to_pi
from src.physics.ecm import (
    DRFMConfig,
    DRFMJammer,
    DRFMState,
    ECMSimulator,
    apply_receiver_hard_limiter,
)
from src.physics.metrics import calculate_pd_swerling, cfar_loss_db
from src.physics.radar_equation import (
    RadarParameters,
    calculate_received_power,
    calculate_snr,
)
from src.physics.rain import ITU_R_P838
from src.physics.rcs import SwerlingModel
from src.tracking.monopulse import MonopulseEstimator

from .events import JAMMED_LOSS_DB, EventKind, SimEvent
from .objects import MotionModel, Radar, SimulationState, Target

# Terrain masking (optional)
try:
    from src.physics.terrain import TerrainMap

    TERRAIN_AVAILABLE = True
except ImportError:
    TERRAIN_AVAILABLE = False
    TerrainMap = None

# Track-While-Scan tracking (optional)
try:
    from src.tracking import TrackManager

    TRACKING_AVAILABLE = True
except ImportError:
    TRACKING_AVAILABLE = False
    TrackManager = None

# Clutter model
try:
    from src.physics.clutter import ClutterModel

    CLUTTER_AVAILABLE = True
except ImportError:
    CLUTTER_AVAILABLE = False
    ClutterModel = None

# Pulse-Doppler processing
try:
    from src.signal.pulse_doppler import PulseDopplerProcessor, RangeDopplerMap

    PULSE_DOPPLER_AVAILABLE = True
except ImportError:
    PULSE_DOPPLER_AVAILABLE = False
    PulseDopplerProcessor = None
    RangeDopplerMap = None


@dataclass
class DetectionResult:
    """
    Detection result for a single target at a point in time.

    Contains both truth (ground truth) and measured (with noise) data.
    """

    target_id: int
    time: float

    # Truth data (exact values)
    true_range_m: float
    true_azimuth_rad: float
    true_elevation_rad: float
    true_velocity_mps: float
    true_rcs_m2: float

    # Measured data (with noise)
    measured_range_m: float
    measured_azimuth_rad: float
    measured_elevation_rad: float

    # Detection status
    snr_db: float
    is_detected: bool
    pd: float  # Probability of detection

    illuminated: bool = True  # Beam swept the target during this step
    terrain_masked: bool = False
    atmospheric_loss_db: float = 0.0
    rain_attenuation_db: float = 0.0
    surface_clutter_loss_db: float = 0.0
    surface_sigma0_db: Optional[float] = None
    surface_cell_area_m2: float = 0.0
    surface_clutter_rcs_m2: float = 0.0
    surface_clutter_model: str = "disabled"
    rain_clutter_loss_db: float = 0.0
    pulse_compression_gain_db: float = 0.0
    jammer_jsr_db: Optional[float] = None
    jammer_loss_db: float = 0.0
    receiver_input_power_dbm: Optional[float] = None
    receiver_headroom_db: Optional[float] = None
    receiver_clipping_loss_db: float = 0.0
    receiver_overloaded: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "target_id": self.target_id,
            "time": self.time,
            "true_range_m": self.true_range_m,
            "true_azimuth_deg": np.degrees(self.true_azimuth_rad),
            "true_elevation_deg": np.degrees(self.true_elevation_rad),
            "measured_range_m": self.measured_range_m,
            "measured_azimuth_deg": np.degrees(self.measured_azimuth_rad),
            "snr_db": self.snr_db,
            "is_detected": self.is_detected,
            "pd": self.pd,
            "illuminated": self.illuminated,
            "terrain_masked": self.terrain_masked,
            "atmospheric_loss_db": self.atmospheric_loss_db,
            "rain_attenuation_db": self.rain_attenuation_db,
            "surface_clutter_loss_db": self.surface_clutter_loss_db,
            "surface_sigma0_db": self.surface_sigma0_db,
            "surface_cell_area_m2": self.surface_cell_area_m2,
            "surface_clutter_rcs_m2": self.surface_clutter_rcs_m2,
            "surface_clutter_model": self.surface_clutter_model,
            "rain_clutter_loss_db": self.rain_clutter_loss_db,
            "pulse_compression_gain_db": self.pulse_compression_gain_db,
            "jammer_jsr_db": self.jammer_jsr_db,
            "jammer_loss_db": self.jammer_loss_db,
            "receiver_input_power_dbm": self.receiver_input_power_dbm,
            "receiver_headroom_db": self.receiver_headroom_db,
            "receiver_clipping_loss_db": self.receiver_clipping_loss_db,
            "receiver_overloaded": self.receiver_overloaded,
        }


@dataclass
class FalseTarget:
    """
    ECM-generated false target (Chaff, DRFM ghost, Decoy).

    These are radar returns from countermeasures that appear as
    additional targets on the radar display.

    Attributes:
        position: [x, y, z] position [m]
        velocity: [vx, vy, vz] velocity [m/s]
        rcs_m2: Radar cross section [m²]
        ecm_type: Type of ECM (chaff, drfm, decoy)
        parent_target_id: ID of the real target that deployed this
        creation_time: Simulation time when created [s]
        lifetime_s: How long the false target persists [s]
    """

    position: np.ndarray
    velocity: np.ndarray
    rcs_m2: float
    ecm_type: str  # 'chaff', 'drfm', 'decoy', 'noise'
    parent_target_id: int
    creation_time: float
    lifetime_s: float = 60.0

    # Unique ID for this false target
    false_id: int = -1

    def is_expired(self, current_time: float) -> bool:
        """Check if false target has expired."""
        return (current_time - self.creation_time) > self.lifetime_s

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for UI."""
        return {
            "target_id": self.false_id,
            "position": self.position.tolist(),
            "velocity": self.velocity.tolist(),
            "rcs_m2": self.rcs_m2,
            "ecm_type": self.ecm_type,
            "is_false_target": True,
            "parent_id": self.parent_target_id,
        }


@dataclass
class SimulationLog:
    """
    Complete simulation log with all detection results.
    """

    detection_history: List[DetectionResult] = field(default_factory=list)

    # Statistics
    total_opportunities: int = 0
    total_detections: int = 0

    @property
    def detection_ratio(self) -> float:
        """Overall detection ratio."""
        if self.total_opportunities == 0:
            return 0.0
        return self.total_detections / self.total_opportunities

    def add_result(self, result: DetectionResult) -> None:
        """Add a detection result to the log."""
        self.detection_history.append(result)
        self.total_opportunities += 1
        if result.is_detected:
            self.total_detections += 1

    def get_target_history(self, target_id: int) -> List[DetectionResult]:
        """Get detection history for a specific target."""
        return [r for r in self.detection_history if r.target_id == target_id]


# Number of events kept in SimulationEngine.event_log.
EVENT_LOG_LENGTH = 5000

# ECM mode in which every jammer uses its own technique.
ECM_PER_TARGET = "per_target"

# Euclidean pre-gate for confirmed-track association [m]; the chi-square gate
# does the statistical work, this only bounds the search.
TRACK_GATE_DISTANCE_M = 5000.0

# Shortest time a track may coast without an update before it is dropped [s].
MIN_TRACK_COAST_TIME_S = 3.0

# SNR reported for a target hidden by terrain [dB].
MASKED_SNR_DB = -100.0

# Rain height h_R = h_0 + 0.36 km with a mid-latitude 0 deg C isotherm h_0 = 3 km.
# Reference: ITU-R P.839-4.
DEFAULT_RAIN_HEIGHT_M = 3360.0

# Clutter velocity spread sigma_v: wooded land in light wind and sea clutter.
# Reference: Nathanson, "Radar Design Principles", 2nd Ed., Table 7.5 / Ch. 7.
LAND_CLUTTER_SIGMA_V_MPS = 0.25
SEA_CLUTTER_SIGMA_V_MPS = 1.0

# The simple engine MTI is a two-pulse (single delay-line) canceller.
MTI_CANCELLER_ORDER = 1

# Frequency-agile band the barrage jammer must cover, as a carrier fraction.
FREQUENCY_AGILITY_BANDWIDTH_FRACTION = 0.1

@dataclass(frozen=True)
class ClutterSector:
    """
    Region of the surface with its own terrain class.

    Azimuths run clockwise from North; the sector spans start -> end clockwise.
    ``sigma0_adjustment_db`` is added to the model sigma0 (e.g. a measured
    calibration offset).
    """

    azimuth_start_deg: float
    azimuth_end_deg: float
    range_start_m: float
    range_end_m: float
    terrain_type: str
    sigma0_adjustment_db: float = 0.0

    def contains(self, azimuth_rad: float, range_m: float) -> bool:
        if not self.range_start_m <= range_m <= self.range_end_m:
            return False
        width = (self.azimuth_end_deg - self.azimuth_start_deg) % 360.0
        offset = (np.degrees(azimuth_rad) - self.azimuth_start_deg) % 360.0
        return offset <= width


@dataclass(frozen=True)
class CFARConfig:
    """Adaptive-threshold detector configuration (loss applied to every look)."""

    cfar_type: str = "CA"
    reference_cells: int = 16
    guard_cells: int = 2


# Oh-1992 bare-soil model validity: 10-70 deg incidence = 20-80 deg grazing.
OH1992_MIN_GRAZING_RAD = np.radians(20.0)
OH1992_MAX_GRAZING_RAD = np.radians(80.0)


class SimulationEngine:
    """
    Main simulation engine - the heartbeat of the radar simulation.

    Orchestrates time evolution, physics calculations, and detection logic.

    Reference: Skolnik, "Radar Handbook", 3rd Ed.
    """

    def __init__(
        self,
        radar: Radar,
        targets: List[Target] = None,
        dt: float = 0.01,
        enable_atmospheric: bool = True,
        detection_threshold_db: float = 13.0,
        range_noise_std_m: float = 50.0,
        angle_noise_std_rad: float = 0.001,
        terrain: "TerrainMap" = None,
        probability_false_alarm: float = 1e-6,
        pulses_integrated: int = 1,
        atmospheric_temperature_c: float = 15.0,
        atmospheric_pressure_hpa: float = 1013.25,
        water_vapor_density_g_m3: float = 7.5,
        enable_clutter: bool = False,
        terrain_type: str = "rural",
        sea_state: int = 3,
        rain_rate_mm_hr: float = 0.0,
        receiver_full_scale_dbm: float = -20.0,
        ground_model: str = "gamma",
        land_gamma_db: Optional[float] = None,
        ground_relative_permittivity: complex = complex(8.0, -0.8),
        ground_rms_height_m: float = 0.01,
        seed: Optional[int] = None,
        scan_gated_detection: bool = True,
        rain_height_m: float = DEFAULT_RAIN_HEIGHT_M,
        clutter_velocity_spread_mps: Optional[float] = None,
        clutter_sectors: Optional[List[ClutterSector]] = None,
        cfar: Optional[CFARConfig] = None,
    ):
        """
        Initialize simulation engine.

        Args:
            radar: Radar object
            targets: List of Target objects
            dt: Time step [s]
            enable_atmospheric: Enable ITU-R atmospheric loss
            detection_threshold_db: Legacy nominal SNR threshold retained for compatibility
            range_noise_std_m: Range measurement noise [m]
            angle_noise_std_rad: Angle measurement noise [rad]
            terrain: Optional TerrainMap for LOS calculations
            probability_false_alarm: Probability of false alarm per resolution cell
            pulses_integrated: Pulses integrated by the square-law detector
            atmospheric_temperature_c: Path temperature [degrees Celsius]
            atmospheric_pressure_hpa: Barometric (total) path pressure [hPa]
            water_vapor_density_g_m3: Water-vapor density [g/m^3]
            enable_clutter: Enable surface and volume clutter losses
            terrain_type: Surface classification used by the clutter model
            sea_state: Douglas sea-state index from 0 through 6
            rain_rate_mm_hr: Rain rate [mm/h]
            receiver_full_scale_dbm: Complex-envelope hard-limiter threshold [dBm]
            ground_model: Land reflectivity model, either gamma or oh1992
            land_gamma_db: Measured or calibrated gamma value for the gamma model [dB]
            ground_relative_permittivity: Passive complex soil permittivity eps'-j eps''
            ground_rms_height_m: RMS soil surface height used by Oh-1992 [m]
            seed: Seed for every random draw (None = non-reproducible)
            scan_gated_detection: Detect a target only while the beam sweeps it
            rain_height_m: Top of the rain layer (ITU-R P.839) [m]
            clutter_velocity_spread_mps: Clutter spectrum std-dev for MTI
                (None = land/sea default by terrain type)
            clutter_sectors: Regions overriding ``terrain_type`` (first match wins)
            cfar: Adaptive-threshold detector; its CFAR loss is applied to Pd
        """
        if dt <= 0.0:
            raise ValueError("dt must be greater than zero")
        if not 0.0 < probability_false_alarm < 1.0:
            raise ValueError("probability_false_alarm must be between 0 and 1")
        if pulses_integrated < 1:
            raise ValueError("pulses_integrated must be at least 1")
        if atmospheric_pressure_hpa <= 0.0:
            raise ValueError("atmospheric_pressure_hpa must be greater than zero")
        if water_vapor_density_g_m3 < 0.0 or rain_rate_mm_hr < 0.0:
            raise ValueError("water-vapor density and rain rate cannot be negative")
        if not 0 <= sea_state <= 6:
            raise ValueError("sea_state must be between 0 and 6")
        ground_model = ground_model.lower()
        if ground_model not in {"gamma", "oh1992"}:
            raise ValueError("ground_model must be 'gamma' or 'oh1992'")
        if ground_relative_permittivity.real <= 1.0 or ground_relative_permittivity.imag > 0.0:
            raise ValueError("ground_relative_permittivity must use eps'-j eps''")
        if ground_rms_height_m <= 0.0:
            raise ValueError("ground_rms_height_m must be greater than zero")
        if rain_height_m <= 0.0:
            raise ValueError("rain_height_m must be greater than zero")
        if clutter_velocity_spread_mps is not None and clutter_velocity_spread_mps <= 0.0:
            raise ValueError("clutter_velocity_spread_mps must be positive")

        self.radar = radar
        self.targets = targets or []
        self.dt = dt
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.scan_gated_detection = scan_gated_detection
        self.rain_height_m = rain_height_m
        self.clutter_velocity_spread_mps = clutter_velocity_spread_mps
        self.clutter_sectors: List[ClutterSector] = list(clutter_sectors or [])
        self.cfar = cfar
        self._cfar_loss_cache: Tuple[Any, float] = (None, 0.0)

        # Snapshots for reset()
        self._initial_radar_state = copy.deepcopy(radar.state)
        self._initial_antenna_azimuth = radar.antenna_azimuth
        self._initial_target_states = {
            target.target_id: copy.deepcopy(target.state) for target in self.targets
        }
        self.enable_atmospheric = enable_atmospheric
        self.detection_threshold_db = detection_threshold_db
        self.range_noise_std = range_noise_std_m
        self.angle_noise_std = angle_noise_std_rad
        self.probability_false_alarm = probability_false_alarm
        self.pulses_integrated = int(pulses_integrated)
        self.atmospheric_temperature_c = atmospheric_temperature_c
        self.atmospheric_pressure_hpa = atmospheric_pressure_hpa
        self.water_vapor_density_g_m3 = water_vapor_density_g_m3

        # Terrain for line-of-sight calculations
        self.terrain = terrain
        self.enable_terrain_masking = terrain is not None

        # Simulation state
        self.current_time = 0.0
        self.state = SimulationState(radar=radar)
        for target in self.targets:
            self.state.add_target(target)

        # Logging
        self.log = SimulationLog()

        # Radar parameters are accessed dynamically via self._radar_params property
        # to ensure real-time synchronization with self.radar changes.

        # ECM false targets (chaff, DRFM ghosts, decoys)
        self.false_targets: List[FalseTarget] = []
        self._next_false_id = 1000  # IDs start at 1000 to distinguish from real targets

        # ═══ PERFORMANCE: Hard limit to prevent freezing ═══
        self.MAX_FALSE_TARGETS = 20  # Never exceed this count

        # ECM state
        self.ecm_active = False
        # ECM_PER_TARGET: every jammer uses its own technique (target.ecm_type);
        # any other value overrides all jammers ('noise_barrage', 'drfm', ...).
        self.ecm_type = ECM_PER_TARGET
        self.ecm_activation_time = 0.0  # When ECM was activated (for RGPO drift)
        self._ecm_simulator = ECMSimulator(radar_wavelength=radar.wavelength)
        self._drfm_jammers: Dict[int, DRFMJammer] = {}

        # ═══ TRACK-WHILE-SCAN (TWS) ═══
        # Last full evaluation per target, reused between beam passes
        self._last_looks: Dict[int, DetectionResult] = {}

        self.tracks: list = []
        # Events raised during the last step, and a bounded history of all events
        self.events: List[SimEvent] = []
        self.event_log: "collections.deque[SimEvent]" = collections.deque(
            maxlen=EVENT_LOG_LENGTH
        )
        self._detected_once: set = set()
        self._detected_last_look: set = set()
        self._masked_targets: set = set()
        self._jammed_targets: set = set()
        self._track_status: Dict[int, str] = {}
        if TRACKING_AVAILABLE:
            self.track_manager = TrackManager(
                gate_distance=TRACK_GATE_DISTANCE_M,
                max_misses=3,  # Missed scans before a confirmed track is dropped
                confirm_hits=3,  # Consecutive scan hits to confirm
                process_noise=10.0,  # CWNA acceleration noise [m/s^2]
                measurement_noise=max(range_noise_std_m, 1.0),
            )
            self.enable_tracking = True
        else:
            self.track_manager = None
            self.enable_tracking = False

        # ═══ ADVANCED MODULE STATE (LPI & Fusion) ═══
        # These are set by the GUI via SimulationThread.set_lpi_mode() / set_fusion_mode()
        self.lpi_enabled = False
        self.lpi_technique = "FHSS"  # 'FHSS', 'DSSS', 'Costas'
        self.fusion_enabled = False
        self.fusion_method = "covariance_intersection"

        # Clutter, MTI, and ECCM state
        # Clutter: Adds environmental noise (ground/sea clutter)
        self.clutter_enabled = enable_clutter
        self.terrain_type = terrain_type
        self.sea_state = sea_state
        self.rain_rate_mm_hr = rain_rate_mm_hr
        self.receiver_full_scale_dbm = receiver_full_scale_dbm
        self.receiver_full_scale_power_watts = 1e-3 * 10.0 ** (
            receiver_full_scale_dbm / 10.0
        )
        self.ground_model = ground_model
        self.land_gamma_db = land_gamma_db
        self.ground_relative_permittivity = ground_relative_permittivity
        self.ground_rms_height_m = ground_rms_height_m
        if enable_clutter and (
            "sea" in terrain_type.lower() or "water" in terrain_type.lower()
        ):
            frequency_ghz = radar.frequency_hz / 1e9
            if not 0.5 <= frequency_ghz <= 35.0:
                raise ValueError(
                    "NRL sea clutter model requires radar frequency from 0.5 to 35 GHz"
                )

        # MTI: Moving Target Indication - filters slow/stationary targets
        self.mti_enabled = False
        self.mti_threshold_mps = 15.0  # Targets slower than this are rejected

        # ECCM: Electronic Counter-Countermeasures
        self.frequency_agility_enabled = False
        self.base_frequency_hz = radar.frequency_hz  # Store original frequency

        # Monopulse angle measurement (replaces the fixed angle noise when enabled)
        self.monopulse_enabled = False
        self._monopulse: Optional[MonopulseEstimator] = None

        # Frame counter for throttled signal processing and display updates
        self._frame_count = 0

        # Store pulse width for rain volume clutter calculation
        self._pulse_width_s = radar.pulse_width_s

        # Pulse-Doppler processing
        self.pulse_doppler_enabled = False
        self._pd_processor: Optional["PulseDopplerProcessor"] = None
        self._rd_map: Optional["RangeDopplerMap"] = None
        self._pd_mti_order = 0  # MTI canceller order
        self._pd_n_pulses = 64  # Coherent pulses per CPI
        self._pd_prf_hz = radar.prf_hz

    def add_target(self, target: Target) -> None:
        """Add a target to the simulation."""
        self.targets.append(target)
        self.state.add_target(target)
        self._initial_target_states[target.target_id] = copy.deepcopy(target.state)

    def set_pulse_doppler_mode(
        self,
        enabled: bool,
        n_pulses: int = 64,
        prf_hz: Optional[float] = None,
        mti_order: int = 0,
    ) -> None:
        """
        Enable or disable signal-level pulse-Doppler processing.

        When enabled, generates a true Range-Doppler map from coherent
        pulse train processing instead of parametric detection only.

        Args:
            enabled: Enable pulse-Doppler processing
            n_pulses: Pulses per CPI (slow-time dimension)
            prf_hz: Pulse repetition frequency [Hz] (None = the radar's PRF)
            mti_order: MTI canceller order (0=off, 1=2-pulse, 2=3-pulse)

        Reference: Richards (2005), "Fundamentals of Radar Signal Processing"
        """
        if prf_hz is None:
            prf_hz = self.radar.prf_hz
        self.pulse_doppler_enabled = enabled and PULSE_DOPPLER_AVAILABLE
        self._pd_n_pulses = n_pulses
        self._pd_prf_hz = prf_hz
        self._pd_mti_order = mti_order

        if self.pulse_doppler_enabled:
            self._pd_processor = PulseDopplerProcessor(
                prf_hz=prf_hz,
                n_pulses=n_pulses,
                n_range_bins=512,
                bandwidth_hz=5e6,
                pulse_width_s=self._pulse_width_s,
                frequency_hz=self.radar.frequency_hz,
                mti_order=mti_order,
                window_type="hamming",
            )
            self._rd_map = None
        else:
            self._pd_processor = None
            self._rd_map = None

    def _run_pulse_doppler(self) -> None:
        """
        Execute pulse-Doppler processing for targets in the antenna beam.

        Only processes when the beam sweeps past targets (simulates
        real radar dwell time). Updates the internal R-D map.

        Reference: Richards (2005), Ch. 3-5
        """
        if not self.pulse_doppler_enabled or self._pd_processor is None:
            return

        # Throttle PD processing to every 5th frame (~6 Hz at 30 FPS)
        if self._frame_count % 5 != 0:
            return

        # Collect targets within the antenna beam or nearby
        ranges_m = []
        velocities_mps = []
        amplitudes = []

        for target in self.targets:
            geom = self.radar.calculate_target_geometry(
                target.position, target.velocity
            )
            r_m = geom["range_m"]
            if r_m < 100.0 or r_m > self._pd_processor.max_instrumented_range_m:
                continue

            # Only targets inside the main lobe contribute to this CPI
            az_diff = abs(wrap_to_pi(geom["azimuth_rad"] - self.radar.antenna_azimuth))
            if az_diff > self.radar.beamwidth_rad:
                continue

            # Single-pulse SNR from the radar equation, coherently integrated
            # over the CPI (gain N; the window loss is applied by the processor).
            single_pulse_snr = max(
                10 ** (self.state.snr_values.get(target.target_id, MASKED_SNR_DB) / 10.0),
                1e-12,
            )
            amplitude = self._pd_processor.amplitude_for_output_snr(
                single_pulse_snr * self._pd_processor.n_pulses
            )

            ranges_m.append(r_m)
            velocities_mps.append(geom["radial_velocity_mps"])
            amplitudes.append(amplitude)

        # Always generate R-D map (with at least noise)
        self._rd_map = self._pd_processor.process_cpi(
            target_ranges_m=np.array(ranges_m) if ranges_m else np.array([]),
            target_velocities_mps=np.array(velocities_mps)
            if velocities_mps
            else np.array([]),
            target_amplitudes=np.array(amplitudes) if amplitudes else np.array([]),
            noise_power=1.0,
            seed=int(self.rng.integers(0, 2**31 - 1)),
        )

    def effective_ecm_type(self, target: Target) -> str:
        """Technique a jammer-equipped target uses under the current ECM mode."""
        if self.ecm_type == ECM_PER_TARGET:
            return target.ecm_type
        return self.ecm_type

    def set_ecm_mode(self, active: bool, ecm_type: str = ECM_PER_TARGET) -> None:
        """
        Switch electronic attack on or off.

        Args:
            active: Master switch for all jammer-equipped targets
            ecm_type: ECM_PER_TARGET ("per_target", default; legacy alias "noise")
                to let every jammer use its own technique, or one technique
                ('noise_barrage', 'noise_spot', 'drfm', 'chaff', 'decoy') for all
        """
        normalized_type = ecm_type.lower().replace(" ", "_")
        if normalized_type in ("noise", "scenario"):
            normalized_type = ECM_PER_TARGET
        previous_type = self.ecm_type
        if active and not self.ecm_active:
            self.ecm_activation_time = self.current_time

        self.ecm_active = active
        self.ecm_type = normalized_type

        if previous_type != normalized_type:
            self.false_targets = []
            for jammer in self._drfm_jammers.values():
                jammer.deactivate()

        if active:
            for target in self.targets:
                if not target.jammer_active or not self.effective_ecm_type(
                    target
                ).startswith("drfm"):
                    continue
                jammer = DRFMJammer(
                    DRFMConfig(
                        power_watts=max(target.jammer_power_watts, 1e-12),
                        gain_over_skin_db=target.drfm_gain_over_skin_db,
                        pull_rate_mps=target.drfm_pull_rate_mps,
                        max_pull_m=target.drfm_max_pull_m,
                        capture_dwell_s=target.drfm_capture_dwell_s,
                        mode=target.drfm_mode,
                        vgpo_accel_hz_per_s=target.drfm_vgpo_rate_hz_per_s,
                        max_doppler_pull_hz=target.drfm_max_doppler_pull_hz,
                        inherent_delay_s=target.drfm_inherent_delay_s,
                    )
                )
                jammer.activate()
                self._drfm_jammers[target.target_id] = jammer

        # Clear false targets when ECM is deactivated
        if not active:
            self.false_targets = []
            self.ecm_activation_time = 0.0
            for jammer in self._drfm_jammers.values():
                jammer.deactivate()

    def generate_ecm_false_targets(self, parent_target: Target) -> List[FalseTarget]:
        """
        Generate false targets based on ECM type.

        5 Distinct ECM Behaviors:
        1. NOISE_BARRAGE: No false targets, wide sector noise (handled by UI)
        2. NOISE_SPOT: No false targets, narrow focused noise (handled by UI)
        3. DRFM: 1-2 ghost targets at delayed range
        4. CHAFF: 3-5 slow-moving dipole clouds
        5. DECOY: 1 high-RCS diverging target

        Reference: Schleher, "Electronic Warfare in the Information Age"

        Args:
            parent_target: The real target deploying ECM

        Returns:
            List of FalseTarget objects (empty for noise types)
        """
        false_targets = []

        if not parent_target.jammer_active:
            return false_targets

        ecm_type = self.effective_ecm_type(parent_target)

        # ═══ CASE 1 & 2: NOISE TYPES ═══
        # Noise Barrage and Noise Spot don't create false targets
        # They create visual strobes only (handled by UI)
        if "noise" in ecm_type:
            return false_targets

        # ═══ CASE 3: CHAFF CLOUD ═══
        elif ecm_type == "chaff" or ecm_type == "chaff_cloud":
            # CHAFF: Deploy 3-5 slow-moving dipole clouds
            # Physics: Chaff falls slowly (wind drift ~5 m/s), RCS ~5-10 m²
            # Key: Velocity drops to near-zero, separates from fast aircraft
            n_clouds = int(self.rng.integers(3, 6))

            for i in range(n_clouds):
                # Random offset from parent (within 500m sphere)
                offset = self.rng.uniform(-300, 300, 3)
                offset[2] = -abs(offset[2]) * 0.3  # Chaff tends to fall (z is up)

                # Wind drift velocity - CRITICAL: near-zero velocity!
                # This makes chaff appear stationary on Doppler display
                wind_vel = np.array(
                    [
                        self.rng.uniform(-3, 3),  # Slow horizontal drift
                        self.rng.uniform(-3, 3),
                        self.rng.uniform(-1, -0.2),  # Slow falling
                    ]
                )

                false_target = FalseTarget(
                    position=parent_target.position + offset,
                    velocity=wind_vel,  # Near-ZERO velocity (key difference from aircraft)
                    rcs_m2=self.rng.uniform(5.0, 12.0),  # Large blooming RCS
                    ecm_type="chaff",
                    parent_target_id=parent_target.target_id,
                    creation_time=self.current_time,
                    lifetime_s=45.0,  # Chaff disperses after ~45s
                    false_id=self._next_false_id,
                )
                self._next_false_id += 1
                false_targets.append(false_target)

        # ═══ CASE 4: DRFM REPEATER with RGPO ═══
        elif ecm_type == "drfm" or ecm_type == "drfm_repeater":
            jammer = self._drfm_jammers.get(parent_target.target_id)
            if jammer is None or not jammer.is_active:
                return false_targets

            line_of_sight = parent_target.position - self.radar.position
            target_range = np.linalg.norm(line_of_sight)
            if target_range <= 0.0:
                return false_targets
            radial_unit = line_of_sight / target_range
            range_offset = jammer.false_range_offset_m
            ghost_position = parent_target.position + radial_unit * range_offset
            ghost_velocity = parent_target.velocity.copy()
            if jammer.state == DRFMState.PULL:
                if jammer.config.mode == "rgpo":
                    ghost_velocity = (
                        ghost_velocity + radial_unit * jammer.config.pull_rate_mps
                    )
                else:
                    # Doppler pull dfd shifts apparent radial velocity by -dfd*lambda/2.
                    ghost_velocity = ghost_velocity - radial_unit * (
                        jammer.pull_offset_hz * self.radar.wavelength / 2.0
                    )

            false_target = FalseTarget(
                position=ghost_position,
                velocity=ghost_velocity,
                rcs_m2=parent_target.rcs_mean
                * 10.0 ** (jammer.config.gain_over_skin_db / 10.0),
                ecm_type="drfm",
                parent_target_id=parent_target.target_id,
                creation_time=self.current_time,
                lifetime_s=max(1.1, 2.0 * self.dt),
                false_id=self._next_false_id,
            )
            self._next_false_id += 1
            false_targets.append(false_target)

        # ═══ CASE 5: DECOY ═══
        elif ecm_type == "decoy":
            # DECOY: Towed or expendable decoy (e.g., ALE-50)
            # Physics: High-RCS target that DIVERGES from main aircraft
            # Key: Maintains high speed but separates by ~5 degrees

            # Calculate divergence direction (perpendicular + slight outward)
            parent_dir = parent_target.velocity / (
                np.linalg.norm(parent_target.velocity) + 1e-6
            )

            # Create perpendicular divergence vector
            perp = np.array([-parent_dir[1], parent_dir[0], 0])
            if np.linalg.norm(perp) < 0.1:
                perp = np.array([1, 0, 0])
            perp = perp / (np.linalg.norm(perp) + 1e-6)

            # Diverge by ~5 degrees (sin(5°) ≈ 0.087)
            diverge_vel = (
                parent_target.velocity
                + perp * np.linalg.norm(parent_target.velocity) * 0.09
            )

            # Position slightly behind and to side
            offset = -parent_dir * 150 + perp * 50  # 150m behind, 50m to side

            false_target = FalseTarget(
                position=parent_target.position + offset,
                velocity=diverge_vel,  # High speed but diverging path
                rcs_m2=parent_target.rcs_mean * 2.5,  # Decoys intentionally high RCS
                ecm_type="decoy",
                parent_target_id=parent_target.target_id,
                creation_time=self.current_time,
                lifetime_s=180.0,  # Decoys persist long time
                false_id=self._next_false_id,
            )
            self._next_false_id += 1
            false_targets.append(false_target)

        return false_targets

    def get_jamming_info(self, target: Target) -> dict:
        """
        Get jamming strobe parameters for UI visualization.

        Returns strobe width and intensity based on ECM type:
        - Noise Barrage: Wide strobe (+/- 15°)
        - Noise Spot: Narrow strobe (+/- 2°), 10x intensity

        Args:
            target: Target with active jammer

        Returns:
            Dict with 'strobe_width_deg', 'intensity', 'ecm_type'
        """
        if not target.jammer_active or not self.ecm_active:
            return {"active": False}

        ecm_type = self.effective_ecm_type(target)

        if ecm_type == "noise_barrage" or ecm_type == "noise":
            return {
                "active": True,
                "ecm_type": "noise_barrage",
                "strobe_width_deg": 30.0,  # Wide +/- 15°
                "intensity": 0.6,
                "is_noise_type": True,
            }
        elif ecm_type == "noise_spot":
            return {
                "active": True,
                "ecm_type": "noise_spot",
                "strobe_width_deg": 4.0,  # Narrow +/- 2°
                "intensity": 1.0,  # 10x power density = max intensity
                "is_noise_type": True,
            }
        else:
            # DRFM, Chaff, Decoy - no noise strobe, show false targets instead
            return {
                "active": True,
                "ecm_type": ecm_type,
                "is_noise_type": False,
                "false_target_count": len(self.false_targets),
            }

    def _beam_swept(self, azimuth_rad: float) -> bool:
        """
        True if the beam centre crossed ``azimuth_rad`` during the last step.

        Each azimuth is crossed exactly once per revolution, so every target gets
        one detection opportunity (one dwell) per scan; the dwell's pulses are
        represented by ``pulses_integrated``. A non-rotating (staring) antenna
        illuminates whatever lies inside its main lobe.
        """
        return self.radar.beam_swept(azimuth_rad)

    def _rain_path_length_m(self, radar_z: float, target_z: float, range_m: float) -> float:
        """Length of the slant path below the rain height (ITU-R P.839 h_R)."""
        low, high = min(radar_z, target_z), max(radar_z, target_z)
        if high <= self.rain_height_m:
            return range_m
        if low >= self.rain_height_m:
            return 0.0
        return range_m * (self.rain_height_m - low) / (high - low)

    def _emit(self, kind: EventKind, message: str, **ids) -> None:
        event = SimEvent(time=self.current_time, kind=kind, message=message, **ids)
        self.events.append(event)
        self.event_log.append(event)

    def _look_events(self, target: Target, result: DetectionResult) -> None:
        tid = target.target_id
        if result.is_detected and tid not in self._detected_once:
            self._detected_once.add(tid)
            self._emit(
                EventKind.FIRST_DETECTION,
                f"{target.name} first detected at {result.true_range_m / 1000:.1f} km",
                target_id=tid,
            )
        was_masked = tid in self._masked_targets
        if result.terrain_masked:
            self._masked_targets.add(tid)
        else:
            self._masked_targets.discard(tid)
        if result.terrain_masked != was_masked:
            kind = EventKind.MASKED if result.terrain_masked else EventKind.UNMASKED
            verb = "hidden by terrain" if result.terrain_masked else "clear of terrain"
            self._emit(kind, f"{target.name} {verb}", target_id=tid)
        jammed = result.jammer_loss_db > JAMMED_LOSS_DB
        was_jammed = tid in self._jammed_targets
        if jammed:
            self._jammed_targets.add(tid)
        else:
            self._jammed_targets.discard(tid)
        if jammed and not was_jammed:
            self._emit(
                EventKind.JAMMED,
                f"{target.name} jamming costs {result.jammer_loss_db:.0f} dB",
                target_id=tid,
            )
        if jammed and result.is_detected and tid not in self._detected_last_look:
            self._emit(
                EventKind.BURN_THROUGH,
                f"Burn-through on {target.name} at {result.true_range_m / 1000:.1f} km",
                target_id=tid,
            )
        if result.is_detected:
            self._detected_last_look.add(tid)
        else:
            self._detected_last_look.discard(tid)

    def _track_events(self) -> None:
        current = {track.id: track.status.value for track in self.tracks}
        for track_id, status in current.items():
            before = self._track_status.get(track_id)
            if before is None:
                self._emit(EventKind.TRACK_INITIATED, f"Track {track_id} initiated", track_id=track_id)
            if status == "confirmed" and before not in ("confirmed", "coasting"):
                self._emit(EventKind.TRACK_CONFIRMED, f"Track {track_id} confirmed", track_id=track_id)
        for track_id, before in self._track_status.items():
            if track_id not in current and before in ("confirmed", "coasting"):
                self._emit(EventKind.TRACK_LOST, f"Track {track_id} lost", track_id=track_id)
        self._track_status = current

    def _monopulse_estimator(self) -> MonopulseEstimator:
        beamwidth_deg = float(np.degrees(self.radar.beamwidth_rad))
        if (
            self._monopulse is None
            or self._monopulse.beamwidth_deg != beamwidth_deg
        ):
            self._monopulse = MonopulseEstimator(beamwidth_deg=beamwidth_deg)
        return self._monopulse

    def _clutter_velocity_spread_mps(self, terrain_type: str) -> float:
        if self.clutter_velocity_spread_mps is not None:
            return self.clutter_velocity_spread_mps
        is_water = "sea" in terrain_type.lower() or "water" in terrain_type.lower()
        return SEA_CLUTTER_SIGMA_V_MPS if is_water else LAND_CLUTTER_SIGMA_V_MPS

    def _surface_class(self, azimuth_rad: float, range_m: float) -> Tuple[str, float]:
        """Terrain type and sigma0 adjustment of the surface cell at (az, range)."""
        for sector in self.clutter_sectors:
            if sector.contains(azimuth_rad, range_m):
                return sector.terrain_type, sector.sigma0_adjustment_db
        return self.terrain_type, 0.0

    @property
    def pulse_compression_gain(self) -> float:
        """Matched-filter time-bandwidth gain tau * B (1 for an unmodulated pulse)."""
        return max(1.0, self.radar.pulse_width_s * self.radar.receiver_bandwidth_hz)

    @property
    def compressed_pulse_width_s(self) -> float:
        """
        Matched-filter output pulse width, which sets the range resolution c*tau/2.

        A pulse-compression waveform whose receiver bandwidth B exceeds 1/tau
        compresses to about 1/B (Richards 2005, Sec. 4.6); an unmodulated pulse
        keeps its length.
        """
        return min(self.radar.pulse_width_s, 1.0 / self.radar.receiver_bandwidth_hz)

    @property
    def active_targets(self) -> List[Target]:
        """Targets deployed at the current simulation time."""
        return [t for t in self.targets if t.is_active(self.current_time)]

    @property
    def cfar_loss_db(self) -> float:
        """CFAR loss of the configured detector at the configured Pfa [dB]."""
        if self.cfar is None:
            return 0.0
        key = (self.cfar, self.probability_false_alarm)
        if self._cfar_loss_cache[0] != key:
            self._cfar_loss_cache = (
                key,
                cfar_loss_db(
                    self.cfar.cfar_type,
                    self.cfar.reference_cells,
                    self.probability_false_alarm,
                ),
            )
        return self._cfar_loss_cache[1]

    def step(self, dt: float = None) -> List[DetectionResult]:
        """
        Advance simulation by one time step.

        Per target: geometry, propagation losses, terrain masking, thermal SNR
        from the aspect-dependent *mean* RCS, surface/volume clutter, noise
        jamming and receiver limiting. The Swerling Pd then averages over the
        RCS fluctuation, so the fluctuation is not sampled a second time. A
        detection opportunity exists only while the rotating beam sweeps the
        target (scan-gated); detected targets yield noisy range/angle
        measurements that feed the track-while-scan tracker.

        Args:
            dt: Time step [s] (uses default if None)

        Returns:
            List of DetectionResult for this time step (one per target)
        """
        if dt is None:
            dt = self.dt

        results = []

        # 1. Update all object kinematics
        previous_time = self.current_time
        self.events = []
        self.state.update_all(dt)
        self.current_time = self.state.time
        for target in self.targets:
            if 0.0 < target.active_from_s and previous_time < target.active_from_s <= self.current_time:
                self._emit(EventKind.DEPLOYED, f"{target.name} deployed", target_id=target.target_id)

        if self.ecm_active:
            for jammer in self._drfm_jammers.values():
                jammer.step(dt)

        self._frame_count += 1
        freq_ghz = self.radar.frequency_hz / 1e9
        radar_z = float(self.radar.position[2])
        tracker_detections: List[Tuple[float, float]] = []

        # 2. Process each deployed target
        for target in self.active_targets:
            geom = self.radar.calculate_target_geometry(
                target.position, target.velocity
            )
            illuminated = (not self.scan_gated_detection) or self._beam_swept(
                geom["azimuth_rad"]
            )
            previous = self._last_looks.get(target.target_id)
            if not illuminated and previous is not None:
                # The radar did not look at this target this step: its SNR is
                # only re-measured on the next look (no new detection decision).
                results.append(
                    dataclasses.replace(
                        previous,
                        time=self.current_time,
                        true_range_m=geom["range_m"],
                        true_azimuth_rad=geom["azimuth_rad"],
                        true_elevation_rad=geom["elevation_rad"],
                        true_velocity_mps=geom["radial_velocity_mps"],
                        measured_range_m=float("nan"),
                        measured_azimuth_rad=float("nan"),
                        measured_elevation_rad=float("nan"),
                        is_detected=False,
                        illuminated=False,
                    )
                )
                self.state.detections[target.target_id] = False
                continue
            range_m = geom["range_m"]
            target_z = float(target.position[2])
            rcs = target.mean_rcs(self.radar.position)

            # Gaseous attenuation (ITU-R P.676 valid for 1-1000 GHz; negligible below)
            atm_loss_db = 0.0
            if self.enable_atmospheric and freq_ghz >= 1.0 and range_m > 100.0:
                atm_loss_db = ITU_R_P676.total_attenuation(
                    range_m / 1000.0,
                    freq_ghz,
                    temperature_c=self.atmospheric_temperature_c,
                    pressure_hpa=ITU_R_P676.dry_air_pressure_hpa(
                        self.atmospheric_pressure_hpa,
                        self.atmospheric_temperature_c,
                        self.water_vapor_density_g_m3,
                    ),
                    water_vapor_density=self.water_vapor_density_g_m3,
                    two_way=True,
                )

            # Rain attenuation over the part of the path below the rain height
            rain_loss_db = 0.0
            rain_path_m = self._rain_path_length_m(radar_z, target_z, range_m)
            if self.rain_rate_mm_hr > 0.0 and rain_path_m > 100.0 and freq_ghz >= 1.0:
                rain_loss_db = ITU_R_P838.path_attenuation(
                    rain_path_m / 1000.0,
                    freq_ghz,
                    self.rain_rate_mm_hr,
                    elevation_angle_deg=geom["elevation_deg"],
                    polarization_tilt_deg=self.radar.polarization_tilt_deg,
                    two_way=True,
                )
            propagation_loss_db = atm_loss_db + rain_loss_db

            # Terrain masking (line of sight)
            terrain_masked = False
            if self.enable_terrain_masking and self.terrain is not None:
                is_visible, _, _ = self.terrain.check_line_of_sight(
                    self.radar.position, target.position
                )
                terrain_masked = not is_visible

            # The SNR chain works with inverse SNR terms referred to the matched-
            # filter output (Richards 2005, Secs. 2.4 and 4.6): thermal noise and
            # noise jamming are white over the receiver band and gain nothing from
            # pulse compression, while clutter is an echo of the same waveform and
            # is compressed like the target (its cell uses the compressed width).
            compression_gain = self.pulse_compression_gain
            inverse_noise = inverse_clutter = inverse_jamming = 0.0
            if terrain_masked:
                snr_db = MASKED_SNR_DB
                input_snr = 0.0
                signal_input_power_watts = None
            else:
                input_snr = 10.0 ** (
                    calculate_snr(
                        self._radar_params,
                        rcs,
                        range_m,
                        atmospheric_loss_db=propagation_loss_db,
                    )
                    / 10.0
                )
                inverse_noise = 1.0 / (input_snr * compression_gain)
                snr_db = -10.0 * np.log10(inverse_noise)
                signal_input_power_watts = calculate_received_power(
                    self._radar_params,
                    rcs,
                    range_m,
                    atmospheric_loss_db=propagation_loss_db,
                )

            # Surface clutter in the target's range cell, entering through the
            # elevation pattern (beam pointed at the target) and reduced by MTI.
            clutter_snr_loss_db = 0.0
            surface_sigma0_db = None
            surface_cell_area_m2 = 0.0
            surface_clutter_rcs_m2 = 0.0
            surface_clutter_model = "disabled"
            if self.clutter_enabled and CLUTTER_AVAILABLE and range_m > 100.0 and not terrain_masked:
                radar_height_m = max(radar_z, 0.0)
                # Surface-cell geometry is undefined at exactly normal incidence
                # (radar overhead), so cap the grazing angle just below 90 deg.
                geometric_grazing = min(
                    np.arcsin(np.clip(radar_height_m / range_m, 0.0, 1.0)),
                    np.radians(89.9),
                )
                terrain_type, sigma0_adjustment_db = self._surface_class(
                    geom["azimuth_rad"], range_m
                )
                is_water = "sea" in terrain_type.lower() or "water" in terrain_type.lower()

                if is_water:
                    model_grazing = np.clip(
                        geometric_grazing, np.radians(0.1), np.radians(60.0)
                    )
                    sigma_h_db = ClutterModel.sea_clutter_sigma0(
                        model_grazing,
                        sea_state=self.sea_state,
                        frequency_ghz=freq_ghz,
                        polarization="HH",
                    )
                    sigma_v_db = ClutterModel.sea_clutter_sigma0(
                        model_grazing,
                        sea_state=self.sea_state,
                        frequency_ghz=freq_ghz,
                        polarization="VV",
                    )
                    surface_clutter_model = "nrl_five_parameter"
                elif self.ground_model == "oh1992" and (
                    OH1992_MIN_GRAZING_RAD <= geometric_grazing <= OH1992_MAX_GRAZING_RAD
                ):
                    # Oh-1992 is only defined for 10-70 deg incidence; outside that
                    # domain the constant-gamma branch below is used instead.
                    model_grazing = geometric_grazing
                    sigma_h_db = ClutterModel.bare_soil_oh1992_sigma0(
                        model_grazing,
                        freq_ghz,
                        self.ground_relative_permittivity,
                        self.ground_rms_height_m,
                        "HH",
                    )
                    sigma_v_db = ClutterModel.bare_soil_oh1992_sigma0(
                        model_grazing,
                        freq_ghz,
                        self.ground_relative_permittivity,
                        self.ground_rms_height_m,
                        "VV",
                    )
                    surface_clutter_model = "oh1992_bare_soil"
                else:
                    model_grazing = max(geometric_grazing, np.radians(0.1))
                    sigma_h_db = ClutterModel.ground_clutter_sigma0(
                        model_grazing,
                        terrain_type=terrain_type,
                        frequency_ghz=freq_ghz,
                        polarization="HH",
                        gamma_db=self.land_gamma_db,
                    )
                    sigma_v_db = sigma_h_db
                    surface_clutter_model = "constant_gamma"

                tilt = np.radians(self.radar.polarization_tilt_deg)
                sigma0_linear = (
                    10.0 ** (sigma_h_db / 10.0) * np.cos(tilt) ** 2
                    + 10.0 ** (sigma_v_db / 10.0) * np.sin(tilt) ** 2
                ) * 10.0 ** (sigma0_adjustment_db / 10.0)
                surface_sigma0_db = float(10.0 * np.log10(sigma0_linear))
                surface_cell_area_m2 = ClutterModel.surface_resolution_cell_area(
                    range_m,
                    self.compressed_pulse_width_s,
                    self.radar.beamwidth_rad,
                    self.radar.beamwidth_el_rad,
                    model_grazing,
                )
                # The clutter patch lies at depression angle -grazing; the beam
                # points at the target's elevation.
                pattern_gain = ClutterModel.elevation_two_way_gain(
                    geom["elevation_rad"] + geometric_grazing,
                    self.radar.beamwidth_el_rad,
                    self.radar.sidelobe_level_db,
                )
                surface_clutter_rcs_m2 = sigma0_linear * surface_cell_area_m2 * pattern_gain
                if self.mti_enabled:
                    improvement_db = ClutterModel.mti_improvement_factor_db(
                        MTI_CANCELLER_ORDER,
                        self.radar.prf_hz,
                        self.radar.wavelength,
                        self._clutter_velocity_spread_mps(terrain_type),
                    )
                    surface_clutter_rcs_m2 *= 10.0 ** (-improvement_db / 10.0)

                if surface_clutter_rcs_m2 > 0.0:
                    thermal_snr_db = snr_db
                    inverse_clutter += surface_clutter_rcs_m2 / rcs
                    snr_db = -10.0 * np.log10(inverse_noise + inverse_clutter)
                    clutter_snr_loss_db = thermal_snr_db - snr_db

            # Noise jamming (self-screening)
            jammer_snr_loss_db = 0.0
            jsr_db = None
            active_ecm_type = self.effective_ecm_type(target)
            if (
                self.ecm_active
                and target.jammer_active
                and target.jammer_power_watts > 0.0
                and "noise" in active_ecm_type
                and not terrain_masked
            ):
                effective_jammer_bandwidth = target.jammer_bandwidth_hz
                if self.frequency_agility_enabled:
                    # The jammer must spread over the agile band (10% of the carrier).
                    effective_jammer_bandwidth = max(
                        effective_jammer_bandwidth,
                        FREQUENCY_AGILITY_BANDWIDTH_FRACTION * self.radar.frequency_hz,
                    )
                jsr_db = self._ecm_simulator.calculate_jsr(
                    radar_pos=self.radar.position,
                    target_pos=target.position,
                    jammer_pos=target.position,
                    radar_power=self.radar.power_watts,
                    radar_gain=10.0 ** (self.radar.antenna_gain_db / 10.0),
                    jammer_power=target.jammer_power_watts,
                    target_rcs=rcs,
                    radar_bandwidth=self.radar.receiver_bandwidth_hz,
                    jammer_bandwidth=effective_jammer_bandwidth,
                    signal_path_loss_db=propagation_loss_db
                    + self.radar.system_losses_db,
                    jammer_path_loss_db=0.5
                    * (propagation_loss_db + self.radar.system_losses_db),
                )
                pre_jamming_snr_db = snr_db
                inverse_jamming = 10.0 ** (jsr_db / 10.0) / compression_gain
                snr_db = -10.0 * np.log10(inverse_noise + inverse_clutter + inverse_jamming)
                jammer_snr_loss_db = pre_jamming_snr_db - snr_db

            # Rain volume clutter in the target's resolution cell
            rain_clutter_loss_db = 0.0
            if (
                self.rain_rate_mm_hr > 0.0
                and CLUTTER_AVAILABLE
                and range_m > 100.0
                and target_z < self.rain_height_m
                and not terrain_masked
            ):
                eta = ClutterModel.rain_reflectivity_marshall_palmer(
                    self.rain_rate_mm_hr, freq_ghz
                )
                rain_clutter_rcs = ClutterModel.volume_clutter_rcs(
                    eta,
                    range_m,
                    self.radar.beamwidth_rad,
                    self.compressed_pulse_width_s,
                    elevation_beamwidth_rad=self.radar.beamwidth_el_rad,
                )
                if rain_clutter_rcs > 0.0:
                    pre_volume_clutter_snr_db = snr_db
                    inverse_clutter += rain_clutter_rcs / rcs
                    snr_db = -10.0 * np.log10(
                        inverse_noise + inverse_clutter + inverse_jamming
                    )
                    rain_clutter_loss_db = pre_volume_clutter_snr_db - snr_db

            # Receiver hard limiter
            receiver_input_power_dbm = None
            receiver_headroom_db = None
            receiver_clipping_loss_db = 0.0
            receiver_overloaded = False
            if signal_input_power_watts is not None:
                # Before compression the clutter echo is spread over the full
                # pulse, i.e. compression_gain times its compressed-cell power.
                inverse_input = (
                    1.0 / input_snr
                    + inverse_jamming * compression_gain
                    + inverse_clutter * compression_gain
                )
                limiter = apply_receiver_hard_limiter(
                    signal_input_power_watts,
                    signal_input_power_watts * inverse_input,
                    self.receiver_full_scale_power_watts,
                )
                snr_db -= limiter.clipping_loss_db
                receiver_input_power_dbm = limiter.input_power_dbm
                receiver_headroom_db = limiter.headroom_db
                receiver_clipping_loss_db = limiter.clipping_loss_db
                receiver_overloaded = limiter.overloaded

            # Detection: Swerling Pd averages over the RCS fluctuation
            pd = self._calculate_pd(snr_db - self.cfar_loss_db, target.swerling_model)
            is_detected = illuminated and bool(self.rng.random() < pd)

            # MTI notch: returns below the threshold radial speed are cancelled
            # together with the clutter (Richards 2005, Sec. 17.1).
            if self.mti_enabled and is_detected:
                if abs(geom["radial_velocity_mps"]) < self.mti_threshold_mps:
                    is_detected = False

            if is_detected:
                measured_range = range_m + self.rng.normal(0.0, self.range_noise_std)
                if self.monopulse_enabled:
                    # Sum/difference estimate relative to the current beam pointing;
                    # the elevation beam is assumed centred on the target.
                    monopulse = self._monopulse_estimator().measure_angle(
                        self.radar.antenna_azimuth,
                        geom["elevation_rad"],
                        geom["azimuth_rad"],
                        geom["elevation_rad"],
                        snr_db=snr_db,
                        rng=self.rng,
                    )
                    measured_az = monopulse.measured_azimuth_rad
                    measured_el = monopulse.measured_elevation_rad
                else:
                    measured_az = wrap_to_pi(
                        geom["azimuth_rad"] + self.rng.normal(0.0, self.angle_noise_std)
                    )
                    measured_el = geom["elevation_rad"] + self.rng.normal(
                        0.0, self.angle_noise_std
                    )
                ground_offset = enu_from_polar(measured_range, measured_az, measured_el)
                tracker_detections.append(
                    (
                        float(self.radar.position[0] + ground_offset[0]),
                        float(self.radar.position[1] + ground_offset[1]),
                    )
                )
            else:
                measured_range = measured_az = measured_el = float("nan")

            result = DetectionResult(
                target_id=target.target_id,
                time=self.current_time,
                true_range_m=range_m,
                true_azimuth_rad=geom["azimuth_rad"],
                true_elevation_rad=geom["elevation_rad"],
                true_velocity_mps=geom["radial_velocity_mps"],
                true_rcs_m2=rcs,
                measured_range_m=measured_range,
                measured_azimuth_rad=measured_az,
                measured_elevation_rad=measured_el,
                snr_db=snr_db,
                is_detected=is_detected,
                pd=pd,
                illuminated=illuminated,
                terrain_masked=terrain_masked,
                atmospheric_loss_db=atm_loss_db,
                rain_attenuation_db=rain_loss_db,
                surface_clutter_loss_db=clutter_snr_loss_db,
                surface_sigma0_db=surface_sigma0_db,
                surface_cell_area_m2=surface_cell_area_m2,
                surface_clutter_rcs_m2=surface_clutter_rcs_m2,
                surface_clutter_model=surface_clutter_model,
                rain_clutter_loss_db=rain_clutter_loss_db,
                pulse_compression_gain_db=10.0 * np.log10(compression_gain),
                jammer_jsr_db=jsr_db,
                jammer_loss_db=jammer_snr_loss_db,
                receiver_input_power_dbm=receiver_input_power_dbm,
                receiver_headroom_db=receiver_headroom_db,
                receiver_clipping_loss_db=receiver_clipping_loss_db,
                receiver_overloaded=receiver_overloaded,
            )

            if illuminated:
                self._look_events(target, result)
            results.append(result)
            self._last_looks[target.target_id] = result
            if illuminated:
                self.log.add_result(result)

            self.state.detections[target.target_id] = is_detected
            self.state.snr_values[target.target_id] = snr_db
            self.state.terrain_masked[target.target_id] = terrain_masked

            # 3. ECM false targets, generated at most once per second
            if self.ecm_active and target.jammer_active:
                time_since_activation = self.current_time - self.ecm_activation_time
                should_generate = int(time_since_activation) != int(
                    time_since_activation - dt
                )
                if should_generate or len(self.false_targets) == 0:
                    self.false_targets.extend(self.generate_ecm_false_targets(target))

        # Enforce FIFO cap on false targets
        while len(self.false_targets) > self.MAX_FALSE_TARGETS:
            self.false_targets.pop(0)

        # 4. Update and clean up false targets
        active_false_targets = []
        for ft in self.false_targets:
            if not ft.is_expired(self.current_time):
                if ft.creation_time < self.current_time:
                    ft.position = ft.position + ft.velocity * dt
                active_false_targets.append(ft)
        self.false_targets = active_false_targets

        # 5. Track-while-scan: only tracks inside the swept sector can be updated
        #    or charged a miss this step.
        if self.enable_tracking and self.track_manager is not None:
            coverage = None
            if self.scan_gated_detection:

                def coverage(position: Tuple[float, float]) -> bool:
                    return self._beam_swept(
                        azimuth_from_enu(
                            position[0] - self.radar.position[0],
                            position[1] - self.radar.position[1],
                        )
                    )

            revisit_s = self.radar.revisit_period_s if self.scan_gated_detection else 0.0
            # A sector raster passes each azimuth twice per cycle at uneven
            # intervals, so only the shorter half-cycle is guarded.
            revisit_guard_s = 0.25 * revisit_s
            self.tracks = self.track_manager.update(
                tracker_detections,
                dt,
                in_coverage=coverage,
                revisit_guard_s=revisit_guard_s,
                max_coast_time_s=max(3.5 * revisit_s, MIN_TRACK_COAST_TIME_S),
            )
            self._track_events()

        # 6. Pulse-Doppler processing
        self._run_pulse_doppler()

        return results

    def run(self, duration_s: float) -> SimulationLog:
        """
        Run simulation for a specified duration.

        Args:
            duration_s: Simulation duration [s]

        Returns:
            SimulationLog with all results
        """
        n_steps = int(round(duration_s / self.dt))

        for _ in range(n_steps):
            self.step()

        return self.log

    def reset(self) -> None:
        """Restore the initial scenario state (kinematics, ECM, tracks, RNG)."""
        self.radar.state = copy.deepcopy(self._initial_radar_state)
        self.radar.antenna_azimuth = self._initial_antenna_azimuth
        self.radar.previous_antenna_azimuth = self._initial_antenna_azimuth
        self.radar.swept_legs = []
        self.radar._sector_direction = 1.0
        for target in self.targets:
            initial = self._initial_target_states.get(target.target_id)
            if initial is not None:
                target.state = copy.deepcopy(initial)
            target.age_s = 0.0
            target._position_history.clear()
        self.current_time = 0.0
        self.state = SimulationState(radar=self.radar)
        for target in self.targets:
            self.state.add_target(target)
        self.log = SimulationLog()
        self.rng = np.random.default_rng(self.seed)
        self.false_targets = []
        self._next_false_id = 1000
        for jammer in self._drfm_jammers.values():
            jammer.deactivate()
        self._drfm_jammers = {}
        self.ecm_active = False
        self.ecm_activation_time = 0.0
        self._frame_count = 0
        self._rd_map = None
        self.tracks = []
        self._last_looks = {}
        self.events = []
        self.event_log.clear()
        self._detected_once = set()
        self._detected_last_look = set()
        self._masked_targets = set()
        self._jammed_targets = set()
        self._track_status = {}
        if self.track_manager is not None:
            self.track_manager.clear()

    def _calculate_pd(
        self,
        snr_db: float,
        swerling_model: SwerlingModel,
        pfa: Optional[float] = None,
    ) -> float:
        """
        Calculate probability of detection for given SNR.

        Uses a square-law detector threshold derived from Pfa. Conditional
        integrated target power follows a non-central chi-square distribution;
        Swerling fluctuations are averaged over their RCS distributions.

        Args:
            snr_db: Signal-to-noise ratio [dB]
            swerling_model: Target fluctuation model
            pfa: Probability of false alarm

        Returns:
            Probability of detection (0-1)
        """
        return calculate_pd_swerling(
            snr_db,
            pfa=self.probability_false_alarm if pfa is None else pfa,
            swerling_case=swerling_model.value,
            n_pulses=self.pulses_integrated,
        )

    @property
    def simulation_time(self) -> float:
        """Current simulation time [s]."""
        return self.current_time

    @property
    def _radar_params(self) -> RadarParameters:
        """
        Dynamic radar parameters synchronized with self.radar in real time.

        Ensures that interactive UI controls or runtime API changes to
        radar frequency, power, bandwidth, or losses are immediately
        reflected in SNR and detection calculations.
        """
        return RadarParameters(
            frequency=self.radar.frequency_hz,
            power_transmitted=self.radar.power_watts,
            antenna_gain_tx=self.radar.antenna_gain_db,
            antenna_gain_rx=self.radar.antenna_gain_db,
            system_losses_tx=self.radar.system_losses_db / 2.0,
            system_losses_rx=self.radar.system_losses_db / 2.0,
            noise_figure=self.radar.noise_figure_db,
            temperature=self.radar.system_temperature_k,
            pulse_width=self.radar.pulse_width_s,
            prf=self.radar.prf_hz,
            noise_bandwidth=self.radar.receiver_bandwidth_hz,
        )


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================


def validate_linear_motion(
    velocity_mps: float = 100.0, duration_s: float = 10.0, dt: float = 0.01
) -> dict:
    """
    Validate constant velocity motion model.

    A target starting at x=0 with velocity v should be at x=v*t after time t.

    Args:
        velocity_mps: Target velocity [m/s]
        duration_s: Simulation duration [s]
        dt: Time step [s]

    Returns:
        Validation result
    """
    # Create target at origin with velocity in x direction
    target = Target(
        target_id=1,
        position=np.array([0.0, 0.0, 0.0]),
        velocity=np.array([velocity_mps, 0.0, 0.0]),
        motion_model=MotionModel.CONSTANT_VELOCITY,
    )

    # Run kinematic updates
    n_steps = int(duration_s / dt)
    for _ in range(n_steps):
        target.update(dt)

    # Expected position
    expected_x = velocity_mps * duration_s
    actual_x = target.position[0]

    # Tolerance (allow small numerical error)
    tolerance_m = 0.1  # 10 cm
    is_valid = abs(actual_x - expected_x) <= tolerance_m

    return {
        "parameters": {
            "velocity_mps": velocity_mps,
            "duration_s": duration_s,
            "dt": dt,
            "n_steps": n_steps,
        },
        "computed_values": {
            "final_position": target.position.tolist(),
            "actual_x": actual_x,
            "expected_x": expected_x,
            "error_m": abs(actual_x - expected_x),
        },
        "validation": {
            "is_valid": is_valid,
            "tolerance_m": tolerance_m,
            "reference": "Kinematic equation: x = x0 + v*t",
        },
    }


def validate_detection_logic(
    close_range_km: float = 10.0, far_range_km: float = 500.0
) -> dict:
    """
    Validate detection logic at different ranges.

    Close targets (high SNR) should be detected.
    Far targets (low SNR) should not be detected.

    Args:
        close_range_km: Close range for high SNR test [km]
        far_range_km: Far range for low SNR test [km]

    Returns:
        Validation result
    """
    # Create radar
    radar = Radar(
        radar_id="test_radar",
        position=np.array([0.0, 0.0, 0.0]),
        frequency_hz=10e9,
        power_watts=100e3,
        antenna_gain_db=30.0,
    )

    # Create close target (should detect)
    close_target = Target(
        target_id=1,
        position=np.array([close_range_km * 1000, 0.0, 0.0]),
        rcs_m2=10.0,
        swerling_model=SwerlingModel.SWERLING_0,  # No fluctuation for test
    )

    # Create far target (should not detect)
    far_target = Target(
        target_id=2,
        position=np.array([far_range_km * 1000, 0.0, 0.0]),
        rcs_m2=1.0,
        swerling_model=SwerlingModel.SWERLING_0,
    )

    # Run simulation
    engine = SimulationEngine(
        radar=radar,
        targets=[close_target, far_target],
        dt=0.1,
        detection_threshold_db=13.0,
    )

    # Single step
    results = engine.step()

    close_result = next(r for r in results if r.target_id == 1)
    far_result = next(r for r in results if r.target_id == 2)

    # Validate
    close_detected = close_result.snr_db > engine.detection_threshold_db
    far_not_detected = far_result.snr_db < engine.detection_threshold_db

    is_valid = close_detected and far_not_detected

    return {
        "parameters": {
            "close_range_km": close_range_km,
            "far_range_km": far_range_km,
            "detection_threshold_db": engine.detection_threshold_db,
        },
        "computed_values": {
            "close_snr_db": close_result.snr_db,
            "far_snr_db": far_result.snr_db,
            "close_above_threshold": close_detected,
            "far_below_threshold": far_not_detected,
        },
        "validation": {
            "is_valid": is_valid,
            "reference": "Radar equation: SNR decreases with R^4",
        },
    }
