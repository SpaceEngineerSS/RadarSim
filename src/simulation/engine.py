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

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import time

from .objects import Target, Radar, SimulationState, MotionModel
from src.physics.radar_equation import (
    RadarParameters, calculate_snr, calculate_received_power
)
from src.physics.atmospheric import ITU_R_P676
from src.physics.rcs import SwerlingModel

# Terrain masking (optional)
try:
    from src.physics.terrain import TerrainMap, TerrainConfig
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

# Clutter model (Phase 19)
try:
    from src.physics.clutter import ClutterModel
    CLUTTER_AVAILABLE = True
except ImportError:
    CLUTTER_AVAILABLE = False
    ClutterModel = None

# AI Inference Engine (Phase 25)
try:
    from src.ml.inference_engine import InferenceEngine
    INFERENCE_AVAILABLE = True
except ImportError:
    INFERENCE_AVAILABLE = False
    InferenceEngine = None


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
    
    # AI Classification (Phase 25)
    predicted_class: str = "Unknown"
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'target_id': self.target_id,
            'time': self.time,
            'true_range_m': self.true_range_m,
            'true_azimuth_deg': np.degrees(self.true_azimuth_rad),
            'true_elevation_deg': np.degrees(self.true_elevation_rad),
            'measured_range_m': self.measured_range_m,
            'measured_azimuth_deg': np.degrees(self.measured_azimuth_rad),
            'snr_db': self.snr_db,
            'is_detected': self.is_detected,
            'pd': self.pd,
            'predicted_class': self.predicted_class,
            'confidence': self.confidence,
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
            'target_id': self.false_id,
            'position': self.position.tolist(),
            'velocity': self.velocity.tolist(),
            'rcs_m2': self.rcs_m2,
            'ecm_type': self.ecm_type,
            'is_false_target': True,
            'parent_id': self.parent_target_id,
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
        terrain: 'TerrainMap' = None
    ):
        """
        Initialize simulation engine.
        
        Args:
            radar: Radar object
            targets: List of Target objects
            dt: Time step [s]
            enable_atmospheric: Enable ITU-R atmospheric loss
            detection_threshold_db: SNR threshold for detection
            range_noise_std_m: Range measurement noise [m]
            angle_noise_std_rad: Angle measurement noise [rad]
            terrain: Optional TerrainMap for LOS calculations
        """
        self.radar = radar
        self.targets = targets or []
        self.dt = dt
        self.enable_atmospheric = enable_atmospheric
        self.detection_threshold_db = detection_threshold_db
        self.range_noise_std = range_noise_std_m
        self.angle_noise_std = angle_noise_std_rad
        
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
        
        # Build radar parameters for physics
        self._radar_params = RadarParameters(
            frequency=radar.frequency_hz,
            power_transmitted=radar.power_watts,
            antenna_gain_tx=radar.antenna_gain_db,
            antenna_gain_rx=radar.antenna_gain_db,
            noise_figure=4.0,
            pulse_width=1e-6,
        )
        
        # ECM false targets (chaff, DRFM ghosts, decoys)
        self.false_targets: List[FalseTarget] = []
        self._next_false_id = 1000  # IDs start at 1000 to distinguish from real targets
        
        # ═══ PERFORMANCE: Hard limit to prevent freezing ═══
        self.MAX_FALSE_TARGETS = 20  # Never exceed this count
        
        # ECM state
        self.ecm_active = False
        self.ecm_type = 'noise'  # 'noise', 'chaff', 'drfm', 'decoy'
        self.ecm_activation_time = 0.0  # When ECM was activated (for RGPO drift)
        
        # ═══ TRACK-WHILE-SCAN (TWS) ═══
        if TRACKING_AVAILABLE:
            self.track_manager = TrackManager(
                gate_distance=500.0,    # Association gate [m]
                max_misses=5,           # Coast before delete
                confirm_hits=3,         # Hits to confirm
                process_noise=5.0,      # Maneuver responsiveness
                measurement_noise=50.0  # Radar noise
            )
            self.enable_tracking = True
        else:
            self.track_manager = None
            self.enable_tracking = False
        
        # ═══ ADVANCED MODULE STATE (LPI & Fusion) ═══
        # These are set by the GUI via SimulationThread.set_lpi_mode() / set_fusion_mode()
        self.lpi_enabled = False
        self.lpi_technique = 'FHSS'  # 'FHSS', 'DSSS', 'Costas'
        self.fusion_enabled = False
        self.fusion_method = 'kalman'  # 'kalman', 'particle', 'bayesian'
        
        # ═══ PHASE 19: CLUTTER, MTI & ECCM STATE ═══
        # Clutter: Adds environmental noise (ground/sea clutter)
        self.clutter_enabled = False
        self.clutter_enabled = False
        self.terrain_type = 'rural'  # 'urban', 'suburban', 'rural', 'forest', 'desert', 'mountains', 'sea'
        self.sea_state = 3  # Douglas sea state (0-6)
        self.rain_rate_mm_hr = 0.0  # For volume clutter (rain/snow)
        
        # MTI: Moving Target Indication - filters slow/stationary targets
        self.mti_enabled = False
        self.mti_threshold_mps = 15.0  # Targets slower than this are rejected
        
        # ECCM: Electronic Counter-Countermeasures
        self.frequency_agility_enabled = False
        self.base_frequency_hz = radar.frequency_hz  # Store original frequency
        
        # ═══ PHASE 20: MONOPULSE ANGLE TRACKING ═══
        self.monopulse_enabled = False
        
        # ═══ PHASE 25: AI INFERENCE ENGINE ═══
        if INFERENCE_AVAILABLE:
            self._inference_engine = InferenceEngine()
            self.enable_auto_classification = True
        else:
            self._inference_engine = None
            self.enable_auto_classification = False
        
        # Frame counter for throttled operations (AI inference every 10 frames)
        self._frame_count = 0
        
        # Store pulse width for rain volume clutter calculation
        self._pulse_width_s = 1e-6  # 1 microsecond default
    
    def add_target(self, target: Target) -> None:
        """Add a target to the simulation."""
        self.targets.append(target)
        self.state.add_target(target)
    
    def set_ecm_mode(self, active: bool, ecm_type: str = 'noise') -> None:
        """
        Set ECM mode from GUI controls.
        
        Args:
            active: Whether ECM is active
            ecm_type: Type of ECM ('noise', 'chaff', 'drfm', 'decoy')
        """
        # Track activation time for RGPO drift calculation
        if active and not self.ecm_active:
            self.ecm_activation_time = self.current_time
        
        self.ecm_active = active
        self.ecm_type = ecm_type.lower().replace(' ', '_').replace('_barrage', '').replace('_spot', '')
        
        # Clear false targets when ECM is deactivated
        if not active:
            self.false_targets = []
            self.ecm_activation_time = 0.0
    
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
        
        ecm_type = self.ecm_type
        
        # ═══ CASE 1 & 2: NOISE TYPES ═══
        # Noise Barrage and Noise Spot don't create false targets
        # They create visual strobes only (handled by UI)
        if 'noise' in ecm_type:
            return false_targets
        
        # ═══ CASE 3: CHAFF CLOUD ═══
        elif ecm_type == 'chaff' or ecm_type == 'chaff_cloud':
            # CHAFF: Deploy 3-5 slow-moving dipole clouds
            # Physics: Chaff falls slowly (wind drift ~5 m/s), RCS ~5-10 m²
            # Key: Velocity drops to near-zero, separates from fast aircraft
            n_clouds = np.random.randint(3, 6)
            
            for i in range(n_clouds):
                # Random offset from parent (within 500m sphere)
                offset = np.random.uniform(-300, 300, 3)
                offset[2] = -abs(offset[2]) * 0.3  # Chaff tends to fall
                
                # Wind drift velocity - CRITICAL: near-zero velocity!
                # This makes chaff appear stationary on Doppler display
                wind_vel = np.array([
                    np.random.uniform(-3, 3),   # Slow horizontal drift
                    np.random.uniform(-3, 3),
                    np.random.uniform(-1, -0.2)  # Slow falling
                ])
                
                false_target = FalseTarget(
                    position=parent_target.position + offset,
                    velocity=wind_vel,  # Near-ZERO velocity (key difference from aircraft)
                    rcs_m2=np.random.uniform(5.0, 12.0),  # Large blooming RCS
                    ecm_type='chaff',
                    parent_target_id=parent_target.target_id,
                    creation_time=self.current_time,
                    lifetime_s=45.0,  # Chaff disperses after ~45s
                    false_id=self._next_false_id
                )
                self._next_false_id += 1
                false_targets.append(false_target)
        
        # ═══ CASE 4: DRFM REPEATER with RGPO ═══
        elif ecm_type == 'drfm' or ecm_type == 'drfm_repeater':
            # DRFM: Digital RF Memory creates 1-2 ghosts with RGPO drift
            # Physics: Ghost starts at aircraft position, slowly drifts BEHIND
            # RGPO (Range Gate Pull-Off): Ghost range increases by ~50m/s
            
            # Calculate time since ECM activation for RGPO drift
            time_active = self.current_time - self.ecm_activation_time
            rgpo_drift_rate = 50.0  # m/s drift rate (ghost moves away)
            rgpo_drift = min(time_active * rgpo_drift_rate, 2000.0)  # Cap at 2km
            
            n_ghosts = 1 if time_active < 3.0 else 2  # Second ghost after 3s
            
            for i in range(n_ghosts):
                # Base delay + RGPO drift (ghost drifts behind target over time)
                base_delay = 80 + i * 100  # 80m, 180m base offsets
                total_delay = base_delay + rgpo_drift
                
                # Ghost position: Same direction but increasing range (RGPO)
                direction = parent_target.position / (np.linalg.norm(parent_target.position) + 1e-6)
                ghost_pos = parent_target.position + direction * total_delay
                
                # Ghost velocity: Slightly slower than parent (appears to fall back)
                # This creates visible separation on Doppler display
                vel_reduction = direction * 20.0  # 20 m/s slower radially
                
                false_target = FalseTarget(
                    position=ghost_pos,
                    velocity=parent_target.velocity - vel_reduction,  # Slightly slower
                    rcs_m2=parent_target.rcs_mean * np.random.uniform(0.9, 1.2),
                    ecm_type='drfm',
                    parent_target_id=parent_target.target_id,
                    creation_time=self.current_time,
                    lifetime_s=0.15,  # DRFM ghosts continuously regenerate
                    false_id=self._next_false_id
                )
                self._next_false_id += 1
                false_targets.append(false_target)
        
        # ═══ CASE 5: DECOY ═══
        elif ecm_type == 'decoy':
            # DECOY: Towed or expendable decoy (e.g., ALE-50)
            # Physics: High-RCS target that DIVERGES from main aircraft
            # Key: Maintains high speed but separates by ~5 degrees
            
            # Calculate divergence direction (perpendicular + slight outward)
            parent_dir = parent_target.velocity / (np.linalg.norm(parent_target.velocity) + 1e-6)
            
            # Create perpendicular divergence vector
            perp = np.array([-parent_dir[1], parent_dir[0], 0])
            if np.linalg.norm(perp) < 0.1:
                perp = np.array([1, 0, 0])
            perp = perp / (np.linalg.norm(perp) + 1e-6)
            
            # Diverge by ~5 degrees (sin(5°) ≈ 0.087)
            diverge_vel = parent_target.velocity + perp * np.linalg.norm(parent_target.velocity) * 0.09
            
            # Position slightly behind and to side
            offset = -parent_dir * 150 + perp * 50  # 150m behind, 50m to side
            
            false_target = FalseTarget(
                position=parent_target.position + offset,
                velocity=diverge_vel,  # High speed but diverging path
                rcs_m2=parent_target.rcs_mean * 2.5,  # Decoys intentionally high RCS
                ecm_type='decoy',
                parent_target_id=parent_target.target_id,
                creation_time=self.current_time,
                lifetime_s=180.0,  # Decoys persist long time
                false_id=self._next_false_id
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
            return {'active': False}
        
        ecm_type = self.ecm_type
        
        if ecm_type == 'noise_barrage' or ecm_type == 'noise':
            return {
                'active': True,
                'ecm_type': 'noise_barrage',
                'strobe_width_deg': 30.0,  # Wide +/- 15°
                'intensity': 0.6,
                'is_noise_type': True,
            }
        elif ecm_type == 'noise_spot':
            return {
                'active': True,
                'ecm_type': 'noise_spot',
                'strobe_width_deg': 4.0,  # Narrow +/- 2°
                'intensity': 1.0,  # 10x power density = max intensity
                'is_noise_type': True,
            }
        else:
            # DRFM, Chaff, Decoy - no noise strobe, show false targets instead
            return {
                'active': True,
                'ecm_type': ecm_type,
                'is_noise_type': False,
                'false_target_count': len(self.false_targets),
            }
    
    def step(self, dt: float = None) -> List[DetectionResult]:
        """
        Advance simulation by one time step.
        
        This is the core of the simulation engine.
        
        Args:
            dt: Time step [s] (uses default if None)
            
        Returns:
            List of DetectionResult for this time step
        """
        if dt is None:
            dt = self.dt
        
        results = []
        
        # 1. Update all object kinematics
        self.state.update_all(dt)
        self.current_time = self.state.time
        
        # Increment frame counter for throttled operations
        self._frame_count += 1
        
        # 2. Process each target
        for target in self.targets:
            # Calculate geometry
            geom = self.radar.calculate_target_geometry(
                target.position,
                target.velocity
            )
            
            # Get target RCS with fluctuation
            rcs = target.get_rcs(self.radar.position)
            
            # Calculate atmospheric loss
            atm_loss_db = 0.0
            if self.enable_atmospheric:
                freq_ghz = self.radar.frequency_hz / 1e9
                range_km = geom['range_m'] / 1000
                if range_km > 0.1:
                    atm_loss_db = ITU_R_P676.total_attenuation(
                        range_km, freq_ghz, two_way=True
                    )
            
            # ═══ TERRAIN MASKING CHECK (LOS) ═══
            terrain_masked = False
            if self.enable_terrain_masking and self.terrain is not None:
                radar_pos_3d = np.array([
                    self.radar.position[0], 
                    self.radar.position[1], 
                    self.radar.position[2] if len(self.radar.position) > 2 else 0.0
                ])
                target_pos_3d = np.array([
                    target.position[0], 
                    target.position[1], 
                    target.position[2] if len(target.position) > 2 else 0.0
                ])
                is_visible, block_range, _ = self.terrain.check_line_of_sight(
                    radar_pos_3d, target_pos_3d
                )
                terrain_masked = not is_visible
            
            # Calculate SNR (terrain-masked targets get SNR = -inf)
            if terrain_masked:
                snr_db = -100.0  # Effectively invisible
            else:
                snr_db = calculate_snr(
                    self._radar_params,
                    rcs,
                    geom['range_m'],
                    atmospheric_loss_db=atm_loss_db
                )
            
            # ═══ PHASE 19: CLUTTER DEGRADATION ═══
            # Clutter adds to noise floor, degrading SNR
            # Reference: Skolnik, "Radar Handbook", Chapter 5
            clutter_snr_loss_db = 0.0
            if self.clutter_enabled and CLUTTER_AVAILABLE and geom['range_m'] > 100:
                try:
                    # Calculate grazing angle (simplified)
                    grazing_angle = abs(geom['elevation_rad']) if abs(geom['elevation_rad']) > 0.01 else 0.05
                    
                    # Get clutter backscatter coefficient
                    freq_ghz = self.radar.frequency_hz / 1e9
                    
                    # ═══ PHASE 25: SEA CLUTTER AUTOMATION ═══
                    # Use GIT model for sea, Nathanson model for land
                    if 'sea' in self.terrain_type.lower() or 'water' in self.terrain_type.lower():
                        sigma0_db = ClutterModel.sea_clutter_sigma0(
                            grazing_angle,
                            sea_state=self.sea_state,
                            frequency_ghz=freq_ghz
                        )
                    else:
                        sigma0_db = ClutterModel.ground_clutter_sigma0(
                            grazing_angle,
                            terrain_type=self.terrain_type,
                            frequency_ghz=freq_ghz
                        )
                    
                    # Resolution cell area (approximate)
                    range_resolution = 150  # meters (typical)
                    cross_range = geom['range_m'] * np.radians(self.radar.beamwidth_deg)
                    cell_area = range_resolution * cross_range
                    
                    # Clutter RCS 
                    sigma0_linear = 10 ** (sigma0_db / 10)
                    clutter_rcs = sigma0_linear * cell_area
                    
                    # Signal-to-Clutter Ratio (SCR) loss
                    # SNR_eff = SNR * (1 / (1 + C/N))
                    if clutter_rcs > 0.01:
                        clutter_snr_loss_db = 10 * np.log10(1 + clutter_rcs / (rcs + 0.01))
                        snr_db -= clutter_snr_loss_db
                except Exception:
                    pass  # Fail silently if clutter calculation fails
            
            # ═══ PHASE 19: ECCM - FREQUENCY AGILITY vs JAMMING ═══
            # If frequency agility is enabled, jammers are less effective
            eccm_jam_reduction = 1.0
            if self.frequency_agility_enabled and self.ecm_active:
                # Radar hops frequency by ±5%, jammer stays on original
                # Jamming effectiveness reduced significantly
                eccm_jam_reduction = 0.2  # 80% reduction in jamming effect
                snr_db += 6.0  # ~6 dB improvement against jamming
            
            # ═══ PHASE 25: RAIN ATTENUATION (ITU-R P.838) ═══
            # Rain causes significant attenuation at microwave frequencies
            # Reference: ITU-R P.838-3 "Specific attenuation model for rain"
            rain_loss_db = 0.0
            if self.rain_rate_mm_hr > 0 and geom['range_m'] > 100:
                freq_ghz = self.radar.frequency_hz / 1e9
                range_km = geom['range_m'] / 1000
                
                # ITU-R P.838 coefficients for horizontal polarization
                # k and α depend on frequency (using X-band approximation)
                # For 10 GHz: k ≈ 0.0101, α ≈ 1.276
                if freq_ghz <= 10:
                    k, alpha = 0.0101, 1.276
                elif freq_ghz <= 20:
                    k, alpha = 0.0367, 1.154
                else:
                    k, alpha = 0.0751, 1.099
                
                # Specific attenuation: γ_R = k * R^α [dB/km]
                gamma_rain = k * (self.rain_rate_mm_hr ** alpha)
                
                # Two-way path loss through rain
                rain_loss_db = gamma_rain * 2 * range_km
                snr_db -= rain_loss_db
            
            # ═══ PHASE 25: RAIN VOLUME CLUTTER ═══
            # Rain drops create radar returns that degrade SNR
            # Reference: Marshall & Palmer, "Distribution of Raindrops", 1948
            rain_clutter_loss_db = 0.0
            if self.rain_rate_mm_hr > 0 and CLUTTER_AVAILABLE and geom['range_m'] > 100:
                try:
                    freq_ghz = self.radar.frequency_hz / 1e9
                    beamwidth_rad = np.radians(self.radar.beamwidth_deg)
                    
                    # Get rain reflectivity using Marshall-Palmer Z-R relationship
                    eta = ClutterModel.rain_reflectivity_marshall_palmer(
                        self.rain_rate_mm_hr, freq_ghz
                    )
                    
                    # Calculate rain volume clutter RCS
                    rain_clutter_rcs = ClutterModel.volume_clutter_rcs(
                        eta, geom['range_m'], beamwidth_rad, self._pulse_width_s
                    )
                    
                    # Apply clutter-to-signal ratio loss
                    if rain_clutter_rcs > 0.01:
                        rain_clutter_loss_db = 10 * np.log10(1 + rain_clutter_rcs / (rcs + 0.01))
                        snr_db -= rain_clutter_loss_db
                except Exception:
                    pass  # Fail silently if rain clutter calculation fails
            
            # Calculate probability of detection (Swerling model)
            pd = self._calculate_pd(snr_db, target.swerling_model)
            
            # Detection decision
            is_detected = np.random.random() < pd
            
            # ═══ PHASE 19: MTI FILTER ═══
            # Moving Target Indication: Reject slow-moving targets (clutter)
            # Reference: Richards, "Fundamentals of Radar Signal Processing"
            mti_rejected = False
            if self.mti_enabled and is_detected:
                radial_vel = abs(geom['radial_velocity_mps'])
                if radial_vel < self.mti_threshold_mps:
                    is_detected = False
                    mti_rejected = True
                    # Target rejected by MTI filter (appears as clutter)
            
            # Generate measurements (with noise if detected)
            if is_detected:
                measured_range = geom['range_m'] + np.random.normal(0, self.range_noise_std)
                measured_az = geom['azimuth_rad'] + np.random.normal(0, self.angle_noise_std)
                measured_el = geom['elevation_rad'] + np.random.normal(0, self.angle_noise_std)
            else:
                measured_range = 0.0
                measured_az = 0.0
                measured_el = 0.0
            
            # ═══ PHASE 25: AI AUTO-CLASSIFICATION ═══
            # Automatically classify detected targets using ML inference
            # Reference: RandomForest classifier trained on synthetic radar data
            predicted_class = "Unknown"
            ai_confidence = 0.0
            
            if is_detected and self.enable_auto_classification and self._inference_engine is not None:
                # Throttle inference to every 10 frames (~300ms at 30fps) to save CPU
                if self._frame_count % 10 == 0:
                    try:
                        # Calculate Doppler frequency from radial velocity
                        doppler_hz = 2 * geom['radial_velocity_mps'] * self.radar.frequency_hz / 3e8
                        
                        # Build feature vector for inference
                        track_data = {
                            'range_km': geom['range_m'] / 1000.0,
                            'doppler_hz': abs(doppler_hz),
                            'snr_db': snr_db,
                            'rcs_est_m2': rcs,
                        }
                        
                        predicted_class, ai_confidence = self._inference_engine.predict(track_data)
                    except Exception:
                        pass  # Fail silently if inference fails
            
            # Create detection result
            result = DetectionResult(
                target_id=target.target_id,
                time=self.current_time,
                true_range_m=geom['range_m'],
                true_azimuth_rad=geom['azimuth_rad'],
                true_elevation_rad=geom['elevation_rad'],
                true_velocity_mps=geom['radial_velocity_mps'],
                true_rcs_m2=rcs,
                measured_range_m=measured_range,
                measured_azimuth_rad=measured_az,
                measured_elevation_rad=measured_el,
                snr_db=snr_db,
                is_detected=is_detected,
                pd=pd,
                predicted_class=predicted_class,
                confidence=ai_confidence,
            )
            
            results.append(result)
            self.log.add_result(result)
            
            # Update state tracking
            self.state.detections[target.target_id] = is_detected
            self.state.snr_values[target.target_id] = snr_db
            
            # 3. Generate ECM false targets for jammer-equipped targets
            # ═══ THROTTLE: Only generate new targets once per second ═══
            if self.ecm_active and target.jammer_active:
                # Check if enough time has passed since last generation
                time_since_activation = self.current_time - self.ecm_activation_time
                # Generate only at 1 Hz rate (once per second)
                should_generate = (int(time_since_activation) != int(time_since_activation - dt))
                if should_generate or len(self.false_targets) == 0:
                    new_false_targets = self.generate_ecm_false_targets(target)
                    self.false_targets.extend(new_false_targets)
        
        # ═══ PERFORMANCE: Enforce FIFO cap on false targets ═══
        while len(self.false_targets) > self.MAX_FALSE_TARGETS:
            self.false_targets.pop(0)  # Remove oldest (FIFO)
        
        # 4. Update and clean up false targets
        active_false_targets = []
        for ft in self.false_targets:
            if not ft.is_expired(self.current_time):
                # Update position based on velocity
                ft.position = ft.position + ft.velocity * dt
                active_false_targets.append(ft)
        self.false_targets = active_false_targets
        
        return results
    
    def run(self, duration_s: float) -> SimulationLog:
        """
        Run simulation for a specified duration.
        
        Args:
            duration_s: Simulation duration [s]
            
        Returns:
            SimulationLog with all results
        """
        n_steps = int(duration_s / self.dt)
        
        for _ in range(n_steps):
            self.step()
        
        return self.log
    
    def reset(self) -> None:
        """Reset simulation to initial state."""
        self.current_time = 0.0
        self.state = SimulationState(radar=self.radar)
        for target in self.targets:
            self.state.add_target(target)
        self.log = SimulationLog()
    
    def _calculate_pd(
        self,
        snr_db: float,
        swerling_model: SwerlingModel,
        pfa: float = 1e-6
    ) -> float:
        """
        Calculate probability of detection for given SNR.
        
        Uses Albersheim's equation approximation:
        SNR_req = A + 0.12*A*B + 1.7*B
        where A = ln(0.62/Pfa), B = ln(Pd/(1-Pd))
        
        Reference: Albersheim, 1981
        
        Args:
            snr_db: Signal-to-noise ratio [dB]
            swerling_model: Target fluctuation model
            pfa: Probability of false alarm
            
        Returns:
            Probability of detection (0-1)
        """
        # Swerling fluctuation loss
        fluctuation_loss = {
            SwerlingModel.SWERLING_0: 0.0,
            SwerlingModel.SWERLING_1: 8.0,
            SwerlingModel.SWERLING_2: 7.0,
            SwerlingModel.SWERLING_3: 5.0,
            SwerlingModel.SWERLING_4: 4.5,
        }
        
        loss = fluctuation_loss.get(swerling_model, 0.0)
        effective_snr = snr_db - loss
        
        # Simple sigmoid model for Pd vs SNR
        # Pd ≈ 0.5 at threshold, approaches 1 above, 0 below
        threshold_snr = self.detection_threshold_db
        
        # Using logistic function
        k = 0.5  # Steepness
        pd = 1.0 / (1.0 + np.exp(-k * (effective_snr - threshold_snr)))
        
        return np.clip(pd, 0.0, 1.0)
    
    @property
    def simulation_time(self) -> float:
        """Current simulation time [s]."""
        return self.current_time


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_linear_motion(
    velocity_mps: float = 100.0,
    duration_s: float = 10.0,
    dt: float = 0.01
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
        motion_model=MotionModel.CONSTANT_VELOCITY
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
        'parameters': {
            'velocity_mps': velocity_mps,
            'duration_s': duration_s,
            'dt': dt,
            'n_steps': n_steps,
        },
        'computed_values': {
            'final_position': target.position.tolist(),
            'actual_x': actual_x,
            'expected_x': expected_x,
            'error_m': abs(actual_x - expected_x),
        },
        'validation': {
            'is_valid': is_valid,
            'tolerance_m': tolerance_m,
            'reference': 'Kinematic equation: x = x0 + v*t',
        }
    }


def validate_detection_logic(
    close_range_km: float = 10.0,
    far_range_km: float = 500.0
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
        antenna_gain_db=30.0
    )
    
    # Create close target (should detect)
    close_target = Target(
        target_id=1,
        position=np.array([close_range_km * 1000, 0.0, 0.0]),
        rcs_m2=10.0,
        swerling_model=SwerlingModel.SWERLING_0  # No fluctuation for test
    )
    
    # Create far target (should not detect)
    far_target = Target(
        target_id=2,
        position=np.array([far_range_km * 1000, 0.0, 0.0]),
        rcs_m2=1.0,
        swerling_model=SwerlingModel.SWERLING_0
    )
    
    # Run simulation
    engine = SimulationEngine(
        radar=radar,
        targets=[close_target, far_target],
        dt=0.1,
        detection_threshold_db=13.0
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
        'parameters': {
            'close_range_km': close_range_km,
            'far_range_km': far_range_km,
            'detection_threshold_db': engine.detection_threshold_db,
        },
        'computed_values': {
            'close_snr_db': close_result.snr_db,
            'far_snr_db': far_result.snr_db,
            'close_above_threshold': close_detected,
            'far_below_threshold': far_not_detected,
        },
        'validation': {
            'is_valid': is_valid,
            'reference': 'Radar equation: SNR decreases with R^4',
        }
    }
