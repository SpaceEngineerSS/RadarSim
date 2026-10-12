"""
Scenario Loader

Parses RadarSim scenario files (YAML or JSON) into typed configuration objects
and builds a fully configured ``SimulationEngine``.

Every key a scenario may contain is listed in ``src.io.scenario_schema``; keys
outside the schema are reported in ``ScenarioLoader.warnings`` (with a "did you
mean" hint) instead of being dropped silently. Supported elements:

    - Radar: waveform, receiver, antenna, scan pattern (circular/sector/stare)
    - Targets: kinematics (CV, CA, coordinated turn), Swerling model, jammers,
      deployment time
    - Expendables: chaff clouds and decoys deployed during the run
    - Environment: atmosphere, rain, sea/land clutter, clutter sectors, terrain
    - Processing: Pfa, pulse integration, CFAR, MTI, monopulse, frequency agility

Coordinates are ENU (x East, y North, z Up) in metres; azimuths run clockwise
from North.

Usage:
    loader = ScenarioLoader("scenarios/f16_vs_sa6.yaml")
    engine = loader.create_simulation_engine()
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from src.io.scenario_schema import validate
from src.physics.constants import BOLTZMANN_CONSTANT

# Jammer ERP used when a scenario enables ECM without specifying power; matches the
# Target default so loader-built and directly-built targets behave the same.
DEFAULT_JAMMER_ERP_WATTS = 1000.0

# Chaff cloud defaults: blooming within about a second, dispersing over minutes,
# falling slowly. Reference: Schleher (1999), Sec. 6.4.
DEFAULT_CHAFF_BLOOM_TIME_S = 1.0
DEFAULT_CHAFF_DECAY_TIME_S = 120.0
DEFAULT_CHAFF_FALL_RATE_MPS = 1.0

ECM_TYPE_ALIASES = {
    "noise": "noise_barrage",
    "noise_jamming": "noise_barrage",
    "barrage": "noise_barrage",
    "noise_barrage": "noise_barrage",
    "spot": "noise_spot",
    "noise_spot": "noise_spot",
    "drfm": "drfm",
    "drfm_repeater": "drfm",
}

SCAN_TYPES = ("circular", "sector", "stare")
CFAR_TYPES = ("CA", "GO", "SO", "OS")
TERRAIN_PRESETS = ("flat", "mountainous", "ambush")


def _vector(mapping: Optional[Dict[str, Any]], keys) -> np.ndarray:
    mapping = mapping or {}
    return np.array([float(mapping.get(key, 0.0)) for key in keys])


def _position(mapping) -> np.ndarray:
    return _vector(mapping, ("x_m", "y_m", "z_m"))


def _velocity(mapping) -> np.ndarray:
    return _vector(mapping, ("vx_mps", "vy_mps", "vz_mps"))


def _propagate(target: "TargetConfig", time_s: float):
    """Position and velocity of a scenario target at ``time_s`` (exact kinematics)."""
    from src.simulation.objects import (
        _update_kinematics_ca,
        _update_kinematics_ct,
        _update_kinematics_cv,
    )

    position = np.array(target.position, dtype=np.float64)
    velocity = np.array(target.velocity, dtype=np.float64)
    if time_s <= 0.0:
        return position, velocity
    if target.turn_rate_deg_s != 0.0:
        return _update_kinematics_ct(
            position, velocity, np.radians(target.turn_rate_deg_s), time_s
        )
    if np.any(target.acceleration != 0.0):
        position, velocity, _ = _update_kinematics_ca(
            position, velocity, np.array(target.acceleration, dtype=np.float64), time_s
        )
        return position, velocity
    return _update_kinematics_cv(position, velocity, time_s)


@dataclass
class RadarConfig:
    """Radar configuration from scenario file."""

    name: str
    frequency_hz: float
    power_watts: float
    antenna_gain_db: float
    beamwidth_az_deg: float
    beamwidth_el_deg: float
    prf_hz: float
    pulse_width_s: float
    noise_figure_db: float
    receiver_bandwidth_hz: float
    system_temperature_k: float
    receiver_full_scale_dbm: float
    system_losses_db: float
    polarization_tilt_deg: float
    position: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    radar_type: str = ""
    height_above_terrain_m: Optional[float] = None
    sidelobe_db: float = -30.0
    scan_type: str = "circular"
    scan_rate_rpm: float = 6.0
    sector_az_min_deg: float = -60.0
    sector_az_max_deg: float = 60.0
    scan_speed_deg_s: Optional[float] = None
    stare_azimuth_deg: float = 0.0

    def __post_init__(self) -> None:
        positive = {
            "frequency_hz": self.frequency_hz,
            "power_watts": self.power_watts,
            "antenna_gain_db": self.antenna_gain_db,
            "beamwidth_az_deg": self.beamwidth_az_deg,
            "beamwidth_el_deg": self.beamwidth_el_deg,
            "prf_hz": self.prf_hz,
            "pulse_width_s": self.pulse_width_s,
            "receiver_bandwidth_hz": self.receiver_bandwidth_hz,
            "system_temperature_k": self.system_temperature_k,
        }
        invalid = [name for name, value in positive.items() if value <= 0.0]
        if invalid:
            raise ValueError(f"Radar parameters must be positive: {', '.join(invalid)}")
        if self.noise_figure_db < 0.0 or self.system_losses_db < 0.0:
            raise ValueError("Noise figure and system losses cannot be negative")
        if self.scan_type not in SCAN_TYPES:
            raise ValueError(f"radar.antenna.scan_type must be one of {SCAN_TYPES}")
        if self.sidelobe_db >= 0.0:
            raise ValueError("radar.antenna.sidelobe_db must be negative")


@dataclass
class TargetConfig:
    """Target configuration from scenario file."""

    name: str
    target_type: str
    rcs_m2: float
    swerling_model: int
    position: np.ndarray
    velocity: np.ndarray
    affiliation: str = "unknown"
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    turn_rate_deg_s: float = 0.0
    deploy_time_s: float = 0.0
    rcs_bloom_time_s: float = 0.0
    rcs_decay_time_s: float = 0.0
    has_ecm: bool = False
    ecm_type: str = "noise_barrage"
    ecm_power_watts: float = DEFAULT_JAMMER_ERP_WATTS
    ecm_bandwidth_hz: float = 100e6
    drfm_gain_over_skin_db: float = 10.0
    drfm_capture_dwell_s: float = 2.0
    drfm_pull_rate_mps: float = 50.0
    drfm_max_pull_m: float = 2000.0
    drfm_mode: str = "rgpo"
    drfm_vgpo_rate_hz_per_s: float = 50.0
    drfm_max_doppler_pull_hz: float = 500.0
    drfm_inherent_delay_s: float = 0.0

    def __post_init__(self) -> None:
        if self.swerling_model not in (0, 1, 2, 3, 4):
            raise ValueError(f"Target '{self.name}': swerling_model must be 0-4")
        if self.rcs_m2 <= 0.0:
            raise ValueError(f"Target '{self.name}': rcs_m2 must be positive")
        if self.deploy_time_s < 0.0:
            raise ValueError(f"Target '{self.name}': deploy_time_s cannot be negative")


@dataclass
class ClutterSectorConfig:
    """Surface region with its own terrain class."""

    name: str
    azimuth_start_deg: float
    azimuth_end_deg: float
    range_start_km: float
    range_end_km: float
    terrain_type: str
    sigma0_adjustment_db: float = 0.0


@dataclass
class TerrainConfigSpec:
    """Digital terrain for line-of-sight masking."""

    preset: Optional[str] = None
    seed: int = 12345
    scale_m: float = 50000.0
    max_height_m: float = 2000.0
    octaves: int = 4
    persistence: float = 0.5
    lacunarity: float = 2.0
    peaks: List[tuple] = field(default_factory=list)


@dataclass
class EnvironmentConfig:
    """Environment configuration from scenario file."""

    temperature_c: float = 15.0
    pressure_hpa: float = 1013.25
    water_vapor_gpm3: float = 7.5
    terrain_type: str = "rural"
    sea_state: int = 0
    rain_rate_mm_hr: float = 0.0
    rain_height_m: Optional[float] = None
    ground_model: str = "gamma"
    land_gamma_db: Optional[float] = None
    ground_relative_permittivity_real: float = 8.0
    ground_relative_permittivity_loss: float = 0.8
    ground_rms_height_m: float = 0.01
    clutter_velocity_spread_mps: Optional[float] = None
    clutter_sectors: List[ClutterSectorConfig] = field(default_factory=list)
    terrain: Optional[TerrainConfigSpec] = None

    def __post_init__(self) -> None:
        if self.pressure_hpa <= 0.0:
            raise ValueError("pressure_hpa must be greater than zero")
        if self.water_vapor_gpm3 < 0.0 or self.rain_rate_mm_hr < 0.0:
            raise ValueError("Water-vapor density and rain rate cannot be negative")
        if not 0 <= self.sea_state <= 6:
            raise ValueError("sea_state must be between 0 and 6")
        if self.ground_model not in {"gamma", "oh1992"}:
            raise ValueError("ground_model must be 'gamma' or 'oh1992'")
        if self.ground_relative_permittivity_real <= 1.0:
            raise ValueError("ground_relative_permittivity_real must be greater than one")
        if self.ground_relative_permittivity_loss < 0.0:
            raise ValueError("ground_relative_permittivity_loss cannot be negative")
        if self.ground_rms_height_m <= 0.0:
            raise ValueError("ground_rms_height_m must be greater than zero")


@dataclass
class ScenarioConfig:
    """Complete scenario configuration."""

    name: str
    description: str
    duration_s: float
    update_rate_hz: float
    radar: RadarConfig
    targets: List[TargetConfig]
    environment: EnvironmentConfig
    enable_atmospheric_loss: bool = True
    enable_clutter: bool = False
    probability_false_alarm: float = 1e-6
    pulses_integrated: int = 1
    seed: Optional[int] = None
    range_noise_std_m: float = 50.0
    angle_noise_std_rad: float = 0.001
    scan_gated_detection: bool = True
    cfar_type: Optional[str] = None
    cfar_guard_cells: int = 2
    cfar_reference_cells: int = 16
    mti_enabled: bool = False
    mti_threshold_mps: float = 15.0
    monopulse_enabled: bool = False
    frequency_agility_enabled: bool = False
    tracker_process_noise_mps2: float = 10.0
    tracker_confirm_hits: int = 3
    tracker_max_misses: int = 3
    required_radar_preset: Optional[str] = None

    def __post_init__(self) -> None:
        if self.duration_s <= 0.0 or self.update_rate_hz <= 0.0:
            raise ValueError("Scenario duration and update rate must be positive")
        if not 0.0 < self.probability_false_alarm < 1.0:
            raise ValueError("probability_false_alarm must be between 0 and 1")
        if self.pulses_integrated < 1:
            raise ValueError("pulses_integrated must be at least 1")
        if self.cfar_type is not None and self.cfar_type not in CFAR_TYPES:
            raise ValueError(f"simulation.cfar.type must be one of {CFAR_TYPES}")


# Backwards-compatible name (the headless runner has its own SimulationConfig).
SimulationConfig = ScenarioConfig


class ScenarioLoader:
    """
    Loads a scenario file and builds the simulation engine.

    Attributes:
        data: Raw parsed file contents
        warnings: Messages about keys the loader does not understand
    """

    def __init__(self, filepath: Optional[str] = None):
        self.filepath = filepath
        self.data: Dict[str, Any] = {}
        self.warnings: List[str] = []
        self._config: Optional[ScenarioConfig] = None

        if filepath:
            self.load(filepath)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScenarioLoader":
        """Build a loader from an in-memory scenario mapping."""
        loader = cls()
        loader.data = data
        loader.warnings = validate(data)
        loader._config = loader._parse_config()
        return loader

    def load(self, filepath: str) -> bool:
        """
        Load and validate a scenario file.

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file is empty, malformed, or has invalid values
            yaml.YAMLError: If the file is not valid YAML/JSON
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Scenario file not found: {filepath}")

        self.filepath = filepath
        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)  # JSON is a subset of YAML 1.2
        if data is None:
            raise ValueError(f"Scenario file is empty: {filepath}")
        if not isinstance(data, dict):
            raise ValueError("A scenario file must contain a mapping at the top level")

        self.data = data
        self.warnings = validate(data)
        self._config = self._parse_config()
        return True

    # ------------------------------------------------------------------ parsing

    def _section(self, *keys: str) -> Dict[str, Any]:
        node: Any = self.data
        for key in keys:
            node = (node or {}).get(key) if isinstance(node, dict) else None
        return node if isinstance(node, dict) else {}

    def _parse_config(self) -> ScenarioConfig:
        scenario = self._section("scenario")
        sim = self._section("simulation")
        cfar = sim.get("cfar") or {}
        mti = sim.get("mti") or {}
        tracker = sim.get("tracker") or {}
        primaries = self._parse_targets()
        targets = primaries + self._parse_expendables(primaries)

        seed = sim.get("seed", scenario.get("seed"))
        cfar_type = cfar.get("type")
        return ScenarioConfig(
            name=scenario.get("name", "Unnamed Scenario"),
            description=scenario.get("description", ""),
            duration_s=float(scenario.get("duration_seconds", 180)),
            update_rate_hz=float(scenario.get("update_rate_hz", 30)),
            radar=self._parse_radar(),
            targets=targets,
            environment=self._parse_environment(),
            enable_atmospheric_loss=bool(sim.get("enable_atmospheric_loss", True)),
            enable_clutter=bool(sim.get("enable_clutter", False)),
            probability_false_alarm=float(sim.get("pfa", 1e-6)),
            pulses_integrated=int(sim.get("pulses_integrated", 1)),
            seed=int(seed) if seed is not None else None,
            range_noise_std_m=float(sim.get("range_noise_std_m", 50.0)),
            angle_noise_std_rad=float(sim.get("angle_noise_std_rad", 0.001)),
            scan_gated_detection=bool(sim.get("scan_gated_detection", True)),
            cfar_type=str(cfar_type).upper() if cfar_type is not None else None,
            cfar_guard_cells=int(cfar.get("guard_cells", 2)),
            cfar_reference_cells=int(cfar.get("reference_cells", 16)),
            mti_enabled=bool(mti.get("enabled", False)),
            mti_threshold_mps=float(mti.get("threshold_mps", 15.0)),
            monopulse_enabled=bool((sim.get("monopulse") or {}).get("enabled", False)),
            frequency_agility_enabled=bool(
                (sim.get("eccm") or {}).get("frequency_agility", False)
            ),
            tracker_process_noise_mps2=float(tracker.get("process_noise_mps2", 10.0)),
            tracker_confirm_hits=int(tracker.get("confirm_hits", 3)),
            tracker_max_misses=int(tracker.get("max_misses", 3)),
            required_radar_preset=scenario.get("required_radar_preset"),
        )

    def _parse_radar(self) -> RadarConfig:
        radar = self._section("radar")
        antenna = radar.get("antenna") or {}
        receiver = radar.get("receiver") or {}
        pulse_width_s = float(radar.get("pulse_width_s", 1e-6))
        bandwidth_hz = float(receiver.get("bandwidth_hz", 1.0 / pulse_width_s))
        temperature_k = float(receiver.get("system_temperature_k", 290.0))
        noise_figure_db = float(receiver.get("noise_figure_db", 4.0))

        if "full_scale_dbm" in receiver:
            full_scale_dbm = float(receiver["full_scale_dbm"])
        elif "dynamic_range_db" in receiver:
            # Full scale = thermal noise floor k*T*B*F plus the dynamic range.
            noise_dbm = 10.0 * np.log10(
                BOLTZMANN_CONSTANT * temperature_k * bandwidth_hz * 1e3
            ) + noise_figure_db
            full_scale_dbm = noise_dbm + float(receiver["dynamic_range_db"])
        else:
            full_scale_dbm = -20.0

        if "scan_rate_rpm" in antenna:
            scan_rate_rpm = float(antenna["scan_rate_rpm"])
        elif "scan_rate_deg_s" in antenna:
            scan_rate_rpm = float(antenna["scan_rate_deg_s"]) / 6.0
        else:
            scan_rate_rpm = float(radar.get("scan_rate_rpm", 6.0))
        scan_speed = antenna.get("scan_speed_deg_s")

        return RadarConfig(
            name=radar.get("name", "Radar"),
            radar_type=str(radar.get("type", "")),
            frequency_hz=float(radar.get("frequency_hz", 10e9)),
            power_watts=float(radar.get("power_watts", 100e3)),
            antenna_gain_db=float(antenna.get("gain_db", 30)),
            beamwidth_az_deg=float(antenna.get("beamwidth_az_deg", 2.0)),
            beamwidth_el_deg=float(antenna.get("beamwidth_el_deg", 3.0)),
            prf_hz=float(radar.get("prf_hz", 1000)),
            pulse_width_s=pulse_width_s,
            noise_figure_db=noise_figure_db,
            receiver_bandwidth_hz=bandwidth_hz,
            system_temperature_k=temperature_k,
            receiver_full_scale_dbm=full_scale_dbm,
            system_losses_db=float(radar.get("system_losses_db", 4.0)),
            polarization_tilt_deg=float(antenna.get("polarization_tilt_deg", 0.0)),
            position=_position(radar.get("position")),
            velocity=_velocity(radar.get("velocity")),
            height_above_terrain_m=(
                float(radar["height_above_terrain_m"])
                if "height_above_terrain_m" in radar
                else None
            ),
            sidelobe_db=float(antenna.get("sidelobe_db", -30.0)),
            scan_type=str(antenna.get("scan_type", "circular")).lower(),
            scan_rate_rpm=scan_rate_rpm,
            sector_az_min_deg=float(antenna.get("sector_az_min_deg", -60.0)),
            sector_az_max_deg=float(antenna.get("sector_az_max_deg", 60.0)),
            scan_speed_deg_s=float(scan_speed) if scan_speed is not None else None,
            stare_azimuth_deg=float(antenna.get("stare_azimuth_deg", 0.0)),
        )

    def _parse_targets(self) -> List[TargetConfig]:
        targets = []
        for idx, t in enumerate(self.data.get("targets") or []):
            drfm = t.get("drfm") or {}
            name = t.get("name", f"Target_{idx + 1}")
            ecm_type = str(t.get("ecm_type", "noise_barrage")).lower()
            if ecm_type not in ECM_TYPE_ALIASES:
                raise ValueError(
                    f"Target '{name}': unknown ecm_type '{ecm_type}' "
                    f"(use one of {sorted(set(ECM_TYPE_ALIASES.values()))})"
                )
            targets.append(
                TargetConfig(
                    name=name,
                    target_type=t.get("type", "aircraft"),
                    affiliation=str(t.get("affiliation", "unknown")).lower(),
                    rcs_m2=float(t.get("rcs_m2", 1.0)),
                    swerling_model=int(t.get("swerling_model", 1)),
                    position=_position(t.get("initial_position")),
                    velocity=_velocity(t.get("velocity")),
                    acceleration=_vector(
                        t.get("acceleration"), ("ax_mps2", "ay_mps2", "az_mps2")
                    ),
                    turn_rate_deg_s=float(t.get("turn_rate_deg_s", 0.0)),
                    deploy_time_s=float(t.get("deploy_time_s", 0.0)),
                    has_ecm=bool(t.get("has_ecm", False)),
                    ecm_type=ECM_TYPE_ALIASES[ecm_type],
                    ecm_power_watts=float(t.get("ecm_power_watts", DEFAULT_JAMMER_ERP_WATTS)),
                    ecm_bandwidth_hz=float(t.get("ecm_bandwidth_hz", 100e6)),
                    drfm_gain_over_skin_db=float(drfm.get("gain_over_skin_db", 10.0)),
                    drfm_capture_dwell_s=float(drfm.get("capture_dwell_s", 2.0)),
                    drfm_pull_rate_mps=float(drfm.get("pull_rate_mps", 50.0)),
                    drfm_max_pull_m=float(drfm.get("max_pull_m", 2000.0)),
                    drfm_mode=str(drfm.get("mode", "rgpo")).lower(),
                    drfm_vgpo_rate_hz_per_s=float(drfm.get("vgpo_rate_hz_per_s", 50.0)),
                    drfm_max_doppler_pull_hz=float(drfm.get("max_doppler_pull_hz", 500.0)),
                    drfm_inherent_delay_s=float(drfm.get("inherent_delay_s", 0.0)),
                )
            )
        return targets

    def _parse_expendables(self, primaries: List[TargetConfig]) -> List[TargetConfig]:
        """
        Chaff clouds and decoys become radar objects deployed mid-run.

        With ``deployed_by`` the object is released at that target's position
        at ``deploy_time_s`` (a decoy without its own velocity inherits the
        carrier's); otherwise ``position``/``initial_position`` is used.
        """
        by_name = {t.name: t for t in primaries}

        def release_state(item: Dict[str, Any], position_key: str):
            deploy_time = float(item.get("deploy_time_s", 0.0))
            carrier_name = item.get("deployed_by")
            if carrier_name is None:
                return _position(item.get(position_key)), None, deploy_time
            if carrier_name not in by_name:
                raise ValueError(f"deployed_by: no target named '{carrier_name}'")
            position, velocity = _propagate(by_name[carrier_name], deploy_time)
            return position, velocity, deploy_time

        ecm = self._section("ecm")
        objects = []
        for idx, chaff in enumerate(ecm.get("chaff_clouds") or []):
            velocity = chaff.get("velocity")
            position, _, deploy_time = release_state(chaff, "position")
            objects.append(
                TargetConfig(
                    name=chaff.get("name", f"Chaff_{idx + 1}"),
                    target_type="chaff",
                    rcs_m2=float(chaff.get("rcs_m2", 50.0)),
                    # Many independently moving dipoles decorrelate pulse to pulse.
                    swerling_model=2,
                    position=position,
                    # Chaff loses the carrier's speed within a second and then
                    # drifts with the wind while falling.
                    velocity=(
                        _velocity(velocity)
                        if velocity is not None
                        else np.array([0.0, 0.0, -DEFAULT_CHAFF_FALL_RATE_MPS])
                    ),
                    deploy_time_s=deploy_time,
                    rcs_bloom_time_s=float(
                        chaff.get("bloom_time_s", DEFAULT_CHAFF_BLOOM_TIME_S)
                    ),
                    rcs_decay_time_s=float(
                        chaff.get("decay_time_s", DEFAULT_CHAFF_DECAY_TIME_S)
                    ),
                )
            )
        for idx, decoy in enumerate(ecm.get("decoys") or []):
            position, carrier_velocity, deploy_time = release_state(
                decoy, "initial_position"
            )
            velocity = decoy.get("velocity")
            objects.append(
                TargetConfig(
                    name=decoy.get("name", f"Decoy_{idx + 1}"),
                    target_type="decoy",
                    rcs_m2=float(decoy.get("rcs_m2", 5.0)),
                    swerling_model=int(decoy.get("swerling_model", 1)),
                    position=position,
                    velocity=(
                        _velocity(velocity)
                        if velocity is not None or carrier_velocity is None
                        else carrier_velocity
                    ),
                    deploy_time_s=deploy_time,
                )
            )
        return objects

    def _parse_environment(self) -> EnvironmentConfig:
        env = self._section("environment")
        clutter = env.get("clutter") or {}
        ground = env.get("ground_surface") or {}
        gamma_value = ground.get("gamma_db")
        rain_height = env.get("rain_height_m")
        spread = clutter.get("velocity_spread_mps")

        sectors = [
            ClutterSectorConfig(
                name=str(sector.get("name", f"Sector_{idx + 1}")),
                azimuth_start_deg=float(sector["azimuth_start_deg"]),
                azimuth_end_deg=float(sector["azimuth_end_deg"]),
                range_start_km=float(sector.get("range_start_km", 0.0)),
                range_end_km=float(sector.get("range_end_km", 1e6)),
                terrain_type=str(sector.get("terrain_type", "rural")),
                sigma0_adjustment_db=float(sector.get("sigma0_adjustment_db", 0.0)),
            )
            for idx, sector in enumerate(clutter.get("sectors") or [])
        ]

        terrain = None
        terrain_data = env.get("terrain")
        if terrain_data:
            preset = terrain_data.get("preset")
            if preset is not None and preset not in TERRAIN_PRESETS:
                raise ValueError(f"environment.terrain.preset must be one of {TERRAIN_PRESETS}")
            terrain = TerrainConfigSpec(
                preset=preset,
                seed=int(terrain_data.get("seed", 12345)),
                scale_m=float(terrain_data.get("scale_m", 50000.0)),
                max_height_m=float(terrain_data.get("max_height_m", 2000.0)),
                octaves=int(terrain_data.get("octaves", 4)),
                persistence=float(terrain_data.get("persistence", 0.5)),
                lacunarity=float(terrain_data.get("lacunarity", 2.0)),
                peaks=[
                    (float(p["x_m"]), float(p["y_m"]), float(p["height_m"]))
                    for p in terrain_data.get("peaks") or []
                ],
            )

        return EnvironmentConfig(
            temperature_c=float(env.get("temperature_c", 15.0)),
            pressure_hpa=float(env.get("pressure_hpa", 1013.25)),
            water_vapor_gpm3=float(env.get("water_vapor_gpm3", 7.5)),
            terrain_type=str(env.get("terrain_type", clutter.get("terrain_type", "rural"))),
            sea_state=int(env.get("sea_state", clutter.get("sea_state", 0))),
            rain_rate_mm_hr=float(env.get("rain_rate_mm_hr", 0.0)),
            rain_height_m=float(rain_height) if rain_height is not None else None,
            ground_model=str(ground.get("model", "gamma")).lower(),
            land_gamma_db=float(gamma_value) if gamma_value is not None else None,
            ground_relative_permittivity_real=float(
                ground.get("relative_permittivity_real", 8.0)
            ),
            ground_relative_permittivity_loss=float(
                ground.get("relative_permittivity_loss", 0.8)
            ),
            ground_rms_height_m=float(ground.get("rms_height_m", 0.01)),
            clutter_velocity_spread_mps=float(spread) if spread is not None else None,
            clutter_sectors=sectors,
            terrain=terrain,
        )

    # --------------------------------------------------------------- accessors

    def get_config(self) -> Optional[ScenarioConfig]:
        """Parsed scenario configuration (None until a file is loaded)."""
        return self._config

    def get_scenario_name(self) -> str:
        return self._config.name if self._config else "Unknown"

    def get_required_preset(self) -> Optional[str]:
        """Radar preset label the scenario was written for (display only)."""
        return self._config.required_radar_preset if self._config else None

    # ---------------------------------------------------------------- building

    def create_simulation_engine(self):
        """
        Build a ``SimulationEngine`` with every configured feature applied.

        Raises:
            ValueError: If no scenario is loaded
        """
        if not self._config:
            raise ValueError("No scenario loaded. Call load() first.")

        from src.physics.rcs import SwerlingModel
        from src.physics.terrain import (
            TerrainConfig,
            TerrainMap,
            create_ambush_terrain,
            create_flat_terrain,
            create_mountainous_terrain,
        )
        from src.simulation.engine import (
            DEFAULT_RAIN_HEIGHT_M,
            CFARConfig,
            ClutterSector,
            SimulationEngine,
        )
        from src.simulation.objects import MotionModel, Radar, ScanType, Target

        cfg = self._config
        env = cfg.environment
        terrain = None
        if env.terrain is not None:
            spec = env.terrain
            if spec.preset == "flat":
                terrain = create_flat_terrain()
            elif spec.preset == "mountainous":
                terrain = create_mountainous_terrain(seed=spec.seed)
            elif spec.preset == "ambush":
                terrain = create_ambush_terrain()
            else:
                terrain = TerrainMap(
                    TerrainConfig(
                        seed=spec.seed,
                        scale=spec.scale_m,
                        max_height=spec.max_height_m,
                        octaves=spec.octaves,
                        persistence=spec.persistence,
                        lacunarity=spec.lacunarity,
                        mountain_peaks=list(spec.peaks),
                    )
                )

        rc = cfg.radar
        radar_position = np.array(rc.position, dtype=np.float64)
        if rc.height_above_terrain_m is not None:
            # Heights are above mean sea level; mount the antenna on the terrain.
            ground = terrain.get_elevation(radar_position[0], radar_position[1]) if terrain else 0.0
            radar_position[2] = ground + rc.height_above_terrain_m
        radar = Radar(
            radar_id=rc.name,
            position=radar_position,
            velocity=rc.velocity,
            frequency_hz=rc.frequency_hz,
            power_watts=rc.power_watts,
            antenna_gain_db=rc.antenna_gain_db,
            beamwidth_deg=rc.beamwidth_az_deg,
            beamwidth_el_deg=rc.beamwidth_el_deg,
            scan_rate_rpm=rc.scan_rate_rpm,
            prf_hz=rc.prf_hz,
            pulse_width_s=rc.pulse_width_s,
            receiver_bandwidth_hz=rc.receiver_bandwidth_hz,
            noise_figure_db=rc.noise_figure_db,
            system_temperature_k=rc.system_temperature_k,
            system_losses_db=rc.system_losses_db,
            polarization_tilt_deg=rc.polarization_tilt_deg,
            sidelobe_level_db=rc.sidelobe_db,
            scan_type=ScanType(rc.scan_type),
            sector_az_min_deg=rc.sector_az_min_deg,
            sector_az_max_deg=rc.sector_az_max_deg,
            scan_speed_deg_s=rc.scan_speed_deg_s,
            stare_azimuth_deg=rc.stare_azimuth_deg,
        )

        targets = []
        for idx, tc in enumerate(cfg.targets):
            if tc.turn_rate_deg_s != 0.0:
                motion = MotionModel.COORDINATED_TURN
            elif np.any(tc.acceleration != 0.0):
                motion = MotionModel.CONSTANT_ACCELERATION
            elif np.allclose(tc.velocity, 0.0):
                motion = MotionModel.STATIC
            else:
                motion = MotionModel.CONSTANT_VELOCITY
            targets.append(
                Target(
                    target_id=idx + 1,
                    name=tc.name,
                    position=tc.position,
                    velocity=tc.velocity,
                    acceleration=tc.acceleration,
                    rcs_m2=tc.rcs_m2,
                    target_type=tc.target_type,
                    affiliation=tc.affiliation,
                    swerling_model=SwerlingModel(tc.swerling_model),
                    motion_model=motion,
                    turn_rate_rad_s=np.radians(tc.turn_rate_deg_s),
                    active_from_s=tc.deploy_time_s,
                    rcs_bloom_time_s=tc.rcs_bloom_time_s,
                    rcs_decay_time_s=tc.rcs_decay_time_s,
                    has_jammer=tc.has_ecm,
                    jammer_power_watts=tc.ecm_power_watts,
                    jammer_bandwidth_hz=tc.ecm_bandwidth_hz,
                    ecm_type=tc.ecm_type,
                    drfm_gain_over_skin_db=tc.drfm_gain_over_skin_db,
                    drfm_capture_dwell_s=tc.drfm_capture_dwell_s,
                    drfm_pull_rate_mps=tc.drfm_pull_rate_mps,
                    drfm_max_pull_m=tc.drfm_max_pull_m,
                    drfm_mode=tc.drfm_mode,
                    drfm_vgpo_rate_hz_per_s=tc.drfm_vgpo_rate_hz_per_s,
                    drfm_max_doppler_pull_hz=tc.drfm_max_doppler_pull_hz,
                    drfm_inherent_delay_s=tc.drfm_inherent_delay_s,
                )
            )

        engine = SimulationEngine(
            radar=radar,
            targets=targets,
            dt=1.0 / cfg.update_rate_hz,
            enable_atmospheric=cfg.enable_atmospheric_loss,
            range_noise_std_m=cfg.range_noise_std_m,
            angle_noise_std_rad=cfg.angle_noise_std_rad,
            terrain=terrain,
            probability_false_alarm=cfg.probability_false_alarm,
            pulses_integrated=cfg.pulses_integrated,
            atmospheric_temperature_c=env.temperature_c,
            atmospheric_pressure_hpa=env.pressure_hpa,
            water_vapor_density_g_m3=env.water_vapor_gpm3,
            enable_clutter=cfg.enable_clutter,
            terrain_type=env.terrain_type,
            sea_state=env.sea_state,
            rain_rate_mm_hr=env.rain_rate_mm_hr,
            receiver_full_scale_dbm=rc.receiver_full_scale_dbm,
            ground_model=env.ground_model,
            land_gamma_db=env.land_gamma_db,
            ground_relative_permittivity=complex(
                env.ground_relative_permittivity_real,
                -env.ground_relative_permittivity_loss,
            ),
            ground_rms_height_m=env.ground_rms_height_m,
            seed=cfg.seed,
            scan_gated_detection=cfg.scan_gated_detection,
            rain_height_m=env.rain_height_m or DEFAULT_RAIN_HEIGHT_M,
            clutter_velocity_spread_mps=env.clutter_velocity_spread_mps,
            clutter_sectors=[
                ClutterSector(
                    azimuth_start_deg=s.azimuth_start_deg,
                    azimuth_end_deg=s.azimuth_end_deg,
                    range_start_m=s.range_start_km * 1000.0,
                    range_end_m=s.range_end_km * 1000.0,
                    terrain_type=s.terrain_type,
                    sigma0_adjustment_db=s.sigma0_adjustment_db,
                )
                for s in env.clutter_sectors
            ],
            cfar=(
                CFARConfig(cfg.cfar_type, cfg.cfar_reference_cells, cfg.cfar_guard_cells)
                if cfg.cfar_type
                else None
            ),
        )
        engine.mti_enabled = cfg.mti_enabled
        engine.mti_threshold_mps = cfg.mti_threshold_mps
        engine.monopulse_enabled = cfg.monopulse_enabled
        engine.frequency_agility_enabled = cfg.frequency_agility_enabled
        if engine.track_manager is not None:
            engine.track_manager.kf.process_noise = cfg.tracker_process_noise_mps2
            engine.track_manager.confirm_hits = cfg.tracker_confirm_hits
            engine.track_manager.max_misses = cfg.tracker_max_misses
        if any(t.has_ecm for t in cfg.targets):
            # Scenario jammers radiate from the start, each with its own technique.
            engine.set_ecm_mode(True)
        return engine


def load_scenario(filepath: str) -> ScenarioConfig:
    """Convenience function: parse a scenario file into a ScenarioConfig."""
    return ScenarioLoader(filepath).get_config()
