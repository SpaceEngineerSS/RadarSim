"""
Scenario Loader

YAML-based scenario configuration parser for RadarSim.

Loads simulation scenarios from YAML files and creates
configured SimulationEngine instances with radar and targets.

Supported scenario elements:
    - Radar configuration (frequency, power, antenna, position)
    - Multiple targets with kinematics and RCS
    - ECM payloads (chaff, decoys, jammers)
    - Environment parameters (atmosphere, terrain)

Migration Note: Extracted from gui/main_window.py ScenarioLoader class.

Usage:
    loader = ScenarioLoader('scenarios/f16_vs_sa6.yaml')
    engine = loader.create_simulation_engine()
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import yaml


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
    system_losses_db: float
    polarization_tilt_deg: float
    position: np.ndarray

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


@dataclass
class TargetConfig:
    """Target configuration from scenario file."""

    name: str
    target_type: str
    rcs_m2: float
    swerling_model: int
    position: np.ndarray
    velocity: np.ndarray
    has_ecm: bool = False
    ecm_type: str = ""
    ecm_power_watts: float = 0.0
    ecm_bandwidth_hz: float = 100e6


@dataclass
class EnvironmentConfig:
    """Environment configuration from scenario file."""

    temperature_c: float = 15.0
    pressure_hpa: float = 1013.25
    water_vapor_gpm3: float = 7.5
    terrain_type: str = "rural"
    sea_state: int = 0
    rain_rate_mm_hr: float = 0.0

    def __post_init__(self) -> None:
        if self.pressure_hpa <= 0.0:
            raise ValueError("pressure_hpa must be greater than zero")
        if self.water_vapor_gpm3 < 0.0 or self.rain_rate_mm_hr < 0.0:
            raise ValueError("Water-vapor density and rain rate cannot be negative")
        if not 0 <= self.sea_state <= 6:
            raise ValueError("sea_state must be between 0 and 6")


@dataclass
class SimulationConfig:
    """Complete simulation configuration."""

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

    def __post_init__(self) -> None:
        if self.duration_s <= 0.0 or self.update_rate_hz <= 0.0:
            raise ValueError("Scenario duration and update rate must be positive")
        if not 0.0 < self.probability_false_alarm < 1.0:
            raise ValueError("probability_false_alarm must be between 0 and 1")
        if self.pulses_integrated < 1:
            raise ValueError("pulses_integrated must be at least 1")


class ScenarioLoader:
    """
    Loads simulation scenarios from YAML files.

    Parses scenario configuration and creates SimulationEngine instances
    with properly configured radar and target objects.

    Usage:
        loader = ScenarioLoader('scenarios/f16_vs_sa6.yaml')
        config = loader.get_config()
        engine = loader.create_simulation_engine()
    """

    def __init__(self, filepath: Optional[str] = None):
        """
        Initialize scenario loader.

        Args:
            filepath: Path to YAML scenario file (optional)
        """
        self.filepath = filepath
        self.data: Dict[str, Any] = {}
        self._config: Optional[SimulationConfig] = None

        if filepath:
            self.load(filepath)

    def load(self, filepath: str) -> bool:
        """
        Load scenario from YAML file.

        Args:
            filepath: Path to YAML scenario file

        Returns:
            True if loaded successfully, False otherwise

        Raises:
            FileNotFoundError: If file doesn't exist
            yaml.YAMLError: If YAML parsing fails
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Scenario file not found: {filepath}")

        self.filepath = filepath

        with open(filepath, "r", encoding="utf-8") as f:
            self.data = yaml.safe_load(f)

        self._config = self._parse_config()
        return True

    def _parse_config(self) -> SimulationConfig:
        """Parse loaded YAML data into SimulationConfig."""
        # Scenario metadata
        scenario = self.data.get("scenario", {})
        name = scenario.get("name", "Unnamed Scenario")
        description = scenario.get("description", "")
        duration = float(scenario.get("duration_seconds", 180))
        update_rate = float(scenario.get("update_rate_hz", 30))

        # Radar configuration
        radar_config = self._parse_radar()

        # Targets
        targets = self._parse_targets()

        # Environment
        environment = self._parse_environment()

        # Simulation parameters
        sim_params = self.data.get("simulation", {})

        return SimulationConfig(
            name=name,
            description=description,
            duration_s=duration,
            update_rate_hz=update_rate,
            radar=radar_config,
            targets=targets,
            environment=environment,
            enable_atmospheric_loss=sim_params.get("enable_atmospheric_loss", True),
            enable_clutter=sim_params.get("enable_clutter", False),
            probability_false_alarm=float(sim_params.get("pfa", 1e-6)),
            pulses_integrated=int(sim_params.get("pulses_integrated", 1)),
        )

    def _parse_radar(self) -> RadarConfig:
        """Parse radar configuration."""
        radar = self.data.get("radar", {})
        antenna = radar.get("antenna", {})
        receiver = radar.get("receiver", {})
        pos = radar.get("position", {})

        return RadarConfig(
            name=radar.get("name", "Radar"),
            frequency_hz=float(radar.get("frequency_hz", 10e9)),
            power_watts=float(radar.get("power_watts", 100e3)),
            antenna_gain_db=float(antenna.get("gain_db", 30)),
            beamwidth_az_deg=float(antenna.get("beamwidth_az_deg", 2.0)),
            beamwidth_el_deg=float(antenna.get("beamwidth_el_deg", 3.0)),
            prf_hz=float(radar.get("prf_hz", 1000)),
            pulse_width_s=float(radar.get("pulse_width_s", 1e-6)),
            noise_figure_db=float(receiver.get("noise_figure_db", 4.0)),
            receiver_bandwidth_hz=float(
                receiver.get(
                    "bandwidth_hz", 1.0 / float(radar.get("pulse_width_s", 1e-6))
                )
            ),
            system_temperature_k=float(receiver.get("system_temperature_k", 290.0)),
            system_losses_db=float(radar.get("system_losses_db", 4.0)),
            polarization_tilt_deg=float(antenna.get("polarization_tilt_deg", 0.0)),
            position=np.array(
                [
                    float(pos.get("x_m", 0)),
                    float(pos.get("y_m", 0)),
                    float(pos.get("z_m", 0)),
                ]
            ),
        )

    def _parse_targets(self) -> List[TargetConfig]:
        """Parse target configurations."""
        targets = []

        for idx, t in enumerate(self.data.get("targets", [])):
            pos = t.get("initial_position", {})
            vel = t.get("velocity", {})

            targets.append(
                TargetConfig(
                    name=t.get("name", f"Target_{idx}"),
                    target_type=t.get("type", "aircraft"),
                    rcs_m2=float(t.get("rcs_m2", 1.0)),
                    swerling_model=int(t.get("swerling_model", 1)),
                    position=np.array(
                        [
                            float(pos.get("x_m", 0)),
                            float(pos.get("y_m", 0)),
                            float(pos.get("z_m", 0)),
                        ]
                    ),
                    velocity=np.array(
                        [
                            float(vel.get("vx_mps", 0)),
                            float(vel.get("vy_mps", 0)),
                            float(vel.get("vz_mps", 0)),
                        ]
                    ),
                    has_ecm=t.get("has_ecm", False),
                    ecm_type=t.get("ecm_type", ""),
                    ecm_power_watts=float(t.get("ecm_power_watts", 0)),
                    ecm_bandwidth_hz=float(t.get("ecm_bandwidth_hz", 100e6)),
                )
            )

        return targets

    def _parse_environment(self) -> EnvironmentConfig:
        """Parse environment configuration."""
        env = self.data.get("environment", {})

        return EnvironmentConfig(
            temperature_c=float(env.get("temperature_c", 15.0)),
            pressure_hpa=float(env.get("pressure_hpa", 1013.25)),
            water_vapor_gpm3=float(env.get("water_vapor_gpm3", 7.5)),
            terrain_type=str(env.get("terrain_type", "rural")),
            sea_state=int(env.get("sea_state", 0)),
            rain_rate_mm_hr=float(env.get("rain_rate_mm_hr", 0.0)),
        )

    def get_config(self) -> Optional[SimulationConfig]:
        """
        Get parsed simulation configuration.

        Returns:
            SimulationConfig or None if not loaded
        """
        return self._config

    def get_scenario_name(self) -> str:
        """Get scenario name."""
        if self._config:
            return self._config.name
        return "Unknown"

    def get_required_preset(self) -> Optional[str]:
        """
        Get required radar preset for this scenario.

        Scenarios can specify a required_radar_preset to auto-configure
        the radar architecture when loaded.

        Returns:
            Preset name string or None if not specified
        """
        scenario = self.data.get("scenario", {})
        return scenario.get("required_radar_preset", None)

    def create_simulation_engine(self):
        """
        Create a SimulationEngine from the loaded scenario.

        Returns:
            Configured SimulationEngine instance

        Raises:
            ValueError: If no scenario is loaded
        """
        if not self._config:
            raise ValueError("No scenario loaded. Call load() first.")

        # Import here to avoid circular dependencies
        from src.physics.rcs import SwerlingModel
        from src.simulation.engine import SimulationEngine
        from src.simulation.objects import MotionModel, Radar, Target

        # Create radar
        radar = Radar(
            radar_id=self._config.radar.name,
            position=self._config.radar.position,
            frequency_hz=self._config.radar.frequency_hz,
            power_watts=self._config.radar.power_watts,
            antenna_gain_db=self._config.radar.antenna_gain_db,
            beamwidth_deg=self._config.radar.beamwidth_az_deg,
            beamwidth_el_deg=self._config.radar.beamwidth_el_deg,
            scan_rate_rpm=6.0,
            prf_hz=self._config.radar.prf_hz,
            pulse_width_s=self._config.radar.pulse_width_s,
            receiver_bandwidth_hz=self._config.radar.receiver_bandwidth_hz,
            noise_figure_db=self._config.radar.noise_figure_db,
            system_temperature_k=self._config.radar.system_temperature_k,
            system_losses_db=self._config.radar.system_losses_db,
            polarization_tilt_deg=self._config.radar.polarization_tilt_deg,
        )

        # Create targets
        targets = []
        for idx, t_config in enumerate(self._config.targets):
            # Map swerling model
            swerling_map = {
                0: SwerlingModel.SWERLING_0,
                1: SwerlingModel.SWERLING_1,
                2: SwerlingModel.SWERLING_2,
                3: SwerlingModel.SWERLING_3,
                4: SwerlingModel.SWERLING_4,
            }
            swerling = swerling_map.get(
                t_config.swerling_model, SwerlingModel.SWERLING_1
            )

            # Determine motion model from velocity
            is_static = np.allclose(t_config.velocity, 0)
            motion = MotionModel.STATIC if is_static else MotionModel.CONSTANT_VELOCITY

            target = Target(
                target_id=idx + 1,
                position=t_config.position,
                velocity=t_config.velocity,
                rcs_m2=t_config.rcs_m2,
                target_type=t_config.target_type,
                swerling_model=swerling,
                motion_model=motion,
                has_jammer=t_config.has_ecm,
                jammer_power_watts=t_config.ecm_power_watts,
                jammer_bandwidth_hz=t_config.ecm_bandwidth_hz,
                ecm_type=t_config.ecm_type or "noise_barrage",
            )
            targets.append(target)

        # Create engine
        engine = SimulationEngine(
            radar=radar,
            targets=targets,
            dt=1.0 / self._config.update_rate_hz,
            enable_atmospheric=self._config.enable_atmospheric_loss,
            probability_false_alarm=self._config.probability_false_alarm,
            pulses_integrated=self._config.pulses_integrated,
            atmospheric_temperature_c=self._config.environment.temperature_c,
            atmospheric_pressure_hpa=self._config.environment.pressure_hpa,
            water_vapor_density_g_m3=self._config.environment.water_vapor_gpm3,
            enable_clutter=self._config.enable_clutter,
            terrain_type=self._config.environment.terrain_type,
            sea_state=self._config.environment.sea_state,
            rain_rate_mm_hr=self._config.environment.rain_rate_mm_hr,
        )

        return engine


def load_scenario(filepath: str) -> SimulationConfig:
    """
    Convenience function to load a scenario file.

    Args:
        filepath: Path to YAML scenario file

    Returns:
        SimulationConfig instance
    """
    loader = ScenarioLoader(filepath)
    return loader.get_config()
