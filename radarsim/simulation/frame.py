"""
Frame contract between the simulation and any display.

``build_frame`` snapshots a ``SimulationEngine`` into plain, immutable data
(floats, tuples, small numpy arrays that are never written again), so the
worker thread can hand it to the GUI without sharing mutable engine state.
Recordings and replays use the same structure.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from radarsim.physics.geometry import enu_from_polar

from .events import SimEvent

Vec3 = Tuple[float, float, float]
Vec2 = Tuple[float, float]


@dataclass(frozen=True)
class RadarFrame:
    name: str
    position: Vec3
    antenna_azimuth: float  # rad, clockwise from North
    swept_legs: Tuple[Tuple[float, float], ...]  # (start, signed sweep) rad
    scan_type: str
    sector: Vec2  # (min, max) rad, sector scan only
    beamwidth_az: float  # rad
    beamwidth_el: float  # rad
    scan_rate_rpm: float
    frequency_hz: float
    power_watts: float
    prf_hz: float
    pulse_width_s: float
    bandwidth_hz: float
    unambiguous_range_m: float
    range_resolution_m: float


@dataclass(frozen=True)
class TargetFrame:
    target_id: int
    name: str
    target_type: str
    affiliation: str
    position: Vec3
    velocity: Vec3
    speed: float
    heading: float  # rad, clockwise from North
    altitude: float
    range_m: float
    azimuth: float
    elevation: float
    radial_velocity: float
    rcs_m2: float
    snr_db: float  # last look
    pd: float  # last look
    looked_at: bool  # beam swept the target this step
    detected: bool  # detected this step
    last_look_detected: bool  # outcome of the most recent look
    last_look_time: float  # simulation time of that look (nan before the first)
    measured_position: Optional[Vec3]  # ENU plot of this step's detection
    masked: bool
    jammer_active: bool
    ecm_type: str
    jammer_loss_db: float
    clutter_loss_db: float


@dataclass(frozen=True)
class TrackFrame:
    track_id: int
    status: str
    position: Vec2
    velocity: Vec2
    heading: float
    speed: float
    hits: int
    misses: int
    covariance: Tuple[Tuple[float, float], Tuple[float, float]]
    history: Tuple[Vec2, ...]


@dataclass(frozen=True)
class FalseTargetFrame:
    false_id: int
    ecm_type: str
    position: Vec3
    velocity: Vec3


@dataclass(frozen=True)
class RangeDopplerFrame:
    data_db: np.ndarray  # (doppler, range)
    range_axis_m: np.ndarray
    velocity_axis_mps: np.ndarray
    blind_speeds_mps: Tuple[float, ...]


@dataclass(frozen=True)
class FrameState:
    time: float
    dt: float
    radar: RadarFrame
    targets: Tuple[TargetFrame, ...]
    tracks: Tuple[TrackFrame, ...]
    false_targets: Tuple[FalseTargetFrame, ...]
    events: Tuple[SimEvent, ...]
    rd_map: Optional[RangeDopplerFrame] = None
    looks: int = 0
    detections: int = 0
    settings: dict = field(default_factory=dict)

    def target(self, target_id: int) -> Optional[TargetFrame]:
        return next((t for t in self.targets if t.target_id == target_id), None)


C_LIGHT = 299_792_458.0


def _vec(values) -> tuple:
    return tuple(float(v) for v in values)


def build_frame(engine, dt: float, include_rd_map: bool = True) -> FrameState:
    """Snapshot ``engine`` after a step (call from the thread that owns it)."""
    radar = engine.radar
    radar_frame = RadarFrame(
        name=str(radar.radar_id),
        position=_vec(radar.position),
        antenna_azimuth=float(radar.antenna_azimuth),
        swept_legs=tuple((float(a), float(b)) for a, b in radar.swept_legs),
        scan_type=radar.scan_type.value,
        sector=(float(radar.sector_az_min_rad), float(radar.sector_az_max_rad)),
        beamwidth_az=float(radar.beamwidth_rad),
        beamwidth_el=float(radar.beamwidth_el_rad),
        scan_rate_rpm=float(radar.scan_rate_rpm),
        frequency_hz=float(radar.frequency_hz),
        power_watts=float(radar.power_watts),
        prf_hz=float(radar.prf_hz),
        pulse_width_s=float(radar.pulse_width_s),
        bandwidth_hz=float(radar.receiver_bandwidth_hz),
        unambiguous_range_m=C_LIGHT / (2.0 * radar.prf_hz),
        range_resolution_m=C_LIGHT * engine.compressed_pulse_width_s / 2.0,
    )

    targets = []
    for target in engine.active_targets:
        geom = radar.calculate_target_geometry(target.position, target.velocity)
        look = engine._last_looks.get(target.target_id)
        looked = bool(look is not None and look.illuminated and look.time == engine.current_time)
        detected = bool(looked and look.is_detected)
        measured = None
        if detected:
            offset = enu_from_polar(
                look.measured_range_m, look.measured_azimuth_rad, look.measured_elevation_rad
            )
            measured = _vec(radar.position + offset)
        targets.append(
            TargetFrame(
                target_id=target.target_id,
                name=target.name,
                target_type=target.target_type,
                affiliation=target.affiliation,
                position=_vec(target.position),
                velocity=_vec(target.velocity),
                speed=float(target.speed),
                heading=float(target.state.heading),
                altitude=float(target.state.altitude),
                range_m=float(geom["range_m"]),
                azimuth=float(geom["azimuth_rad"]),
                elevation=float(geom["elevation_rad"]),
                radial_velocity=float(geom["radial_velocity_mps"]),
                rcs_m2=float(target.rcs_mean),
                snr_db=float(look.snr_db) if look is not None else float("nan"),
                pd=float(look.pd) if look is not None else 0.0,
                looked_at=looked,
                detected=detected,
                last_look_detected=bool(look.is_detected) if look is not None else False,
                last_look_time=float(look.time) if look is not None and look.illuminated else float("nan"),
                measured_position=measured,
                masked=bool(look.terrain_masked) if look is not None else False,
                jammer_active=bool(target.has_jammer and target.jammer_active and engine.ecm_active),
                ecm_type=engine.effective_ecm_type(target) if target.has_jammer else "",
                jammer_loss_db=float(look.jammer_loss_db) if look is not None else 0.0,
                clutter_loss_db=float(look.surface_clutter_loss_db) if look is not None else 0.0,
            )
        )

    tracks = []
    for track in engine.tracks:
        covariance = np.asarray(track.state.P)[:2, :2]
        tracks.append(
            TrackFrame(
                track_id=int(track.id),
                status=track.status.value,
                position=_vec(track.position),
                velocity=_vec(track.velocity),
                heading=float(track.heading_rad),
                speed=float(track.speed_mps),
                hits=int(track.hits),
                misses=int(track.misses),
                covariance=(_vec(covariance[0]), _vec(covariance[1])),
                history=tuple(_vec(p) for p in track.history[-30:]),
            )
        )

    false_targets = tuple(
        FalseTargetFrame(
            false_id=int(ft.false_id),
            ecm_type=ft.ecm_type,
            position=_vec(ft.position),
            velocity=_vec(ft.velocity),
        )
        for ft in engine.false_targets
    )

    rd_frame = None
    rdm = engine._rd_map
    if include_rd_map and engine.pulse_doppler_enabled and rdm is not None:
        rd_frame = RangeDopplerFrame(
            data_db=np.array(rdm.data_db, copy=True),
            range_axis_m=np.array(rdm.range_axis_m, copy=True),
            velocity_axis_mps=np.array(rdm.velocity_axis_mps, copy=True),
            blind_speeds_mps=tuple(float(v) for v in rdm.blind_speeds_mps),
        )

    return FrameState(
        time=float(engine.current_time),
        dt=float(dt),
        radar=radar_frame,
        targets=tuple(targets),
        tracks=tuple(tracks),
        false_targets=false_targets,
        events=tuple(engine.events),
        rd_map=rd_frame,
        looks=int(engine.log.total_opportunities),
        detections=int(engine.log.total_detections),
        settings={
            "clutter": bool(engine.clutter_enabled),
            "mti": bool(engine.mti_enabled),
            "monopulse": bool(engine.monopulse_enabled),
            "frequency_agility": bool(engine.frequency_agility_enabled),
            "pulse_doppler": bool(engine.pulse_doppler_enabled),
            "ecm": bool(engine.ecm_active),
            "cfar": engine.cfar.cfar_type if engine.cfar is not None else "",
            "cfar_loss_db": float(engine.cfar_loss_db),
            "pfa": float(engine.probability_false_alarm),
        },
    )
