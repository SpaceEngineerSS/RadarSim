"""
Range-Doppler Map Widget

High-fidelity 2D FFT output display showing targets in range-velocity space.

Phase 26 Upgrade: Now supports DUAL modes:
    - [PULSE-DOPPLER]: Displays actual processed R-D map from coherent
      pulse train processing (CPI → Matched Filter → MTI → Doppler FFT)
    - [SYNTHETIC]: Legacy mode with Gaussian blob visualization

Features:
    - Range on X-axis [km]
    - Doppler/Velocity on Y-axis [m/s]
    - Intensity as color (aerospace colormap)
    - Detection markers at local maxima
    - Blind speed markers (v_blind = λ·PRF/2)
    - MTI rejection band indicator

Reference: Richards, "Fundamentals of Radar Signal Processing", 2nd Ed., Ch. 4
Developed by Mehmet Gümüş (github.com/SpaceEngineerSS)

Migration Note: Ported from gui/widgets/range_doppler.py with
    PyQt6 modernization, SimulationState integration, and Phase 26
    pulse-Doppler signal-level processing support.
"""

import time
from typing import Any, Dict, List

import numpy as np
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

try:
    import pyqtgraph as pg

    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False


class RangeDopplerScope(QWidget):
    """
    Range-Doppler Map (2D FFT output visualization).

    Supports two display modes:
        [PULSE-DOPPLER]: Real processed R-D map from signal chain
        [SYNTHETIC]: Gaussian blob approximation from kinematic data

    Displays targets in range-velocity space with:
        - Range on X-axis [km]
        - Doppler velocity on Y-axis [m/s]
        - Intensity as color (aerospace colormap)
        - Blind speed markers and MTI notch indicators

    Reference: Richards, "Fundamentals of Radar Signal Processing"
    """

    # Theme colors
    BACKGROUND_COLOR = "#0a1510"
    GRID_COLOR = "#1a3525"
    TEXT_COLOR = "#00dd66"

    def __init__(
        self,
        parent: QWidget = None,
        max_range_km: float = 150.0,
        max_velocity_mps: float = 500.0,
        n_range_bins: int = 128,  # Reduced from 256 for performance
        n_doppler_bins: int = 64,  # Reduced from 128 for performance
    ):
        """
        Initialize Range-Doppler map widget.

        Args:
            parent: Parent Qt widget
            max_range_km: Maximum range [km]
            max_velocity_mps: Maximum velocity [m/s]
            n_range_bins: Number of range bins
            n_doppler_bins: Number of Doppler bins
        """
        super().__init__(parent)

        self.max_range_km = max_range_km
        self.max_velocity_mps = max_velocity_mps
        self.n_range_bins = n_range_bins
        self.n_doppler_bins = n_doppler_bins

        # Data array
        self.rd_map = np.zeros((n_doppler_bins, n_range_bins))

        # Detection threshold
        self.threshold_db = 10.0

        # Noise floor for synthetic data
        self.noise_floor_db = -40.0

        # ═══ PERFORMANCE: Frame rate limiting ═══
        self._last_update_time = 0.0
        self._min_update_interval = 1.0 / 30.0  # 30 FPS max
        self._frame_count = 0

        # ═══ Phase 26: Processing mode tracking ═══
        self._pd_mode = False
        self._blind_speed_lines = []
        self._mti_notch_fill = None

        # Setup UI
        self._setup_ui()

    def _setup_ui(self):
        """Create UI components."""
        self.setMinimumHeight(250)
        self.setStyleSheet(f"background-color: {self.BACKGROUND_COLOR};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(2)

        # Header (dynamically updated for processing mode)
        self._header_label = QLabel("RANGE-DOPPLER MAP [SYNTHETIC]")
        self._header_label.setStyleSheet(
            f"""
            QLabel {{
                color: {self.TEXT_COLOR};
                font-family: 'Consolas', monospace;
                font-size: 12px;
                font-weight: bold;
                padding: 5px;
                background-color: rgba(0, 50, 25, 150);
            }}
        """
        )
        self._header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._header_label)

        if PYQTGRAPH_AVAILABLE:
            self._setup_plot(layout)
        else:
            no_pg = QLabel("PyQtGraph required for Range-Doppler display")
            no_pg.setStyleSheet("color: #ff5555; font-size: 12px;")
            layout.addWidget(no_pg)

    def _setup_plot(self, layout: QVBoxLayout):
        """Setup the pyqtgraph plot widget."""
        pg.setConfigOptions(antialias=True)

        # Create plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground(QColor(10, 21, 16))

        # Labels
        self.plot_widget.setLabel("left", "Velocity", units="m/s", color="#888888")
        self.plot_widget.setLabel("bottom", "Range", units="km", color="#888888")

        # Set axis ranges
        self.plot_widget.setXRange(0, self.max_range_km)
        self.plot_widget.setYRange(-self.max_velocity_mps, self.max_velocity_mps)

        # Style axes
        for axis in ["left", "bottom"]:
            self.plot_widget.getAxis(axis).setPen(pg.mkPen(color=self.GRID_COLOR))
            self.plot_widget.getAxis(axis).setTextPen(pg.mkPen(color="#888888"))

        # Grid
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)

        layout.addWidget(self.plot_widget)

        # Create colormap (aerospace style)
        self._create_colormap()

        # Create image item for RD map
        self._create_image_item()

        # Create detection overlay
        self._create_detection_overlay()

        # Zero velocity reference line
        self.zero_line = pg.InfiniteLine(
            pos=0,
            angle=0,
            pen=pg.mkPen(color=(100, 100, 100, 100), width=1, style=Qt.PenStyle.DashLine),
        )
        self.plot_widget.addItem(self.zero_line)

    def _create_colormap(self):
        """Create aerospace-style colormap: black -> blue -> cyan -> green -> yellow -> red."""
        colors = [
            (0.00, (0, 0, 0)),  # Black (noise floor)
            (0.10, (0, 0, 50)),  # Dark blue
            (0.25, (0, 50, 100)),  # Blue
            (0.40, (0, 100, 100)),  # Cyan
            (0.55, (0, 150, 50)),  # Green
            (0.70, (100, 150, 0)),  # Yellow-green
            (0.85, (200, 100, 0)),  # Orange
            (1.00, (255, 50, 50)),  # Red (strong returns)
        ]

        positions = [c[0] for c in colors]
        rgb_colors = [c[1] for c in colors]

        self.colormap = pg.ColorMap(positions, rgb_colors)
        self.lookup_table = self.colormap.getLookupTable(nPts=256)

    def _create_image_item(self):
        """Create the image item for the RD map."""
        self.image_item = pg.ImageItem()
        self.plot_widget.addItem(self.image_item)

        # Set colormap
        self.image_item.setLookupTable(self.lookup_table)

        # Set transform to match coordinate system
        from PyQt6.QtGui import QTransform

        transform = QTransform()
        transform.scale(
            self.max_range_km / self.n_range_bins, 2 * self.max_velocity_mps / self.n_doppler_bins
        )
        transform.translate(0, -self.n_doppler_bins / 2)
        self.image_item.setTransform(transform)

        # Initialize with zeros
        self.image_item.setImage(self.rd_map.T)

    def _create_detection_overlay(self):
        """Create overlay for detected targets."""
        self.detection_scatter = pg.ScatterPlotItem(
            size=14, symbol="+", pen=pg.mkPen(color="#ffffff", width=2), brush=pg.mkBrush(None)
        )
        self.plot_widget.addItem(self.detection_scatter)

    def _update_blind_speed_markers(self, blind_speeds_mps: List[float]):
        """
        Draw blind speed markers as horizontal dashed lines.

        v_blind = λ·PRF/2 (first blind speed)
        Reference: Richards (2005), Eq. 3.16
        """
        # Remove old markers
        for line in self._blind_speed_lines:
            self.plot_widget.removeItem(line)
        self._blind_speed_lines.clear()

        for v_blind in blind_speeds_mps:
            if v_blind > self.max_velocity_mps:
                continue
            for sign in [1, -1]:
                line = pg.InfiniteLine(
                    pos=sign * v_blind,
                    angle=0,
                    pen=pg.mkPen(color=(255, 100, 100, 120), width=1, style=Qt.PenStyle.DashLine),
                )
                self.plot_widget.addItem(line)
                self._blind_speed_lines.append(line)

    def _update_mti_notch_indicator(self, mti_order: int, prf_hz: float, wavelength_m: float):
        """
        Draw MTI rejection band near zero Doppler.

        The clutter notch width depends on MTI order:
            1st order: Δf_3dB ≈ 0.45·PRF
            2nd order: Δf_3dB ≈ 0.27·PRF

        Reference: Richards (2005), Ch. 3.4
        """
        if self._mti_notch_fill is not None:
            self.plot_widget.removeItem(self._mti_notch_fill)
            self._mti_notch_fill = None

        if mti_order == 0:
            return

        # Clutter notch width in velocity
        if mti_order == 1:
            notch_frac = 0.45  # 2-pulse
        else:
            notch_frac = 0.27  # 3-pulse

        notch_vel = wavelength_m * prf_hz * notch_frac / 2.0

        # Draw shaded region
        notch_region = pg.LinearRegionItem(
            values=(-notch_vel, notch_vel),
            orientation="horizontal",
            brush=pg.mkBrush(255, 50, 50, 30),
            pen=pg.mkPen(color=(255, 50, 50, 80), width=1),
            movable=False,
        )
        self.plot_widget.addItem(notch_region)
        self._mti_notch_fill = notch_region

    @pyqtSlot(dict)
    def update_display(self, state: Dict[str, Any]):
        """
        Update display with new simulation state.

        Routes to either pulse-Doppler (actual R-D map) or synthetic mode.
        Throttled to 30 FPS for performance.

        Args:
            state: State dictionary from SimulationWorker
        """
        if not PYQTGRAPH_AVAILABLE:
            return

        # ═══ PERFORMANCE: Frame rate limiting ═══
        current_time = time.perf_counter()
        if current_time - self._last_update_time < self._min_update_interval:
            return  # Skip this frame
        self._last_update_time = current_time
        self._frame_count += 1

        # ═══ Phase 26: Route to correct rendering mode ═══
        pd_enabled = state.get("pulse_doppler_enabled", False)
        rd_map_data = state.get("rd_map")

        if pd_enabled and rd_map_data is not None:
            self._render_pulse_doppler(state, rd_map_data)
        else:
            self._render_synthetic(state)

    def _render_pulse_doppler(self, state: Dict[str, Any], rd_map_data):
        """
        Render actual processed Range-Doppler map from pulse-Doppler pipeline.

        Displays the magnitude output (dB) of the Doppler FFT with
        blind speed markers and MTI rejection band overlay.

        Reference: Richards (2005), Ch. 4
        """
        # Update header
        if not self._pd_mode:
            self._pd_mode = True
            self._header_label.setText("RANGE-DOPPLER MAP [PULSE-DOPPLER]")
            self._header_label.setStyleSheet(
                """
                QLabel {
                    color: #00ffaa;
                    font-family: 'Consolas', monospace;
                    font-size: 12px;
                    font-weight: bold;
                    padding: 5px;
                    background-color: rgba(0, 80, 40, 180);
                }
            """
            )

        # Convert R-D map data
        rd_db = np.array(rd_map_data, dtype=np.float32)
        pd_meta = state.get("pd_metadata", {})

        # Normalize to [0, 255] for colormap
        data_min = np.percentile(rd_db, 5)  # Use percentile to avoid outlier domination
        data_max = np.max(rd_db)
        dynamic_range = max(data_max - data_min, 1.0)

        normalized = np.clip((rd_db - data_min) / dynamic_range, 0.0, 1.0)

        # Update image transform to match R-D map dimensions
        n_doppler, n_range = rd_db.shape

        if n_doppler != self.n_doppler_bins or n_range != self.n_range_bins:
            self.n_doppler_bins = n_doppler
            self.n_range_bins = n_range

        # Get actual axis limits from metadata
        if pd_meta:
            range_axis = pd_meta.get("range_axis_m", [])
            vel_axis = pd_meta.get("velocity_axis_mps", [])
            if range_axis:
                max_range_km = range_axis[-1] / 1000.0
                if max_range_km > 0:
                    self.max_range_km = max_range_km
            if vel_axis:
                max_vel = max(abs(vel_axis[0]), abs(vel_axis[-1]))
                if max_vel > 0:
                    self.max_velocity_mps = max_vel

        self._update_transform()
        self.plot_widget.setXRange(0, self.max_range_km)
        self.plot_widget.setYRange(-self.max_velocity_mps, self.max_velocity_mps)

        # Display the R-D map
        self.image_item.setImage((normalized * 255).T)

        # Update detection markers from targets
        detection_spots = []
        for target in state.get("targets", []):
            if target.get("is_detected", False):
                range_km = target.get("range_km", 0)
                radial_vel = target.get("radial_velocity_mps", 0)
                detection_spots.append({"pos": (range_km, radial_vel), "size": 14})
        self.detection_scatter.setData(detection_spots)

        # Update blind speed markers and MTI notch (only on mode change)
        if pd_meta:
            blind_speeds = pd_meta.get("blind_speeds_mps", [])
            if blind_speeds:
                self._update_blind_speed_markers(blind_speeds)

            mti_order = pd_meta.get("mti_order", 0)
            prf_hz = pd_meta.get("prf_hz", 1000.0)
            wavelength_m = pd_meta.get("wavelength_m", 0.03)
            self._update_mti_notch_indicator(mti_order, prf_hz, wavelength_m)

    def _render_synthetic(self, state: Dict[str, Any]):
        """
        Legacy synthetic rendering mode (Gaussian blobs).

        Used when pulse-Doppler processing is disabled. Creates
        synthetic Range-Doppler map by placing Gaussian blobs
        at each target's (range, radial_velocity) coordinates.
        """
        # Update header if switching from PD mode
        if self._pd_mode:
            self._pd_mode = False
            self._header_label.setText("RANGE-DOPPLER MAP [SYNTHETIC]")
            self._header_label.setStyleSheet(
                f"""
                QLabel {{
                    color: {self.TEXT_COLOR};
                    font-family: 'Consolas', monospace;
                    font-size: 12px;
                    font-weight: bold;
                    padding: 5px;
                    background-color: rgba(0, 50, 25, 150);
                }}
            """
            )
            # Remove PD overlays
            for line in self._blind_speed_lines:
                self.plot_widget.removeItem(line)
            self._blind_speed_lines.clear()
            if self._mti_notch_fill:
                self.plot_widget.removeItem(self._mti_notch_fill)
                self._mti_notch_fill = None

        # ═══ PERFORMANCE: Conditional handling based on ECM type ═══
        try:
            is_jammed = state.get("jamming_active", False)
            ecm_type = state.get("ecm_type", "noise") or "noise"
            is_noise_type = "noise" in ecm_type.lower()

            if is_jammed and is_noise_type:
                if self._frame_count % 3 != 0:
                    return
                self.rd_map = np.full(
                    (self.n_doppler_bins, self.n_range_bins),
                    0.7,
                    dtype=np.float32,
                )
                noise_mask = np.random.random((self.n_doppler_bins, self.n_range_bins)) < 0.1
                self.rd_map[noise_mask] = 0.9 + 0.1 * np.random.random(noise_mask.sum())
                self.image_item.setImage((self.rd_map * 255).T.astype(np.uint8))
                return
        except Exception:
            pass

        # Reset map to noise floor
        self.rd_map = np.random.normal(
            self.noise_floor_db, 3, (self.n_doppler_bins, self.n_range_bins)
        ).astype(np.float32)

        targets = state.get("targets", [])
        false_targets = state.get("false_targets", []) or []
        detection_spots = []

        for target in targets:
            range_km = target.get("range_km", 0)
            radial_vel = target.get("radial_velocity_mps", 0)
            snr_db = target.get("snr_db", 0)
            is_detected = target.get("is_detected", False)

            if range_km > self.max_range_km or abs(radial_vel) > self.max_velocity_mps:
                continue

            range_idx = int((range_km / self.max_range_km) * self.n_range_bins)
            vel_idx = int(
                ((radial_vel + self.max_velocity_mps) / (2 * self.max_velocity_mps))
                * self.n_doppler_bins
            )

            range_idx = max(0, min(self.n_range_bins - 1, range_idx))
            vel_idx = max(0, min(self.n_doppler_bins - 1, vel_idx))

            blob_width_r = min(5, max(2, int(snr_db / 8)))
            blob_width_v = min(5, max(2, int(snr_db / 8)))

            for di in range(-blob_width_v, blob_width_v + 1):
                for dj in range(-blob_width_r, blob_width_r + 1):
                    vi = vel_idx + di
                    ri = range_idx + dj

                    if 0 <= vi < self.n_doppler_bins and 0 <= ri < self.n_range_bins:
                        intensity = snr_db * np.exp(-0.5 * (di**2 + dj**2) / 4)
                        self.rd_map[vi, ri] = max(
                            self.rd_map[vi, ri], self.noise_floor_db + intensity
                        )

            if is_detected:
                detection_spots.append({"pos": (range_km, radial_vel), "size": 14})

        # ═══ Render FALSE TARGETS (Chaff/DRFM/Decoy) ═══
        try:
            for ft in false_targets:
                if not isinstance(ft, dict):
                    continue
                ft_pos = np.array(ft.get("position", [0, 0, 0]))
                ft_vel = np.array(ft.get("velocity", [0, 0, 0]))
                ft_rcs = ft.get("rcs_m2", 5.0)
                ft_ecm_type = ft.get("ecm_type", "chaff")

                ft_range_km = np.linalg.norm(ft_pos) / 1000.0
                ft_radial_vel = (
                    np.linalg.norm(ft_vel) * np.sign(ft_vel[0]) if len(ft_vel) > 0 else 0
                )

                if ft_range_km > self.max_range_km or abs(ft_radial_vel) > self.max_velocity_mps:
                    continue

                range_idx = int((ft_range_km / self.max_range_km) * self.n_range_bins)
                vel_idx = int(
                    ((ft_radial_vel + self.max_velocity_mps) / (2 * self.max_velocity_mps))
                    * self.n_doppler_bins
                )

                range_idx = max(0, min(self.n_range_bins - 1, range_idx))
                vel_idx = max(0, min(self.n_doppler_bins - 1, vel_idx))

                blob_width = max(2, int(ft_rcs / 3))
                snr_db = 10 + ft_rcs

                for di in range(-blob_width, blob_width + 1):
                    for dj in range(-blob_width, blob_width + 1):
                        vi = vel_idx + di
                        ri = range_idx + dj

                        if 0 <= vi < self.n_doppler_bins and 0 <= ri < self.n_range_bins:
                            intensity = snr_db * np.exp(-0.5 * (di**2 + dj**2) / 4)
                            self.rd_map[vi, ri] = max(
                                self.rd_map[vi, ri], self.noise_floor_db + intensity
                            )

                detection_spots.append(
                    {
                        "pos": (ft_range_km, ft_radial_vel),
                        "size": 10,
                        "symbol": "x" if ft_ecm_type == "chaff" else "d",
                    }
                )
        except Exception:
            pass

        # Normalize and update image
        data_min = np.min(self.rd_map)
        data_max = np.max(self.rd_map)

        if data_max > data_min:
            normalized = (self.rd_map - data_min) / (data_max - data_min)
        else:
            normalized = np.zeros_like(self.rd_map)

        self.image_item.setImage((normalized * 255).T)
        self.detection_scatter.setData(detection_spots)

    def set_max_range(self, range_km: float):
        """Set maximum display range."""
        self.max_range_km = range_km
        self.plot_widget.setXRange(0, range_km)
        self._update_transform()

    def set_max_velocity(self, velocity_mps: float):
        """Set maximum display velocity."""
        self.max_velocity_mps = velocity_mps
        self.plot_widget.setYRange(-velocity_mps, velocity_mps)
        self._update_transform()

    def _update_transform(self):
        """Update image transform after range/velocity change."""
        from PyQt6.QtGui import QTransform

        transform = QTransform()
        transform.scale(
            self.max_range_km / self.n_range_bins, 2 * self.max_velocity_mps / self.n_doppler_bins
        )
        transform.translate(0, -self.n_doppler_bins / 2)
        self.image_item.setTransform(transform)

    def clear_display(self):
        """Clear the display."""
        self.rd_map = np.zeros((self.n_doppler_bins, self.n_range_bins))
        if PYQTGRAPH_AVAILABLE:
            self.image_item.setImage(self.rd_map.T)
            self.detection_scatter.setData([])
