"""
Main Window

Application shell for RadarSim GUI.

Components:
    - PPI Scope (main radar display)
    - B-Scope (AESA radar display)
    - A-Scope (diagnostic view)
    - Control panel (frequency, power, range)
    - PlaybackPanel (replay mode)
    - AnalysisPanel (post-flight analysis)
    - Status bar

Modes:
    - LIVE: SimulationWorker runs physics loop
    - REPLAY: ReplayLoader provides historical data

Architecture: Model-View-Controller (MVC)
- GUI only visualizes, never calculates physics
- Components extracted to src/ui/panels/ for modularity
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, 
    QDockWidget, QStatusBar, QSplitter,
    QMessageBox, QFileDialog, QTabWidget
)
from PyQt6.QtCore import Qt, pyqtSlot, QSettings
from PyQt6.QtGui import QAction
from typing import Optional
from enum import Enum

# Local UI components
from .ppi_scope import PPIScope
from .b_scope import BScope
from .a_scope import AScope
from .range_doppler import RangeDopplerScope
from .rhi_scope import RHIScope
from .tactical_3d import TacticalMap3D
from .thread_manager import SimulationThread, create_demo_scenario
from .playback_panel import PlaybackPanel
from .analysis_panel import AnalysisPanel
from .analysis_window import AnalysisWindow
from .sar_viewer import SARViewer

# Extracted panel components
from .panels import ControlPanel, TargetInspector

# I/O modules
from src.io.replay_loader import ReplayLoader
from src.io.scenario_loader import ScenarioLoader
from src.simulation.recorder import FlightRecorder


class SimulationMode(Enum):
    """Simulation operating modes."""
    LIVE = "live"       # Real-time physics simulation
    REPLAY = "replay"   # Playing back recorded session


class MainWindow(QMainWindow):
    """
    Main application window for RadarSim.
    
    Features:
        - PPI Scope (central display for mechanical radars)
        - B-Scope (for AESA/electronic radars)
        - A-Scope (bottom dock)
        - Control panel (right dock)
        - Target Inspector (left dock)
        - Dark military theme
    """
    
    def __init__(self) -> None:
        super().__init__()
        
        self.setWindowTitle("RadarSim - Professional Radar Simulation")
        self.setMinimumSize(1200, 800)
        
        # Apply dark theme
        self._apply_dark_theme()
        
        # Create simulation
        self.engine = create_demo_scenario()
        self.sim_thread: Optional[SimulationThread] = None
        
        # Replay mode state
        self.mode = SimulationMode.LIVE
        self.replay_loader: Optional[ReplayLoader] = None
        
        # Recording state
        self.recorder = FlightRecorder(output_dir="output")
        self.is_recording = False
        
        # ═══ ADVANCED MODULE STATE ═══
        self.lpi_enabled = False
        self.lpi_technique = 'FHSS'  # FHSS, DSSS, Costas
        self.fusion_enabled = False
        self.fusion_method = 'kalman'  # kalman, particle, bayesian
        
        # SAR Viewer window
        self.sar_viewer = None
        
        # Analysis window (separate window for post-flight analysis)
        self.analysis_window = AnalysisWindow(self)
        
        # Setup UI
        self._setup_ui()
        self._setup_menu()
        self._setup_status_bar()
        
        # ═══ SETTINGS PERSISTENCE ═══
        self._load_settings()
        
        # Start simulation in LIVE mode
        self._start_simulation()
    
    def _apply_dark_theme(self) -> None:
        """Apply dark military theme."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0a1510;
            }
            QDockWidget {
                color: #00dd66;
                font-family: 'Consolas', monospace;
            }
            QDockWidget::title {
                background-color: #002815;
                padding: 5px;
            }
        """)
    
    def _setup_ui(self) -> None:
        """Setup the main UI layout with tabbed displays."""
        # Central widget with tabbed interface
        central = QWidget()
        self.setCentralWidget(central)
        
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # ═══ TABBED DISPLAY INTERFACE ═══
        self.display_tabs = QTabWidget()
        self.display_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #00aa55;
                background-color: #0a1510;
            }
            QTabBar::tab {
                background-color: #001a0d;
                color: #00aa55;
                padding: 8px 16px;
                border: 1px solid #00aa55;
                border-bottom: none;
                font-family: 'Consolas', monospace;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background-color: #002815;
                color: #00ff88;
            }
            QTabBar::tab:hover {
                background-color: #003322;
            }
        """)
        
        # ═══ TAB 1: SCOPE VIEW (PPI/B-Scope + A-Scope) ═══
        scope_tab = QWidget()
        scope_layout = QVBoxLayout(scope_tab)
        scope_layout.setContentsMargins(0, 0, 0, 0)
        
        # Splitter for primary scope and A-scope
        self.scope_splitter = QSplitter(Qt.Orientation.Vertical)
        
        # PPI Scope (main display for mechanical radars)
        self.ppi_scope = PPIScope(max_range_km=150.0)
        self.scope_splitter.addWidget(self.ppi_scope)
        
        # B-Scope (for AESA/electronic radars)
        self.b_scope = BScope(max_range_km=200.0, azimuth_limits=(-60.0, 60.0))
        self.b_scope.hide()  # Hidden by default (PPI is default)
        self.scope_splitter.addWidget(self.b_scope)
        
        # Track active display type
        self.active_display = "ppi"  # "ppi" or "b_scope"
        
        # A-Scope (diagnostic)
        self.a_scope = AScope(max_range_km=150.0)
        self.scope_splitter.addWidget(self.a_scope)
        
        # Set splitter sizes: [PPI, B-Scope(hidden), A-Scope]
        # 60% PPI, 0% B-Scope (hidden), 40% A-Scope
        self.scope_splitter.setSizes([500, 0, 300])
        self.scope_splitter.setCollapsible(0, False)  # PPI cannot be collapsed
        self.scope_splitter.setCollapsible(1, True)   # B-Scope can be hidden
        self.scope_splitter.setCollapsible(2, False)  # A-Scope cannot be collapsed
        
        # Ensure A-Scope has minimum height
        self.a_scope.setMinimumHeight(180)
        
        scope_layout.addWidget(self.scope_splitter)
        self.display_tabs.addTab(scope_tab, "📡 SCOPE")
        
        # ═══ TAB 2: RHI (Elevation View) ═══
        self.rhi_scope = RHIScope(max_range_km=150.0, max_altitude_m=15000.0)
        self.display_tabs.addTab(self.rhi_scope, "📐 ELEVATION (RHI)")
        
        # ═══ TAB 3: 3D TACTICAL MAP ═══
        self.tactical_3d = TacticalMap3D(max_range_km=150.0, max_altitude_km=15.0)
        self.display_tabs.addTab(self.tactical_3d, "🗺️ 3D TACTICAL")
        
        main_layout.addWidget(self.display_tabs)
        
        # Right dock for controls
        control_dock = QDockWidget("CONTROLS", self)
        control_dock.setObjectName("controls_dock")
        control_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        
        self.control_panel = ControlPanel()
        self.control_panel.set_freq_callback(self._on_freq_changed)
        self.control_panel.set_power_callback(self._on_power_changed)
        self.control_panel.set_range_callback(self._on_range_changed)
        self.control_panel.set_speed_callback(self._on_speed_changed)
        self.control_panel.set_arch_callback(self._on_arch_preset_changed)
        self.control_panel.play_btn.clicked.connect(self._on_play_toggled)
        self.control_panel.reset_btn.clicked.connect(self._on_reset)
        
        control_dock.setWidget(self.control_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, control_dock)
        
        # Left dock for Target Inspector
        self.inspector_dock = QDockWidget("TARGET", self)
        self.inspector_dock.setObjectName("inspector_dock")
        self.inspector_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.inspector_dock.setMaximumWidth(350)  # Prevent over-expansion
        
        self.target_inspector = TargetInspector()
        self.inspector_dock.setWidget(self.target_inspector)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.inspector_dock)
        
        # Connect PPI target selection to inspector
        self.ppi_scope.target_selected.connect(self._on_target_selected)
        
        # Connect ECM state changes to simulation thread
        self.target_inspector.ecm_changed.connect(self._on_ecm_state_changed)
        
        # Store target data for lookup
        self._target_data_cache = {}
        
        # Bottom dock for Playback Panel
        self.playback_dock = QDockWidget("REPLAY", self)
        self.playback_dock.setObjectName("playback_dock")
        self.playback_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        
        self.playback_panel = PlaybackPanel()
        self.playback_panel.file_load_requested.connect(self._on_load_recording)
        self.playback_panel.time_changed.connect(self._on_replay_time_changed)
        self.playback_panel.stop_requested.connect(self._on_replay_stopped)
        self.playback_panel.set_enabled(False)  # Disabled until file loaded
        
        self.playback_dock.setWidget(self.playback_panel)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.playback_dock)
        
        # Left dock for Analysis Panel (hidden by default)
        self.analysis_dock = QDockWidget("ANALYSIS", self)
        self.analysis_dock.setObjectName("analysis_dock")
        self.analysis_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable |
            QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        self.analysis_dock.setMaximumWidth(350)  # Prevent over-expansion
        
        self.analysis_panel = AnalysisPanel()
        self.analysis_dock.setWidget(self.analysis_panel)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.analysis_dock)
        self.analysis_dock.hide()  # Hidden until replay mode
        
        # Right dock for Range-Doppler Scope
        self.rd_dock = QDockWidget("RANGE-DOPPLER", self)
        self.rd_dock.setObjectName("rd_dock")
        self.rd_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable |
            QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        
        self.rd_scope = RangeDopplerScope(max_range_km=150.0, max_velocity_mps=500.0)
        self.rd_dock.setWidget(self.rd_scope)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.rd_dock)
        
        # Tabify Range-Doppler with Controls dock
        self.tabifyDockWidget(control_dock, self.rd_dock)
        control_dock.raise_()  # Show controls by default
    
    def _setup_menu(self) -> None:
        """Setup menu bar."""
        menubar = self.menuBar()
        menubar.setStyleSheet("""
            QMenuBar {
                background-color: #001a0d;
                color: #00dd66;
                font-family: 'Consolas', monospace;
            }
            QMenuBar::item:selected {
                background-color: #003322;
            }
            QMenu {
                background-color: #001a0d;
                color: #00dd66;
            }
            QMenu::item:selected {
                background-color: #003322;
            }
        """)
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        # Load Scenario action
        load_scenario_action = QAction("&Load Scenario...", self)
        load_scenario_action.setShortcut("Ctrl+O")
        load_scenario_action.triggered.connect(self._on_load_scenario)
        file_menu.addAction(load_scenario_action)
        
        # Save Scenario action (Phase 22)
        save_scenario_action = QAction("&Save Scenario As...", self)
        save_scenario_action.setShortcut("Ctrl+Shift+S")
        save_scenario_action.triggered.connect(self._on_save_scenario)
        file_menu.addAction(save_scenario_action)
        
        file_menu.addSeparator()
        
        # Recording controls
        self.start_rec_action = QAction("⏺ Start &Recording", self)
        self.start_rec_action.setShortcut("Ctrl+R")
        self.start_rec_action.triggered.connect(self._on_start_recording)
        file_menu.addAction(self.start_rec_action)
        
        self.stop_rec_action = QAction("⏹ Stop Recordin&g", self)
        self.stop_rec_action.triggered.connect(self._on_stop_recording)
        self.stop_rec_action.setEnabled(False)
        file_menu.addAction(self.stop_rec_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Simulation menu
        sim_menu = menubar.addMenu("&Simulation")
        
        start_action = QAction("&Start", self)
        start_action.triggered.connect(self._start_simulation)
        sim_menu.addAction(start_action)
        
        stop_action = QAction("Sto&p", self)
        stop_action.triggered.connect(self._stop_simulation)
        sim_menu.addAction(stop_action)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        
        zoom_in = QAction("Zoom &In", self)
        zoom_in.setShortcut("Ctrl++")
        view_menu.addAction(zoom_in)
        
        zoom_out = QAction("Zoom &Out", self)
        zoom_out.setShortcut("Ctrl+-")
        view_menu.addAction(zoom_out)
        
        view_menu.addSeparator()
        
        # Show Analysis action (opens separate window)
        show_analysis = QAction("Show &Analysis Window", self)
        show_analysis.triggered.connect(lambda: self.analysis_window.show() or self.analysis_window.raise_())
        view_menu.addAction(show_analysis)
        
        view_menu.addSeparator()
        
        # Reset Layout action
        reset_layout = QAction("&Reset Layout", self)
        reset_layout.setShortcut("Ctrl+Shift+R")
        reset_layout.triggered.connect(self._reset_layout)
        view_menu.addAction(reset_layout)
        
        # ═══ ADVANCED MENU ═══
        advanced_menu = menubar.addMenu("&Advanced")
        
        # LPI Mode submenu
        lpi_menu = advanced_menu.addMenu("📡 LPI Mode")
        
        self.lpi_action = QAction("Enable &LPI Mode", self)
        self.lpi_action.setCheckable(True)
        self.lpi_action.setChecked(False)
        self.lpi_action.triggered.connect(self._on_lpi_toggled)
        lpi_menu.addAction(self.lpi_action)
        
        lpi_menu.addSeparator()
        
        # LPI Technique selection
        self.lpi_fhss = QAction("FHSS (Frequency Hopping)", self)
        self.lpi_fhss.setCheckable(True)
        self.lpi_fhss.setChecked(True)
        self.lpi_fhss.triggered.connect(lambda: self._set_lpi_technique('FHSS'))
        lpi_menu.addAction(self.lpi_fhss)
        
        self.lpi_dsss = QAction("DSSS (Direct Sequence)", self)
        self.lpi_dsss.setCheckable(True)
        self.lpi_dsss.triggered.connect(lambda: self._set_lpi_technique('DSSS'))
        lpi_menu.addAction(self.lpi_dsss)
        
        self.lpi_costas = QAction("Costas Array", self)
        self.lpi_costas.setCheckable(True)
        self.lpi_costas.triggered.connect(lambda: self._set_lpi_technique('Costas'))
        lpi_menu.addAction(self.lpi_costas)
        
        advanced_menu.addSeparator()
        
        # Sensor Fusion
        fusion_menu = advanced_menu.addMenu("🔗 Sensor Fusion")
        
        self.fusion_action = QAction("Enable &Sensor Fusion", self)
        self.fusion_action.setCheckable(True)
        self.fusion_action.setChecked(False)
        self.fusion_action.triggered.connect(self._on_fusion_toggled)
        fusion_menu.addAction(self.fusion_action)
        
        fusion_menu.addSeparator()
        
        # Fusion Method selection
        self.fusion_kalman = QAction("Kalman Filter", self)
        self.fusion_kalman.setCheckable(True)
        self.fusion_kalman.setChecked(True)
        self.fusion_kalman.triggered.connect(lambda: self._set_fusion_method('kalman'))
        fusion_menu.addAction(self.fusion_kalman)
        
        self.fusion_particle = QAction("Particle Filter", self)
        self.fusion_particle.setCheckable(True) 
        self.fusion_particle.triggered.connect(lambda: self._set_fusion_method('particle'))
        fusion_menu.addAction(self.fusion_particle)
        
        self.fusion_bayesian = QAction("Bayesian Fusion", self)
        self.fusion_bayesian.setCheckable(True)
        self.fusion_bayesian.triggered.connect(lambda: self._set_fusion_method('bayesian'))
        fusion_menu.addAction(self.fusion_bayesian)
        
        advanced_menu.addSeparator()
        
        # SAR Image Generator
        self.sar_action = QAction("🛰️ Generate &SAR Image...", self)
        self.sar_action.setShortcut("Ctrl+Shift+S")
        self.sar_action.triggered.connect(self._on_generate_sar)
        advanced_menu.addAction(self.sar_action)
        
        advanced_menu.addSeparator()
        
        # ═══ PHASE 19: CLUTTER, MTI & ECCM ═══
        
        # Environmental Clutter
        self.clutter_action = QAction("🌧️ Enable Environmental &Clutter", self)
        self.clutter_action.setCheckable(True)
        self.clutter_action.setChecked(False)
        self.clutter_action.triggered.connect(self._on_clutter_toggled)
        advanced_menu.addAction(self.clutter_action)
        
        # MTI Filter
        self.mti_action = QAction("🔍 Enable &MTI Filter", self)
        self.mti_action.setCheckable(True)
        self.mti_action.setChecked(False)
        self.mti_action.setToolTip("Moving Target Indication: Filters slow/stationary targets")
        self.mti_action.triggered.connect(self._on_mti_toggled)
        advanced_menu.addAction(self.mti_action)
        
        # ECCM: Frequency Agility
        self.eccm_action = QAction("⚡ ECCM: Frequency &Agility", self)
        self.eccm_action.setCheckable(True)
        self.eccm_action.setChecked(False)
        self.eccm_action.setToolTip("Counter-jamming: Hop frequency to defeat spot jammers")
        self.eccm_action.triggered.connect(self._on_eccm_toggled)
        advanced_menu.addAction(self.eccm_action)
        
        advanced_menu.addSeparator()
        
        # ═══ PHASE 20: MONOPULSE TRACKING ═══
        self.monopulse_action = QAction("🎯 Enable &Monopulse Tracking", self)
        self.monopulse_action.setCheckable(True)
        self.monopulse_action.setChecked(False)
        self.monopulse_action.setToolTip("Sum/Difference pattern: Sub-beamwidth angular accuracy")
        self.monopulse_action.triggered.connect(self._on_monopulse_toggled)
        advanced_menu.addAction(self.monopulse_action)
    
    def _setup_status_bar(self) -> None:
        """Setup status bar."""
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet("""
            QStatusBar {
                background-color: #001a0d;
                color: #00aa55;
                font-family: 'Consolas', monospace;
                font-size: 11px;
            }
        """)
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("RadarSim Ready | Press PLAY to start simulation")
    
    # ═══════════════════════════════════════════════════════════════════════
    # SIMULATION CONTROL
    # ═══════════════════════════════════════════════════════════════════════
    
    def _start_simulation(self) -> None:
        """Start the simulation thread."""
        if self.sim_thread is not None:
            self._stop_simulation()
        
        self.sim_thread = SimulationThread(
            self.engine,
            update_rate_hz=30.0
        )
        
        # Connect signals
        self.sim_thread.update_data.connect(self._on_update)
        self.sim_thread.error.connect(self._on_error)
        
        self.sim_thread.start()
        self.status_bar.showMessage("Simulation RUNNING")
    
    def _stop_simulation(self) -> None:
        """Stop the simulation thread."""
        if self.sim_thread:
            self.sim_thread.stop()
            self.sim_thread.wait(1000)
            self.sim_thread = None
        self.status_bar.showMessage("Simulation STOPPED")
    
    @pyqtSlot(dict)
    def _on_update(self, state: dict) -> None:
        """Handle simulation state update."""
        # Route to active display
        if self.active_display == "b_scope":
            self.b_scope.update_display(state)
        else:
            self.ppi_scope.update_display(state)
        
        self.a_scope.update_display(state)
        self.rd_scope.update_display(state)  # Range-Doppler scope
        self.rhi_scope.update_display(state)  # RHI (Elevation) scope
        self.tactical_3d.update_display(state)  # 3D Tactical map
        
        # Record state if recording is active
        if self.is_recording:
            self.recorder.record_state(state)
        
        # Cache target data for inspector lookup
        targets = state.get('targets', [])
        for target in targets:
            self._target_data_cache[target['id']] = target
        
        # Update inspector if target is selected
        if self.ppi_scope.selected_target_id is not None:
            target_data = self._target_data_cache.get(self.ppi_scope.selected_target_id)
            if target_data:
                self.target_inspector.update_target(target_data)
        
        # Update status bar
        time_s = state.get('time', 0)
        detections = state.get('detection_count', 0)
        total = state.get('total_targets', 0)
        self.status_bar.showMessage(
            f"TIME: {time_s:.1f}s | "
            f"TARGETS: {total} | "
            f"DETECTIONS: {detections}"
        )
    
    @pyqtSlot(str)
    def _on_error(self, error_msg: str) -> None:
        """Handle simulation error."""
        self.status_bar.showMessage(f"ERROR: {error_msg}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # TARGET & ECM HANDLERS
    # ═══════════════════════════════════════════════════════════════════════
    
    @pyqtSlot(int)
    def _on_target_selected(self, target_id: int) -> None:
        """Handle target selection from PPI scope."""
        target_data = self._target_data_cache.get(target_id)
        self.target_inspector.update_target(target_data)
        self.status_bar.showMessage(f"TARGET {target_id} SELECTED")
    
    @pyqtSlot(bool, str, int)
    def _on_ecm_state_changed(self, active: bool, ecm_type: str, target_id: int) -> None:
        """
        Handle ECM state change from TargetInspector.
        
        Forwards the ECM state to the simulation thread for processing.
        """
        if self.sim_thread is not None:
            self.sim_thread.set_ecm_state(active, ecm_type, target_id)
            print(f"[ECM] Forwarded to engine: Active={active}, Type={ecm_type}, Target={target_id}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # CONTROL PANEL CALLBACKS
    # ═══════════════════════════════════════════════════════════════════════
    
    def _on_freq_changed(self, frequency_hz: float) -> None:
        """
        Handle frequency slider change.
        
        Updates radar frequency in real-time, affecting:
        - Wavelength (λ = c/f)
        - Atmospheric attenuation (especially at 60 GHz O₂ resonance)
        - Doppler resolution
        """
        if self.engine and hasattr(self.engine, 'radar'):
            self.engine.radar.frequency_hz = frequency_hz
            freq_ghz = frequency_hz / 1e9
            self.status_bar.showMessage(f"FREQ: {freq_ghz:.1f} GHz")
    
    def _on_power_changed(self, power_watts: float) -> None:
        """
        Handle power slider change.
        
        Updates radar transmit power in real-time, affecting:
        - SNR (directly proportional)
        - Detection range (∝ P^1/4)
        """
        if self.engine and hasattr(self.engine, 'radar'):
            self.engine.radar.power_watts = power_watts
            power_kw = power_watts / 1000
            self.status_bar.showMessage(f"POWER: {power_kw:.0f} kW")
    
    def _on_range_changed(self, range_km: float) -> None:
        """Handle range slider change."""
        self.ppi_scope.set_max_range(range_km)
    
    def _on_speed_changed(self, speed: float) -> None:
        """Handle speed slider change."""
        if self.sim_thread:
            self.sim_thread.set_speed(speed)
    
    def _on_arch_preset_changed(self, preset) -> None:
        """
        Handle radar architecture preset change from dropdown.
        
        Updates:
            1. PPI scope scan mode (CIRCULAR/SECTOR/STARE)
            2. Engine radar parameters (frequency, power, wavelength)
            3. Status bar with new physics values
        """
        try:
            # Import DisplayType for checking
            from src.physics.radar_equation import DisplayType
            
            # 1. Switch display based on preset display type
            self._switch_display(preset.display_type)
            
            # 2. Update scope settings based on display type
            if preset.display_type == DisplayType.B_SCOPE:
                # Update B-Scope settings
                if preset.sector_limits:
                    self.b_scope.set_azimuth_limits(
                        preset.sector_limits[0], 
                        preset.sector_limits[1]
                    )
                self.b_scope.set_max_range(preset.max_range_km)
            else:
                # Update PPI scope scan mode
                scan_type_str = preset.scan_type.value
                self.ppi_scope.set_scan_mode(
                    scan_type=scan_type_str,
                    sector_limits=preset.sector_limits,
                    scan_speed_deg_s=preset.scan_speed_deg_s,
                    stare_angle_deg=0.0
                )
                self.ppi_scope.set_max_range(preset.max_range_km)
            
            # 3. Update engine radar parameters (if engine exists)
            if hasattr(self, 'engine') and self.engine:
                self.engine.radar.frequency_hz = preset.frequency_hz
                self.engine.radar.power_watts = preset.peak_power_watts
            
            # 4. Show status
            wavelength_cm = preset.wavelength_m * 100
            display_name = "B-SCOPE" if preset.display_type == DisplayType.B_SCOPE else "PPI"
            self.status_bar.showMessage(
                f"RADAR: {preset.name} | {display_name} | λ={wavelength_cm:.1f}cm | G={preset.gain_db:.1f}dB"
            )
            
            print(f"[ARCH] Switched to {display_name} display, limits={preset.sector_limits}")
            
        except Exception as e:
            print(f"[ARCH] Error applying preset to scope: {e}")
    
    def _switch_display(self, display_type) -> None:
        """
        Switch between PPI and B-Scope displays.
        
        Args:
            display_type: DisplayType enum (PPI or B_SCOPE)
        """
        from src.physics.radar_equation import DisplayType
        
        if display_type == DisplayType.B_SCOPE:
            if self.active_display != "b_scope":
                self.ppi_scope.hide()
                self.b_scope.show()
                self.active_display = "b_scope"
                print("[DISPLAY] Switched to B-SCOPE (Cartesian)")
        else:
            if self.active_display != "ppi":
                self.b_scope.hide()
                self.ppi_scope.show()
                self.active_display = "ppi"
                print("[DISPLAY] Switched to PPI (Polar)")
        
        # Restore splitter sizes after display switch to prevent layout breakage
        if hasattr(self, 'scope_splitter'):
            self.scope_splitter.setSizes([500, 0, 300])
    
    def _on_play_toggled(self, checked: bool) -> None:
        """Handle play/pause button."""
        if self.sim_thread:
            if checked:
                self.sim_thread.resume()
                self.control_panel.play_btn.setText("▶ PLAY")
            else:
                self.sim_thread.pause()
                self.control_panel.play_btn.setText("❚❚ PAUSE")
    
    def _on_reset(self) -> None:
        """Reset simulation."""
        self._switch_to_live_mode()
    
    # ═══════════════════════════════════════════════════════════════════════
    # RECORDING HANDLERS
    # ═══════════════════════════════════════════════════════════════════════
    
    def _on_start_recording(self) -> None:
        """Start recording simulation data."""
        if self.is_recording:
            return
        
        # Build config from current radar
        config = {
            'radar_frequency_hz': self.engine.radar.frequency_hz,
            'radar_power_watts': self.engine.radar.power_watts,
            'radar_gain_db': self.engine.radar.antenna_gain_db,
        }
        
        self.recorder.start_recording(config)
        self.is_recording = True
        
        # Update menu items
        self.start_rec_action.setEnabled(False)
        self.stop_rec_action.setEnabled(True)
        
        self.status_bar.showMessage("⏺ RECORDING...")
        print("[RECORDING] Started")
    
    def _on_stop_recording(self) -> None:
        """Stop recording and save to HDF5."""
        if not self.is_recording:
            return
        
        filepath = self.recorder.stop_recording()
        self.is_recording = False
        
        # Update menu items
        self.start_rec_action.setEnabled(True)
        self.stop_rec_action.setEnabled(False)
        
        if filepath:
            self.status_bar.showMessage(f"✓ Saved: {filepath}")
            QMessageBox.information(
                self,
                "Recording Saved",
                f"Recording saved to:\n{filepath}\n\nUse File > Load Scenario to replay."
            )
            print(f"[RECORDING] Saved to: {filepath}")
        else:
            self.status_bar.showMessage("Recording failed - no data")
    
    # ═══════════════════════════════════════════════════════════════════════
    # SCENARIO & REPLAY HANDLERS
    # ═══════════════════════════════════════════════════════════════════════
    
    def _on_load_scenario(self) -> None:
        """Handle File > Load Scenario action."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Load Scenario",
            "scenarios",
            "YAML Files (*.yaml *.yml);;All Files (*)"
        )
        
        if not filepath:
            return
        
        try:
            # Load scenario
            loader = ScenarioLoader(filepath)
            scenario_name = loader.get_scenario_name()
            
            # Stop current simulation
            self._stop_simulation()
            
            # Create new engine from scenario
            self.engine = loader.create_simulation_engine()
            
            # ═══ TERRAIN INTEGRATION ═══
            # Pass terrain to visualization components
            if hasattr(self.engine, 'terrain') and self.engine.terrain is not None:
                terrain = self.engine.terrain
                # Pass to RHI scope for elevation profile
                if hasattr(self.rhi_scope, 'set_terrain'):
                    self.rhi_scope.set_terrain(terrain)
                    print(f"[TERRAIN] Passed to RHI scope")
                # Pass to 3D tactical map for surface rendering
                if hasattr(self.tactical_3d, 'set_terrain_from_map'):
                    self.tactical_3d.set_terrain_from_map(terrain)
                    print(f"[TERRAIN] Passed to 3D tactical map")
            
            # Sync control panel with loaded radar parameters
            self.control_panel.update_from_radar(self.engine.radar)
            
            # Auto-switch radar preset if scenario specifies one
            required_preset = loader.get_required_preset()
            if required_preset:
                # Find and set the preset in dropdown
                index = self.control_panel.arch_combo.findText(required_preset)
                if index >= 0:
                    self.control_panel.arch_combo.setCurrentIndex(index)
                    print(f"[SCENARIO] Auto-configured: {required_preset}")
            
            # Start simulation with new scenario
            self._start_simulation()
            
            # ═══ CRITICAL: Restore UI layout after all changes ═══
            from PyQt6.QtWidgets import QApplication
            QApplication.processEvents()  # Let UI fully update
            
            # Ensure Analysis dock stays hidden (it's only for REPLAY mode)
            self.analysis_dock.hide()
            
            # Restore splitter sizes
            if hasattr(self, 'scope_splitter'):
                self.scope_splitter.setSizes([500, 0, 300])
                self.a_scope.setMinimumHeight(180)
            
            self.status_bar.showMessage(f"LOADED: {scenario_name}")
            
        except FileNotFoundError:
            QMessageBox.critical(self, "Error", f"File not found:\n{filepath}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load scenario:\n{str(e)}")
    
    def _switch_to_live_mode(self) -> None:
        """Switch to LIVE simulation mode."""
        # Close any open replay
        if self.replay_loader:
            self.replay_loader.close()
            self.replay_loader = None
        
        self.mode = SimulationMode.LIVE
        
        # Reset UI
        self.playback_panel.stop()
        self.playback_panel.set_enabled(False)
        self.analysis_dock.hide()
        
        # Restart live simulation
        self.engine = create_demo_scenario()
        self._start_simulation()
        
        self.status_bar.showMessage("MODE: LIVE SIMULATION")
    
    def _switch_to_replay_mode(self) -> None:
        """Switch to REPLAY mode."""
        # Stop live simulation
        self._stop_simulation()
        
        self.mode = SimulationMode.REPLAY
        
        # Enable replay controls (analysis dock stays hidden - user can open via View menu)
        self.playback_panel.set_enabled(True)
        
        self.status_bar.showMessage("MODE: REPLAY")
    
    @pyqtSlot(str)
    def _on_load_recording(self, filepath: str) -> None:
        """Load an HDF5 recording file."""
        try:
            # Close any existing loader
            if self.replay_loader:
                self.replay_loader.close()
            
            # Load new file
            self.replay_loader = ReplayLoader(filepath)
            
            # Switch to replay mode
            self._switch_to_replay_mode()
            
            # Setup playback panel
            self.playback_panel.set_duration(self.replay_loader.duration)
            self.playback_panel.set_time(0)
            
            # Open analysis in separate window
            self.analysis_window.set_loader(self.replay_loader)
            self.analysis_window.show()
            self.analysis_window.raise_()
            
            # Show initial state
            self._on_replay_time_changed(0.0)
            
            # Restore splitter layout
            if hasattr(self, 'scope_splitter'):
                self.scope_splitter.setSizes([500, 0, 300])
            
            self.status_bar.showMessage(
                f"LOADED: {filepath} | DURATION: {self.replay_loader.duration:.1f}s"
            )
            
        except FileNotFoundError:
            QMessageBox.critical(self, "Error", f"File not found:\n{filepath}")
        except ValueError as e:
            QMessageBox.critical(self, "Error", f"Invalid file format:\n{str(e)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load file:\n{str(e)}")
    
    @pyqtSlot(float)
    def _on_replay_time_changed(self, t: float) -> None:
        """Handle timeline scrubbing in replay mode."""
        if self.mode != SimulationMode.REPLAY or self.replay_loader is None:
            return
        
        # Get interpolated state at time t
        state = self.replay_loader.get_state_at_time(t)
        
        # Convert to dict format expected by PPI/A-scope
        state_dict = state.to_dict()
        
        # Update displays
        self.ppi_scope.update_display(state_dict)
        self.a_scope.update_display(state_dict)
        
        # Update analysis time marker
        self.analysis_panel.set_current_time(t)
        
        # Cache target data for inspector
        for target in state_dict.get('targets', []):
            self._target_data_cache[target['id']] = target
        
        # Update inspector if target is selected
        if self.ppi_scope.selected_target_id is not None:
            target_data = self._target_data_cache.get(self.ppi_scope.selected_target_id)
            if target_data:
                self.target_inspector.update_target(target_data)
    
    @pyqtSlot()
    def _on_replay_stopped(self) -> None:
        """Handle replay stop."""
        self.status_bar.showMessage("REPLAY STOPPED")
    
    # ═══════════════════════════════════════════════════════════════════════
    # SETTINGS PERSISTENCE
    # ═══════════════════════════════════════════════════════════════════════
    
    def _load_settings(self) -> None:
        """Load saved settings from QSettings."""
        settings = QSettings("RadarSim", "RadarSimulator")
        
        # Window geometry
        geometry = settings.value("window/geometry")
        if geometry:
            self.restoreGeometry(geometry)
        
        # Window geometry only (NOT dock state - causes layout issues)
        
        # Radar preset
        preset_index = settings.value("radar/preset_index", 0, type=int)
        if hasattr(self, 'control_panel') and self.control_panel:
            if preset_index < self.control_panel.arch_combo.count():
                self.control_panel.arch_combo.setCurrentIndex(preset_index)
        
        print("[SETTINGS] Loaded saved configuration")
    
    def _save_settings(self) -> None:
        """Save current settings to QSettings."""
        settings = QSettings("RadarSim", "RadarSimulator")
        
        # Window geometry
        settings.setValue("window/geometry", self.saveGeometry())
        
        # Note: NOT saving dock state - causes layout issues when loading scenarios
        
        # Radar preset
        if hasattr(self, 'control_panel') and self.control_panel:
            settings.setValue("radar/preset_index", 
                            self.control_panel.arch_combo.currentIndex())
        
        print("[SETTINGS] Saved configuration")
    
    def _reset_layout(self) -> None:
        """Reset window layout to defaults by clearing saved settings."""
        # Clear ALL saved settings
        settings = QSettings("RadarSim", "RadarSimulator")
        settings.clear()
        settings.sync()  # Force write to disk
        
        # Show message and prompt restart
        QMessageBox.information(
            self,
            "Reset Complete",
            "Settings cleared!\n\nPlease restart the application for changes to take effect."
        )
        print("[SETTINGS] Settings completely cleared!")

    # ═══════════════════════════════════════════════════════════════════════
    # ADVANCED MODULE HANDLERS
    # ═══════════════════════════════════════════════════════════════════════
    
    def _on_lpi_toggled(self, checked: bool) -> None:
        """Handle LPI Mode toggle."""
        self.lpi_enabled = checked
        
        # Update simulation engine if available
        if self.sim_thread and hasattr(self.sim_thread, 'set_lpi_mode'):
            self.sim_thread.set_lpi_mode(checked, self.lpi_technique)
        
        status = "ACTIVE" if checked else "OFF"
        self.status_bar.showMessage(f"LPI MODE: {status} | Technique: {self.lpi_technique}")
        print(f"[LPI] Mode={'ON' if checked else 'OFF'}, Technique={self.lpi_technique}")
    
    def _set_lpi_technique(self, technique: str) -> None:
        """Set LPI waveform technique."""
        self.lpi_technique = technique
        
        # Update checkmarks
        self.lpi_fhss.setChecked(technique == 'FHSS')
        self.lpi_dsss.setChecked(technique == 'DSSS')
        self.lpi_costas.setChecked(technique == 'Costas')
        
        # Notify engine if LPI is enabled
        if self.lpi_enabled and self.sim_thread:
            if hasattr(self.sim_thread, 'set_lpi_mode'):
                self.sim_thread.set_lpi_mode(True, technique)
        
        self.status_bar.showMessage(f"LPI Technique: {technique}")
        print(f"[LPI] Technique set to: {technique}")
    
    def _on_fusion_toggled(self, checked: bool) -> None:
        """Handle Sensor Fusion toggle."""
        self.fusion_enabled = checked
        
        # Update simulation engine if available
        if self.sim_thread and hasattr(self.sim_thread, 'set_fusion_mode'):
            self.sim_thread.set_fusion_mode(checked, self.fusion_method)
        
        status = "ACTIVE" if checked else "OFF"
        self.status_bar.showMessage(f"SENSOR FUSION: {status} | Method: {self.fusion_method}")
        print(f"[FUSION] Mode={'ON' if checked else 'OFF'}, Method={self.fusion_method}")
    
    def _set_fusion_method(self, method: str) -> None:
        """Set sensor fusion method."""
        self.fusion_method = method
        
        # Update checkmarks
        self.fusion_kalman.setChecked(method == 'kalman')
        self.fusion_particle.setChecked(method == 'particle')
        self.fusion_bayesian.setChecked(method == 'bayesian')
        
        # Notify engine if fusion is enabled
        if self.fusion_enabled and self.sim_thread:
            if hasattr(self.sim_thread, 'set_fusion_mode'):
                self.sim_thread.set_fusion_mode(True, method)
        
        self.status_bar.showMessage(f"Fusion Method: {method}")
        print(f"[FUSION] Method set to: {method}")
    
    def _on_generate_sar(self) -> None:
        """Handle SAR Image generation."""
        # Create or show SAR viewer
        if self.sar_viewer is None:
            self.sar_viewer = SARViewer(self)
        
        # Generate SAR image using advanced module
        try:
            from src.advanced import AdvancedSARISAR
            import numpy as np
            
            # Get current targets from cache
            targets = list(self._target_data_cache.values())
            
            if not targets:
                QMessageBox.warning(
                    self,
                    "No Targets",
                    "No targets available for SAR imaging.\n\n"
                    "Load a scenario with targets first."
                )
                return
            
            # Create SAR processor
            sar = AdvancedSARISAR(
                fc=10e9,
                bandwidth=100e6,
                prf=1000,
                platform_velocity=100,
                synthetic_aperture=100
            )
            
            # Extract target positions and RCS
            positions = []
            rcs_values = []
            for t in targets:
                pos = t.get('position_m', [0, 0, 0])
                if isinstance(pos, (list, tuple)):
                    positions.append(pos)
                else:
                    positions.append([pos, 0, 0])
                rcs_values.append(t.get('rcs_m2', 1.0))
            
            if positions:
                target_positions = np.array(positions)
                target_rcs = np.array(rcs_values)
                
                # Generate SAR raw data and process
                raw_data = sar.generate_sar_raw_data(target_positions, target_rcs)
                sar_image = sar.range_doppler_algorithm(raw_data)
                quality = sar.calculate_image_quality(sar_image)
                
                # Display in viewer
                self.sar_viewer.update_image(sar_image, quality)
                self.status_bar.showMessage(f"SAR Image generated | SNR: {quality.get('SNR_dB', 0):.1f} dB")
            else:
                # Generate demo image
                self.sar_viewer._generate_demo_image()
                self.status_bar.showMessage("SAR Demo Image generated")
                
        except ImportError as e:
            print(f"[SAR] Advanced module not available: {e}")
            # Generate demo image as fallback
            self.sar_viewer._generate_demo_image()
            self.status_bar.showMessage("SAR Demo Image generated (advanced module unavailable)")
        except Exception as e:
            print(f"[SAR] Error generating image: {e}")
            self.sar_viewer._generate_demo_image()
            self.status_bar.showMessage(f"SAR Demo Image generated")
        
        # Show the viewer
        self.sar_viewer.show()
        self.sar_viewer.raise_()
    
    # ═══ PHASE 19: CLUTTER, MTI & ECCM HANDLERS ═══
    
    def _on_clutter_toggled(self, checked: bool) -> None:
        """Handle Clutter toggle."""
        if self.sim_thread and hasattr(self.sim_thread, 'set_clutter_mode'):
            self.sim_thread.set_clutter_mode(checked, 'rural')
        
        status = "ACTIVE" if checked else "OFF"
        self.status_bar.showMessage(f"CLUTTER: {status}")
    
    def _on_mti_toggled(self, checked: bool) -> None:
        """Handle MTI Filter toggle."""
        if self.sim_thread and hasattr(self.sim_thread, 'set_mti_mode'):
            self.sim_thread.set_mti_mode(checked)
        
        status = "ACTIVE (filtering slow targets)" if checked else "OFF"
        self.status_bar.showMessage(f"MTI FILTER: {status}")
    
    def _on_eccm_toggled(self, checked: bool) -> None:
        """Handle ECCM Frequency Agility toggle."""
        if self.sim_thread and hasattr(self.sim_thread, 'set_eccm_agility'):
            self.sim_thread.set_eccm_agility(checked)
        
        status = "ACTIVE (frequency hopping)" if checked else "OFF"
        self.status_bar.showMessage(f"ECCM: {status}")
    
    def _on_monopulse_toggled(self, checked: bool) -> None:
        """Handle Monopulse Tracking toggle."""
        if self.sim_thread and hasattr(self.sim_thread, 'set_monopulse_mode'):
            self.sim_thread.set_monopulse_mode(checked)
        
        status = "ACTIVE (precision angle tracking)" if checked else "OFF"
        self.status_bar.showMessage(f"MONOPULSE: {status}")
    
    # ═══ PHASE 22: SCENARIO EXPORT ═══
    
    def _on_save_scenario(self) -> None:
        """Save current simulation state to YAML."""
        from PyQt6.QtWidgets import QFileDialog, QInputDialog
        
        # Get filename
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save Scenario",
            "scenarios/custom_scenario.yaml",
            "YAML Files (*.yaml *.yml)"
        )
        
        if not filepath:
            return
        
        # Get scenario name
        name, ok = QInputDialog.getText(
            self,
            "Scenario Name",
            "Enter scenario name:",
            text="Custom Scenario"
        )
        
        if not ok:
            name = "Custom Scenario"
        
        try:
            from src.io.exporter import export_scenario_to_yaml
            
            if self.sim_thread and self.sim_thread.engine:
                success = export_scenario_to_yaml(
                    self.sim_thread.engine,
                    filepath,
                    scenario_name=name
                )
                
                if success:
                    self.status_bar.showMessage(f"Scenario saved: {filepath}")
                else:
                    self.status_bar.showMessage("Failed to save scenario")
            else:
                self.status_bar.showMessage("No active simulation to save")
        except ImportError as e:
            print(f"[EXPORT] Exporter not available: {e}")
    
    # ═══ PHASE 22: KEYBOARD SHORTCUTS ═══
    
    def keyPressEvent(self, event) -> None:
        """Handle global keyboard shortcuts."""
        key = event.key()
        
        # Space - Play/Pause toggle
        if key == Qt.Key.Key_Space and not event.modifiers():
            if self.sim_thread and self.sim_thread.isRunning():
                self.control_panel._toggle_pause()
            else:
                self._start_simulation()
            return
        
        # R - Reset (stop and restart)
        if key == Qt.Key.Key_R and not event.modifiers():
            self._stop_simulation()
            self.status_bar.showMessage("Simulation reset")
            return
        
        # 1-4 - Tab switching
        if key == Qt.Key.Key_1 and not event.modifiers():
            self.display_tabs.setCurrentIndex(0)  # PPI
            return
        if key == Qt.Key.Key_2 and not event.modifiers():
            self.display_tabs.setCurrentIndex(1)  # RHI
            return
        if key == Qt.Key.Key_3 and not event.modifiers():
            self.display_tabs.setCurrentIndex(2)  # 3D Tactical
            return
        if key == Qt.Key.Key_4 and not event.modifiers():
            if self.display_tabs.count() > 3:
                self.display_tabs.setCurrentIndex(3)
            return
        
        # F11 - Toggle Fullscreen
        if key == Qt.Key.Key_F11:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
            return
        
        # Default handler
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        """Handle window close."""
        # Auto-save recording if active
        if self.is_recording:
            filepath = self.recorder.stop_recording()
            if filepath:
                print(f"[RECORDING] Auto-saved on exit: {filepath}")
        
        self._save_settings()
        self._stop_simulation()
        if self.replay_loader:
            self.replay_loader.close()
        event.accept()


