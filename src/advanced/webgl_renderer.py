"""
Gelişmiş 3D Görselleştirme Modülü (WebGL/Three.js Tabanlı)

Bu modül, radar simülasyonu için gelişmiş 3D görselleştirme sağlar.

Bilimsel Temeller:
- IEEE Transactions on Visualization and Computer Graphics
- WebGL Programming Guide, Addison-Wesley, 2013
- Three.js Documentation, Real-time 3D Graphics

Özellikler:
- Gerçek zamanlı 3D radar görselleştirme
- Radar lobları ve beam patterns
- Hedef izleri ve trajectory visualization
- Sinyal güç haritaları
- Clutter bölgeleri
- Interactive camera controls
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Tuple, List, Optional, Dict, Any
import json

class Advanced3DRenderer:
    """Gelişmiş 3D görselleştirme sınıfı"""
    
    def __init__(self, scene_size: float = 1000, fps: int = 30):
        self.scene_size = scene_size
        self.fps = fps
        self.fig = None
        self.ax = None
        self.animation = None
        
        # 3D nesneler
        self.radar_position = np.array([0, 0, 0])
        self.targets = []
        self.missiles = []
        self.radar_beams = []
        self.target_trajectories = []
        
    def create_3d_scene(self) -> None:
        """3D sahne oluşturur"""
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Sahne ayarları
        self.ax.set_xlim([-self.scene_size, self.scene_size])
        self.ax.set_ylim([-self.scene_size, self.scene_size])
        self.ax.set_zlim([0, self.scene_size])
        
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title('Gelişmiş Radar Arayıcı Başlık Simülasyonu - 3D Görünüm')
        
    def plot_radar_system(self, radar_pos: np.ndarray, 
                         radar_range: float = 500) -> None:
        """Radar sistemini 3D'de çizer"""
        # Radar pozisyonu
        self.ax.scatter(radar_pos[0], radar_pos[1], radar_pos[2], 
                       c='red', s=100, marker='^', label='Radar')
        
        # Radar menzil küresi
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = radar_pos[0] + radar_range * np.outer(np.cos(u), np.sin(v))
        y = radar_pos[1] + radar_range * np.outer(np.sin(u), np.sin(v))
        z = radar_pos[2] + radar_range * np.outer(np.ones(np.size(u)), np.cos(v))
        
        self.ax.plot_surface(x, y, z, alpha=0.1, color='red')
        
    def plot_radar_beam(self, radar_pos: np.ndarray, 
                       beam_direction: np.ndarray,
                       beam_width: float = np.pi/6,
                       beam_range: float = 300) -> None:
        """Radar beam'ini 3D'de çizer"""
        # Beam yönü
        beam_end = radar_pos + beam_direction * beam_range
        
        # Ana beam çizgisi
        self.ax.plot([radar_pos[0], beam_end[0]], 
                    [radar_pos[1], beam_end[1]], 
                    [radar_pos[2], beam_end[2]], 
                    'r-', linewidth=2, alpha=0.8)
        
        # Beam konisi
        cone_points = self.generate_beam_cone(radar_pos, beam_direction, 
                                            beam_width, beam_range)
        self.ax.plot_trisurf(cone_points[:, 0], cone_points[:, 1], cone_points[:, 2], 
                           alpha=0.2, color='red')
        
    def plot_targets(self, targets: List[Dict[str, Any]]) -> None:
        """Hedefleri 3D'de çizer"""
        for target in targets:
            pos = target['position']
            vel = target['velocity']
            target_type = target.get('type', 'unknown')
            
            # Hedef pozisyonu
            color = self.get_target_color(target_type)
            self.ax.scatter(pos[0], pos[1], pos[2], 
                          c=color, s=50, marker='o', label=f'Target ({target_type})')
            
            # Hız vektörü
            if np.linalg.norm(vel) > 0:
                vel_end = pos + vel * 5
                self.ax.plot([pos[0], vel_end[0]], 
                           [pos[1], vel_end[1]], 
                           [pos[2], vel_end[2]], 
                           color=color, alpha=0.7, linewidth=1)
                
    def plot_missiles(self, missiles: List[Dict[str, Any]]) -> None:
        """Füzeleri 3D'de çizer"""
        for missile in missiles:
            pos = missile['position']
            vel = missile['velocity']
            
            # Füze pozisyonu
            self.ax.scatter(pos[0], pos[1], pos[2], 
                          c='orange', s=30, marker='s', label='Missile')
            
            # Füze hız vektörü
            if np.linalg.norm(vel) > 0:
                vel_end = pos + vel * 3
                self.ax.plot([pos[0], vel_end[0]], 
                           [pos[1], vel_end[1]], 
                           [pos[2], vel_end[2]], 
                           color='orange', alpha=0.8, linewidth=2)
                
    def plot_target_trajectories(self, trajectories: List[List[np.ndarray]]) -> None:
        """Hedef izlerini 3D'de çizer"""
        colors = ['blue', 'green', 'purple', 'brown', 'pink']
        
        for i, trajectory in enumerate(trajectories):
            if len(trajectory) > 1:
                trajectory_array = np.array(trajectory)
                color = colors[i % len(colors)]
                
                self.ax.plot(trajectory_array[:, 0], 
                           trajectory_array[:, 1], 
                           trajectory_array[:, 2], 
                           color=color, alpha=0.6, linewidth=1)
                
    def plot_signal_power_map(self, radar_pos: np.ndarray, 
                            power_map: np.ndarray,
                            grid_size: int = 50) -> None:
        """Sinyal güç haritasını 3D'de çizer"""
        x = np.linspace(-self.scene_size/2, self.scene_size/2, grid_size)
        y = np.linspace(-self.scene_size/2, self.scene_size/2, grid_size)
        X, Y = np.meshgrid(x, y)
        
        # Z koordinatı (güç değeri)
        Z = power_map
        
        # Güç haritası
        surf = self.ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.3)
        self.fig.colorbar(surf, ax=self.ax, shrink=0.5, aspect=5)
        
    def plot_clutter_regions(self, clutter_data: List[Dict[str, Any]]) -> None:
        """Clutter bölgelerini 3D'de çizer"""
        for clutter in clutter_data:
            center = clutter['center']
            radius = clutter['radius']
            intensity = clutter['intensity']
            
            # Clutter küresi
            u = np.linspace(0, 2 * np.pi, 10)
            v = np.linspace(0, np.pi, 10)
            x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
            y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
            z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
            
            alpha = min(0.3, intensity / 100)
            self.ax.plot_surface(x, y, z, alpha=alpha, color='gray')
            
    def create_interactive_plotly_scene(self) -> go.Figure:
        """Interactive Plotly 3D sahne oluşturur"""
        fig = go.Figure()
        
        # Radar sistemi
        fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0],
            mode='markers',
            marker=dict(size=10, color='red', symbol='diamond'),
            name='Radar'
        ))
        
        # Menzil küresi
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = 500 * np.outer(np.cos(u), np.sin(v)).flatten()
        y = 500 * np.outer(np.sin(u), np.sin(v)).flatten()
        z = 500 * np.outer(np.ones(np.size(u)), np.cos(v)).flatten()
        
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(size=1, color='red', opacity=0.1),
            name='Radar Range'
        ))
        
        # Sahne ayarları
        fig.update_layout(
            title='Interactive 3D Radar Simulation',
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)',
                aspectmode='cube'
            ),
            width=800,
            height=600
        )
        
        return fig
        
    def animate_3d_scene(self, simulation_data: List[Dict[str, Any]], 
                        duration: float = 10) -> FuncAnimation:
        """3D sahneyi animate eder"""
        def update(frame):
            self.ax.clear()
            
            # Sahne ayarları
            self.ax.set_xlim([-self.scene_size, self.scene_size])
            self.ax.set_ylim([-self.scene_size, self.scene_size])
            self.ax.set_zlim([0, self.scene_size])
            
            # Mevcut frame verisi
            frame_data = simulation_data[frame]
            
            # Radar sistemi
            self.plot_radar_system(frame_data['radar_position'])
            
            # Radar beam
            if 'radar_beam' in frame_data:
                self.plot_radar_beam(frame_data['radar_position'], 
                                   frame_data['radar_beam']['direction'])
            
            # Hedefler
            if 'targets' in frame_data:
                self.plot_targets(frame_data['targets'])
            
            # Füzeler
            if 'missiles' in frame_data:
                self.plot_missiles(frame_data['missiles'])
            
            # İzler
            if 'trajectories' in frame_data:
                self.plot_target_trajectories(frame_data['trajectories'])
            
            self.ax.set_xlabel('X (m)')
            self.ax.set_ylabel('Y (m)')
            self.ax.set_zlabel('Z (m)')
            self.ax.set_title(f'Radar Simulation - Frame {frame}')
            
        # Animation oluştur
        total_frames = len(simulation_data)
        interval = duration * 1000 / total_frames  # milliseconds
        
        self.animation = FuncAnimation(self.fig, update, frames=total_frames,
                                     interval=interval, repeat=True)
        
        return self.animation
        
    def generate_beam_cone(self, radar_pos: np.ndarray, 
                          beam_direction: np.ndarray,
                          beam_width: float, beam_range: float) -> np.ndarray:
        """Radar beam konisi üretir"""
        # Koninin taban noktaları
        n_points = 20
        angles = np.linspace(0, 2 * np.pi, n_points)
        
        # Beam yönünü normalize et
        beam_dir_norm = beam_direction / np.linalg.norm(beam_direction)
        
        # Dik vektörler bul
        if np.abs(beam_dir_norm[2]) < 0.9:
            perp1 = np.array([-beam_dir_norm[1], beam_dir_norm[0], 0])
        else:
            perp1 = np.array([1, 0, 0])
        
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(beam_dir_norm, perp1)
        
        # Koninin taban noktaları
        cone_points = []
        for angle in angles:
            # Taban noktası
            radius = beam_range * np.tan(beam_width)
            point = (radar_pos + beam_dir_norm * beam_range + 
                    radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2))
            cone_points.append(point)
        
        # Radar pozisyonu ve taban noktaları
        all_points = [radar_pos] + cone_points
        
        return np.array(all_points)
        
    def get_target_color(self, target_type: str) -> str:
        """Hedef tipine göre renk döndürür"""
        color_map = {
            'aircraft': 'blue',
            'missile': 'red',
            'drone': 'green',
            'ship': 'purple',
            'ground': 'brown',
            'unknown': 'gray'
        }
        return color_map.get(target_type, 'gray')
        
    def export_to_html(self, filename: str = 'radar_3d_simulation.html') -> None:
        """3D sahneyi HTML olarak export eder"""
        if self.fig is not None:
            # Plotly figure oluştur
            plotly_fig = self.create_interactive_plotly_scene()
            plotly_fig.write_html(filename)
            print(f"3D sahne {filename} dosyasına kaydedildi.")
            
    def save_animation(self, filename: str = 'radar_animation.gif') -> None:
        """Animation'ı GIF olarak kaydeder"""
        if self.animation is not None:
            self.animation.save(filename, writer='pillow', fps=self.fps)
            print(f"Animation {filename} dosyasına kaydedildi.")

class WebGLRenderer:
    """WebGL tabanlı renderer (Three.js benzeri)"""
    
    def __init__(self):
        self.scene_data = {
            'radar': None,
            'targets': [],
            'missiles': [],
            'beams': [],
            'trajectories': []
        }
        
    def create_webgl_scene(self) -> str:
        """WebGL sahne HTML kodu üretir"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Radar 3D Simulation</title>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
            <style>
                body { margin: 0; }
                canvas { display: block; }
                #info {
                    position: absolute;
                    top: 10px;
                    left: 10px;
                    color: white;
                    font-family: Arial, sans-serif;
                    background: rgba(0,0,0,0.7);
                    padding: 10px;
                    border-radius: 5px;
                }
            </style>
        </head>
        <body>
            <div id="info">
                <h3>Radar 3D Simulation</h3>
                <p>Mouse: Rotate | Scroll: Zoom | Right-click: Pan</p>
                <p id="stats"></p>
            </div>
            <script>
                // Three.js scene setup
                const scene = new THREE.Scene();
                const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 2000);
                const renderer = new THREE.WebGLRenderer();
                renderer.setSize(window.innerWidth, window.innerHeight);
                document.body.appendChild(renderer.domElement);
                
                // Controls
                const controls = new THREE.OrbitControls(camera, renderer.domElement);
                controls.enableDamping = true;
                controls.dampingFactor = 0.05;
                
                // Lighting
                const ambientLight = new THREE.AmbientLight(0x404040);
                scene.add(ambientLight);
                const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
                directionalLight.position.set(1, 1, 1);
                scene.add(directionalLight);
                
                // Radar system
                const radarGeometry = new THREE.ConeGeometry(10, 20, 8);
                const radarMaterial = new THREE.MeshPhongMaterial({color: 0xff0000});
                const radar = new THREE.Mesh(radarGeometry, radarMaterial);
                radar.position.set(0, 0, 0);
                scene.add(radar);
                
                // Radar range sphere
                const rangeGeometry = new THREE.SphereGeometry(500, 32, 32);
                const rangeMaterial = new THREE.MeshBasicMaterial({
                    color: 0xff0000,
                    transparent: true,
                    opacity: 0.1,
                    wireframe: true
                });
                const rangeSphere = new THREE.Mesh(rangeGeometry, rangeMaterial);
                scene.add(rangeSphere);
                
                // Camera position
                camera.position.set(200, 200, 200);
                camera.lookAt(0, 0, 0);
                
                // Animation loop
                function animate() {
                    requestAnimationFrame(animate);
                    controls.update();
                    renderer.render(scene, camera);
                }
                animate();
                
                // Window resize
                window.addEventListener('resize', onWindowResize, false);
                function onWindowResize() {
                    camera.aspect = window.innerWidth / window.innerHeight;
                    camera.updateProjectionMatrix();
                    renderer.setSize(window.innerWidth, window.innerHeight);
                }
            </script>
        </body>
        </html>
        """
        return html_template

# Test fonksiyonu
if __name__ == "__main__":
    # 3D renderer testi
    renderer = Advanced3DRenderer()
    renderer.create_3d_scene()
    
    # Test verileri
    radar_pos = np.array([0, 0, 0])
    targets = [
        {'position': np.array([100, 200, 50]), 'velocity': np.array([10, 20, 0]), 'type': 'aircraft'},
        {'position': np.array([-50, 150, 30]), 'velocity': np.array([-5, 15, 0]), 'type': 'missile'}
    ]
    missiles = [
        {'position': np.array([0, 0, 10]), 'velocity': np.array([0, 100, 0])}
    ]
    
    # Sahneyi çiz
    renderer.plot_radar_system(radar_pos)
    renderer.plot_targets(targets)
    renderer.plot_missiles(missiles)
    
    # Radar beam
    beam_direction = np.array([0, 1, 0])
    renderer.plot_radar_beam(radar_pos, beam_direction)
    
    plt.legend()
    plt.show()
    
    # WebGL renderer testi
    webgl_renderer = WebGLRenderer()
    html_code = webgl_renderer.create_webgl_scene()
    
    with open('radar_3d_webgl.html', 'w') as f:
        f.write(html_code)
    
    print("3D görselleştirme modülü test edildi.")
    print("WebGL sahne radar_3d_webgl.html dosyasına kaydedildi.") 