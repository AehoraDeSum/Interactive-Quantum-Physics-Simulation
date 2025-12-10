import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib import cm
import taichi as ti
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QTabWidget, QGroupBox, QLabel,
                             QLineEdit, QComboBox, QPushButton, QTextEdit,
                             QSpinBox, QDoubleSpinBox, QProgressBar, QGridLayout,
                             QSplitter, QFrame, QMessageBox, QScrollArea)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPalette, QColor

# Initialize Taichi
ti.init(arch=ti.cpu, default_fp=ti.f32)


class QuantumSimulator:
    """Quantum system simulator using Taichi for high-performance computation."""
    
    def __init__(self):
        self.positions = None
        self.spins = None
        self.energy_levels = None

    def initialize_system(self, n_particles):
        """Initialize quantum system with given number of particles."""
        self.positions = ti.Vector.field(3, dtype=ti.f32, shape=n_particles)
        self.spins = ti.Vector.field(2, dtype=ti.f32, shape=n_particles)
        self.energy_levels = ti.field(dtype=ti.f32, shape=n_particles)

        @ti.kernel
        def init_kernel():
            for i in range(n_particles):
                r = ti.sqrt(-2 * ti.log(ti.random())) * 0.5
                theta = 2 * 3.14159 * ti.random()
                self.positions[i] = ti.Vector([r * ti.cos(theta), r * ti.sin(theta), 0.0])
                self.spins[i] = ti.Vector([ti.random(), ti.random()]).normalized()
                self.energy_levels[i] = (i + 1) * 0.1

        init_kernel()

    def evolve_system(self, dt):
        """Evolve quantum system by time step dt."""
        @ti.kernel
        def evolve_kernel(dt: ti.f32):
            for i in range(self.positions.shape[0]):
                force = -self.positions[i] * 0.1
                if ti.random() < 0.05:
                    self.spins[i] = ti.Vector([self.spins[i][1], self.spins[i][0]])
                self.positions[i] += force * dt

        evolve_kernel(dt)

    def get_data(self):
        """Extract simulation data as numpy arrays."""
        return {
            'positions': self.positions.to_numpy(),
            'spins': self.spins.to_numpy(),
            'energy_levels': self.energy_levels.to_numpy()
        }


class QuantumSimulationThread(QThread):
    """Thread for running quantum simulations."""
    
    update_signal = pyqtSignal(dict)
    finished_signal = pyqtSignal(dict)

    def __init__(self, params, quantum_simulator):
        super().__init__()
        self.params = params
        self.quantum_simulator = quantum_simulator
        self.running = True

    def run(self):
        """Main simulation loop."""
        n_particles = self.params['particle_count']
        frames = 50
        entanglement_history = []

        for frame in range(frames):
            if not self.running:
                break

            self.quantum_simulator.evolve_system(0.1)
            data = self.quantum_simulator.get_data()
            pos_np = data['positions']
            spin_np = data['spins']

            # Calculate entanglement entropy
            if n_particles >= 2:
                entanglement = 0.0
                for i in range(0, n_particles - 1, 2):
                    correlation = np.dot(spin_np[i], spin_np[i + 1])
                    entanglement += (1.0 - abs(correlation)) * 0.5
                entanglement /= (n_particles // 2)
                entanglement_history.append(entanglement)
            else:
                entanglement_history.append(0.0)

            self.update_signal.emit({
                'frame': frame,
                'positions': pos_np,
                'spins': spin_np,
                'entanglement': entanglement_history[-1],
                'energy_levels': data['energy_levels']
            })

            time.sleep(0.1)

        result = {
            'type': 'quantum',
            'entanglement': np.mean(entanglement_history) if entanglement_history else 0.0,
            'final_entanglement': entanglement_history[-1] if entanglement_history else 0.0,
            'particle_count': n_particles,
            'parameters': self.params,
            'entanglement_history': entanglement_history
        }

        self.finished_signal.emit(result)

    def stop(self):
        """Stop simulation thread."""
        self.running = False


class CosmicSimulationThread(QThread):
    """Thread for cosmic ray detection simulations."""
    
    update_signal = pyqtSignal(dict)
    finished_signal = pyqtSignal(dict)

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.running = True
        self.cosmic_data = None

    def generate_cosmic_data(self):
        """Generate synthetic cosmic ray data."""
        n_events = int(self.params['detection_time'] * 20)
        energies = np.random.power(2.7, n_events) * 1000
        times = np.cumsum(np.random.exponential(0.1, n_events))

        x = np.random.normal(self.params['coord_x'], 2, n_events)
        y = np.random.normal(self.params['coord_y'], 2, n_events)
        z = np.random.normal(self.params['coord_z'], 1, n_events)

        particle_types = []
        for energy in energies:
            if energy < 1:
                types = ['proton'] * 70 + ['gamma'] * 20 + ['neutrino'] * 10
            elif energy < 10:
                types = ['proton'] * 50 + ['gamma'] * 30 + ['neutrino'] * 20
            else:
                types = ['proton'] * 40 + ['gamma'] * 35 + ['neutrino'] * 25
            particle_types.append(np.random.choice(types))

        mask = energies >= self.params['energy_threshold']
        if self.params['ray_type'] != 'all':
            mask = mask & (np.array(particle_types) == self.params['ray_type'])

        filtered_data = np.array(list(zip(
            times[mask], x[mask], y[mask], z[mask],
            energies[mask], [particle_types[i] for i in range(len(mask)) if mask[i]]
        )), dtype=[
            ('time', 'f4'), ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('energy', 'f4'), ('particle_type', 'U10')
        ])

        return filtered_data

    def run(self):
        """Main cosmic ray simulation loop."""
        self.cosmic_data = self.generate_cosmic_data()
        total_time = self.params['detection_time']
        frames = min(int(total_time * 2), 100)

        for frame in range(frames):
            if not self.running:
                break

            current_time = (frame / frames) * total_time
            detected_events = self.cosmic_data[self.cosmic_data['time'] <= current_time]

            stats = {
                'total_events': len(detected_events),
                'current_time': current_time,
                'detection_rate': len(detected_events) / max(current_time, 0.1),
                'max_energy': np.max(detected_events['energy']) if len(detected_events) > 0 else 0,
                'mean_energy': np.mean(detected_events['energy']) if len(detected_events) > 0 else 0,
                'particle_types': detected_events['particle_type'] if len(detected_events) > 0 else []
            }

            self.update_signal.emit({
                'frame': frame,
                'current_time': current_time,
                'detected_events': detected_events,
                'stats': stats
            })

            time.sleep(0.2)

        final_events = self.cosmic_data[self.cosmic_data['time'] <= total_time]
        result = {
            'type': 'cosmic',
            'total_events': len(final_events),
            'max_energy': np.max(final_events['energy']) if len(final_events) > 0 else 0,
            'mean_energy': np.mean(final_events['energy']) if len(final_events) > 0 else 0,
            'detection_rate': len(final_events) / total_time if total_time > 0 else 0,
            'particle_distribution': dict(zip(*np.unique(final_events['particle_type'], return_counts=True))) if len(
                final_events) > 0 else {},
            'parameters': self.params
        }

        self.finished_signal.emit(result)

    def stop(self):
        """Stop cosmic ray simulation."""
        self.running = False


class QuantumVisualizationWidget(QWidget):
    """Widget for quantum simulation visualization."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(10, 8), facecolor='#1e1e1e')
        self.canvas = FigureCanvas(self.figure)
        self.entanglement_history = []
        self.setup_plots()
        
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def setup_plots(self):
        """Initialize quantum visualization plots."""
        self.figure.clear()
        self.ax1 = self.figure.add_subplot(221, projection='3d')
        self.ax2 = self.figure.add_subplot(222)
        self.ax3 = self.figure.add_subplot(223)
        self.ax4 = self.figure.add_subplot(224)

        # Set dark theme for plots
        axes = [self.ax1, self.ax2, self.ax3, self.ax4]
        for i, ax in enumerate(axes):
            ax.set_facecolor('#2d2d2d')
            ax.spines['bottom'].set_color('#888888')
            ax.spines['top'].set_color('#888888') 
            ax.spines['right'].set_color('#888888')
            ax.spines['left'].set_color('#888888')
            ax.tick_params(colors='#888888')
            ax.xaxis.label.set_color('#ffffff')
            ax.yaxis.label.set_color('#ffffff')
            ax.title.set_color('#ffffff')
            
            # Sadece 3D eksen için z-axis ayarı
            if i == 0:  # ax1 is 3D
                ax.zaxis.label.set_color('#ffffff')
                ax.zaxis.set_tick_params(color='#888888')

        self.ax1.set_title('Quantum Particle Distribution')
        self.ax2.set_title('Spin States')
        self.ax3.set_title('Energy Levels')
        self.ax4.set_title('Entanglement Evolution')
        self.entanglement_history.clear()

    def update_plots(self, data):
        """Update quantum plots with new data."""
        positions = data['positions']
        spins = data['spins']
        energy_levels = data['energy_levels']
        entanglement = data['entanglement']

        self.entanglement_history.append(entanglement)

        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()

        # Set dark theme again after clearing
        axes = [self.ax1, self.ax2, self.ax3, self.ax4]
        for i, ax in enumerate(axes):
            ax.set_facecolor('#2d2d2d')
            ax.spines['bottom'].set_color('#888888')
            ax.spines['top'].set_color('#888888') 
            ax.spines['right'].set_color('#888888')
            ax.spines['left'].set_color('#888888')
            ax.tick_params(colors='#888888')
            ax.xaxis.label.set_color('#ffffff')
            ax.yaxis.label.set_color('#ffffff')
            ax.title.set_color('#ffffff')
            
            # Sadece 3D eksen için z-axis ayarı
            if i == 0:  # ax1 is 3D
                ax.zaxis.label.set_color('#ffffff')
                ax.zaxis.set_tick_params(color='#888888')

        # 1. 3D Particle Distribution
        if len(positions) > 0:
            colors = plt.cm.plasma(energy_levels / max(energy_levels) if max(energy_levels) > 0 else energy_levels)
            self.ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                             c=colors, s=30, alpha=0.8)
        self.ax1.set_title('Quantum Particle Distribution')
        self.ax1.set_xlabel('X')
        self.ax1.set_ylabel('Y')
        self.ax1.set_zlabel('Z')

        # 2. Spin Distribution
        if len(spins) > 0:
            theta = np.arctan2(spins[:, 1], spins[:, 0])
            self.ax2.scatter(np.cos(theta), np.sin(theta),
                             c=energy_levels, cmap='hsv', alpha=0.7, s=20)
            circle = plt.Circle((0, 0), 1, fill=False, color='white', linestyle='--')
            self.ax2.add_patch(circle)
        self.ax2.set_xlim([-1.2, 1.2])
        self.ax2.set_ylim([-1.2, 1.2])
        self.ax2.set_title('Spin States (Bloch Sphere)')
        self.ax2.set_aspect('equal')
        self.ax2.grid(True, color='#444444')

        # 3. Energy Levels
        if len(energy_levels) > 0:
            self.ax3.hist(energy_levels, bins=20, alpha=0.7, color='purple', edgecolor='white')
        self.ax3.set_title('Energy Level Distribution')
        self.ax3.set_xlabel('Energy')
        self.ax3.set_ylabel('Particle Count')
        self.ax3.grid(True, color='#444444')

        # 4. Entanglement Evolution
        if len(self.entanglement_history) > 0:
            self.ax4.plot(self.entanglement_history, 'g-', linewidth=2)
            self.ax4.set_ylim([0, 1.0])
        self.ax4.set_title('Entanglement Entropy Time Evolution')
        self.ax4.set_xlabel('Time Step')
        self.ax4.set_ylabel('Entanglement Entropy')
        self.ax4.grid(True, color='#444444')

        self.figure.tight_layout(rect=[0, 0, 1, 0.96])
        self.canvas.draw()


class CosmicVisualizationWidget(QWidget):
    """Widget for cosmic ray visualization."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(12, 8), facecolor='#1e1e1e')
        self.canvas = FigureCanvas(self.figure)
        self.setup_plots()
        
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def setup_plots(self):
        """Initialize cosmic ray visualization plots."""
        self.figure.clear()

        self.ax1 = self.figure.add_subplot(231, projection='3d')
        self.ax2 = self.figure.add_subplot(232)
        self.ax3 = self.figure.add_subplot(233)
        self.ax4 = self.figure.add_subplot(234)
        self.ax5 = self.figure.add_subplot(235)
        self.ax6 = self.figure.add_subplot(236)

        # Set dark theme for all plots
        axes = [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5, self.ax6]
        for i, ax in enumerate(axes):
            ax.set_facecolor('#2d2d2d')
            ax.spines['bottom'].set_color('#888888')
            ax.spines['top'].set_color('#888888') 
            ax.spines['right'].set_color('#888888')
            ax.spines['left'].set_color('#888888')
            ax.tick_params(colors='#888888')
            ax.xaxis.label.set_color('#ffffff')
            ax.yaxis.label.set_color('#ffffff')
            ax.title.set_color('#ffffff')
            
            # Sadece 3D eksen için z-axis ayarı
            if i == 0:  # ax1 is 3D
                ax.zaxis.label.set_color('#ffffff')
                ax.zaxis.set_tick_params(color='#888888')

        self.ax1.set_title('3D Detection Area')
        self.ax2.set_title('Energy Distribution')
        self.ax3.set_title('Time Series')
        self.ax4.set_title('Angle Distribution')
        self.ax5.set_title('Particle Type')
        self.ax6.set_title('Statistics')
        self.ax6.axis('off')

    def update_plots(self, data):
        """Update cosmic ray plots with new data."""
        detected_events = data['detected_events']
        stats = data['stats']
        current_time = data['current_time']

        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        self.ax5.clear()
        self.ax6.clear()

        # Set dark theme again after clearing
        axes = [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5, self.ax6]
        for i, ax in enumerate(axes):
            ax.set_facecolor('#2d2d2d')
            ax.spines['bottom'].set_color('#888888')
            ax.spines['top'].set_color('#888888') 
            ax.spines['right'].set_color('#888888')
            ax.spines['left'].set_color('#888888')
            ax.tick_params(colors='#888888')
            ax.xaxis.label.set_color('#ffffff')
            ax.yaxis.label.set_color('#ffffff')
            ax.title.set_color('#ffffff')
            
            # Sadece 3D eksen için z-axis ayarı
            if i == 0:  # ax1 is 3D
                ax.zaxis.label.set_color('#ffffff')
                ax.zaxis.set_tick_params(color='#888888')

        # 1. 3D Detection Area
        if len(detected_events) > 0:
            energy_values = detected_events['energy']
            if len(energy_values) > 0 and np.max(energy_values) > 0:
                colors = plt.cm.viridis(energy_values / np.max(energy_values))
                sizes = np.sqrt(energy_values) * 10
                self.ax1.scatter(detected_events['x'], detected_events['y'], detected_events['z'],
                                 c=colors, s=sizes, alpha=0.7)

        self.ax1.set_xlim([-10, 10])
        self.ax1.set_ylim([-10, 10])
        self.ax1.set_zlim([-10, 10])
        self.ax1.set_title('3D Cosmic Ray Detection Area')
        self.ax1.set_xlabel('X (km)')
        self.ax1.set_ylabel('Y (km)')
        self.ax1.set_zlabel('Z (km)')

        # 2. Energy Distribution
        if len(detected_events) > 0:
            self.ax2.hist(detected_events['energy'], bins=20, alpha=0.7, color='red', edgecolor='white')
        self.ax2.set_title('Cosmic Ray Energy Distribution')
        self.ax2.set_xlabel('Energy (GeV)')
        self.ax2.set_ylabel('Event Count')
        self.ax2.grid(True, color='#444444')

        # 3. Time Series
        if len(detected_events) > 0:
            time_points = np.linspace(0, current_time, 20)
            event_counts = [np.sum(detected_events['time'] <= t) for t in time_points]
            self.ax3.plot(time_points, event_counts, 'cyan', linewidth=2)
        self.ax3.set_title('Cumulative Detection Count')
        self.ax3.set_xlabel('Time (s)')
        self.ax3.set_ylabel('Total Events')
        self.ax3.grid(True, color='#444444')

        # 4. Angle Distribution
        if len(detected_events) > 0:
            theta = np.arctan2(detected_events['y'], detected_events['x'])
            self.ax4.hist(theta, bins=20, alpha=0.7, color='lime', edgecolor='white')
        self.ax4.set_title('Arrival Angle Distribution')
        self.ax4.set_xlabel('Angle (radians)')
        self.ax4.set_ylabel('Event Count')
        self.ax4.grid(True, color='#444444')

        # 5. Particle Type
        if len(detected_events) > 0:
            types, counts = np.unique(detected_events['particle_type'], return_counts=True)
            colors = ['cyan', 'magenta', 'yellow', 'orange', 'pink']
            self.ax5.bar(range(len(types)), counts, alpha=0.7, color=colors[:len(types)], edgecolor='white')
            self.ax5.set_xticks(range(len(types)))
            self.ax5.set_xticklabels(types, rotation=45, color='white')
        self.ax5.set_title('Particle Type Distribution')
        self.ax5.set_ylabel('Event Count')
        self.ax5.grid(True, axis='y', color='#444444')

        # 6. Statistics
        stats_text = f"""
STATISTICS:
• Total Events: {stats['total_events']}
• Current Time: {current_time:.1f} s
• Detection Rate: {stats['detection_rate']:.1f} events/s
"""
        if stats['total_events'] > 0:
            stats_text += f"""• Maximum Energy: {stats['max_energy']:.1f} GeV
• Mean Energy: {stats['mean_energy']:.1f} GeV
• Different Types: {len(np.unique(stats['particle_types']))}"""

        self.ax6.text(0.1, 0.5, stats_text, transform=self.ax6.transAxes, fontsize=10,
                      bbox=dict(boxstyle="round", facecolor='#333333', edgecolor='#666666', alpha=0.9),
                      color='white')
        self.ax6.axis('off')

        self.figure.tight_layout(rect=[0, 0, 1, 0.96])
        self.canvas.draw()


class QuantumTab(QWidget):
    """Tab for quantum simulation interface."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.quantum_simulator = QuantumSimulator()
        self.simulation_thread = None
        self.init_ui()

    def init_ui(self):
        """Initialize quantum tab UI."""
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Left panel for controls
        left_panel = QWidget()
        left_panel.setStyleSheet("background-color: #2d2d2d; border-radius: 10px; padding: 10px;")
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(10)

        # Parameters group
        params_group = QGroupBox("Quantum Simulation Parameters")
        params_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555555;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #4CAF50;
            }
        """)
        params_layout = QGridLayout()
        params_layout.setSpacing(8)

        # Parameter controls
        self.particle_count = QSpinBox()
        self.particle_count.setRange(1, 1000)
        self.particle_count.setValue(100)
        self.particle_count.setStyleSheet("""
            QSpinBox {
                background-color: #3d3d3d;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 5px;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                background-color: #555555;
                border: 1px solid #666666;
                border-radius: 2px;
            }
        """)
        params_layout.addWidget(QLabel("Particle Count:"), 0, 0)
        params_layout.addWidget(self.particle_count, 0, 1)

        self.energy_level = QDoubleSpinBox()
        self.energy_level.setRange(0.1, 1000)
        self.energy_level.setValue(13.6)
        self.energy_level.setDecimals(1)
        self.energy_level.setStyleSheet("""
            QDoubleSpinBox {
                background-color: #3d3d3d;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 5px;
            }
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                background-color: #555555;
                border: 1px solid #666666;
                border-radius: 2px;
            }
        """)
        params_layout.addWidget(QLabel("Energy Level (eV):"), 1, 0)
        params_layout.addWidget(self.energy_level, 1, 1)

        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(0, 10000)
        self.temperature.setValue(300)
        self.temperature.setStyleSheet("""
            QDoubleSpinBox {
                background-color: #3d3d3d;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 5px;
            }
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                background-color: #555555;
                border: 1px solid #666666;
                border-radius: 2px;
            }
        """)
        params_layout.addWidget(QLabel("Temperature (K):"), 2, 0)
        params_layout.addWidget(self.temperature, 2, 1)

        self.magnetic_field = QDoubleSpinBox()
        self.magnetic_field.setRange(0, 10)
        self.magnetic_field.setValue(1.0)
        self.magnetic_field.setDecimals(1)
        self.magnetic_field.setStyleSheet("""
            QDoubleSpinBox {
                background-color: #3d3d3d;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 5px;
            }
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                background-color: #555555;
                border: 1px solid #666666;
                border-radius: 2px;
            }
        """)
        params_layout.addWidget(QLabel("Magnetic Field (T):"), 3, 0)
        params_layout.addWidget(self.magnetic_field, 3, 1)

        # Dropdown controls
        params_layout.addWidget(QLabel("Potential Type:"), 4, 0)
        self.potential_type = QComboBox()
        self.potential_type.addItems(["harmonic", "square", "free"])
        self.potential_type.setStyleSheet("""
            QComboBox {
                background-color: #3d3d3d;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 5px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 2px solid #ffffff;
                border-bottom: 2px solid #ffffff;
                width: 5px;
                height: 5px;
                transform: rotate(-45deg);
                margin-right: 5px;
            }
            QComboBox QAbstractItemView {
                background-color: #3d3d3d;
                color: #ffffff;
                selection-background-color: #4CAF50;
            }
        """)
        params_layout.addWidget(self.potential_type, 4, 1)

        params_layout.addWidget(QLabel("Entanglement:"), 5, 0)
        self.entanglement = QComboBox()
        self.entanglement.addItems(["yes", "no"])
        self.entanglement.setStyleSheet(self.potential_type.styleSheet())
        params_layout.addWidget(self.entanglement, 5, 1)

        params_layout.addWidget(QLabel("Quantum State:"), 6, 0)
        self.quantum_state = QComboBox()
        self.quantum_state.addItems(["ground", "excited", "superposition", "entangled"])
        self.quantum_state.setStyleSheet(self.potential_type.styleSheet())
        params_layout.addWidget(self.quantum_state, 6, 1)

        params_group.setLayout(params_layout)
        left_layout.addWidget(params_group)

        # Control buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        self.start_button = QPushButton("Start Simulation")
        self.start_button.clicked.connect(self.start_simulation)
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 12px;
                border-radius: 6px;
                border: none;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #666666;
                color: #999999;
            }
        """)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_simulation)
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                padding: 12px;
                border-radius: 6px;
                border: none;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
            QPushButton:disabled {
                background-color: #666666;
                color: #999999;
            }
        """)
        
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        left_layout.addLayout(button_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #555555;
                border-radius: 4px;
                text-align: center;
                background-color: #3d3d3d;
                color: white;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 4px;
            }
        """)
        left_layout.addWidget(self.progress_bar)

        # Results text
        results_label = QLabel("Simulation Results:")
        results_label.setStyleSheet("color: #ffffff; font-weight: bold;")
        left_layout.addWidget(results_label)
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(200)
        self.result_text.setStyleSheet("""
            QTextEdit {
                background-color: #3d3d3d;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 6px;
                padding: 8px;
                font-family: 'Courier New', monospace;
            }
        """)
        left_layout.addWidget(self.result_text)

        left_panel.setLayout(left_layout)
        left_panel.setMaximumWidth(400)

        # Visualization widget
        self.visualization_widget = QuantumVisualizationWidget()

        # Combine layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(self.visualization_widget)
        self.setLayout(main_layout)

    def start_simulation(self):
        """Start quantum simulation."""
        params = {
            'particle_count': self.particle_count.value(),
            'energy_level': self.energy_level.value(),
            'temperature': self.temperature.value(),
            'magnetic_field': self.magnetic_field.value(),
            'potential_type': self.potential_type.currentText(),
            'entanglement': self.entanglement.currentText(),
            'quantum_state': self.quantum_state.currentText()
        }

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setValue(0)

        # Initialize quantum system
        self.quantum_simulator.initialize_system(params['particle_count'])
        self.visualization_widget.setup_plots()

        # Start simulation thread
        self.simulation_thread = QuantumSimulationThread(params, self.quantum_simulator)
        self.simulation_thread.update_signal.connect(self.on_simulation_update)
        self.simulation_thread.finished_signal.connect(self.on_simulation_finished)
        self.simulation_thread.start()

        self.result_text.append("Quantum simulation started...")

    def stop_simulation(self):
        """Stop quantum simulation."""
        if self.simulation_thread:
            self.simulation_thread.stop()
            self.simulation_thread.wait()

        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setValue(100)
        self.result_text.append("Simulation stopped.")

    def on_simulation_update(self, data):
        """Handle simulation update signals."""
        frame = data['frame']
        progress = min(100, int((frame + 1) * 2))
        self.progress_bar.setValue(progress)
        self.visualization_widget.update_plots(data)

    def on_simulation_finished(self, result):
        """Handle simulation completion."""
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setValue(100)

        self.result_text.append("\n=== QUANTUM SIMULATION RESULTS ===")
        self.result_text.append(f"Particle Count: {result['particle_count']}")
        self.result_text.append(f"Entanglement Level: {result['final_entanglement']:.4f}")
        self.result_text.append(f"Energy Level: {result['parameters']['energy_level']} eV")
        self.result_text.append(f"Quantum State: {result['parameters']['quantum_state']}")
        self.result_text.append("Simulation completed!")

        if self.parent:
            self.parent.simulation_data['quantum'] = result


class CosmicTab(QWidget):
    """Tab for cosmic ray analysis interface."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.simulation_thread = None
        self.init_ui()

    def init_ui(self):
        """Initialize cosmic tab UI."""
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Left panel for controls
        left_panel = QWidget()
        left_panel.setStyleSheet("background-color: #2d2d2d; border-radius: 10px; padding: 10px;")
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(10)

        # Parameters group
        params_group = QGroupBox("Cosmic Ray Detection Parameters")
        params_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555555;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #2196F3;
            }
        """)
        params_layout = QGridLayout()
        params_layout.setSpacing(8)

        # Parameter controls
        self.detection_time = QDoubleSpinBox()
        self.detection_time.setRange(1, 3600)
        self.detection_time.setValue(60)
        self.detection_time.setStyleSheet("""
            QDoubleSpinBox {
                background-color: #3d3d3d;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 5px;
            }
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                background-color: #555555;
                border: 1px solid #666666;
                border-radius: 2px;
            }
        """)
        params_layout.addWidget(QLabel("Detection Time (s):"), 0, 0)
        params_layout.addWidget(self.detection_time, 0, 1)

        self.energy_threshold = QDoubleSpinBox()
        self.energy_threshold.setRange(0.1, 1000)
        self.energy_threshold.setValue(1.0)
        self.energy_threshold.setDecimals(1)
        self.energy_threshold.setStyleSheet("""
            QDoubleSpinBox {
                background-color: #3d3d3d;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 5px;
            }
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                background-color: #555555;
                border: 1px solid #666666;
                border-radius: 2px;
            }
        """)
        params_layout.addWidget(QLabel("Energy Threshold (GeV):"), 1, 0)
        params_layout.addWidget(self.energy_threshold, 1, 1)

        self.coord_x = QDoubleSpinBox()
        self.coord_x.setRange(-1000, 1000)
        self.coord_x.setValue(0)
        self.coord_x.setStyleSheet(self.detection_time.styleSheet())
        params_layout.addWidget(QLabel("X Coordinate (km):"), 2, 0)
        params_layout.addWidget(self.coord_x, 2, 1)

        self.coord_y = QDoubleSpinBox()
        self.coord_y.setRange(-1000, 1000)
        self.coord_y.setValue(0)
        self.coord_y.setStyleSheet(self.detection_time.styleSheet())
        params_layout.addWidget(QLabel("Y Coordinate (km):"), 3, 0)
        params_layout.addWidget(self.coord_y, 3, 1)

        self.coord_z = QDoubleSpinBox()
        self.coord_z.setRange(-1000, 1000)
        self.coord_z.setValue(0)
        self.coord_z.setStyleSheet(self.detection_time.styleSheet())
        params_layout.addWidget(QLabel("Z Coordinate (km):"), 4, 0)
        params_layout.addWidget(self.coord_z, 4, 1)

        self.atmosphere_depth = QDoubleSpinBox()
        self.atmosphere_depth.setRange(100, 10000)
        self.atmosphere_depth.setValue(1000)
        self.atmosphere_depth.setStyleSheet(self.detection_time.styleSheet())
        params_layout.addWidget(QLabel("Atmosphere Depth (g/cm²):"), 5, 0)
        params_layout.addWidget(self.atmosphere_depth, 5, 1)

        # Ray type dropdown
        params_layout.addWidget(QLabel("Ray Type:"), 6, 0)
        self.ray_type = QComboBox()
        self.ray_type.addItems(["all", "proton", "gamma", "neutrino"])
        self.ray_type.setStyleSheet("""
            QComboBox {
                background-color: #3d3d3d;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 5px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 2px solid #ffffff;
                border-bottom: 2px solid #ffffff;
                width: 5px;
                height: 5px;
                transform: rotate(-45deg);
                margin-right: 5px;
            }
            QComboBox QAbstractItemView {
                background-color: #3d3d3d;
                color: #ffffff;
                selection-background-color: #2196F3;
            }
        """)
        params_layout.addWidget(self.ray_type, 6, 1)

        params_group.setLayout(params_layout)
        left_layout.addWidget(params_group)

        # Control buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        self.start_button = QPushButton("Start Detection")
        self.start_button.clicked.connect(self.start_simulation)
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                padding: 12px;
                border-radius: 6px;
                border: none;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #666666;
                color: #999999;
            }
        """)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_simulation)
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                padding: 12px;
                border-radius: 6px;
                border: none;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
            QPushButton:disabled {
                background-color: #666666;
                color: #999999;
            }
        """)
        
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        left_layout.addLayout(button_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #555555;
                border-radius: 4px;
                text-align: center;
                background-color: #3d3d3d;
                color: white;
            }
            QProgressBar::chunk {
                background-color: #2196F3;
                border-radius: 4px;
            }
        """)
        left_layout.addWidget(self.progress_bar)

        # Results text
        results_label = QLabel("Detection Results:")
        results_label.setStyleSheet("color: #ffffff; font-weight: bold;")
        left_layout.addWidget(results_label)
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(200)
        self.result_text.setStyleSheet("""
            QTextEdit {
                background-color: #3d3d3d;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 6px;
                padding: 8px;
                font-family: 'Courier New', monospace;
            }
        """)
        left_layout.addWidget(self.result_text)

        left_panel.setLayout(left_layout)
        left_panel.setMaximumWidth(400)

        # Visualization widget
        self.visualization_widget = CosmicVisualizationWidget()

        # Combine layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(self.visualization_widget)
        self.setLayout(main_layout)

    def start_simulation(self):
        """Start cosmic ray detection."""
        params = {
            'detection_time': self.detection_time.value(),
            'energy_threshold': self.energy_threshold.value(),
            'coord_x': self.coord_x.value(),
            'coord_y': self.coord_y.value(),
            'coord_z': self.coord_z.value(),
            'atmosphere_depth': self.atmosphere_depth.value(),
            'ray_type': self.ray_type.currentText()
        }

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setValue(0)

        self.visualization_widget.setup_plots()
        self.simulation_thread = CosmicSimulationThread(params)
        self.simulation_thread.update_signal.connect(self.on_simulation_update)
        self.simulation_thread.finished_signal.connect(self.on_simulation_finished)
        self.simulation_thread.start()

        self.result_text.append("Cosmic ray detection started...")

    def stop_simulation(self):
        """Stop cosmic ray detection."""
        if self.simulation_thread:
            self.simulation_thread.stop()
            self.simulation_thread.wait()

        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setValue(100)
        self.result_text.append("Detection stopped.")

    def on_simulation_update(self, data):
        """Handle detection update signals."""
        frame = data['frame']
        progress = min(100, int((frame + 1)))
        self.progress_bar.setValue(progress)
        self.visualization_widget.update_plots(data)

    def on_simulation_finished(self, result):
        """Handle detection completion."""
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setValue(100)

        self.result_text.append("\n=== COSMIC RAY DETECTION RESULTS ===")
        self.result_text.append(f"Total Event Count: {result['total_events']}")
        self.result_text.append(f"Maximum Energy: {result['max_energy']:.1f} GeV")
        self.result_text.append(f"Detection Rate: {result['detection_rate']:.1f} events/s")
        self.result_text.append(f"Detection Time: {result['parameters']['detection_time']} s")
        self.result_text.append(f"Ray Type: {result['parameters']['ray_type']}")
        self.result_text.append("Detection completed!")

        if self.parent:
            self.parent.simulation_data['cosmic'] = result


class ReportTab(QWidget):
    """Tab for scientific report generation."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.init_ui()

    def init_ui(self):
        """Initialize report tab UI."""
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Title
        title_label = QLabel("SCIENTIFIC REPORT GENERATOR")
        title_label.setFont(QFont("Arial", 18, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #FF6B6B; padding: 15px; background-color: #2d2d2d; border-radius: 10px;")
        layout.addWidget(title_label)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(20)
        
        self.create_report_button = QPushButton("📄 Create Report")
        self.create_report_button.clicked.connect(self.create_report)
        self.create_report_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.create_report_button.setStyleSheet("""
            QPushButton {
                background-color: #FF6B6B;
                color: white;
                font-weight: bold;
                padding: 15px 30px;
                border-radius: 8px;
                border: none;
                min-width: 180px;
            }
            QPushButton:hover {
                background-color: #ff5252;
            }
        """)
        
        self.export_button = QPushButton("💾 Export Report")
        self.export_button.clicked.connect(self.export_report)
        self.export_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.export_button.setStyleSheet("""
            QPushButton {
                background-color: #4ECDC4;
                color: white;
                font-weight: bold;
                padding: 15px 30px;
                border-radius: 8px;
                border: none;
                min-width: 180px;
            }
            QPushButton:hover {
                background-color: #3db8af;
            }
        """)
        
        button_layout.addStretch()
        button_layout.addWidget(self.create_report_button)
        button_layout.addWidget(self.export_button)
        button_layout.addStretch()
        
        layout.addLayout(button_layout)

        # Report text area
        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setFont(QFont("Courier New", 10))
        self.report_text.setStyleSheet("""
            QTextEdit {
                background-color: #2d2d2d;
                color: #ffffff;
                border: 2px solid #555555;
                border-radius: 10px;
                padding: 15px;
                font-family: 'Courier New', monospace;
            }
        """)
        layout.addWidget(self.report_text)

        self.setLayout(layout)

    def create_report(self):
        """Generate scientific report from simulation data."""
        if not self.parent or not self.parent.simulation_data:
            QMessageBox.warning(self, "Warning", "No simulation data yet!\nPlease run a simulation first.")
            return

        self.report_text.clear()
        report = "=" * 60 + "\n"
        report += "         SCIENTIFIC SIMULATION ANALYSIS REPORT\n"
        report += "=" * 60 + "\n\n"

        if 'quantum' in self.parent.simulation_data:
            data = self.parent.simulation_data['quantum']
            report += "🔬 QUANTUM SYSTEM ANALYSIS\n"
            report += "─" * 40 + "\n"
            report += f"• Particle Count: {data['particle_count']}\n"
            report += f"• Entanglement Level: {data['entanglement']:.4f}\n"
            report += f"• Energy Level: {data['parameters']['energy_level']} eV\n"
            report += f"• Temperature: {data['parameters']['temperature']} K\n"
            report += f"• Magnetic Field: {data['parameters']['magnetic_field']} T\n"
            report += f"• Quantum State: {data['parameters']['quantum_state']}\n\n"

        if 'cosmic' in self.parent.simulation_data:
            data = self.parent.simulation_data['cosmic']
            report += "🌌 COSMIC RAY ANALYSIS\n"
            report += "─" * 40 + "\n"
            report += f"• Total Event Count: {data['total_events']}\n"
            report += f"• Maximum Energy: {data['max_energy']:.1f} GeV\n"
            report += f"• Detection Rate: {data['detection_rate']:.1f} events/s\n"
            report += f"• Detection Time: {data['parameters']['detection_time']} s\n"
            report += f"• Energy Threshold: {data['parameters']['energy_threshold']} GeV\n"
            report += f"• Ray Type: {data['parameters']['ray_type']}\n\n"

        report += "🏆 SCIENTIFIC CONCLUSIONS\n"
        report += "─" * 40 + "\n"
        report += "• Simulations completed successfully\n"
        report += "• Obtained data is suitable for scientific analysis\n"
        report += "• System operated stably and efficiently\n\n"

        report += "🚀 RECOMMENDATIONS\n"
        report += "─" * 40 + "\n"
        report += "• Try longer simulation durations\n"
        report += "• Test different parameter combinations\n"
        report += "• Compare with real experimental data\n"

        self.report_text.setPlainText(report)

    def export_report(self):
        """Export report to text file."""
        if self.report_text.toPlainText():
            filename = f"scientific_report_{time.strftime('%Y%m%d_%H%M%S')}.txt"
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.report_text.toPlainText())
                QMessageBox.information(self, "Success", f"Report saved to '{filename}'.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not save report: {str(e)}")
        else:
            QMessageBox.warning(self, "Warning", "Please create a report first!")


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.simulation_data = {}
        self.init_ui()

    def init_ui(self):
        """Initialize main window UI."""
        self.setWindowTitle("Interactive Scientific Simulation Platform - GitHub: AehoraDeSum")
        self.setGeometry(100, 100, 1600, 1000)
        
        # Set dark theme for main window
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QTabWidget::pane {
                border: 1px solid #444444;
                background-color: #2d2d2d;
                border-radius: 8px;
            }
            QTabBar::tab {
                background-color: #3d3d3d;
                color: #cccccc;
                padding: 10px 20px;
                margin-right: 2px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
            }
            QTabBar::tab:selected {
                background-color: #2d2d2d;
                color: #ffffff;
                font-weight: bold;
            }
            QTabBar::tab:hover:!selected {
                background-color: #4d4d4d;
            }
            QLabel {
                color: #ffffff;
            }
            QStatusBar {
                background-color: #2d2d2d;
                color: #cccccc;
                border-top: 1px solid #444444;
            }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header with GitHub profile
        header_widget = QWidget()
        header_widget.setStyleSheet("background-color: #121212; border-bottom: 2px solid #444444;")
        header_layout = QVBoxLayout()
        header_layout.setContentsMargins(20, 15, 20, 15)
        
        title_label = QLabel("🧬 INTERACTIVE SCIENTIFIC SIMULATION PLATFORM")
        title_label.setFont(QFont("Arial", 22, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #4CAF50; padding: 5px;")
        
        subtitle_label = QLabel("Computational Modeling of Quantum & Cosmic Phenomena")
        subtitle_label.setFont(QFont("Arial", 14))
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setStyleSheet("color: #888888; padding: 5px;")
        
        github_label = QLabel("Developed by: <a href='https://github.com/AehoraDeSum' style='color: #4ECDC4;'>AehoraDeSum</a>")
        github_label.setFont(QFont("Arial", 12))
        github_label.setAlignment(Qt.AlignCenter)
        github_label.setTextFormat(Qt.RichText)
        github_label.setTextInteractionFlags(Qt.TextBrowserInteraction)
        github_label.setOpenExternalLinks(True)
        github_label.setStyleSheet("color: #4ECDC4; padding: 10px; background-color: #2d2d2d; border-radius: 6px; margin: 5px 100px;")
        
        header_layout.addWidget(title_label)
        header_layout.addWidget(subtitle_label)
        header_layout.addWidget(github_label)
        header_widget.setLayout(header_layout)
        
        layout.addWidget(header_widget)

        # Tab widget for different modules
        self.tabs = QTabWidget()
        self.tabs.setFont(QFont("Arial", 10))
        
        self.quantum_tab = QuantumTab(self)
        self.tabs.addTab(self.quantum_tab, "🔬 Quantum Simulation")
        
        self.cosmic_tab = CosmicTab(self)
        self.tabs.addTab(self.cosmic_tab, "🌌 Cosmic Ray Analysis")
        
        self.report_tab = ReportTab(self)
        self.tabs.addTab(self.report_tab, "📊 Scientific Report")
        
        layout.addWidget(self.tabs)

        # Status bar
        status_text = "Ready | GitHub: https://github.com/AehoraDeSum | Scientific Simulation Platform v1.0"
        self.statusBar().showMessage(status_text)

    def closeEvent(self, event):
        """Handle application close event."""
        for tab in [self.quantum_tab, self.cosmic_tab]:
            if hasattr(tab, 'simulation_thread') and tab.simulation_thread and tab.simulation_thread.isRunning():
                tab.simulation_thread.stop()
                tab.simulation_thread.wait()

        event.accept()


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Set dark theme palette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(30, 30, 30))
    palette.setColor(QPalette.WindowText, QColor(220, 220, 220))
    palette.setColor(QPalette.Base, QColor(45, 45, 45))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, QColor(0, 0, 0))
    palette.setColor(QPalette.ToolTipText, QColor(220, 220, 220))
    palette.setColor(QPalette.Text, QColor(220, 220, 220))
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, QColor(220, 220, 220))
    palette.setColor(QPalette.BrightText, QColor(255, 255, 255))
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
    
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor(127, 127, 127))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(127, 127, 127))
    
    app.setPalette(palette)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()