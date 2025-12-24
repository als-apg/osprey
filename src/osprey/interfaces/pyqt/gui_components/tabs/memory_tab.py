"""Memory Monitoring Tab for Osprey GUI.

This tab monitors memory usage for:
- GUI process itself
- Framework-spawned processes
- Docker/Podman containers started by the framework
"""

import psutil
import subprocess
from datetime import datetime
from typing import Dict, List, Optional
from collections import deque
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox,
    QProgressBar
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QColor, QFont

from osprey.utils.logger import get_logger
from osprey.interfaces.pyqt.gui_constants import MEMORY, COLORS

logger = get_logger("memory_tab")


class MemoryTab(QWidget):
    """Tab for monitoring memory usage of GUI and framework processes."""
    
    def __init__(self, parent=None):
        """Initialize the Memory Monitoring tab.
        
        Args:
            parent: Parent OspreyGUI instance
        """
        super().__init__(parent)
        self.parent_gui = parent
        self.gui_process = psutil.Process()
        self.tracked_pids = set()  # PIDs we're tracking
        self.container_ids = set()  # Container IDs we're tracking
        self.monitoring_enabled = False
        self.update_timer = None
        
        # Memory trend tracking
        self.memory_history = deque(maxlen=20)  # Keep last 20 measurements
        self.last_trend_check = None
        self.trend_start_time = datetime.now()  # Track when monitoring started
        self.trend_warmup_seconds = 120  # Wait 2 minutes before showing trend
        
        self.setup_ui()
        
        # Start monitoring if enabled in settings
        if self.parent_gui and self.parent_gui.settings_manager.get('memory_monitor_enabled', True):
            self.start_monitoring()
    
    def setup_ui(self):
        """Setup the memory monitoring tab UI."""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Header with controls
        header_layout = QHBoxLayout()
        
        title_label = QLabel("Memory Monitor")
        title_label.setStyleSheet("color: #00FF00; font-weight: bold; font-size: 14px;")
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        # Start/Stop button
        self.toggle_button = QPushButton("⏸️ Pause")
        self.toggle_button.setCheckable(True)
        self.toggle_button.setStyleSheet("background-color: #4A5568; color: #FFFFFF;")
        self.toggle_button.clicked.connect(self.toggle_monitoring)
        header_layout.addWidget(self.toggle_button)
        
        # Refresh button
        refresh_button = QPushButton("🔄 Refresh")
        refresh_button.setStyleSheet("background-color: #4A5568; color: #FFFFFF;")
        refresh_button.clicked.connect(self.update_memory_stats)
        header_layout.addWidget(refresh_button)
        
        layout.addLayout(header_layout)
        
        # Summary section
        summary_group = QGroupBox("System Memory Summary")
        summary_group.setStyleSheet("""
            QGroupBox {
                color: #00FFFF;
                font-weight: bold;
                border: 1px solid #3F3F46;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        summary_layout = QVBoxLayout()
        summary_group.setLayout(summary_layout)
        
        # System memory bar
        system_mem_layout = QHBoxLayout()
        system_mem_label = QLabel("System Memory:")
        system_mem_label.setStyleSheet("color: #FFFFFF;")
        system_mem_layout.addWidget(system_mem_label)
        
        self.system_memory_bar = QProgressBar()
        self.system_memory_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #3F3F46;
                border-radius: 3px;
                text-align: center;
                background-color: #1E1E1E;
                color: #FFFFFF;
            }
            QProgressBar::chunk {
                background-color: #00FF00;
            }
        """)
        system_mem_layout.addWidget(self.system_memory_bar)
        
        self.system_memory_label = QLabel("0 MB / 0 MB (0%)")
        self.system_memory_label.setStyleSheet("color: #FFFFFF; font-family: monospace;")
        self.system_memory_label.setMinimumWidth(200)
        system_mem_layout.addWidget(self.system_memory_label)
        
        summary_layout.addLayout(system_mem_layout)
        
        # Framework total memory
        framework_mem_layout = QHBoxLayout()
        framework_mem_label = QLabel("Framework Total:")
        framework_mem_label.setStyleSheet("color: #FFFFFF;")
        framework_mem_layout.addWidget(framework_mem_label)
        
        self.framework_memory_bar = QProgressBar()
        self.framework_memory_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #3F3F46;
                border-radius: 3px;
                text-align: center;
                background-color: #1E1E1E;
                color: #FFFFFF;
            }
            QProgressBar::chunk {
                background-color: #00FFFF;
            }
        """)
        framework_mem_layout.addWidget(self.framework_memory_bar)
        
        self.framework_memory_label = QLabel("0 MB")
        self.framework_memory_label.setStyleSheet("color: #00FFFF; font-family: monospace;")
        self.framework_memory_label.setMinimumWidth(200)
        framework_mem_layout.addWidget(self.framework_memory_label)
        
        summary_layout.addLayout(framework_mem_layout)
        
        # Warning/Critical thresholds info (store as instance variable for updates)
        self.threshold_label = QLabel()
        warning_mb = self.parent_gui.settings_manager.get('memory_warning_threshold_mb', MEMORY.WARNING_THRESHOLD_MB) if self.parent_gui else MEMORY.WARNING_THRESHOLD_MB
        critical_mb = self.parent_gui.settings_manager.get('memory_critical_threshold_mb', MEMORY.CRITICAL_THRESHOLD_MB) if self.parent_gui else MEMORY.CRITICAL_THRESHOLD_MB
        self.threshold_label.setText(
            f"⚠️ Warning: {warning_mb} MB  |  🔴 Critical: {critical_mb} MB"
        )
        self.threshold_label.setStyleSheet("color: #FFA500; font-size: 10px;")
        summary_layout.addWidget(self.threshold_label)
        
        # Memory trend indicator
        trend_layout = QHBoxLayout()
        trend_label = QLabel("Memory Trend:")
        trend_label.setStyleSheet("color: #FFFFFF;")
        trend_layout.addWidget(trend_label)
        
        self.trend_indicator = QLabel("Collecting data...")
        self.trend_indicator.setStyleSheet("color: #808080; font-family: monospace;")
        trend_layout.addWidget(self.trend_indicator)
        
        trend_layout.addStretch()
        summary_layout.addLayout(trend_layout)
        
        layout.addWidget(summary_group)
        
        # Process table
        process_group = QGroupBox("Process Details")
        process_group.setStyleSheet("""
            QGroupBox {
                color: #FFD700;
                font-weight: bold;
                border: 1px solid #3F3F46;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        process_layout = QVBoxLayout()
        process_group.setLayout(process_layout)
        
        self.process_table = QTableWidget()
        self.process_table.setColumnCount(6)
        self.process_table.setHorizontalHeaderLabels([
            'Type', 'PID/ID', 'Name', 'Memory (MB)', 'CPU %', 'Status'
        ])
        self.process_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.process_table.setStyleSheet("""
            QTableWidget {
                background-color: #1E1E1E;
                color: #FFFFFF;
                border: 1px solid #3F3F46;
                gridline-color: #3F3F46;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QHeaderView::section {
                background-color: #2D2D30;
                color: #FFFFFF;
                padding: 5px;
                border: 1px solid #3F3F46;
                font-weight: bold;
            }
        """)
        self.process_table.setAlternatingRowColors(True)
        self.process_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.process_table.setSelectionBehavior(QTableWidget.SelectRows)
        
        process_layout.addWidget(self.process_table)
        
        layout.addWidget(process_group)
        
        # Status bar at bottom
        self.status_label = QLabel("Monitoring paused")
        self.status_label.setStyleSheet("color: #808080; font-size: 10px; padding: 5px;")
        layout.addWidget(self.status_label)
    
    def start_monitoring(self):
        """Start automatic memory monitoring."""
        if self.update_timer is None:
            self.update_timer = QTimer(self)
            self.update_timer.timeout.connect(self.update_memory_stats)
        
        interval_seconds = self.parent_gui.settings_manager.get(
            'memory_check_interval_seconds', 
            MEMORY.CHECK_INTERVAL_SECONDS
        ) if self.parent_gui else MEMORY.CHECK_INTERVAL_SECONDS
        
        self.update_timer.start(interval_seconds * 1000)
        self.monitoring_enabled = True
        self.toggle_button.setText("⏸️ Pause")
        self.toggle_button.setChecked(False)
        self.status_label.setText(f"Monitoring active (updates every {interval_seconds}s)")
        self.status_label.setStyleSheet("color: #00FF00; font-size: 10px; padding: 5px;")
        
        # Initial update
        self.update_memory_stats()
    
    def stop_monitoring(self):
        """Stop automatic memory monitoring."""
        if self.update_timer:
            self.update_timer.stop()
        
        self.monitoring_enabled = False
        self.toggle_button.setText("▶️ Resume")
        self.toggle_button.setChecked(True)
        self.status_label.setText("Monitoring paused")
        self.status_label.setStyleSheet("color: #808080; font-size: 10px; padding: 5px;")
    
    def toggle_monitoring(self):
        """Toggle monitoring on/off."""
        if self.monitoring_enabled:
            self.stop_monitoring()
        else:
            self.start_monitoring()
    
    def update_memory_stats(self):
        """Update all memory statistics."""
        try:
            # Update system memory
            self._update_system_memory()
            
            # Collect process information
            processes = self._collect_process_info()
            
            # Update process table
            self._update_process_table(processes)
            
            # Update framework total
            total_framework_mb = sum(p['memory_mb'] for p in processes)
            self._update_framework_memory(total_framework_mb)
            
            # Update memory trend
            self._update_memory_trend(total_framework_mb)
            
            # Check thresholds and update status
            self._check_thresholds(total_framework_mb)
            
        except Exception as e:
            logger.error(f"Error updating memory stats: {e}")
            self.status_label.setText(f"Error: {e}")
            self.status_label.setStyleSheet("color: #FF0000; font-size: 10px; padding: 5px;")
    
    def _update_system_memory(self):
        """Update system memory display."""
        mem = psutil.virtual_memory()
        used_mb = mem.used / (1024 * 1024)
        total_mb = mem.total / (1024 * 1024)
        percent = mem.percent
        
        self.system_memory_bar.setValue(int(percent))
        self.system_memory_label.setText(f"{used_mb:.0f} MB / {total_mb:.0f} MB ({percent:.1f}%)")
        
        # Color code based on usage
        if percent > 90:
            chunk_color = '#FF0000'
        elif percent > 75:
            chunk_color = '#FFA500'
        else:
            chunk_color = '#00FF00'
        
        self.system_memory_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid #3F3F46;
                border-radius: 3px;
                text-align: center;
                background-color: #1E1E1E;
                color: #FFFFFF;
            }}
            QProgressBar::chunk {{
                background-color: {chunk_color};
            }}
        """)
    
    def _collect_process_info(self) -> List[Dict]:
        """Collect information about all tracked processes.
        
        Returns:
            List of process info dictionaries
        """
        processes = []
        
        # Add GUI process
        try:
            gui_info = self._get_process_info(self.gui_process, 'GUI')
            if gui_info:
                processes.append(gui_info)
        except Exception as e:
            logger.debug(f"Error getting GUI process info: {e}")
        
        # Add child processes
        try:
            for child in self.gui_process.children(recursive=True):
                child_info = self._get_process_info(child, 'Child')
                if child_info:
                    processes.append(child_info)
                    self.tracked_pids.add(child.pid)
        except Exception as e:
            logger.debug(f"Error getting child processes: {e}")
        
        # Add Docker containers
        docker_containers = self._get_docker_containers()
        processes.extend(docker_containers)
        
        # Add Podman containers
        podman_containers = self._get_podman_containers()
        processes.extend(podman_containers)
        
        return processes
    
    def _get_process_info(self, process: psutil.Process, proc_type: str) -> Optional[Dict]:
        """Get information about a single process.
        
        Args:
            process: psutil.Process instance
            proc_type: Type label for the process
            
        Returns:
            Dictionary with process info or None if error
        """
        try:
            mem_info = process.memory_info()
            memory_mb = mem_info.rss / (1024 * 1024)
            cpu_percent = process.cpu_percent(interval=0.1)
            
            return {
                'type': proc_type,
                'pid': process.pid,
                'name': process.name(),
                'memory_mb': memory_mb,
                'cpu_percent': cpu_percent,
                'status': process.status()
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None
    
    def _get_docker_containers(self) -> List[Dict]:
        """Get memory usage of Docker containers started by framework.
        
        Returns:
            List of container info dictionaries
        """
        containers = []
        try:
            result = subprocess.run(
                ['docker', 'ps', '--format', '{{.ID}}|{{.Names}}|{{.Status}}'],
                capture_output=True,
                text=True,
                timeout=2
            )
            
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if not line:
                        continue
                    
                    parts = line.split('|')
                    if len(parts) >= 3:
                        container_id = parts[0]
                        name = parts[1]
                        status = parts[2]
                        
                        # Get memory stats
                        stats_result = subprocess.run(
                            ['docker', 'stats', container_id, '--no-stream', '--format', '{{.MemUsage}}'],
                            capture_output=True,
                            text=True,
                            timeout=2
                        )
                        
                        if stats_result.returncode == 0:
                            mem_usage = stats_result.stdout.strip()
                            try:
                                used_str = mem_usage.split('/')[0].strip()
                                if 'MiB' in used_str:
                                    memory_mb = float(used_str.replace('MiB', ''))
                                elif 'GiB' in used_str:
                                    memory_mb = float(used_str.replace('GiB', '')) * 1024
                                else:
                                    memory_mb = 0
                                
                                containers.append({
                                    'type': 'Docker',
                                    'pid': container_id[:12],
                                    'name': name,
                                    'memory_mb': memory_mb,
                                    'cpu_percent': 0,
                                    'status': status
                                })
                                self.container_ids.add(container_id)
                            except:
                                pass
        
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        except Exception as e:
            logger.debug(f"Error getting Docker containers: {e}")
        
        return containers
    
    def _get_podman_containers(self) -> List[Dict]:
        """Get memory usage of Podman containers started by framework.
        
        Returns:
            List of container info dictionaries
        """
        containers = []
        try:
            result = subprocess.run(
                ['podman', 'ps', '--format', '{{.ID}}|{{.Names}}|{{.Status}}'],
                capture_output=True,
                text=True,
                timeout=2
            )
            
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if not line:
                        continue
                    
                    parts = line.split('|')
                    if len(parts) >= 3:
                        container_id = parts[0]
                        name = parts[1]
                        status = parts[2]
                        
                        # Get memory stats
                        stats_result = subprocess.run(
                            ['podman', 'stats', container_id, '--no-stream', '--format', '{{.MemUsage}}'],
                            capture_output=True,
                            text=True,
                            timeout=2
                        )
                        
                        if stats_result.returncode == 0:
                            mem_usage = stats_result.stdout.strip()
                            try:
                                used_str = mem_usage.split('/')[0].strip()
                                if 'MB' in used_str:
                                    memory_mb = float(used_str.replace('MB', ''))
                                elif 'GB' in used_str:
                                    memory_mb = float(used_str.replace('GB', '')) * 1024
                                else:
                                    memory_mb = 0
                                
                                containers.append({
                                    'type': 'Podman',
                                    'pid': container_id[:12],
                                    'name': name,
                                    'memory_mb': memory_mb,
                                    'cpu_percent': 0,
                                    'status': status
                                })
                                self.container_ids.add(container_id)
                            except:
                                pass
        
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        except Exception as e:
            logger.debug(f"Error getting Podman containers: {e}")
        
        return containers
    
    def _update_process_table(self, processes: List[Dict]):
        """Update the process table with current data.
        
        Args:
            processes: List of process info dictionaries
        """
        self.process_table.setRowCount(len(processes))
        
        for row, proc in enumerate(processes):
            # Type
            type_item = QTableWidgetItem(proc['type'])
            if proc['type'] == 'GUI':
                type_item.setForeground(QColor(COLORS.INFO_MESSAGE))
            elif proc['type'] in ('Docker', 'Podman'):
                type_item.setForeground(QColor('#FF00FF'))
            else:
                type_item.setForeground(QColor(COLORS.WARNING_MESSAGE))
            self.process_table.setItem(row, 0, type_item)
            
            # PID/ID
            pid_item = QTableWidgetItem(str(proc['pid']))
            pid_item.setForeground(QColor('#FFFFFF'))
            pid_item.setFont(QFont("Monospace", 9))
            self.process_table.setItem(row, 1, pid_item)
            
            # Name
            name_item = QTableWidgetItem(proc['name'])
            name_item.setForeground(QColor('#00FFFF'))
            self.process_table.setItem(row, 2, name_item)
            
            # Memory
            memory_item = QTableWidgetItem(f"{proc['memory_mb']:.1f}")
            memory_item.setForeground(QColor('#FFD700'))
            memory_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.process_table.setItem(row, 3, memory_item)
            
            # CPU
            cpu_item = QTableWidgetItem(f"{proc['cpu_percent']:.1f}")
            cpu_item.setForeground(QColor('#00FF00'))
            cpu_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.process_table.setItem(row, 4, cpu_item)
            
            # Status
            status_item = QTableWidgetItem(proc['status'])
            status_item.setForeground(QColor('#808080'))
            self.process_table.setItem(row, 5, status_item)
    
    def _update_framework_memory(self, total_mb: float):
        """Update framework total memory display.
        
        Args:
            total_mb: Total memory usage in MB
        """
        system_total_mb = psutil.virtual_memory().total / (1024 * 1024)
        percent = (total_mb / system_total_mb) * 100
        
        self.framework_memory_bar.setValue(int(min(percent, 100)))
        self.framework_memory_label.setText(f"{total_mb:.1f} MB ({percent:.2f}% of system)")
        
        # Color code based on thresholds
        warning_mb = self.parent_gui.settings_manager.get(
            'memory_warning_threshold_mb', 
            MEMORY.WARNING_THRESHOLD_MB
        ) if self.parent_gui else MEMORY.WARNING_THRESHOLD_MB
        
        critical_mb = self.parent_gui.settings_manager.get(
            'memory_critical_threshold_mb',
            MEMORY.CRITICAL_THRESHOLD_MB
        ) if self.parent_gui else MEMORY.CRITICAL_THRESHOLD_MB
        
        if total_mb >= critical_mb:
            color = '#FF0000'
        elif total_mb >= warning_mb:
            color = '#FFA500'
        else:
            color = '#00FFFF'
        
        self.framework_memory_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid #3F3F46;
                border-radius: 3px;
                text-align: center;
                background-color: #1E1E1E;
                color: #FFFFFF;
            }}
            QProgressBar::chunk {{
                background-color: {color};
            }}
        """)
    
    def _update_memory_trend(self, total_mb: float):
        """Update memory trend analysis.
        
        Args:
            total_mb: Current total framework memory usage in MB
        """
        # Add current measurement to history
        timestamp = datetime.now()
        self.memory_history.append({
            'timestamp': timestamp,
            'memory_mb': total_mb
        })
        
        # Check if we're still in warmup period (2 minutes after startup)
        elapsed_seconds = (timestamp - self.trend_start_time).total_seconds()
        
        if elapsed_seconds < self.trend_warmup_seconds:
            # Still in warmup - show countdown
            remaining = self.trend_warmup_seconds - elapsed_seconds
            minutes = int(remaining // 60)
            seconds = int(remaining % 60)
            self.trend_indicator.setText(
                f"Warming up... {minutes}m {seconds}s remaining (collecting baseline data)"
            )
            self.trend_indicator.setStyleSheet("color: #808080; font-family: monospace;")
            return
        
        # Need at least 5 measurements for trend analysis
        if len(self.memory_history) < 5:
            self.trend_indicator.setText(f"Collecting data... ({len(self.memory_history)}/5)")
            self.trend_indicator.setStyleSheet("color: #808080; font-family: monospace;")
            return
        
        # Calculate trend
        trend_info = self._calculate_trend()
        
        # Update display
        if trend_info['direction'] == 'increasing':
            icon = "📈"
            color = "#FFA500"  # Orange for increasing
            if trend_info['rate'] > 10:  # More than 10 MB/min
                icon = "⚠️📈"
                color = "#FF6B6B"  # Bright coral red for rapidly increasing (more readable)
        elif trend_info['direction'] == 'decreasing':
            icon = "📉"
            color = "#00FF00"  # Green for decreasing
        else:  # stable
            icon = "➡️"
            color = "#00FFFF"  # Cyan for stable
        
        trend_text = (
            f"{icon} {trend_info['direction'].upper()} "
            f"({trend_info['rate']:+.1f} MB/min over {trend_info['duration']:.0f}s)"
        )
        
        self.trend_indicator.setText(trend_text)
        self.trend_indicator.setStyleSheet(f"color: {color}; font-family: monospace; font-weight: bold;")
    
    def _calculate_trend(self) -> Dict:
        """Calculate memory usage trend from history.
        
        Returns:
            Dictionary with trend information
        """
        if len(self.memory_history) < 2:
            return {
                'direction': 'unknown',
                'rate': 0.0,
                'duration': 0.0
            }
        
        # Get first and last measurements
        first = self.memory_history[0]
        last = self.memory_history[-1]
        
        # Calculate time difference in seconds
        duration = (last['timestamp'] - first['timestamp']).total_seconds()
        
        if duration == 0:
            return {
                'direction': 'stable',
                'rate': 0.0,
                'duration': 0.0
            }
        
        # Calculate memory change
        memory_change = last['memory_mb'] - first['memory_mb']
        
        # Calculate rate in MB per minute
        rate_per_minute = (memory_change / duration) * 60
        
        # Determine direction (threshold: 0.5 MB/min to avoid noise)
        if rate_per_minute > 0.5:
            direction = 'increasing'
        elif rate_per_minute < -0.5:
            direction = 'decreasing'
        else:
            direction = 'stable'
        
        return {
            'direction': direction,
            'rate': rate_per_minute,
            'duration': duration
        }
    
    def _check_thresholds(self, total_mb: float):
        """Check memory thresholds and update status.
        
        Args:
            total_mb: Total framework memory usage in MB
        """
        warning_mb = self.parent_gui.settings_manager.get(
            'memory_warning_threshold_mb',
            MEMORY.WARNING_THRESHOLD_MB
        ) if self.parent_gui else MEMORY.WARNING_THRESHOLD_MB
        
        critical_mb = self.parent_gui.settings_manager.get(
            'memory_critical_threshold_mb',
            MEMORY.CRITICAL_THRESHOLD_MB
        ) if self.parent_gui else MEMORY.CRITICAL_THRESHOLD_MB
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if total_mb >= critical_mb:
            status_text = f"🔴 CRITICAL: Framework using {total_mb:.1f} MB (threshold: {critical_mb} MB) - {timestamp}"
            self.status_label.setText(status_text)
            self.status_label.setStyleSheet("color: #FF0000; font-size: 10px; padding: 5px; font-weight: bold;")
            logger.warning(f"Memory usage critical: {total_mb:.1f} MB")
        elif total_mb >= warning_mb:
            status_text = f"⚠️ WARNING: Framework using {total_mb:.1f} MB (threshold: {warning_mb} MB) - {timestamp}"
            self.status_label.setText(status_text)
            self.status_label.setStyleSheet("color: #FFA500; font-size: 10px; padding: 5px; font-weight: bold;")
        elif self.monitoring_enabled:
            interval_seconds = self.parent_gui.settings_manager.get(
                'memory_check_interval_seconds',
                MEMORY.CHECK_INTERVAL_SECONDS
            ) if self.parent_gui else MEMORY.CHECK_INTERVAL_SECONDS
            status_text = f"✅ Normal: {total_mb:.1f} MB - Last update: {timestamp} (updates every {interval_seconds}s)"
            self.status_label.setText(status_text)
            self.status_label.setStyleSheet("color: #00FF00; font-size: 10px; padding: 5px;")
    
    def update_threshold_display(self):
        """Update the threshold display with current settings."""
        if not self.parent_gui:
            return
        
        warning_mb = self.parent_gui.settings_manager.get('memory_warning_threshold_mb', MEMORY.WARNING_THRESHOLD_MB)
        critical_mb = self.parent_gui.settings_manager.get('memory_critical_threshold_mb', MEMORY.CRITICAL_THRESHOLD_MB)
        
        self.threshold_label.setText(
            f"⚠️ Warning: {warning_mb} MB  |  🔴 Critical: {critical_mb} MB"
        )
        
        # Force an immediate update of memory stats to reflect new thresholds
        self.update_memory_stats()
    
    def closeEvent(self, event):
        """Handle tab close - stop monitoring."""
        self.stop_monitoring()
        super().closeEvent(event)