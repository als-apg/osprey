"""Plot Viewer Widget for Osprey PyQt GUI.

This module provides a dedicated widget for displaying plots inline
in the conversation, with support for multiple images and click-to-enlarge.
"""

from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from osprey.interfaces.pyqt.image_display import ClickableImageLabel, ImageViewerDialog
from osprey.utils.logger import get_logger

logger = get_logger("plot_viewer")


class PlotViewerWidget(QWidget):
    """Widget for displaying one or more plots inline in the conversation."""

    def __init__(self, image_paths: list[Path], parent=None):
        super().__init__(parent)
        self.image_paths = image_paths
        self.setup_ui()

    def setup_ui(self):
        """Setup the plot viewer UI."""
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        self.setLayout(layout)

        # Set background and border
        self.setStyleSheet("""
            PlotViewerWidget {
                background-color: #2D2D30;
                border: 2px solid #00FFFF;
                border-radius: 5px;
            }
        """)

        # Header
        header = QLabel(
            f"📊 Plot{'s' if len(self.image_paths) > 1 else ''} ({len(self.image_paths)})"
        )
        header.setStyleSheet(
            "color: #00FFFF; font-weight: bold; font-size: 12px; background: transparent; border: none;"
        )
        layout.addWidget(header)

        # Scroll area for images
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background-color: #1E1E1E; border: 1px solid #3F3F46;")

        # Container for images
        images_container = QWidget()
        images_layout = QVBoxLayout()
        images_layout.setSpacing(15)
        images_container.setLayout(images_layout)

        # Add each image
        for image_path in self.image_paths:
            image_widget = self.create_image_display(image_path)
            if image_widget:
                images_layout.addWidget(image_widget)

        scroll.setWidget(images_container)
        layout.addWidget(scroll)

        # Set reasonable size
        self.setMinimumHeight(200)
        self.setMaximumHeight(500)

    def create_image_display(self, image_path: Path) -> QWidget:
        """Create a display widget for a single image."""
        try:
            if not image_path.exists():
                logger.warning(f"Image not found: {image_path}")
                return None

            pixmap = QPixmap(str(image_path))
            if pixmap.isNull():
                logger.warning(f"Failed to load image: {image_path}")
                return None

            # Container
            container = QFrame()
            container.setStyleSheet("""
                QFrame {
                    background-color: #1E1E1E;
                    border: 1px solid #3F3F46;
                    border-radius: 3px;
                    padding: 5px;
                }
            """)
            layout = QVBoxLayout()
            layout.setContentsMargins(5, 5, 5, 5)
            container.setLayout(layout)

            # Image label (clickable)
            image_label = ClickableImageLabel()

            # Scale to fit (max 550px wide, 350px tall for inline display)
            max_width = 550
            max_height = 350
            if pixmap.width() > max_width or pixmap.height() > max_height:
                scaled_pixmap = pixmap.scaled(
                    max_width, max_height, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            else:
                scaled_pixmap = pixmap

            image_label.setPixmap(scaled_pixmap)
            image_label.setAlignment(Qt.AlignCenter)
            image_label.setStyleSheet("""
                QLabel {
                    background-color: #000000;
                    border: 1px solid #3F3F46;
                }
                QLabel:hover {
                    border: 1px solid #00FFFF;
                }
            """)

            # Connect click to open full size (modeless)
            def open_full_size():
                dialog = ImageViewerDialog(image_path, self)
                dialog.show()  # Use show() instead of exec_() for modeless dialog

            image_label.clicked.connect(open_full_size)
            image_label.setToolTip("Click to view full size")

            layout.addWidget(image_label)

            # Info row
            info_layout = QHBoxLayout()

            # Filename
            filename_label = QLabel(f"📄 {image_path.name}")
            filename_label.setStyleSheet("color: #00FFFF; font-size: 10px;")
            info_layout.addWidget(filename_label)

            info_layout.addStretch()

            # Dimensions
            dims_label = QLabel(f"{pixmap.width()}×{pixmap.height()}px")
            dims_label.setStyleSheet("color: #808080; font-size: 9px;")
            info_layout.addWidget(dims_label)

            # View button
            view_btn = QPushButton("🔍 Full Size")
            view_btn.setStyleSheet(
                "background-color: #4A5568; color: #FFFFFF; font-size: 9px; padding: 3px 8px;"
            )
            view_btn.setMaximumWidth(100)
            view_btn.clicked.connect(open_full_size)
            info_layout.addWidget(view_btn)

            # Open file button
            open_btn = QPushButton("📂 Open")
            open_btn.setStyleSheet(
                "background-color: #4A5568; color: #FFFFFF; font-size: 9px; padding: 3px 8px;"
            )
            open_btn.setMaximumWidth(80)

            def open_file():
                import platform
                import subprocess

                try:
                    if platform.system() == "Darwin":  # macOS
                        subprocess.run(["open", str(image_path)])
                    elif platform.system() == "Windows":
                        subprocess.run(["start", str(image_path)], shell=True)
                    else:  # Linux
                        subprocess.run(["xdg-open", str(image_path)])
                except Exception as e:
                    logger.error(f"Failed to open file: {e}")

            open_btn.clicked.connect(open_file)
            info_layout.addWidget(open_btn)

            layout.addLayout(info_layout)

            # Path (truncated)
            path_str = str(image_path)
            if len(path_str) > 80:
                path_str = "..." + path_str[-77:]
            path_label = QLabel(f"📍 {path_str}")
            path_label.setStyleSheet("color: #606060; font-size: 8px; font-family: monospace;")
            path_label.setWordWrap(True)
            layout.addWidget(path_label)

            return container

        except Exception as e:
            logger.error(f"Error creating image display for {image_path}: {e}")
            return None
