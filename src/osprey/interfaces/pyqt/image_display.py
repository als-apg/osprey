"""Image Display Handler for Osprey PyQt GUI.

This module provides utilities for detecting and displaying images (plots)
in the conversation display, with support for agent-specific plot directories.
"""

import re
from pathlib import Path
from typing import Optional
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget, QPushButton, QDialog
from PyQt5.QtGui import QPixmap, QCursor
from PyQt5.QtCore import Qt, pyqtSignal

from osprey.utils.logger import get_logger

logger = get_logger("image_display")


class ImageViewerDialog(QDialog):
    """Modeless dialog for viewing images in full size."""

    def __init__(self, image_path: Path, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Image Viewer - {image_path.name}")
        self.setModal(False)  # Make it modeless so user can interact with main window
        self.setAttribute(Qt.WA_DeleteOnClose)  # Clean up when closed

        layout = QVBoxLayout()
        self.setLayout(layout)

        # Image label
        image_label = QLabel()
        pixmap = QPixmap(str(image_path))

        # Scale to reasonable size while maintaining aspect ratio
        max_width = 1200
        max_height = 900
        if pixmap.width() > max_width or pixmap.height() > max_height:
            pixmap = pixmap.scaled(max_width, max_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        image_label.setPixmap(pixmap)
        image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(image_label)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)

        # Set dialog size
        self.resize(pixmap.width() + 40, pixmap.height() + 80)


class ClickableImageLabel(QLabel):
    """QLabel that emits a signal when clicked."""

    clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCursor(QCursor(Qt.PointingHandCursor))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class ImageDisplayHandler:
    """Handles detection and display of images in conversation messages."""

    # Patterns to detect image references in messages
    IMAGE_PATH_PATTERNS = [
        r'(?:at|File|Plot|Image|path|saved at):\s*([^\s]+\.(?:png|jpg|jpeg|gif|svg))',  # "at path/to/image.png" or "File: path/to/image.png"
        r'([^\s]*_agent_data/plots/[^\s]+\.(?:png|jpg|jpeg|gif))',  # Direct path to plots (with or without prefix)
        r'`([^`]+\.(?:png|jpg|jpeg|gif|svg))`',  # Backtick-wrapped paths
        r'saved (?:at|to)\s+([^\s]+\.(?:png|jpg|jpeg|gif|svg))',  # "saved at path/to/image.png"
    ]

    @staticmethod
    def extract_image_paths(message: str, agent_name: Optional[str] = None) -> list[Path]:
        """
        Extract image paths from a message.

        Args:
            message: The message text to search
            agent_name: Optional agent name to help locate agent-specific plots

        Returns:
            List of Path objects for found images
        """
        image_paths = []

        # Try each pattern
        for pattern in ImageDisplayHandler.IMAGE_PATH_PATTERNS:
            matches = re.finditer(pattern, message, re.IGNORECASE)
            for match in matches:
                path_str = match.group(1)
                image_path = Path(path_str)

                # If path is relative, try to resolve it
                if not image_path.is_absolute():
                    # Try relative to current working directory
                    if image_path.exists():
                        image_paths.append(image_path.resolve())
                        continue

                    # Try relative to agent directory if agent_name provided
                    if agent_name:
                        agent_path = Path(agent_name) / path_str
                        if agent_path.exists():
                            image_paths.append(agent_path.resolve())
                            continue

                    # Try common locations
                    for base_dir in [Path.cwd(), Path.cwd() / "aps-control-assistant", Path.cwd() / "its-control-assistant"]:
                        full_path = base_dir / path_str
                        if full_path.exists():
                            image_paths.append(full_path.resolve())
                            break
                elif image_path.exists():
                    image_paths.append(image_path)

        return image_paths

    @staticmethod
    def create_image_widget(image_path: Path, max_width: int = 600, max_height: int = 400) -> Optional[QWidget]:
        """
        Create a widget displaying an image with click-to-enlarge functionality.

        Args:
            image_path: Path to the image file
            max_width: Maximum width for thumbnail
            max_height: Maximum height for thumbnail

        Returns:
            QWidget containing the image, or None if image cannot be loaded
        """
        try:
            if not image_path.exists():
                logger.warning(f"Image file not found: {image_path}")
                return None

            pixmap = QPixmap(str(image_path))
            if pixmap.isNull():
                logger.warning(f"Failed to load image: {image_path}")
                return None

            # Create container widget
            container = QWidget()
            layout = QVBoxLayout()
            layout.setContentsMargins(5, 5, 5, 5)
            container.setLayout(layout)

            # Create clickable image label
            image_label = ClickableImageLabel()

            # Scale image to fit max dimensions while maintaining aspect ratio
            if pixmap.width() > max_width or pixmap.height() > max_height:
                scaled_pixmap = pixmap.scaled(max_width, max_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            else:
                scaled_pixmap = pixmap

            image_label.setPixmap(scaled_pixmap)
            image_label.setAlignment(Qt.AlignCenter)
            image_label.setStyleSheet("""
                QLabel {
                    border: 2px solid #3F3F46;
                    background-color: #1E1E1E;
                    padding: 5px;
                }
                QLabel:hover {
                    border: 2px solid #00FFFF;
                }
            """)

            # Connect click to open full-size viewer
            def open_full_size():
                dialog = ImageViewerDialog(image_path, container)
                dialog.exec_()

            image_label.clicked.connect(open_full_size)
            image_label.setToolTip(f"Click to view full size\n{image_path.name}")

            layout.addWidget(image_label)

            # Add caption with filename
            caption = QLabel(f"📊 {image_path.name}")
            caption.setStyleSheet("color: #00FFFF; font-size: 10px; font-style: italic;")
            caption.setAlignment(Qt.AlignCenter)
            layout.addWidget(caption)

            # Add "Open in viewer" button
            open_btn = QPushButton("🔍 View Full Size")
            open_btn.setMaximumWidth(150)
            open_btn.setStyleSheet("background-color: #4A5568; color: #FFFFFF; font-size: 9px;")
            open_btn.clicked.connect(open_full_size)
            layout.addWidget(open_btn, alignment=Qt.AlignCenter)

            logger.info(f"Created image widget for: {image_path}")
            return container

        except Exception as e:
            logger.error(f"Error creating image widget for {image_path}: {e}")
            return None

    @staticmethod
    def should_display_inline(message: str) -> bool:
        """
        Check if a message likely contains image references that should be displayed.

        Args:
            message: The message text

        Returns:
            True if message appears to reference images
        """
        # Check for common image-related keywords
        image_keywords = ['plot', 'image', 'chart', 'graph', 'figure', '.png', '.jpg', '.jpeg']
        message_lower = message.lower()

        return any(keyword in message_lower for keyword in image_keywords)
