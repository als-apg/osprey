"""
About Dialog for Osprey PyQt GUI

This module provides the About dialog showing version information
and system details for the Osprey Framework.
"""

import platform

from PyQt5.QtCore import PYQT_VERSION_STR, QT_VERSION_STR
from PyQt5.QtWidgets import QDialog, QPushButton, QTextBrowser, QVBoxLayout


def show_about_dialog(parent, gui_version: str):
    """
    Show the about dialog as a non-modal window.

    Args:
        parent: Parent widget (typically the main GUI window)
        gui_version: Version string for the GUI interface
    """
    from osprey.interfaces.pyqt.version_info import get_all_versions

    # Get comprehensive version information
    versions = get_all_versions()
    osprey_version = versions["osprey"]
    python_version = versions["python"]

    qt_version = QT_VERSION_STR
    pyqt_version = PYQT_VERSION_STR
    os_info = f"{platform.system()} {platform.release()}"

    # Build core dependencies HTML
    core_deps_html = ""
    for pkg, ver in versions["core"].items():
        core_deps_html += f"<li>{pkg}: {ver}</li>\n"

    # Build optional dependencies HTML (only installed ones)
    optional_deps_html = ""
    installed_optional = {
        pkg: ver for pkg, ver in versions["optional"].items() if ver != "Not installed"
    }
    if installed_optional:
        for pkg, ver in installed_optional.items():
            optional_deps_html += f"<li>{pkg}: {ver}</li>\n"

    # Create a non-modal dialog
    dialog = QDialog(parent)
    dialog.setWindowTitle("About Osprey Framework")
    dialog.setModal(False)  # Non-modal - can be moved and interact with main window
    dialog.resize(600, 600)

    layout = QVBoxLayout()
    dialog.setLayout(layout)

    # Use QTextBrowser for rich text display
    text_browser = QTextBrowser()
    text_browser.setOpenExternalLinks(True)
    text_browser.setHtml(f"""
        <p><b>Osprey Framework Version:</b> {osprey_version}</p>
        <p><b>PyQt GUI Interface Version:</b> {gui_version}</p>
        <hr>
        <p><b>System Information:</b></p>
        <ul>
        <li>Python: {python_version}</li>
        <li>Qt: {qt_version}</li>
        <li>PyQt: {pyqt_version}</li>
        <li>OS: {os_info}</li>
        </ul>
        <hr>
        <p><b>Core Dependencies:</b></p>
        <ul>
        {core_deps_html}
        </ul>
        {f"<hr><p><b>Optional Dependencies (Installed):</b></p><ul>{optional_deps_html}</ul>" if optional_deps_html else ""}
    """)
    layout.addWidget(text_browser)

    # Close button
    close_button = QPushButton("Close")
    close_button.clicked.connect(dialog.close)
    layout.addWidget(close_button)

    # Show non-modal dialog
    dialog.show()
