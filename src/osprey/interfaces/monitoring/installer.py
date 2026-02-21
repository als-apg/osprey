"""Binary installer for the OSPREY monitoring stack.

Downloads platform-appropriate binaries for OTEL Collector, Prometheus,
and Grafana OSS to ``~/.osprey/monitoring/bin/``.
"""

from __future__ import annotations

import logging
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import click

logger = logging.getLogger(__name__)

# Pinned versions for reproducibility
OTEL_COLLECTOR_VERSION = "0.114.0"
PROMETHEUS_VERSION = "2.54.1"
GRAFANA_VERSION = "11.3.0"

INSTALL_DIR = Path.home() / ".osprey" / "monitoring" / "bin"


def _platform_arch() -> tuple[str, str]:
    """Return (os, arch) strings matching release naming conventions."""
    os_name = {"darwin": "darwin", "linux": "linux", "win32": "windows"}.get(
        sys.platform, sys.platform
    )
    machine = platform.machine().lower()
    arch = {"x86_64": "amd64", "amd64": "amd64", "arm64": "arm64", "aarch64": "arm64"}.get(
        machine, machine
    )
    return os_name, arch


def _download(url: str, dest: Path) -> None:
    """Download a URL to a local path with progress feedback."""
    click.echo(f"  Downloading {url}")
    urlretrieve(url, dest)


def _make_executable(path: Path) -> None:
    """Make a file executable."""
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _clear_quarantine(path: Path) -> None:
    """Remove macOS Gatekeeper quarantine attribute."""
    if sys.platform == "darwin":
        try:
            subprocess.run(
                ["xattr", "-d", "com.apple.quarantine", str(path)],
                capture_output=True,
                check=False,
            )
        except FileNotFoundError:
            pass


def _install_otel_collector(os_name: str, arch: str, force: bool) -> None:
    """Download and install the OTEL Collector Contrib binary."""
    binary = INSTALL_DIR / "otelcol-contrib"
    if binary.exists() and not force:
        click.echo(f"  otelcol-contrib already installed at {binary}")
        return

    ext = "tar.gz" if os_name != "windows" else "zip"
    filename = f"otelcol-contrib_{OTEL_COLLECTOR_VERSION}_{os_name}_{arch}.{ext}"
    url = (
        f"https://github.com/open-telemetry/opentelemetry-collector-releases"
        f"/releases/download/v{OTEL_COLLECTOR_VERSION}/{filename}"
    )

    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / filename
        _download(url, archive)

        if ext == "tar.gz":
            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(tmp, filter="data")
        else:
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(tmp)

        src = Path(tmp) / "otelcol-contrib"
        if not src.exists():
            # Some releases nest inside a directory
            for candidate in Path(tmp).rglob("otelcol-contrib"):
                if candidate.is_file():
                    src = candidate
                    break

        shutil.copy2(src, binary)
        _make_executable(binary)
        _clear_quarantine(binary)

    click.echo(f"  Installed otelcol-contrib {OTEL_COLLECTOR_VERSION}")


def _install_prometheus(os_name: str, arch: str, force: bool) -> None:
    """Download and install the Prometheus binary."""
    binary = INSTALL_DIR / "prometheus"
    if binary.exists() and not force:
        click.echo(f"  prometheus already installed at {binary}")
        return

    dirname = f"prometheus-{PROMETHEUS_VERSION}.{os_name}-{arch}"
    ext = "tar.gz" if os_name != "windows" else "zip"
    filename = f"{dirname}.{ext}"
    url = (
        f"https://github.com/prometheus/prometheus/releases"
        f"/download/v{PROMETHEUS_VERSION}/{filename}"
    )

    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / filename
        _download(url, archive)

        if ext == "tar.gz":
            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(tmp, filter="data")
        else:
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(tmp)

        src = Path(tmp) / dirname / "prometheus"
        shutil.copy2(src, binary)
        _make_executable(binary)
        _clear_quarantine(binary)

    click.echo(f"  Installed prometheus {PROMETHEUS_VERSION}")


def _install_grafana(os_name: str, arch: str, force: bool) -> None:
    """Download and install the Grafana OSS standalone binary."""
    binary = INSTALL_DIR / "grafana"
    if binary.exists() and not force:
        click.echo(f"  grafana already installed at {binary}")
        return

    # Grafana uses slightly different naming
    gf_os = {"darwin": "darwin", "linux": "linux", "windows": "windows"}.get(os_name, os_name)
    filename = f"grafana-{GRAFANA_VERSION}.{gf_os}-{arch}.tar.gz"
    url = f"https://dl.grafana.com/oss/release/{filename}"

    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / filename
        _download(url, archive)

        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(tmp, filter="data")

        # Grafana extracts to grafana-vX.Y.Z/ or grafana-X.Y.Z/
        grafana_dir = None
        for candidate in Path(tmp).iterdir():
            if candidate.is_dir() and candidate.name.startswith("grafana"):
                grafana_dir = candidate
                break

        if grafana_dir is None:
            raise FileNotFoundError("Could not find grafana directory in archive")

        src = grafana_dir / "bin" / "grafana"
        if not src.exists():
            src = grafana_dir / "bin" / "grafana-server"

        shutil.copy2(src, binary)
        _make_executable(binary)
        _clear_quarantine(binary)

        # Also copy the public/ and conf/ directories needed by Grafana
        grafana_home = INSTALL_DIR.parent / "grafana-home"
        if grafana_home.exists():
            shutil.rmtree(grafana_home)
        grafana_home.mkdir(parents=True)

        for subdir in ("public", "conf"):
            src_dir = grafana_dir / subdir
            if src_dir.exists():
                shutil.copytree(src_dir, grafana_home / subdir)

    click.echo(f"  Installed grafana {GRAFANA_VERSION}")


def install_monitoring_binaries(force: bool = False) -> None:
    """Download and install all monitoring stack binaries.

    Args:
        force: Re-download even if binaries already exist.
    """
    INSTALL_DIR.mkdir(parents=True, exist_ok=True)
    os_name, arch = _platform_arch()

    click.echo(f"Installing monitoring binaries for {os_name}/{arch}...")
    click.echo(f"Install directory: {INSTALL_DIR}")

    _install_otel_collector(os_name, arch, force)
    _install_prometheus(os_name, arch, force)
    _install_grafana(os_name, arch, force)

    click.echo("All monitoring binaries installed.")


def resolve_binary(name: str) -> Path | None:
    """Resolve a monitoring binary, preferring PATH over ~/.osprey.

    Args:
        name: Binary name (e.g. "prometheus", "otelcol-contrib", "grafana").

    Returns:
        Path to the binary, or None if not found.
    """
    # Check PATH first
    on_path = shutil.which(name)
    if on_path:
        return Path(on_path)

    # Fall back to installed location
    installed = INSTALL_DIR / name
    if installed.exists() and installed.is_file():
        return installed

    return None
