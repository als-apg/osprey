"""One live server per OSPREY interface, for the browser suites.

Each launcher is a context manager yielding the server's base URL. It isolates
cwd to ``tmp_path`` first (matters for ariel's config lookup and artifacts'
index dir) and builds the app with the same signature the interface's own
tests use (see ``test_app_setup_parametrized.py``).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from osprey.interfaces._serving import run_app_server
from tests.interfaces._hub import hub_server


@contextmanager
def launch_web_terminal(tmp_path, monkeypatch) -> Iterator[str]:
    monkeypatch.chdir(tmp_path)
    with hub_server(tmp_path) as (base_url, _app):
        yield base_url


@contextmanager
def launch_artifacts(tmp_path, monkeypatch) -> Iterator[str]:
    monkeypatch.chdir(tmp_path)
    from osprey.interfaces.artifacts.app import create_app

    app = create_app(workspace_root=tmp_path)
    with run_app_server(app) as base_url:
        yield base_url


@contextmanager
def launch_ariel(tmp_path, monkeypatch) -> Iterator[str]:
    monkeypatch.chdir(tmp_path)
    from osprey.interfaces.ariel.app import create_app

    app = create_app()
    with run_app_server(app) as base_url:
        yield base_url


@contextmanager
def launch_channel_finder(tmp_path, monkeypatch) -> Iterator[str]:
    monkeypatch.chdir(tmp_path)
    from osprey.interfaces.channel_finder.app import create_app

    app = create_app(project_cwd=str(tmp_path))
    with run_app_server(app) as base_url:
        yield base_url


@contextmanager
def launch_lattice_dashboard(tmp_path, monkeypatch) -> Iterator[str]:
    monkeypatch.chdir(tmp_path)
    from osprey.interfaces.lattice_dashboard.app import create_app

    app = create_app(workspace_root=tmp_path)
    with run_app_server(app) as base_url:
        yield base_url


@contextmanager
def launch_okf_panel(tmp_path, monkeypatch) -> Iterator[str]:
    monkeypatch.chdir(tmp_path)
    from osprey.interfaces.okf_panel.app import create_app

    # Build over the shared on-disk fixture bundle the panel's own unit tests use
    # (tests/interfaces/okf_panel/fixtures/bundle) so GET / renders real content.
    bundle = Path(__file__).parent / "okf_panel" / "fixtures" / "bundle"
    app = create_app(bundle_path=str(bundle))
    with run_app_server(app) as base_url:
        yield base_url
