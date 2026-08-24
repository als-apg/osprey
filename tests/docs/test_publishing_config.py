"""Gate: ``docs/source/conf.py`` derives its version from ``osprey.version``.

The published site keys everything off Sphinx's ``release``: the root shows the
newest tag, ``main`` publishes at ``/latest/`` as the literal ``dev`` so the
pydata-sphinx-theme development banner and the switcher's development entry fire.

``conf.py`` is executed with :mod:`runpy` and the version source is patched as
*module attributes* on :mod:`osprey.version` — not as names in the exec'd
namespace. ``conf.py`` binds ``get_running_version``/``is_release`` at exec time,
so replacing the module attributes is what the import actually picks up, and it
sidesteps the ``functools.cache`` wrapped around the real
:func:`osprey.version.get_running_version`.
"""

from __future__ import annotations

import importlib.util
import runpy
import sys
from pathlib import Path
from typing import Any

import osprey.version

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONF_PY = _REPO_ROOT / "docs" / "source" / "conf.py"
_DOCS_PUBLISH_PY = _REPO_ROOT / "scripts" / "ci" / "docs_publish.py"


def _load_docs_publish():
    """Load ``scripts/ci/docs_publish.py`` by path (``scripts/`` is not a package).

    The module is registered in :data:`sys.modules` before execution because it defines a
    dataclass, and ``dataclasses`` resolves annotations through the module entry.
    """
    spec = importlib.util.spec_from_file_location("docs_publish", _DOCS_PUBLISH_PY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_conf(monkeypatch, *, released: bool, running: str) -> dict[str, Any]:
    """Execute ``conf.py`` with a pinned version source and return its namespace."""
    monkeypatch.setattr(osprey.version, "is_release", lambda: released)
    monkeypatch.setattr(osprey.version, "get_running_version", lambda: running)
    monkeypatch.chdir(_CONF_PY.parent)

    saved_path = list(sys.path)
    try:
        return runpy.run_path(str(_CONF_PY))
    finally:
        sys.path[:] = saved_path


def test_release_build_uses_the_running_version(monkeypatch):
    ns = run_conf(monkeypatch, released=True, running="9.9.9")

    assert ns["release"] == "9.9.9"
    assert ns["version"] == "9.9.9"
    assert ns["html_theme_options"]["switcher"]["version_match"] == "9.9.9"
    assert "|release| replace:: v9.9.9" in ns["rst_prolog"]
    assert "|version| replace:: 9.9.9" in ns["rst_prolog"]


def test_development_build_publishes_the_dev_literal(monkeypatch):
    ns = run_conf(monkeypatch, released=False, running="9.9.9.post7+g1234567")

    assert ns["release"] == "dev"
    assert ns["version"] == "dev"
    assert ns["html_theme_options"]["switcher"]["version_match"] == "dev"
    assert "|release| replace:: dev" in ns["rst_prolog"]
    assert "vdev" not in ns["rst_prolog"]


def test_version_warning_banner_is_enabled(monkeypatch):
    ns = run_conf(monkeypatch, released=False, running="9.9.9.post7+g1234567")

    assert ns["html_theme_options"]["show_version_warning_banner"] is True


def test_every_build_canonicalises_to_the_stable_root(monkeypatch):
    """Snapshot and ``/latest/`` builds alike point their canonical link at the root."""
    docs_publish = _load_docs_publish()

    for released, running in ((True, "9.9.9"), (False, "9.9.9.post7+g1234567")):
        ns = run_conf(monkeypatch, released=released, running=running)
        assert ns["html_baseurl"] == docs_publish.SITE_URL


def test_switcher_json_url_matches_the_ci_site_url(monkeypatch):
    """``conf.py`` and the CI publish helper must agree on where the site lives."""
    docs_publish = _load_docs_publish()
    ns = run_conf(monkeypatch, released=True, running="9.9.9")

    assert (
        ns["html_theme_options"]["switcher"]["json_url"]
        == docs_publish.SITE_URL + "_static/versions.json"
    )


def test_conf_no_longer_shells_out_to_git():
    source = _CONF_PY.read_text(encoding="utf-8")

    assert "subprocess" not in source
    assert "git describe" not in source
