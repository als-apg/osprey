"""The startup line that names both Claude Code binaries.

A project that pins ``claude_code.cli_version`` runs one CLI in the terminal
(spawned from the argv :mod:`osprey.utils.claude_launcher` builds) and another
in the chat (the binary bundled inside the Agent SDK package). The server says
so at startup rather than leaving the pair to be inferred from a behaviour that
only one view has.

The bundle is probed by actually running a stand-in binary, not by patching the
subprocess away: what is being tested is that a real ``--version`` call is made
once, survives every way it can fail, and is not repeated.
"""

from __future__ import annotations

import logging
import os

import pytest

from osprey.interfaces.web_terminal import app as web_app
from osprey.utils import claude_launcher

#: The bundle's stand-in reports this; the pin below deliberately does not
#: match it, so the mismatch path is the interesting one.
BUNDLE_VERSION = "2.1.228"
PINNED_VERSION = "2.1.146"

LAUNCHER_LOGGER = claude_launcher.logger.name
APP_LOGGER = web_app.logger.name

#: The stand-in bundle is a shell script, so these need a POSIX shell.
posix_only = pytest.mark.skipif(os.name != "posix", reason="the stand-in bundle is a shell script")


@pytest.fixture(autouse=True)
def clear_bundle_cache():
    """Drop the per-process probe cache around every test.

    The cache is the point of the function, so it cannot be disabled — but a
    value cached by one test would make the next one assert nothing.
    """
    claude_launcher.bundled_cli_version.cache_clear()
    yield
    claude_launcher.bundled_cli_version.cache_clear()


def _fake_bundle(tmp_path, body: str, name: str = "claude"):
    """Write an executable stand-in for the SDK's bundled binary."""
    path = tmp_path / name
    path.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _install_bundle(monkeypatch, path) -> None:
    """Point the launcher's bundle lookup at ``path`` (or nothing)."""
    monkeypatch.setattr(claude_launcher, "bundled_cli_path", lambda: path)


# ---- Reading the bundled version ----


@posix_only
def test_the_bundled_version_is_read_from_the_binary(tmp_path, monkeypatch):
    """The probe runs the binary and parses the semver out of its output."""
    _install_bundle(monkeypatch, _fake_bundle(tmp_path, f'echo "{BUNDLE_VERSION} (Claude Code)"'))
    assert claude_launcher.bundled_cli_version() == BUNDLE_VERSION


@posix_only
def test_the_binary_is_run_once_per_process(tmp_path, monkeypatch):
    """A second caller gets the cached answer, not a second spawn."""
    marker = tmp_path / "calls"
    _install_bundle(
        monkeypatch,
        _fake_bundle(tmp_path, f'echo x >> "{marker}"\necho "{BUNDLE_VERSION}"'),
    )

    assert claude_launcher.bundled_cli_version() == BUNDLE_VERSION
    assert claude_launcher.bundled_cli_version() == BUNDLE_VERSION

    assert marker.read_text(encoding="utf-8").count("x") == 1


def test_no_bundle_is_an_unknown_version_not_an_error(monkeypatch):
    """An SDK without a bundle leaves the chat's version simply unknown."""
    _install_bundle(monkeypatch, None)
    assert claude_launcher.bundled_cli_version() is None


@posix_only
def test_a_binary_that_fails_reports_no_version(tmp_path, monkeypatch):
    """A non-zero exit is not a version, and is not an exception either."""
    _install_bundle(monkeypatch, _fake_bundle(tmp_path, "exit 3"))
    assert claude_launcher.bundled_cli_version() is None


@posix_only
def test_unparseable_output_reports_no_version(tmp_path, monkeypatch):
    """Output with no semver in it answers ``None`` rather than a fragment."""
    _install_bundle(monkeypatch, _fake_bundle(tmp_path, 'echo "not a version"'))
    assert claude_launcher.bundled_cli_version() is None


def test_a_missing_binary_reports_no_version(tmp_path, monkeypatch):
    """A path that does not exist fails at spawn; the caller still gets a start."""
    _install_bundle(monkeypatch, tmp_path / "does-not-exist")
    assert claude_launcher.bundled_cli_version() is None


def test_the_real_lookup_agrees_with_the_installed_sdk():
    """The bundle is resolved from the installed package, not a guessed path."""
    path = claude_launcher.bundled_cli_path()
    if path is None:
        pytest.skip("the installed Agent SDK ships no bundled CLI")
    assert path.is_file()
    assert path.parent.name == "_bundled"


# ---- Reading the pinned version out of the argv ----


def test_a_pinned_argv_reports_its_version():
    """The pin the launcher wrote into the argv is the pin read back out."""
    argv = claude_launcher.build_claude_launch_argv({"cli_version": PINNED_VERSION})
    assert claude_launcher.argv_cli_version(argv) == PINNED_VERSION


def test_an_unpinned_argv_reports_nothing():
    """``claude`` off PATH carries no version, and none is invented for it."""
    argv = claude_launcher.build_claude_launch_argv({})
    assert claude_launcher.argv_cli_version(argv) is None


def test_a_shell_override_reports_nothing():
    """A configured ``shell`` is not a Claude Code launch at all."""
    assert claude_launcher.argv_cli_version(["/bin/bash", "-l"]) is None


# ---- The startup line ----


@posix_only
def test_both_versions_are_logged(tmp_path, monkeypatch, caplog):
    """One line names the terminal's argv and the chat's bundled version."""
    _install_bundle(monkeypatch, _fake_bundle(tmp_path, f'echo "{BUNDLE_VERSION}"'))
    argv = claude_launcher.build_claude_launch_argv({"cli_version": BUNDLE_VERSION})

    with caplog.at_level(logging.INFO, logger=APP_LOGGER):
        web_app._log_claude_cli_versions(argv)

    logged = caplog.text
    assert "npx" in logged
    assert BUNDLE_VERSION in logged
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


@posix_only
def test_a_pin_that_differs_from_the_bundle_warns(tmp_path, monkeypatch, caplog):
    """The whole point: two builds under one server, said out loud."""
    _install_bundle(monkeypatch, _fake_bundle(tmp_path, f'echo "{BUNDLE_VERSION}"'))
    argv = claude_launcher.build_claude_launch_argv({"cli_version": PINNED_VERSION})

    with caplog.at_level(logging.INFO, logger=APP_LOGGER):
        web_app._log_claude_cli_versions(argv)

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    assert PINNED_VERSION in warnings[0].getMessage()
    assert BUNDLE_VERSION in warnings[0].getMessage()


@posix_only
def test_an_unpinned_launch_never_warns(tmp_path, monkeypatch, caplog):
    """An unprobed PATH ``claude`` is unknown, and unknown is not a mismatch."""
    _install_bundle(monkeypatch, _fake_bundle(tmp_path, f'echo "{BUNDLE_VERSION}"'))
    argv = claude_launcher.build_claude_launch_argv({})

    with caplog.at_level(logging.INFO, logger=APP_LOGGER):
        web_app._log_claude_cli_versions(argv)

    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert BUNDLE_VERSION in caplog.text


def test_an_unreadable_bundle_still_logs_the_argv(monkeypatch, caplog):
    """With nothing to compare against, the terminal's argv is still recorded."""
    _install_bundle(monkeypatch, None)
    argv = claude_launcher.build_claude_launch_argv({"cli_version": PINNED_VERSION})

    with caplog.at_level(logging.INFO, logger=APP_LOGGER):
        web_app._log_claude_cli_versions(argv)

    assert PINNED_VERSION in caplog.text
    assert "unknown" in caplog.text
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_the_probe_failure_path_is_quiet(monkeypatch, caplog, tmp_path):
    """A bundle that cannot be run is debug noise, never a startup warning."""
    _install_bundle(monkeypatch, tmp_path / "does-not-exist")

    with caplog.at_level(logging.DEBUG, logger=LAUNCHER_LOGGER):
        assert claude_launcher.bundled_cli_version() is None

    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
