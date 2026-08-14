"""Tests for :func:`osprey.cli.build_environment.freeze_base_environment`.

The venv-base cases build a *real* virtual environment and fabricate
``.dist-info`` directories inside it: the freeze reads package metadata through
a subprocess against that interpreter, so a fake environment has to be real
enough for ``importlib.metadata`` to walk. Fabricated metadata keeps the tests
offline and deterministic (no installs, no network).

The offender-reporting cases patch the probe seam instead — what is under test
there is that *every* offender is named, which needs a controlled package set.
"""

from __future__ import annotations

import json
import sys
import venv
from pathlib import Path

import pytest

from osprey.cli import build_environment
from osprey.cli.build_environment import freeze_base_environment
from osprey.errors import BuildProfileError

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def base_venv(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Create a real, empty venv. Returns ``(interpreter, site_packages)``."""
    root = tmp_path_factory.mktemp("base-venv") / "venv"
    venv.create(root, with_pip=False, symlinks=True)
    python = root / "bin" / "python"
    site_packages = next(iter(root.glob("lib/python*/site-packages")))
    return python, site_packages


@pytest.fixture
def clean_site_packages(base_venv: tuple[Path, Path]) -> Path:
    """The module-scoped venv's site-packages, emptied of fabricated dists."""
    _, site_packages = base_venv
    for entry in site_packages.glob("*.dist-info"):
        for child in entry.iterdir():
            child.unlink()
        entry.rmdir()
    return site_packages


def write_dist(
    site_packages: Path, name: str, version: str, *, direct_url: dict | None = None
) -> None:
    """Fabricate a minimal installed-distribution record."""
    dist_info = site_packages / f"{name.replace('-', '_')}-{version}.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n", encoding="utf-8"
    )
    if direct_url is not None:
        (dist_info / "direct_url.json").write_text(json.dumps(direct_url), encoding="utf-8")


def record(name: str, version: str, direct_url: dict | None = None) -> dict:
    """A probe record as :func:`_probe_base_distributions` returns them."""
    return {
        "name": name,
        "version": version,
        "direct_url": None if direct_url is None else json.dumps(direct_url),
    }


def fake_probe(monkeypatch: pytest.MonkeyPatch, records: list[dict]) -> None:
    """Make the base look like a venv holding exactly *records*."""
    monkeypatch.setattr(build_environment, "_base_is_venv", lambda _python: True)
    monkeypatch.setattr(build_environment, "_probe_base_distributions", lambda _python: records)


# --------------------------------------------------------------------------
# Base detection
# --------------------------------------------------------------------------


def test_non_venv_base_returns_empty() -> None:
    """A bare interpreter has no installed set to freeze."""
    if sys.prefix == sys.base_prefix:
        pytest.skip("test is not running inside a venv")
    base_executable = getattr(sys, "_base_executable", None)
    if not base_executable or not Path(base_executable).exists():
        pytest.skip("no non-venv interpreter available on this host")

    assert freeze_base_environment(Path(base_executable), []) == []


def test_unusable_interpreter_raises() -> None:
    """A base that cannot be executed fails the build rather than freezing nothing."""
    with pytest.raises(BuildProfileError, match="Cannot probe with base interpreter"):
        freeze_base_environment(Path("/nonexistent/bin/python"), [])


# --------------------------------------------------------------------------
# Enumeration against a real venv
# --------------------------------------------------------------------------


def test_venv_base_pins_installed_distributions(
    base_venv: tuple[Path, Path], clean_site_packages: Path
) -> None:
    python, _ = base_venv
    write_dist(clean_site_packages, "lmfit", "1.3.2")
    write_dist(clean_site_packages, "Alpha-Pkg", "0.1.0")

    assert freeze_base_environment(python, []) == ["Alpha-Pkg==0.1.0", "lmfit==1.3.2"]


def test_inherit_exclude_drops_named_distributions(
    base_venv: tuple[Path, Path], clean_site_packages: Path
) -> None:
    """Exclusion matches on the PEP 503 canonical name, not the literal string."""
    python, _ = base_venv
    write_dist(clean_site_packages, "lmfit", "1.3.2")
    write_dist(clean_site_packages, "Alpha-Pkg", "0.1.0")

    assert freeze_base_environment(python, ["Alpha_Pkg"]) == ["lmfit==1.3.2"]


def test_osprey_itself_is_never_frozen(
    base_venv: tuple[Path, Path], clean_site_packages: Path
) -> None:
    """osprey is installed by the build, not inherited — and its editable install
    must not be reported as an offender either."""
    python, _ = base_venv
    write_dist(clean_site_packages, "lmfit", "1.3.2")
    write_dist(
        clean_site_packages,
        "osprey-framework",
        "2026.7.0",
        direct_url={"url": "file:///src/osprey", "dir_info": {"editable": True}},
    )

    assert freeze_base_environment(python, []) == ["lmfit==1.3.2"]


def test_freeze_logs_duration(
    base_venv: tuple[Path, Path],
    clean_site_packages: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    python, _ = base_venv
    write_dist(clean_site_packages, "lmfit", "1.3.2")

    # DEBUG: the freeze duration is build-internal detail, below the phase lines
    # the verb now prints.
    with caplog.at_level("DEBUG", logger="build"):
        freeze_base_environment(python, [])

    assert any("Froze 1 packages" in r.message and "s)" in r.message for r in caplog.records)


# --------------------------------------------------------------------------
# Offender reporting — all offenders, never just the first
# --------------------------------------------------------------------------


def test_direct_url_installs_report_every_offender(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_probe(
        monkeypatch,
        [
            record("alpha", "1.0.0"),
            record("bravo", "2.0.0", {"url": "file:///src/bravo", "dir_info": {"editable": True}}),
            record("charlie", "3.0.0", {"url": "git+https://example.invalid/charlie.git"}),
            record("delta", "4.0.0", {"url": "https://example.invalid/delta-4.0.0.whl"}),
        ],
    )

    with pytest.raises(BuildProfileError) as exc:
        freeze_base_environment(Path("/base/bin/python"), [])

    message = str(exc.value)
    for name in ("bravo", "charlie", "delta"):
        assert name in message, f"{name} missing from offender report"
    assert "3 package(s)" in message
    assert "editable install from /src/bravo" in message
    assert "git+https://example.invalid/charlie.git" in message
    assert "inherit_exclude" in message
    assert "alpha" not in message  # index installs are not offenders


def test_version_conflicts_report_every_offender(monkeypatch: pytest.MonkeyPatch) -> None:
    from packaging.specifiers import SpecifierSet

    fake_probe(
        monkeypatch,
        [
            record("alpha", "1.0.0"),
            record("bravo", "1.0.0"),
            record("charlie", "0.9.0"),
        ],
    )
    monkeypatch.setattr(
        build_environment,
        "_osprey_requirement_specifiers",
        lambda: {
            "alpha": SpecifierSet(">=1.0"),
            "bravo": SpecifierSet(">=2.0"),
            "charlie": SpecifierSet("==1.2.3"),
        },
    )

    with pytest.raises(BuildProfileError) as exc:
        freeze_base_environment(Path("/base/bin/python"), [])

    message = str(exc.value)
    assert "bravo 1.0.0 installed" in message
    assert "charlie 0.9.0 installed" in message
    assert "2 package(s)" in message
    assert "alpha" not in message  # satisfies osprey's requirement


def test_both_offender_classes_reported_together(monkeypatch: pytest.MonkeyPatch) -> None:
    from packaging.specifiers import SpecifierSet

    fake_probe(
        monkeypatch,
        [
            record("bravo", "1.0.0"),
            record("charlie", "3.0.0", {"url": "file:///src/charlie"}),
        ],
    )
    monkeypatch.setattr(
        build_environment,
        "_osprey_requirement_specifiers",
        lambda: {"bravo": SpecifierSet(">=2.0")},
    )

    with pytest.raises(BuildProfileError) as exc:
        freeze_base_environment(Path("/base/bin/python"), [])

    message = str(exc.value)
    assert "no reproducible source" in message
    assert "Conflicts with osprey-framework's own requirements" in message
    assert "bravo" in message and "charlie" in message
    # The escape-hatch snippet lists both, so one edit clears the build.
    assert "      - bravo" in message
    assert "      - charlie" in message


def test_inherit_exclude_clears_offenders(monkeypatch: pytest.MonkeyPatch) -> None:
    """The documented escape hatch has to actually work: excluded packages are
    dropped before reproducibility is judged."""
    from packaging.specifiers import SpecifierSet

    fake_probe(
        monkeypatch,
        [
            record("alpha", "1.0.0"),
            record("bravo", "1.0.0"),
            record("charlie", "3.0.0", {"url": "file:///src/charlie"}),
        ],
    )
    monkeypatch.setattr(
        build_environment,
        "_osprey_requirement_specifiers",
        lambda: {"bravo": SpecifierSet(">=2.0")},
    )

    assert freeze_base_environment(Path("/base/bin/python"), ["bravo", "Charlie"]) == [
        "alpha==1.0.0"
    ]


# --------------------------------------------------------------------------
# osprey's own requirement map
# --------------------------------------------------------------------------


def test_osprey_requirement_specifiers_skips_extras_gated_requirements() -> None:
    """Extras osprey does not install by default must not manufacture conflicts."""
    pins = build_environment._osprey_requirement_specifiers()

    assert pins, "osprey-framework metadata should be readable in the test environment"
    assert "click" in pins
    assert "pytest" not in pins  # dev extra only
