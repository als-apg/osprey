"""The BLUESKY panel's registration, and the one-release ``results`` alias.

The RESULTS panel became BLUESKY when that bundle absorbed the queue view, so
the panel id moved in three places at once: the build profile's ``web_panels``
list, the ``web.panels.<id>`` entries ``_inject_bluesky_panels`` writes into a
built ``config.yml``, and the sidecar mount the web terminal proxies to. This
module covers the three questions that rename raises:

(a) a profile written today registers ``bluesky`` — and does not resurrect a
    ``results`` entry alongside it (two tabs onto one bundle);
(b) a profile written before the rename still BUILDS, and says so out loud —
    at both surfaces that can see the old spelling: profile validation (which
    sees the ``web_panels`` list) and the injector (which sees a
    ``web.panels.results.*`` override already merged into config.yml);
(c) a config.yml carrying the old ``web.panels.results.path`` still resolves to
    a real, serving panel, because the sidecar mounts the SAME bundle at
    ``/results`` and ``/bluesky`` for exactly this release. That is asserted
    against the sidecar's own mount table by identity, never against a bundle
    directory name — the alias must survive a bundle rename, which is how it
    came to exist in the first place.

The rest of ``_inject_bluesky_panels`` (template copy, service config,
deployed_services, url derivation, override precedence) is covered by
``test_inject_bluesky_panels.py``.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import pytest
from ruamel.yaml import YAML

from osprey.cli.build_injectors import _inject_bluesky_panels
from osprey.cli.build_profile import BlueskyPanelsConfig, BuildProfile


def _write_config(project_path: Path, *, panels: dict | None = None) -> None:
    """Write a minimal config.yml, optionally pre-seeded with web.panels entries."""
    yaml = YAML()
    config: dict = {"deployed_services": [], "services": {}}
    if panels is not None:
        config["web"] = {"panels": panels}
    with open(project_path / "config.yml", "w") as fh:
        yaml.dump(config, fh)


def _read_panels(project_path: Path) -> dict:
    yaml = YAML()
    with open(project_path / "config.yml") as fh:
        return yaml.load(fh)["web"]["panels"]


@pytest.fixture
def project_path(tmp_path: Path) -> Path:
    path = tmp_path / "project"
    path.mkdir()
    return path


# ---------------------------------------------------------------------------
# (a) A profile written today renders `bluesky`
# ---------------------------------------------------------------------------


def test_injector_registers_the_bluesky_panel(project_path: Path) -> None:
    """The panel id, label, and mount path all move together to BLUESKY."""
    _write_config(project_path)

    _inject_bluesky_panels(BlueskyPanelsConfig(port=8095), project_path=project_path)

    bluesky = _read_panels(project_path)["bluesky"]
    assert bluesky["url"] == "${BLUESKY_PANELS_URL:-http://localhost:8095}"
    assert bluesky["path"] == "/bluesky/"
    assert bluesky["label"] == "BLUESKY"


def test_injector_does_not_register_results_on_a_fresh_build(project_path: Path) -> None:
    """The alias is a migration accommodation, not a second panel.

    Creating ``results`` here would give every NEW project two tabs onto one
    bundle — and would keep the deprecated id alive past the release that
    retires it, since a rebuild would always put it back.
    """
    _write_config(project_path)

    _inject_bluesky_panels(BlueskyPanelsConfig(), project_path=project_path)

    assert "results" not in _read_panels(project_path)


def test_injector_warns_about_nothing_on_a_fresh_build(
    project_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Negative control for the deprecation warning below: a build with no old
    spelling anywhere must be silent, or the warning says nothing when it fires."""
    _write_config(project_path)

    with caplog.at_level(logging.WARNING, logger="osprey.cli.build_injectors"):
        _inject_bluesky_panels(BlueskyPanelsConfig(), project_path=project_path)

    assert "deprecated" not in caplog.text


# ---------------------------------------------------------------------------
# (b) A profile written before the rename still builds, loudly
# ---------------------------------------------------------------------------


def test_profile_listing_results_still_validates(tmp_path: Path) -> None:
    """A pre-rename ``web_panels: [results]`` is accepted, not a build failure.

    ``results`` is url-less in the profile (its URL is derived post-build from
    the sidecar port), so without the alias entry in the url-less special case
    this profile would fail validation outright with "Unknown web_panel".
    """
    profile = BuildProfile(
        name="legacy",
        web_panels=["plan", "results"],
        bluesky_panels=BlueskyPanelsConfig(),
    )

    with pytest.warns(UserWarning):
        profile.validate(tmp_path)  # must not raise


def test_profile_listing_results_warns_naming_the_rename_and_the_window(
    tmp_path: Path,
) -> None:
    """The warning has to be actionable on its own: what the id became, and how
    long the old one keeps working."""
    profile = BuildProfile(
        name="legacy",
        web_panels=["results"],
        bluesky_panels=BlueskyPanelsConfig(),
    )

    with pytest.warns(UserWarning) as record:
        profile.validate(tmp_path)

    message = str(record[0].message)
    assert "results" in message
    assert "bluesky" in message
    assert "ONE release" in message


def test_profile_listing_bluesky_validates_without_warning(tmp_path: Path) -> None:
    """The new spelling is url-less-legal too — and silent."""
    profile = BuildProfile(
        name="modern",
        web_panels=["plan", "bluesky"],
        bluesky_panels=BlueskyPanelsConfig(),
    )

    with warnings.catch_warnings():
        # A UserWarning here means the new spelling is being deprecated too.
        warnings.simplefilter("error", UserWarning)
        profile.validate(tmp_path)


def test_injector_warns_on_a_pre_rename_results_override(
    project_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The injector's own view of the old spelling: a ``web.panels.results.*``
    override merged into config.yml earlier in the build (or a config.yml built
    before the rename and rebuilt in place)."""
    _write_config(project_path, panels={"results": {"label": "RESULTS"}})

    with caplog.at_level(logging.WARNING, logger="osprey.cli.build_injectors"):
        _inject_bluesky_panels(BlueskyPanelsConfig(), project_path=project_path)

    assert "web.panels.results is deprecated" in caplog.text
    assert "web.panels.bluesky" in caplog.text
    assert "ONE release" in caplog.text


def test_injector_keeps_the_legacy_results_entry_resolvable(project_path: Path) -> None:
    """Warn AND alias: the old entry keeps its explicit label, gains the derived
    url/path it lacks, and the new panel is registered beside it. A rebuild that
    quietly dropped the operator's existing tab would be the same upgrade
    breakage the alias exists to prevent."""
    _write_config(project_path, panels={"results": {"label": "RESULTS"}})

    _inject_bluesky_panels(BlueskyPanelsConfig(port=8095), project_path=project_path)

    panels = _read_panels(project_path)
    assert panels["results"]["label"] == "RESULTS"
    assert panels["results"]["path"] == "/results/"
    assert panels["results"]["url"] == "${BLUESKY_PANELS_URL:-http://localhost:8095}"
    assert panels["bluesky"]["label"] == "BLUESKY"


# ---------------------------------------------------------------------------
# (c) The old config.yml still yields a WORKING tab, via the sidecar alias
# ---------------------------------------------------------------------------


def test_a_legacy_results_path_resolves_to_the_bluesky_bundle(project_path: Path) -> None:
    """The alias asserted by IDENTITY against the sidecar's mount table.

    What makes the deprecated tab work is not that ``/results`` is mounted, but
    that it is mounted onto the SAME bundle as ``/bluesky`` — so a config.yml
    still saying ``path: /results/`` reaches the panel the operator expects
    rather than a stale copy. Naming the bundle directory here would pin the
    wrong thing: the directory has already been renamed once, and the alias has
    to survive that.
    """
    from osprey.interfaces.bluesky_panels.app import _PANEL_MOUNTS

    _write_config(project_path, panels={"results": {"label": "RESULTS"}})
    _inject_bluesky_panels(BlueskyPanelsConfig(), project_path=project_path)

    panels = _read_panels(project_path)
    bundle_of = dict(_PANEL_MOUNTS)

    legacy_mount = panels["results"]["path"].rstrip("/")
    current_mount = panels["bluesky"]["path"].rstrip("/")
    assert legacy_mount in bundle_of, (
        f"config.yml points the deprecated tab at {legacy_mount}, which the sidecar "
        f"does not mount: {sorted(bundle_of)}"
    )
    assert bundle_of[legacy_mount] == bundle_of[current_mount]


def test_a_legacy_results_path_actually_serves_the_panel(project_path: Path) -> None:
    """The operator-visible half: the registered path returns the panel's HTML.

    A mount can be registered against an empty directory and still 404 (see
    ``tests/interfaces/bluesky_panels/test_health.py``), which would leave this
    config.yml pointing a tab at nothing.
    """
    from fastapi.testclient import TestClient

    from osprey.interfaces.bluesky_panels.app import app

    _write_config(project_path, panels={"results": {"label": "RESULTS"}})
    _inject_bluesky_panels(BlueskyPanelsConfig(), project_path=project_path)

    panels = _read_panels(project_path)

    with TestClient(app) as client:
        for panel_id in ("results", "bluesky"):
            response = client.get(panels[panel_id]["path"])
            assert response.status_code == 200, panel_id
            assert "text/html" in response.headers["content-type"], panel_id
