"""``osprey build``'s voice for a panel the web terminal will not serve.

A render's ``panels/`` is judged by the web terminal's own discovery, so the
build names exactly the bundles the server would refuse: one warning per
panel, every reason in it, and nothing when discovery is off.
"""

from __future__ import annotations

import yaml

from osprey.deployment.web_terminals.lint import PANEL_NOT_SERVED, panels_not_served
from osprey.interfaces.design_system.panels.validator import validate_panel
from osprey.interfaces.web_terminal.panel_discovery import configured_panel_ids, discover_panels
from tests.interfaces.web_terminal.test_panel_discovery import (
    _UNMARKED_HTML,
    _UNSTYLED_CLASS_HTML,
    _manifest,
    _write_panel,
)

_ON = {"allow_runtime_panels": True}


def _render(tmp_path, web: dict | None = None):
    config = {"project_name": "demo"} if web is None else {"project_name": "demo", "web": web}
    (tmp_path / "config.yml").write_text(yaml.safe_dump(config), encoding="utf-8")
    return tmp_path


def test_a_refused_panel_is_one_warning_naming_it_and_every_reason(tmp_path):
    render = _render(tmp_path, _ON)
    _write_panel(render / "panels", "unmarked", manifest=_manifest(), html=_UNMARKED_HTML)
    [finding] = panels_not_served(render, shown_as="build")
    assert finding.severity == "warn"
    assert finding.code == "web_terminals.panel_not_served" == PANEL_NOT_SERVED
    assert finding.message.startswith(
        "Panel 'unmarked' will not be served: "
        "build/panels/unmarked/index.html: never sets the page-ready mark"
    )


def test_two_reasons_are_joined_in_one_line(tmp_path):
    render = _render(tmp_path, _ON)
    html = _UNMARKED_HTML.replace("<h1>", '<h1 style="color: #ff0000">')
    bundle = _write_panel(render / "panels", "two", manifest=_manifest(), html=html)
    reasons = validate_panel(bundle)
    assert len(reasons) == 2
    [finding] = panels_not_served(render, shown_as="build")
    expected = "; ".join(
        f"build/{bundle.relative_to(render).as_posix()}{r.source[len(str(bundle)) :]}: {r.message}"
        for r in reasons
    )
    assert finding.message == f"Panel 'two' will not be served: {expected}"


def test_a_served_panel_and_a_missing_panels_dir_say_nothing(tmp_path):
    render = _render(tmp_path, _ON)
    assert panels_not_served(render, shown_as="build") == []
    _write_panel(render / "panels", "demo", manifest=_manifest())
    assert panels_not_served(render, shown_as="build") == []


def test_a_panel_whose_findings_only_report_is_not_named(tmp_path):
    render = _render(tmp_path, _ON)
    _write_panel(render / "panels", "demo", manifest=_manifest(), html=_UNSTYLED_CLASS_HTML)
    assert panels_not_served(render, shown_as="build") == []


def test_discovery_off_says_nothing(tmp_path):
    render = _render(tmp_path, {})
    _write_panel(render / "panels", "unmarked", manifest=_manifest(), html=_UNMARKED_HTML)
    assert panels_not_served(render, shown_as="build") == []
    _render(tmp_path, {"allow_runtime_panels": False})
    assert panels_not_served(render, shown_as="build") == []


def test_an_id_the_render_configures_is_named(tmp_path):
    render = _render(tmp_path, {**_ON, "panels": {"demo": {"url": "http://x/", "label": "D"}}})
    _write_panel(render / "panels", "demo", manifest=_manifest())
    [finding] = panels_not_served(render, shown_as="build")
    assert "web.panels" in finding.message


def test_the_build_says_what_the_server_would_refuse(tmp_path):
    web = {**_ON, "panels": {"taken": {"url": "http://x/", "label": "T"}}}
    render = _render(tmp_path, web)
    _write_panel(render / "panels", "good", manifest=_manifest(pid="good"))
    _write_panel(render / "panels", "unmarked", manifest=_manifest(pid="u"), html=_UNMARKED_HTML)
    _write_panel(render / "panels", "mine", manifest=_manifest(pid="taken"))
    _write_panel(render / "panels", "broken", manifest="{ not json ")
    said = {f.message.split("'")[1] for f in panels_not_served(render, shown_as="build")}
    verdict = discover_panels(render / "panels", taken=configured_panel_ids(web))
    assert said == {r.id for r in verdict.refused} == {"unmarked", "mine", "broken"}


def test_an_unreadable_config_says_nothing(tmp_path):
    (tmp_path / "config.yml").write_text("web: [unclosed\n", encoding="utf-8")
    _write_panel(tmp_path / "panels", "unmarked", manifest=_manifest(), html=_UNMARKED_HTML)
    assert panels_not_served(tmp_path, shown_as="build") == []
    (tmp_path / "config.yml").write_text("- a list\n", encoding="utf-8")
    assert panels_not_served(tmp_path, shown_as="build") == []
