"""The author's per-service axes must survive the injector that rewrites their block.

Eight deploy-time services are wired by a *dedicated* injector — bluesky,
bluesky_web, gchat_bridge, nextcloud_bridge, teams_bridge, virtual_accelerator,
mongodb and archiver_recorder. Each builds its ``services.<name>`` block from its own
profile block and installs it with a whole-VALUE assignment, which runs AFTER
both spellings of an author's per-service key have already landed in ``config.yml``:

* nested — ``services.<name>.config.<key>``, written by ``_inject_profile_services``;
* dotted — ``config: {"services.<name>.<key>": ...}``, merged by
  ``build_cmd._apply_config_overrides``.

The keys that belong to the author rather than to any injector — ``env:`` (a
name list), ``network:`` (the attachment the template renders) and ``http:``
(what the deploy summary prints) — must therefore be carried across that
replacement, or they are accepted at validation, written to the rendered config
and then silently dropped a few steps later, with no error and no effect.

What these tests pin:

* every one of the eight carries a declared ``env:`` list and a ``network:``
  through its injector, in both spellings and in author order;
* the keys the injector *derives* (a port, a trigger, a path) are still
  regenerated — carrying the authored key must not turn the block into an
  append-only accumulation of whatever a previous build left;
* a service that declares nothing gets none of the authored keys, so a
  config.yml that never carried one renders byte-for-byte what it did before;
* ``_inject_profile_services`` fills only the GAP: the nested spelling it builds
  its block from still outranks a dotted override sitting in the block being
  replaced.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml as pyyaml

from osprey.cli.build_injectors import (
    _inject_bluesky,
    _inject_bluesky_web,
    _inject_gchat_bridge,
    _inject_nextcloud_bridge,
    _inject_profile_services,
    _inject_teams_bridge,
    _inject_va,
    _inject_va_archiver,
)
from osprey.cli.build_profile_archiver import VAArchiverConfig
from osprey.cli.build_profile_schema import (
    BlueskyConfig,
    BlueskyWebConfig,
    GChatBridgeProfileConfig,
    NextcloudBridgeProfileConfig,
    ServiceDef,
    TeamsBridgeProfileConfig,
    VAConfig,
)
from osprey.port_layout import default_port

_DECLARED = ["SITE_HTTP_PROXY", "FACILITY_TZ"]

#: One entry per dedicated injector, as ``(service key, call, derived key)``.
#: ``derived key`` is a key the injector regenerates from its own profile block,
#: asserted alongside the carried one so a fix that stopped rewriting the block
#: at all would not read as a pass.
_INJECTORS: tuple[tuple[str, Any, str], ...] = (
    (
        "bluesky",
        lambda path: _inject_bluesky(BlueskyConfig(port=default_port("bluesky")), path),
        "port",
    ),
    (
        "bluesky_web",
        lambda path: _inject_bluesky_web(BlueskyWebConfig(port=default_port("bluesky_web")), path),
        "port",
    ),
    ("virtual_accelerator", lambda path: _inject_va(VAConfig(port=5064), path), "port"),
    (
        "gchat_bridge",
        lambda path: _inject_gchat_bridge(GChatBridgeProfileConfig(), path),
        "trigger",
    ),
    (
        "nextcloud_bridge",
        lambda path: _inject_nextcloud_bridge(NextcloudBridgeProfileConfig(), path),
        "trigger",
    ),
    (
        "teams_bridge",
        lambda path: _inject_teams_bridge(TeamsBridgeProfileConfig(), path),
        "trigger",
    ),
    ("mongodb", lambda path: _inject_va_archiver(VAArchiverConfig(), path), "port_host"),
    ("archiver_recorder", lambda path: _inject_va_archiver(VAArchiverConfig(), path), "path"),
)

_INJECTOR_IDS = [entry[0] for entry in _INJECTORS]


def _project(tmp_path: Path, services: dict[str, Any]) -> Path:
    """A built project whose config.yml already carries the given service blocks.

    This is the state the dedicated injectors run against: both env-axis
    spellings have been merged into ``services.<name>`` by earlier build steps,
    and the injector is about to replace the block wholesale.
    """
    document: dict[str, Any] = {"services": services, "deployed_services": []}
    (tmp_path / "config.yml").write_text(pyyaml.safe_dump(document), encoding="utf-8")
    return tmp_path


def _services(project_path: Path) -> dict[str, Any]:
    text = (project_path / "config.yml").read_text(encoding="utf-8")
    return pyyaml.safe_load(text)["services"]


@pytest.mark.parametrize(("name", "inject", "derived"), _INJECTORS, ids=_INJECTOR_IDS)
def test_a_declared_passthrough_survives_the_injector(tmp_path, name, inject, derived):
    """The whole point: the names the author wrote reach the rendered config.

    Without this the compose template's env-axis macro reads a block that no
    longer has an ``env`` key, renders nothing, and the container never receives
    the variable — with nothing anywhere saying the declaration was dropped.
    """
    project = _project(tmp_path, {name: {"env": list(_DECLARED)}})

    inject(project)

    block = _services(project)[name]
    assert block["env"] == _DECLARED, f"{name} dropped its declared passthrough names"
    assert derived in block, f"{name} no longer writes its own derived {derived!r} key"


@pytest.mark.parametrize(("name", "inject", "derived"), _INJECTORS, ids=_INJECTOR_IDS)
def test_a_service_that_declares_nothing_gets_no_env_key(tmp_path, name, inject, derived):  # noqa: ARG001 - a column of the shared _INJECTORS table
    """No empty ``env: []`` may appear in a config.yml that never carried one.

    The axis is additive by construction — the macro renders nothing for a
    service that declares nothing — and the carry-forward has to keep it that
    way, or every deployment's rendered config would grow a key it never asked
    for and every golden would move for a feature nobody switched on.
    """
    project = _project(tmp_path, {})

    inject(project)

    assert "env" not in _services(project)[name]


@pytest.mark.parametrize(("name", "inject", "derived"), _INJECTORS, ids=_INJECTOR_IDS)
def test_an_authored_network_survives_the_injector(tmp_path, name, inject, derived):
    """A dotted ``network: host`` reaches the render beside the derived keys.

    The network axis is read by each service's compose template off its own
    block, so a dropped key leaves the service on the compose bridge while the
    profile says host.
    """
    project = _project(tmp_path, {name: {"network": "host"}})

    inject(project)

    block = _services(project)[name]
    assert block["network"] == "host", f"{name} dropped its authored network"
    assert derived in block, f"{name} no longer writes its own derived {derived!r} key"


@pytest.mark.parametrize(("name", "inject", "derived"), _INJECTORS, ids=_INJECTOR_IDS)
def test_a_service_that_declares_nothing_gets_no_network_or_http_key(
    tmp_path,
    name,
    inject,
    derived,  # noqa: ARG001 - a column of the shared _INJECTORS table
):
    """Carrying the author's axes adds no key the author did not write."""
    project = _project(tmp_path, {})

    inject(project)

    block = _services(project)[name]
    assert "network" not in block
    assert "http" not in block


@pytest.mark.parametrize(("name", "inject", "derived"), _INJECTORS, ids=_INJECTOR_IDS)
def test_the_derived_keys_are_still_regenerated(tmp_path, name, inject, derived):
    """A stale value from a previous build must not survive alongside the carry.

    The block is rebuilt from the profile on every build precisely so a port or
    a trigger that moved in the profile moves in the render. Carrying the
    authored key must not turn the replacement into a merge.
    """
    project = _project(
        tmp_path, {name: {"env": list(_DECLARED), derived: "stale-value-from-a-previous-build"}}
    )

    inject(project)

    block = _services(project)[name]
    assert block[derived] != "stale-value-from-a-previous-build"
    assert block["env"] == _DECLARED


def test_author_order_is_preserved(tmp_path):
    """The rendered ``environment:`` block shows the author's order, so keep it."""
    reversed_names = list(reversed(_DECLARED))
    project = _project(tmp_path, {"bluesky": {"env": reversed_names}})

    _inject_bluesky(BlueskyConfig(), project)

    assert _services(project)["bluesky"]["env"] == reversed_names


def test_a_non_mapping_block_is_replaced_rather_than_read(tmp_path):
    """A hand-edited config.yml must not turn a build into a crash."""
    project = _project(tmp_path, {"bluesky": "not-a-mapping"})

    _inject_bluesky(BlueskyConfig(port=default_port("bluesky")), project)

    assert _services(project)["bluesky"]["port"] == default_port("bluesky")


# ---------------------------------------------------------------------------
# The profile-service injector fills only the gap
# ---------------------------------------------------------------------------


def test_a_profile_service_carries_a_dotted_declaration(tmp_path):
    """The dotted spelling lands in the block BEFORE this injector replaces it.

    ``_apply_config_overrides`` merges ``services.<name>.env`` into config.yml
    early in the build, and this injector then rebuilds the block from
    ``svc_def.config`` — which holds the nested spelling and nothing else. So
    the dotted spelling was dropped here for exactly the same reason it was
    dropped by the seven dedicated injectors.
    """
    project = _project(tmp_path, {"qmd": {"env": list(_DECLARED)}})
    services = {"qmd": ServiceDef(template="osprey.qmd", config={})}

    _inject_profile_services(tmp_path, project, services)

    assert _services(project)["qmd"]["env"] == _DECLARED


def test_the_nested_spelling_still_outranks_a_dotted_override(tmp_path):
    """Filling the gap must not quietly invert an existing precedence.

    A profile that spells the axis both ways resolved to the nested value before
    this fix, because the injector's own block won. Carrying unconditionally
    would have promoted the dotted spelling instead — a behaviour change nobody
    asked for, in a build that previously worked.
    """
    project = _project(tmp_path, {"qmd": {"env": ["DOTTED_ONLY"]}})
    services = {"qmd": ServiceDef(template="osprey.qmd", config={"env": ["NESTED_WINS"]})}

    _inject_profile_services(tmp_path, project, services)

    assert _services(project)["qmd"]["env"] == ["NESTED_WINS"]


def test_a_profile_service_carries_a_dotted_network_and_http(tmp_path):
    """Both dotted axes sitting in the block survive the profile-service rewrite."""
    project = _project(tmp_path, {"archive": {"network": "host", "http": True}})
    services = {"archive": ServiceDef(template="osprey.archive", config={})}

    _inject_profile_services(tmp_path, project, services)

    block = _services(project)["archive"]
    assert block["network"] == "host"
    assert block["http"] is True


def test_the_nested_spelling_still_outranks_a_dotted_network(tmp_path):
    """The nested ``network:`` the injector builds from wins over a dotted one."""
    project = _project(tmp_path, {"qmd": {"network": "host"}})
    services = {"qmd": ServiceDef(template="osprey.qmd", config={"network": "bridge"})}

    _inject_profile_services(tmp_path, project, services)

    assert _services(project)["qmd"]["network"] == "bridge"


def test_a_profile_service_carries_a_dotted_bind_declaration(tmp_path):
    """A dotted ``listens`` / ``bind_env`` survives the profile-service rewrite."""
    project = _project(
        tmp_path,
        {
            "site_poller": {"network": "host", "listens": False},
            "site_api": {"network": "host", "bind_env": "SITE_BIND"},
        },
    )
    template = tmp_path / "profile" / "services" / "site"
    template.mkdir(parents=True)
    (template / "docker-compose.yml.j2").write_text("services: {}\n", encoding="utf-8")
    services = {
        "site_poller": ServiceDef(template="services/site", config={}),
        "site_api": ServiceDef(template="services/site", config={}),
    }

    _inject_profile_services(tmp_path / "profile", project, services)

    rendered = _services(project)
    assert rendered["site_poller"]["listens"] is False
    assert rendered["site_api"]["bind_env"] == "SITE_BIND"
