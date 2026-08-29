"""Every resolver that holds a config derives its port from *that* config's base.

FR3, stated once and checked over every resolver at once: a caller holding a
loaded config must never fall back to :data:`~osprey.port_layout.DEFAULT_PORT_BASE`.
With ``deployment.port_base: 20000`` set and **no port key in the resolver's own
section**, each resolver below must land on its 20xxx slot. A resolver that
reaches for the layout's default base instead lands on 10xxx and fails here.

This is the number-list-independent half of Goal 2. Nothing in this file
hard-codes a port where a slot name exists: every expectation is
``default_port(slot, index, base=BASE)`` at the base
:func:`~osprey.port_layout.resolve_port_base` read out of :data:`CONFIG`. Rename
an offset in the layout and this file follows; move a resolver back onto the
default base and it does not.

Each case records, in ``base_arrives_as``, how the resolved base actually
reaches the resolver — the config it is handed, an explicit ``base=`` keyword,
or a config it loads for itself — because that is the seam a regression would
break.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from osprey.port_layout import (
    DEFAULT_PORT_BASE,
    SLOTS_BY_NAME,
    default_port,
    index_bounds,
    resolve_port_base,
)
from osprey.registry.web import FRAMEWORK_WEB_SERVERS

#: The base every case resolves against. Deliberately not the layout default:
#: a resolver that ignored the config would produce a 10xxx port and every
#: assertion below would fail with both numbers on screen.
PORT_BASE = 20000

#: The one config shape handed to (or loaded by) every resolver under test. It
#: sets the base and nothing else — no ``services.*.port``, no
#: ``modules.web_terminals.*_base_port``, no ``<panel>.web.port`` — so the only
#: number any resolver can produce is a layout derivation.
CONFIG: dict[str, Any] = {"deployment": {"port_base": PORT_BASE}}


def _base() -> int:
    """The resolved base, taken the way production takes it."""
    return resolve_port_base(CONFIG)


#: Env vars that would pre-empt a resolver's config read. Cleared for every
#: case: a developer's ambient ``OSPREY_ARIEL_PORT`` must not be able to make
#: this file pass.
_AMBIENT_ENV_VARS = (
    "ARIEL_DB_PASSWORD",
    "ARIEL_DATABASE_HOST",
    "GRAPHDB_PASSWORD",
    "OSPREY_ARCHIVER_MONGODB_HOST",
    "OSPREY_ARCHIVER_MONGODB_PORT",
    "OSPREY_WEB_PORT",
    "OSPREY_TERMINAL_WEB_PORT",
    "DISPATCH_WORKER_PORT",
    "DISPATCH_WORKER_BIND",
    "BLUESKY_BRIDGE_URL",
)


@pytest.fixture(autouse=True)
def no_ambient_port_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip every env var that overrides a port before the config is consulted."""
    from osprey.build.claude_code_telemetry import OPENOBSERVE_PORT_ENV_VAR

    names = {
        *_AMBIENT_ENV_VARS,
        OPENOBSERVE_PORT_ENV_VAR,
        *(definition.port_env_var for definition in FRAMEWORK_WEB_SERVERS.values()),
    }
    for name in names:
        monkeypatch.delenv(name, raising=False)


def _pin_loaded_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``load_osprey_config()`` return :data:`CONFIG`.

    The resolvers that load the project config themselves (rather than being
    handed one) import :func:`osprey.utils.workspace.load_osprey_config` inside
    the function body, so patching it at its source reaches all of them.
    """
    import osprey.utils.workspace

    monkeypatch.setattr(osprey.utils.workspace, "load_osprey_config", lambda *a, **k: dict(CONFIG))


# ---------------------------------------------------------------------------
# One case per resolver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    """One resolver, and how the resolved base reaches it.

    Attributes:
        id: Test id — the resolver's own name, so a failure names the site.
        base_arrives_as: The call signature that carries the base. Documentation
            for the reader of a failure, and the thing a regression changes.
        run: Takes ``(tmp_path, monkeypatch)`` and returns ``(actual, expected)``,
            where *expected* is computed from :mod:`osprey.port_layout`.
        slots: Layout slots the case's expectation is built from. Asserted to be
            real slot names, so a renamed slot fails loudly instead of silently
            dropping the case's coverage.
    """

    id: str
    base_arrives_as: str
    run: Callable[[Path, pytest.MonkeyPatch], tuple[Any, Any]]
    slots: tuple[str, ...] = field(default=())


def _ariel_dsn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any]:
    """``resolve_ariel_dsn`` with no ``services.postgresql.port_host``."""
    from osprey.services.ariel_search.config import resolve_ariel_dsn

    actual = resolve_ariel_dsn({}, None, env={}, base=_base())
    expected = (
        f"postgresql://ariel:ariel@localhost:{default_port('postgres', base=PORT_BASE)}/ariel"
    )
    return actual, expected


def _mongodb_archiver_connector(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any]:
    """The archiver connector refuses a missing port rather than guessing one.

    The one B-row that does *not* end in a 20xxx number, and deliberately:
    ``osprey-connectors`` ships as its own wheel with no dependency on
    ``osprey``, so it cannot import :mod:`osprey.port_layout` and cannot compute
    ``port_base + 801``. Guessing would dial a port belonging to some other
    deployment's block, so ``archiver.mongodb_archiver.port`` is required and
    its absence is refused — with the refusal naming both the config key the
    build writes and the env var a containerized consumer sets.
    """
    from osprey_connectors.archiver.mongodb_archiver_connector import (
        MongoDBArchiverConnector,
    )

    connector = MongoDBArchiverConnector()
    with pytest.raises(ValueError) as excinfo:
        asyncio.run(
            connector.connect(
                {
                    "host": "localhost",
                    "name": "osprey_archiver",
                    "collection": "pv_history",
                    "username": "osprey",
                }
            )
        )

    message = str(excinfo.value)
    named = ("archiver.mongodb_archiver.port", "OSPREY_ARCHIVER_MONGODB_PORT")
    return tuple(token for token in named if token in message), named


def _simulation_archiver_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any]:
    """``simulation.apply.archiver_store_config`` dials the store it publishes."""
    from osprey.simulation.apply import archiver_store_config

    config = {**CONFIG, "archiver": {"mongodb_archiver": {"host": "localhost"}}}
    store = archiver_store_config(config, tmp_path)
    assert store is not None
    return store["port"], default_port("mongo", base=PORT_BASE)


def _openobserve_published(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any]:
    """``openobserve_published_port`` with no ``services.openobserve.port``."""
    from osprey.build.claude_code_telemetry import openobserve_published_port

    return openobserve_published_port(CONFIG), default_port("openobserve", base=PORT_BASE)


def _openobserve_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any]:
    """``resolve_openobserve_port`` with no env override and no config key."""
    from osprey.build.claude_code_telemetry import resolve_openobserve_port

    return resolve_openobserve_port(CONFIG), default_port("openobserve", base=PORT_BASE)


def _web_server_address(key: str) -> Callable[[Path, pytest.MonkeyPatch], tuple[Any, Any]]:
    """Build the ``resolve_web_server_address`` case for one companion server."""

    def run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any]:
        from osprey.registry.web import resolve_web_server_address

        definition = FRAMEWORK_WEB_SERVERS[key]
        family = definition.port_family or key
        _host, port = resolve_web_server_address(key, CONFIG)
        return port, default_port(family, 0, base=PORT_BASE)

    return run


def _web_port(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any]:
    """``resolve_web_port`` with neither ``--port`` nor ``web_terminal.port``."""
    from osprey.cli.web_cmd import resolve_web_port

    actual = resolve_web_port(None, None, base=_base(), env={})
    return actual, default_port("web", 0, base=PORT_BASE)


def _nginx_port(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any]:
    """``resolve_nginx_port`` with no ``modules.web_terminals.nginx_port``."""
    from osprey.deployment.web_terminals.ports import resolve_nginx_port

    return resolve_nginx_port(CONFIG), default_port("nginx", base=PORT_BASE)


def _base_ports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any]:
    """``base_ports_from_config`` with a stanza carrying no ``*_base_port`` key."""
    from osprey.deployment.web_terminals.ports import FAMILY_BASE_FIELDS, base_ports_from_config

    actual = base_ports_from_config({"enabled": True}, base=_base())
    expected = {
        family: default_port(family, 0, base=PORT_BASE)
        for family in dict.fromkeys(FAMILY_BASE_FIELDS.values())
    }
    return actual, expected


def _mcp_web_terminal_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any]:
    """``mcp_server.http.web_terminal_url`` with no ``web_terminal.port``."""
    from osprey.mcp_server.http import web_terminal_url

    _pin_loaded_config(monkeypatch)
    return web_terminal_url(), f"http://127.0.0.1:{default_port('web', 0, base=PORT_BASE)}"


def _scaffold_openobserve_probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any]:
    """The deploy scaffold probes the telemetry store inside the profile's block."""
    from osprey.cli.deploy_scaffold_templates import _service_probes

    profile = {
        "config": {
            "deployment.port_base": PORT_BASE,
            "deployed_services": ["openobserve"],
        }
    }
    ports = [probe.port for probe in _service_probes(profile)]
    return ports, [default_port("openobserve", base=PORT_BASE)]


def _scaffold_dispatcher_probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any]:
    """The deploy scaffold probes the dispatcher inside the profile's block."""
    from osprey.cli.deploy_scaffold_templates import _dispatch_probes

    profile = {
        "dispatch": {"enabled": True},
        "config": {"deployment.port_base": PORT_BASE},
    }
    ports = [probe.port for probe in _dispatch_probes(profile)]
    return ports, [default_port("dispatcher", base=PORT_BASE)]


def _scaffold_web_probes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any]:
    """The scaffold's landing page and user 0's terminal, both at the profile's base."""
    from osprey.cli.deploy_scaffold_templates import _web_probes

    profile = {
        "config": {
            "deployment.port_base": PORT_BASE,
            "modules.web_terminals": {"enabled": True, "users": ["alice"]},
        }
    }
    ports = sorted(probe.port for probe in _web_probes(profile))
    expected = sorted(
        [default_port("nginx", base=PORT_BASE), default_port("web", 0, base=PORT_BASE)]
    )
    return ports, expected


def _health_ariel_dsn_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any]:
    """The health cross-check names the port an empty postgresql block implies.

    ``services.postgresql: {}`` sets no ``port_host``, which is not an unknown
    port but a known one — the ``postgres`` slot of this deployment's block. The
    check derives it rather than skipping, so a stale hand-written DSN is still
    caught, and the WARNING it prints has to name that derived number.
    """
    from osprey.health.core.configuration import _check_ariel_dsn_port

    config = {
        **CONFIG,
        "ariel": {"database": {"uri": "postgresql://ariel:ariel@localhost:5432/ariel"}},
        "services": {"postgresql": {}},
    }
    rows = _check_ariel_dsn_port(config)
    assert len(rows) == 1, rows
    assert rows[0].status.name == "WARNING", rows[0]

    match = re.search(r"is (\d+)$", rows[0].message)
    assert match, f"the warning does not end by naming the derived port: {rows[0].message!r}"
    return int(match.group(1)), default_port("postgres", base=PORT_BASE)


def _graphdb_connection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any]:
    """``resolve_graphdb_connection`` with no ``services.graphdb.port_host``."""
    from osprey.deployment.graphdb_service import resolve_graphdb_connection

    connection = resolve_graphdb_connection({}, env={}, base=_base())
    return connection.uri, f"bolt://localhost:{default_port('graphdb_bolt', base=PORT_BASE)}"


def _qmd_service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any]:
    """``resolve_qmd_service_config`` with a ``services.qmd`` block naming no port."""
    from osprey.deployment.qmd_service import resolve_qmd_service_config

    resolved = resolve_qmd_service_config({**CONFIG, "services": {"qmd": {}}})
    assert resolved is not None
    return resolved.port, default_port("qmd", base=PORT_BASE)


def _bluesky_bridge_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any]:
    """``bridge_url_from_config`` with no ``services.bluesky.port``."""
    from osprey.bluesky_bridge_connection import bridge_url_from_config

    return (
        bridge_url_from_config(CONFIG),
        f"http://127.0.0.1:{default_port('bluesky', base=PORT_BASE)}",
    )


def _auth_sidecar_port(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any]:
    """The web-terminal auth sidecar with no ``modules.web_terminals.auth.port``."""
    from osprey.deployment.web_terminals.render import _auth_tls_context

    context = _auth_tls_context({"enabled": True}, base=_base())
    return context["auth_port"], default_port("auth", base=PORT_BASE)


def _dispatcher_worker_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any]:
    """The dispatcher's fallback ``DISPATCH_TARGET`` is worker 1 in its own block."""
    from osprey.dispatch.server import _default_worker_port

    _pin_loaded_config(monkeypatch)
    return _default_worker_port(), default_port("worker", 1, base=PORT_BASE)


def _worker_entrypoint_port(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any]:
    """A hand-started worker binds worker 1 of its own block, not the default one."""
    import osprey.mcp_server.dispatch_worker.__main__ as worker_main

    _pin_loaded_config(monkeypatch)
    bound: dict[str, Any] = {}
    monkeypatch.setattr(
        worker_main.uvicorn, "run", lambda *a, **kwargs: bound.update(kwargs), raising=True
    )

    worker_main.main()
    return bound.get("port"), default_port("worker", 1, base=PORT_BASE)


def _panel_cli_port_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any]:
    """No panel command freezes a port in its ``--port`` default.

    A click default is evaluated at import time, before any config is in hand,
    so a number there could only ever be the layout's default base. ``None``
    keeps the decision in the command body, where the config — and therefore
    the deployment's own base — is reachable.
    """
    from osprey.cli.ariel import web_command as ariel_web
    from osprey.cli.artifacts_cmd import web as artifacts_web
    from osprey.cli.channel_finder_cmd import web as channel_finder_web
    from osprey.cli.web_cmd import web as terminal_web

    commands = {
        "osprey ariel web": ariel_web,
        "osprey artifacts web": artifacts_web,
        "osprey channel-finder web": channel_finder_web,
        "osprey web": terminal_web,
    }
    actual = {
        label: next(p.default for p in command.params if p.name == "port")
        for label, command in commands.items()
    }
    return actual, dict.fromkeys(commands, None)


CASES: tuple[Case, ...] = (
    Case(
        "resolve_ariel_dsn",
        "base= keyword, from the caller's resolve_port_base(config)",
        _ariel_dsn,
        ("postgres",),
    ),
    Case(
        "mongodb_archiver_connector",
        "no base at all — the key is required and its absence refused",
        _mongodb_archiver_connector,
    ),
    Case(
        "simulation.apply.archiver_store_config",
        "the config it is handed",
        _simulation_archiver_store,
        ("mongo",),
    ),
    Case(
        "openobserve_published_port",
        "the config it is handed",
        _openobserve_published,
        ("openobserve",),
    ),
    Case(
        "resolve_openobserve_port",
        "the config it is handed",
        _openobserve_runtime,
        ("openobserve",),
    ),
    *(
        Case(
            f"resolve_web_server_address[{key}]",
            "the config it is handed",
            _web_server_address(key),
            (FRAMEWORK_WEB_SERVERS[key].port_family or key,),
        )
        for key in sorted(FRAMEWORK_WEB_SERVERS)
    ),
    Case("resolve_web_port", "base= keyword, no default", _web_port, ("web",)),
    Case("resolve_nginx_port", "the config it is handed", _nginx_port, ("nginx",)),
    Case(
        "base_ports_from_config",
        "base= keyword, no default",
        _base_ports,
        ("web", "artifact", "ariel", "lattice", "channel_finder", "okf", "system_health"),
    ),
    Case(
        "mcp_server.http.web_terminal_url",
        "the config it loads for itself",
        _mcp_web_terminal_url,
        ("web",),
    ),
    Case(
        "deploy_scaffold_templates._service_probes",
        "the profile's config: overlay, re-wrapped for resolve_port_base",
        _scaffold_openobserve_probe,
        ("openobserve",),
    ),
    Case(
        "deploy_scaffold_templates._dispatch_probes",
        "the profile's config: overlay, re-wrapped for resolve_port_base",
        _scaffold_dispatcher_probe,
        ("dispatcher",),
    ),
    Case(
        "deploy_scaffold_templates._web_probes",
        "the profile's config: overlay, re-wrapped for resolve_port_base",
        _scaffold_web_probes,
        ("nginx", "web"),
    ),
    Case(
        "health.configuration._check_ariel_dsn_port",
        "the config it is handed",
        _health_ariel_dsn_check,
        ("postgres",),
    ),
    Case(
        "resolve_graphdb_connection",
        "base= keyword, from the caller's resolve_port_base(config)",
        _graphdb_connection,
        ("graphdb_bolt",),
    ),
    Case(
        "resolve_qmd_service_config",
        "the config it is handed",
        _qmd_service,
        ("qmd",),
    ),
    Case(
        "bluesky_bridge_connection.bridge_url_from_config",
        "the config it is handed",
        _bluesky_bridge_url,
        ("bluesky",),
    ),
    Case(
        "web_terminals.render._auth_tls_context",
        "base= keyword, from the render's resolve_port_base(root)",
        _auth_sidecar_port,
        ("auth",),
    ),
    Case(
        "dispatch.server._default_worker_port",
        "the config it loads for itself",
        _dispatcher_worker_target,
        ("worker",),
    ),
    Case(
        "dispatch_worker.__main__.main",
        "the config it loads for itself",
        _worker_entrypoint_port,
        ("worker",),
    ),
    Case(
        "panel CLI --port defaults",
        "none — the default is None so the body resolves it",
        _panel_cli_port_defaults,
    ),
)


@pytest.mark.unit
@pytest.mark.parametrize("case", CASES, ids=[case.id for case in CASES])
def test_resolver_derives_its_port_from_the_configured_base(
    case: Case, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each resolver lands in the 20xxx block, never on the layout's default base.

    The expectation is computed from :mod:`osprey.port_layout` at
    :data:`PORT_BASE`, so this test carries no number list of its own — it fails
    when a resolver stops honouring the base it can see, not when the layout
    changes.
    """
    actual, expected = case.run(tmp_path, monkeypatch)

    assert actual == expected, (
        f"{case.id} did not derive its port from deployment.port_base={PORT_BASE} "
        f"(base reaches it as: {case.base_arrives_as})"
    )


@pytest.mark.unit
def test_the_configured_base_is_not_the_layout_default() -> None:
    """The premise of every case above: 20000 and the default base differ.

    Without this, a resolver that ignored the config entirely would still pass —
    the whole file would assert nothing.
    """
    assert _base() == PORT_BASE
    assert PORT_BASE != DEFAULT_PORT_BASE


@pytest.mark.unit
@pytest.mark.parametrize(
    "case", [case for case in CASES if case.slots], ids=[c.id for c in CASES if c.slots]
)
def test_every_case_names_real_layout_slots(case: Case) -> None:
    """A case whose slot was renamed must fail, not quietly stop covering anything.

    Each slot is probed at the first index of its own band —
    :func:`~osprey.port_layout.index_bounds`, not a hard-coded 0 — because the
    worker band starts at 1 and index 0 is not a port it has.
    """
    for slot in case.slots:
        entry = SLOTS_BY_NAME[slot]
        first = index_bounds(entry)[0]
        assert default_port(slot, first, base=PORT_BASE) == PORT_BASE + entry.offset + first
