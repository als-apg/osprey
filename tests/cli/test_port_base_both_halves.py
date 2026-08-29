"""One knob moves a whole deployment — both halves of it, in agreement.

``osprey set config.deployment.port_base=20000`` followed by ``osprey build``
has to move two things that are rendered by different code on different passes:

* the **config half** — the numbers written into ``build/config.yml``, which is
  what every service reads its own port from; and
* the **compose half** — the numbers the compose render derives, which is what
  the container runtime actually publishes on the host.

They are produced separately, so they can disagree separately, and a deployment
whose config says 20800 while its compose publishes 10800 is one where nothing
answers on the port the operator was told to open. This module builds a real
deployment repo at a moved base and asserts the two halves against each other
and against :func:`osprey.port_layout.default_port` — never against a literal,
so a slot's offset is stated once, in the layout, and read here.

The sweep at the end is the part that catches what named assertions cannot: a
resolver that could not reach the config and fell back to
:data:`~osprey.port_layout.DEFAULT_PORT_BASE` lands in 10000–10999, a band this
deployment has abandoned. Nothing the framework publishes may be found there.
The virtual accelerator's Channel Access port (5064) is the one port the base
cannot move, and it is outside that band anyway, so it needs no exemption.

The base here is deliberately far from the default: a number that came from the
layout default instead of from the profile cannot pass by coincidence.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli import build_cmd
from osprey.cli.set_cmd import set as set_command
from osprey.deployment.web_terminals.ports import base_ports_from_config, resolve_nginx_port
from osprey.deployment.web_terminals.render import render_web_terminals
from osprey.port_layout import (
    BLOCK_SIZE,
    DEFAULT_PORT_BASE,
    PORT_BASE_CONFIG_KEY,
    default_port,
    resolve_port_base,
)
from tests.fixtures.lifecycle_repo import EXEMPLAR_DIRNAME, build_exemplar_repo

#: The base this deployment moves to. Two blocks away from the default, so no
#: assertion below can be satisfied by a number that never moved.
MOVED_BASE = 20000

#: A base under 1024, which no deployment can bind without privileges.
UNUSABLE_BASE = 1000

#: The band this deployment has abandoned by moving. Any framework host port
#: found in here came from the layout default rather than from the profile.
ABANDONED_BAND = range(DEFAULT_PORT_BASE, DEFAULT_PORT_BASE + BLOCK_SIZE)

#: Web-terminal env variable to the layout slot whose band it is drawn from.
#: The compose render writes one of these per user per family, so this table
#: turns the rendered stack back into "which slot, at which index".
PANEL_ENV_SLOTS = {
    "OSPREY_WEB_PORT": "web",
    "OSPREY_TERMINAL_WEB_PORT": "web",
    "OSPREY_ARTIFACT_SERVER_PORT": "artifact",
    "OSPREY_ARIEL_PORT": "ariel",
    "OSPREY_LATTICE_DASHBOARD_PORT": "lattice",
    "OSPREY_CHANNEL_FINDER_PORT": "channel_finder",
    "OSPREY_FACILITY_KNOWLEDGE_PORT": "okf",
    "OSPREY_HEALTH_PORT": "system_health",
}

#: Every ``services.*`` port the exemplar deploys, as ``(slot, dotted key)``.
#: The slot names the layout row the number must come from; the key is where
#: the rendered config spells it.
SERVICE_PORT_KEYS = (
    ("postgres", "services.postgresql.port_host"),
    ("mongo", "services.mongodb.port_host"),
    ("graphdb_bolt", "services.graphdb.port_host"),
    ("graphdb_http", "services.graphdb.http_port_host"),
    ("openobserve", "services.openobserve.port"),
    ("qmd", "services.qmd.port"),
    ("dispatcher", "services.event_dispatcher.port"),
    ("bluesky", "services.bluesky.port"),
    ("tiled", "services.bluesky.tiled_port"),
    ("bluesky_web", "services.bluesky_web.port"),
)

#: The compose file each store/service publishes from, so the compose half can
#: be read back per service rather than as one undifferentiated pile.
SERVICE_COMPOSE_DIRS = (
    "postgresql",
    "mongodb",
    "graphdb",
    "openobserve",
    "qmd",
    "event_dispatcher",
    "bluesky",
    "bluesky_web",
)


def _slot_port(slot: str, index: int = 0) -> int:
    """The port a slot takes at this test's moved base.

    Args:
        slot: Slot name, as spelled in :data:`osprey.port_layout.LAYOUT`.
        index: Position within the slot's band.

    Returns:
        ``MOVED_BASE + offset + index``.
    """
    return default_port(slot, index, base=MOVED_BASE)


def _dotted(config: Any, key: str) -> Any:
    """Read a dotted key out of a rendered config.

    Args:
        config: The mapping parsed from ``build/config.yml``.
        key: A dotted path such as ``services.postgresql.port_host``.

    Returns:
        The value at that path, or ``None`` if any step of it is missing.
    """
    node: Any = config
    for part in key.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node


def _host_port(entry: Any) -> int | None:
    """The HOST half of one compose ``ports:`` entry.

    Compose publishes as ``[bind:]host:container[/proto]``, and only the host
    half is what ``port_base`` moves — the container half is the port the
    service listens on inside its own namespace and is not the layout's
    business.

    Args:
        entry: One element of a compose service's ``ports:`` list.

    Returns:
        The host port, or ``None`` for an entry that publishes nothing
        parseable (a bare container port, or a long-form mapping).
    """
    if isinstance(entry, dict):
        published = entry.get("published")
        return int(published) if str(published).isdigit() else None
    fields = str(entry).split("/")[0].split(":")
    if len(fields) < 2:
        return None
    return int(fields[-2]) if fields[-2].isdigit() else None


def _config_port_values(config: Any, prefix: str = "") -> dict[str, int]:
    """Every port-shaped number in a rendered config, keyed by its dotted path.

    "Port-shaped" is a leaf whose key mentions a port and whose value is an
    integer in the TCP range. That is deliberately wider than the layout's own
    keys: the sweep exists to find a number nobody thought to name, so it must
    not be limited to the names already known.

    Args:
        config: The mapping parsed from a rendered ``config.yml``.
        prefix: Dotted path of ``config`` within its document; internal.

    Returns:
        ``{dotted key: port}`` for every such leaf.
    """
    found: dict[str, int] = {}
    if isinstance(config, dict):
        for key, value in config.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict | list):
                found.update(_config_port_values(value, path))
            elif (
                "port" in str(key).lower()
                and isinstance(value, int)
                and not isinstance(value, bool)
                and 1 <= value <= 65535
            ):
                found[path] = value
    elif isinstance(config, list):
        for index, item in enumerate(config):
            found.update(_config_port_values(item, f"{prefix}[{index}]"))
    return found


def _web_render_ports(artifacts: dict[str, str]) -> dict[str, int]:
    """Every host port the web-terminal render puts on the host, by where it sits.

    The web stack runs on the host network, so it publishes nothing through a
    compose ``ports:`` list: its host ports are the ``OSPREY_*_PORT`` variables
    it hands each container, the ``--port`` the auth sidecar is launched on, the
    loopback addresses its health probes and landing URLs name, and the ports
    nginx listens on. All four are read, because a render that moved only some
    of them is exactly the failure this module is for.

    Args:
        artifacts: The render's output, keyed by relative path — the mapping
            :func:`osprey.deployment.web_terminals.render.render_web_terminals`
            returns.

    Returns:
        ``{where it was found: port}``, where the key is a human-readable
        location so a failure names the line rather than just the number.
    """
    found: dict[str, int] = {}
    for name, text in artifacts.items():
        for match in re.finditer(r"\b(OSPREY_[A-Z0-9_]*PORT)=(\d+)\b", text):
            found[f"{name}: {match.group(1)}={match.group(2)}"] = int(match.group(2))
        for match in re.finditer(r"--port[\"',\s]+(\d+)", text):
            found[f"{name}: --port {match.group(1)}"] = int(match.group(1))
        for match in re.finditer(r"\b\d+\.\d+\.\d+\.\d+:(\d{2,5})\b", text):
            found[f"{name}: address :{match.group(1)}"] = int(match.group(1))
        for match in re.finditer(r"^\s*listen\s+(?:\[::\]:)?(\d+)\s*;", text, re.MULTILINE):
            found[f"{name}: listen {match.group(1)}"] = int(match.group(1))
    return found


@dataclass(frozen=True)
class Rendered:
    """One real ``osprey build``, with both halves of its render read back.

    Attributes:
        repo: The deployment repo the build ran in.
        config: ``build/config.yml``, parsed.
        composes: Each service's rendered compose document, keyed by the
            service directory name under ``build/services/``.
        web: The web-terminal render, keyed by relative path.
    """

    repo: Path
    config: dict[str, Any]
    composes: dict[str, dict[str, Any]]
    web: dict[str, str]

    def published(self, service_dir: str) -> list[int]:
        """The host ports one service's compose publishes.

        Args:
            service_dir: Directory name under ``build/services/``.

        Returns:
            Every host port in that document, in document order.
        """
        document = self.composes.get(service_dir, {})
        return [
            port
            for body in (document.get("services") or {}).values()
            for entry in (body.get("ports") or [])
            if (port := _host_port(entry)) is not None
        ]


def _run_build(repo: Path):
    """Run ``osprey build`` in *repo*, in process.

    ``--skip-deps``/``--skip-lifecycle`` drop the virtualenv install and the
    profile's shell phases. Neither renders a port, and both cost minutes.

    Args:
        repo: The deployment repo to build.

    Returns:
        The Click result, so a caller can assert on the exit code and output.
    """
    previous = Path.cwd()
    os.chdir(repo)
    try:
        return CliRunner().invoke(build_cmd.build, ["--skip-deps", "--skip-lifecycle"])
    finally:
        os.chdir(previous)


def _read_render(repo: Path) -> Rendered:
    """Read both halves of the render a build left in *repo*.

    Args:
        repo: A deployment repo that has been built.

    Returns:
        The parsed config, the service compose documents, and the
        web-terminal render derived from that same config — so the two halves
        compared below provably come from one build.
    """
    build = repo / "build"
    config = yaml.safe_load((build / "config.yml").read_text(encoding="utf-8"))
    composes = {
        path.parent.name: yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for path in sorted((build / "services").glob("*/docker-compose.yml"))
    }
    return Rendered(repo=repo, config=config, composes=composes, web=render_web_terminals(config))


@pytest.fixture(scope="module")
def dotted_render(tmp_path_factory: pytest.TempPathFactory) -> Rendered:
    """A deployment moved to :data:`MOVED_BASE` the way an operator moves one.

    ``osprey set`` writes the dotted key into the profile's ``config:`` block;
    the build then resolves it. Module-scoped because the build is the
    expensive part and every assertion here asks about the same run.
    """
    repo = build_exemplar_repo(tmp_path_factory.mktemp("dotted") / EXEMPLAR_DIRNAME)
    written = CliRunner().invoke(
        set_command,
        ["--repo", str(repo), f"config.{PORT_BASE_CONFIG_KEY}={MOVED_BASE}"],
        catch_exceptions=False,
    )
    assert written.exit_code == 0, written.output

    built = _run_build(repo)
    assert built.exit_code == 0, built.output
    return _read_render(repo)


@pytest.fixture(scope="module")
def nested_render(tmp_path_factory: pytest.TempPathFactory) -> Rendered:
    """The same move, spelled as a nested block instead of a dotted key.

    A profile is a hand-edited document, so an author may nest ``deployment:``
    under ``config:`` rather than dot it. Both spellings address one subtree and
    must therefore land on one answer.
    """
    repo = build_exemplar_repo(tmp_path_factory.mktemp("nested") / EXEMPLAR_DIRNAME)
    profile = repo / "profile.yml"
    text = profile.read_text(encoding="utf-8")
    assert "\nconfig:\n" in text, "the exemplar profile must carry a top-level config: block"
    profile.write_text(
        text.replace("\nconfig:\n", f"\nconfig:\n  deployment:\n    port_base: {MOVED_BASE}\n", 1),
        encoding="utf-8",
    )

    built = _run_build(repo)
    assert built.exit_code == 0, built.output
    return _read_render(repo)


# ── The config half ──────────────────────────────────────────────────────────


class TestTheConfigHalfMoves:
    """``build/config.yml`` carries the moved base, key by key."""

    def test_the_base_itself_is_recorded(self, dotted_render: Rendered) -> None:
        """The render says which base it was built at, not only its results."""
        assert _dotted(dotted_render.config, PORT_BASE_CONFIG_KEY) == MOVED_BASE

    @pytest.mark.parametrize(
        ("slot", "key"), SERVICE_PORT_KEYS, ids=[k for _, k in SERVICE_PORT_KEYS]
    )
    def test_each_service_key_sits_on_its_slot(
        self, dotted_render: Rendered, slot: str, key: str
    ) -> None:
        """Every store and service the preset deploys is at ``base + offset``.

        Parametrized one key per case so a failure names the key and the number
        it came out at, which is the whole diagnostic: a wrong number here is a
        resolver that read a base this deployment is not running at.
        """
        assert _dotted(dotted_render.config, key) == _slot_port(slot)

    def test_the_worker_band_starts_above_the_dispatcher(self, dotted_render: Rendered) -> None:
        """Worker 1 is the first member of the dispatch band, at ``base + 11``.

        Asserted beside the dispatcher because the two are one story: a worker
        base that did not move while the dispatcher did would leave the
        dispatcher forwarding to a port nothing listens on.
        """
        config = dotted_render.config
        assert _dotted(config, "services.event_dispatcher.port") == _slot_port("dispatcher")
        assert _dotted(config, "services.dispatch_worker.worker_port_base") == _slot_port(
            "worker", 1
        )

    def test_the_artifact_server_moves_with_its_family(self, dotted_render: Rendered) -> None:
        """The artifact gallery's own key is the panel family's first port."""
        assert _dotted(dotted_render.config, "artifact_server.port") == _slot_port("artifact", 0)

    def test_the_landing_page_and_the_first_terminal_move(self, dotted_render: Rendered) -> None:
        """nginx at the base, user 0's terminal one hundred above it.

        These two are what an operator is handed after a build — the address
        they open, and the address the first person on the roster opens. Read
        through the resolvers the runtime uses, not off the keys: a profile
        that leaves ``nginx_port`` and ``web_base_port`` unspelled is handed
        the block's gateway and web slots at ITS base, and the build does not
        write those keys for it — so the config half of this pair is the base
        alone, and the resolvers are where it has to land.
        """
        config = dotted_render.config
        web_terminals = _dotted(config, "modules.web_terminals") or {}
        assert resolve_nginx_port(config) == _slot_port("nginx")
        base_ports = base_ports_from_config(web_terminals, base=resolve_port_base(config))
        assert base_ports["web"] == _slot_port("web", 0)


# ── The compose half ─────────────────────────────────────────────────────────


class TestTheComposeHalfAgrees:
    """What the runtime publishes is what the config says it publishes."""

    def test_postgres_publishes_the_moved_host_port_onto_its_own(
        self, dotted_render: Rendered
    ) -> None:
        """``20800:5432`` — the host half moved, the container half did not.

        Spelled as a pair rather than as one number because both properties
        matter: moving the container half too would renumber the port Postgres
        itself listens on, which no client expects.
        """
        document = dotted_render.composes["postgresql"]
        entries = [
            str(entry)
            for body in (document.get("services") or {}).values()
            for entry in (body.get("ports") or [])
        ]
        assert any(entry.endswith(f"{_slot_port('postgres')}:5432") for entry in entries), entries

    @pytest.mark.parametrize("service_dir", SERVICE_COMPOSE_DIRS)
    def test_every_published_host_port_is_a_number_the_config_names(
        self, dotted_render: Rendered, service_dir: str
    ) -> None:
        """The two halves agree, per service.

        Read as a set membership rather than positionally: a compose file may
        publish more than one port (graphdb publishes two), and which order
        they appear in is not a contract.
        """
        named = {
            _dotted(dotted_render.config, key)
            for _, key in SERVICE_PORT_KEYS
            if _dotted(dotted_render.config, key) is not None
        }
        published = dotted_render.published(service_dir)
        assert published, f"{service_dir} publishes nothing"
        assert set(published) <= named, (
            f"{service_dir} publishes {sorted(set(published) - named)}, "
            f"which build/config.yml does not name"
        )


# ── The panel half ───────────────────────────────────────────────────────────


class TestThePanelsMove:
    """The web-terminal render is derived from the same rendered config."""

    def test_nginx_listens_on_the_base_itself(self, dotted_render: Rendered) -> None:
        """The landing page is the first port of the block, by definition."""
        listens = {
            int(match)
            for match in re.findall(
                r"^\s*listen\s+(?:\[::\]:)?(\d+)\s*;",
                dotted_render.web["nginx/nginx.conf"],
                re.MULTILINE,
            )
        }
        assert listens == {_slot_port("nginx")}

    def test_the_auth_sidecar_sits_one_above_it(self, dotted_render: Rendered) -> None:
        """The gateway tier is two ports, and both belong to the block."""
        ports = _web_render_ports({"web": dotted_render.web["docker-compose.web.yml"]})
        assert _slot_port("auth") in ports.values(), ports

    def test_every_family_places_every_user_on_its_own_slot(self, dotted_render: Rendered) -> None:
        """User *i* is at ``family base + i``, for every family and every user.

        This is the assertion the per-user layout exists for: removing a user
        must not shift anyone else, which is only true while each family's
        ports are its own band read by index.
        """
        document = yaml.safe_load(dotted_render.web["docker-compose.web.yml"])
        terminals = [
            (name, body)
            for name, body in (document.get("services") or {}).items()
            if name.startswith("web-")
        ]
        assert terminals, "the exemplar must render at least one web terminal"

        offences: list[str] = []
        for index, (name, body) in enumerate(terminals):
            for entry in body.get("environment") or []:
                variable, _, value = str(entry).partition("=")
                slot = PANEL_ENV_SLOTS.get(variable)
                if slot is None or not value.isdigit():
                    continue
                expected = _slot_port(slot, index)
                if int(value) != expected:
                    offences.append(f"{name} {variable}={value}, expected {expected}")
        assert not offences, "Panel ports off their slot:\n  " + "\n  ".join(offences)


# ── Nothing left behind at the default base ──────────────────────────────────


class TestNothingIsLeftAtTheDefaultBase:
    """The sweep: a number in 10000–10999 is a resolver that missed the config.

    Named assertions can only check the keys someone thought of. These three
    check the whole render, which is what catches a port filled in from the
    layout default by code that could not reach the profile.
    """

    def test_no_config_port_is_in_the_abandoned_band(self, dotted_render: Rendered) -> None:
        """Every port-shaped number in ``build/config.yml``, not only the known keys."""
        stranded = {
            key: port
            for key, port in _config_port_values(dotted_render.config).items()
            if port in ABANDONED_BAND
        }
        assert not stranded, f"config keys still at the default base: {stranded}"

    def test_no_published_host_port_is_in_the_abandoned_band(self, dotted_render: Rendered) -> None:
        """Every host port every rendered service compose publishes."""
        stranded = {
            service: [port for port in dotted_render.published(service) if port in ABANDONED_BAND]
            for service in dotted_render.composes
        }
        stranded = {service: ports for service, ports in stranded.items() if ports}
        assert not stranded, f"compose publishes still at the default base: {stranded}"

    def test_no_web_render_port_is_in_the_abandoned_band(self, dotted_render: Rendered) -> None:
        """Every host port the web-terminal render puts on the host."""
        stranded = {
            where: port
            for where, port in _web_render_ports(dotted_render.web).items()
            if port in ABANDONED_BAND
        }
        assert not stranded, f"web render still at the default base: {stranded}"


# ── The second spelling ──────────────────────────────────────────────────────


class TestTheNestedSpellingResolvesTheSame:
    """A nested block and a dotted key address one subtree, so one answer."""

    def test_the_two_spellings_render_the_same_config_ports(
        self, dotted_render: Rendered, nested_render: Rendered
    ) -> None:
        """Every port-shaped number in the render, compared whole.

        Compared as the whole table rather than key by key: the risk is not
        that one key disagrees, it is that a spelling reaches some resolvers
        and not others — which only a full comparison can see.
        """
        assert _config_port_values(nested_render.config) == _config_port_values(
            dotted_render.config
        )

    def test_the_two_spellings_publish_the_same_host_ports(
        self, dotted_render: Rendered, nested_render: Rendered
    ) -> None:
        """The compose half follows the config half under either spelling."""
        assert {service: nested_render.published(service) for service in SERVICE_COMPOSE_DIRS} == {
            service: dotted_render.published(service) for service in SERVICE_COMPOSE_DIRS
        }

    def test_the_two_spellings_render_the_same_web_stack_ports(
        self, dotted_render: Rendered, nested_render: Rendered
    ) -> None:
        """And so does the panel half."""
        assert _web_render_ports(nested_render.web) == _web_render_ports(dotted_render.web)


# ── The base that cannot be honored ──────────────────────────────────────────


class TestAnUnusableBaseStopsTheBuild:
    """A base the deployment could not bind refuses; it does not fall back."""

    def test_a_base_below_1024_is_refused_and_nothing_is_rendered(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The refusal names the key and the number, and leaves no ``build/``.

        Falling back to the default here would publish the deployment at 10000
        while its profile says 1000, and the first sign of it would be a
        service answering on a port nobody named. So the build stops, and it
        stops before it has written anything at all.

        The message is read off the log rather than off the runner's captured
        output: ``osprey build`` reports through the framework's own console,
        so its stdout carries only Click's ``Aborted!``.
        """
        repo = build_exemplar_repo(tmp_path / EXEMPLAR_DIRNAME)
        written = CliRunner().invoke(
            set_command,
            ["--repo", str(repo), f"config.{PORT_BASE_CONFIG_KEY}={UNUSABLE_BASE}"],
            catch_exceptions=False,
        )
        assert written.exit_code == 0, written.output

        with caplog.at_level(logging.ERROR):
            built = _run_build(repo)

        assert built.exit_code != 0, built.output
        refusal = "\n".join(
            record.getMessage() for record in caplog.records if record.levelno >= logging.ERROR
        )
        assert PORT_BASE_CONFIG_KEY in refusal, refusal
        assert str(UNUSABLE_BASE) in refusal, refusal
        assert not (repo / "build").exists(), "a refused base must render nothing"
