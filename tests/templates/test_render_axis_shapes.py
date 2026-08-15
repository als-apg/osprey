"""Byte-for-byte baselines for the deployment shapes the axes actually produce.

``test_render_defaults_golden.py`` pins every bundled template with the axes
UNSET — the shape a deployment that never heard of them gets. This suite pins
the shapes that appear once they are set: the dispatch pair on the host
namespace, the chat bridges beside it, a repo carrying a shared env-chain file,
and a two-worker stack fanning out across host ports.

Each scenario is a small, named delta over that same default context, rendered
through the same context builder and the same Environment (both imported from
the defaults module, so a scenario golden and a default golden can differ only
where the scenario differs). A scenario pins only the templates its delta
reaches: four templates honor the network axis
(``event_dispatcher``, ``dispatch_worker``, and the two bridges) and one renders
the env chain, so rendering the other eight per scenario would commit eight
copies of a file the defaults already pin.

**What these catch that the substring suites cannot.** ``network: host`` is not
one edit to one line — it moves a service's network attachment, deletes its
published ports, deletes the file-level ``networks:`` stanza, narrows two bind
addresses, and rewrites four addresses that other services are reached at. The
per-block assertions in ``tests/deployment/test_compose_generator.py`` prove
each of those individually; only a whole-file baseline proves that flipping the
axis does those things AND NOTHING ELSE. On the host namespace a stray leftover
is not a cosmetic drift — a ``ports:`` block compose rejects, or an address
naming a compose service that resolves to nothing there, is a stack that comes
up looking healthy and cannot route a single run.

**Same-mode per scenario, by design.** The build refuses a topology that puts a
host-mode service beside a network-joined consumer of it
(``_dispatch_parity_errors`` in ``cli/build_cmd.py``, symmetric in both
directions). A fixture mixing the two would be pinning a render no deploy can
reach, so every scenario here declares one mode for the whole dispatch stack.
The stores (``mongodb``, ``openobserve``) stay network-joined in every
scenario because their templates carry no axis at all: they publish on the
host, which is exactly why the host-mode worker is told to reach them at
``localhost``. Telemetry is left off and ``services.openobserve.port`` at its
default 5080, so no scenario trips the build's OTEL-endpoint check either.

**Update discipline** — the defaults module's applies here verbatim, with this
suite's own regeneration command::

    PYTHONPATH=src ./.venv/bin/python tests/templates/test_render_axis_shapes.py

Never hand-edit a golden. Regenerate in the SAME reviewed change as the
template edit that moved it, then account for every changed byte.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import yaml

# Imported by bare module name, not as ``tests.templates.…``: this directory
# carries no ``__init__.py``, so pytest puts it on ``sys.path`` and names the
# module by its basename — the same name the regeneration entry point below
# resolves when the file is run directly.
from test_render_defaults_golden import (
    _GOLDEN_DIR as _DEFAULTS_DIR,
)
from test_render_defaults_golden import (
    _PROJECT_NAME,
    _pinned_context,
    _render_templates,
    _service_keys,
)

_GOLDEN_DIR = Path(__file__).parent / "render_axis_shapes"

#: The chain file a repo carries when nothing shared it — the default probe
#: shape, and the one every scenario but ``env-shared-chain`` renders against.
_LOCAL_ENV_CHAIN = ("./.env",)


def _dispatch_defaults():
    """The shipped dispatch knobs, so a changed default moves these goldens.

    The scenario overlays below mirror what ``_inject_dispatch`` writes into a
    built project's ``config.yml`` for the pair. Reading the values off
    ``DispatchConfig`` rather than restating them is what keeps a golden a
    record of the SHIPPED shape: raise ``worker_port_base`` in the schema and
    this suite fails until the goldens are regenerated, instead of quietly
    pinning a port no build emits any more.
    """
    from osprey.cli.build_profile_schema import DispatchConfig

    return DispatchConfig(triggers="tutorial_triggers.yml")


def _pair_blocks(*, on_host: bool, worker_count: int = 1) -> dict[str, dict]:
    """The two ``services.<half>`` blocks a dispatch build writes.

    Only the keys a compose render reads — ``path`` and ``additional_dirs`` are
    consumed by the build-directory staging, never by a template. The host-only
    keys (``network`` and the worker's ``worker_port_stride``) are written ONLY
    under host, exactly as ``_inject_dispatch`` writes them: with the axis unset
    a built ``config.yml`` is byte-for-byte what it was before the knob existed.
    """
    from osprey.cli.build_injectors import _WORKER_PORT_STRIDE

    dispatch = _dispatch_defaults()
    dispatcher: dict = {
        "port": dispatch.dispatcher_port,
        "facility_name": dispatch.facility_name,
        "pv_strip_prefix": dispatch.pv_strip_prefix,
    }
    worker: dict = {
        "worker_count": worker_count,
        "worker_port_base": dispatch.worker_port_base,
        "workspace_mode": dispatch.workspace_mode,
        "timeout_sec": dispatch.timeout_sec,
        "inactivity_sec": dispatch.inactivity_sec,
    }
    if on_host:
        dispatcher["network"] = "host"
        worker["network"] = "host"
        worker["worker_port_stride"] = _WORKER_PORT_STRIDE
    return {"event_dispatcher": dispatcher, "dispatch_worker": worker}


@dataclass(frozen=True)
class Scenario:
    """One deployment shape, its config delta, and the templates it pins."""

    name: str
    """Directory under ``render_axis_shapes/`` holding this shape's goldens."""

    services: dict[str, dict]
    """``services.<key>`` blocks overlaid on the all-empty default context."""

    deployed: tuple[str, ...]
    """``deployed_services`` — what the templates gate their addresses on."""

    templates: tuple[str, ...]
    """Service keys whose rendered compose file this shape pins."""

    chain: tuple[str, ...] = (".env",)
    """Env-chain files the probe repo carries, in chain order."""

    expected_chain: tuple[str, ...] = field(default=_LOCAL_ENV_CHAIN)
    """The ``env_file:`` entries the render must list for :attr:`chain`."""


SCENARIOS: tuple[Scenario, ...] = (
    # The pair itself on the host namespace, with both stores deployed so the
    # worker's two address rewrites are in the render rather than merely
    # possible. This is the shape `dispatch.network: host` produces.
    Scenario(
        name="pair-on-host",
        services=_pair_blocks(on_host=True),
        deployed=("event_dispatcher", "dispatch_worker", "mongodb", "openobserve"),
        templates=("event_dispatcher", "dispatch_worker"),
    ),
    # A single service template flipped to host beside a same-mode pair. Both
    # bridges are pinned because they are twins by intent — the axis and the two
    # dispatch addresses are meant to be spelled identically in each, and only a
    # baseline of both catches the day one of them drifts.
    Scenario(
        name="bridge-on-host",
        services={
            **_pair_blocks(on_host=True),
            "gchat_bridge": {"network": "host"},
            "nextcloud_bridge": {"network": "host"},
        },
        deployed=(
            "event_dispatcher",
            "dispatch_worker",
            "gchat_bridge",
            "nextcloud_bridge",
        ),
        templates=("gchat_bridge", "nextcloud_bridge"),
    ),
    # A repo carrying committed defaults alongside its local secrets. Axes
    # unset: this shape is about the env chain, and the worker is the one
    # template that renders it.
    Scenario(
        name="env-shared-chain",
        services=_pair_blocks(on_host=False),
        deployed=("event_dispatcher", "dispatch_worker"),
        templates=("dispatch_worker",),
        chain=(".env.shared", ".env"),
        expected_chain=("./.env.shared", "./.env"),
    ),
    # Two workers sharing the host's one port space, which is the only mode
    # where the per-index walk is observable: on the compose bridge every worker
    # listens on the base port and is told apart by its service name.
    Scenario(
        name="two-workers-on-host",
        services=_pair_blocks(on_host=True, worker_count=2),
        deployed=("event_dispatcher", "dispatch_worker"),
        templates=("dispatch_worker",),
    ),
)

#: Every (scenario, service key) pair this suite pins, for parametrization.
CASES: tuple[tuple[Scenario, str], ...] = tuple(
    (scenario, key) for scenario in SCENARIOS for key in scenario.templates
)


def _case_id(case: tuple[Scenario, str]) -> str:
    scenario, key = case
    return f"{scenario.name}/{key}"


@contextmanager
def _probe_repo(chain: tuple[str, ...]) -> Iterator[Path]:
    """A deployment repo carrying exactly *chain*.

    The repo must EXIST: ``resolve_repo_root`` falls back to the working
    directory for a ``project_root`` that does not, which would make the chain
    probe read whatever the test happened to run from and the goldens
    machine-dependent.
    """
    with tempfile.TemporaryDirectory() as directory:
        repo_root = Path(directory)
        for name in chain:
            (repo_root / name).write_text(f"OSPREY_GOLDEN_FIXTURE={name}\n", encoding="utf-8")
        yield repo_root


def _scenario_context(scenario: Scenario, repo_root: Path) -> dict:
    """The default render context with *scenario*'s delta applied.

    The overlay REPLACES a service's block rather than merging into it: the
    default blocks are empty, so there is nothing to merge, and a replace makes
    the scenario's own declaration the whole truth about that service.
    """
    return _pinned_context(
        {
            "project_name": _PROJECT_NAME,
            "project_root": str(repo_root),
            "services": {key: {} for key in _service_keys()} | scenario.services,
            "deployment": {},
            "system": {"timezone": "UTC"},
            "deployed_services": list(scenario.deployed),
        }
    )


def _render_scenario(scenario: Scenario) -> dict[str, str]:
    """Render *scenario*'s pinned templates, keyed by golden filename."""
    rel_paths = [f"services/{key}/docker-compose.yml.j2" for key in scenario.templates]
    with _probe_repo(scenario.chain) as repo_root:
        return _render_templates(_scenario_context(scenario, repo_root), rel_paths)


@pytest.fixture(scope="module")
def rendered() -> dict[str, dict[str, str]]:
    """Every scenario's renders, once for the module, keyed by scenario name."""
    return {scenario.name: _render_scenario(scenario) for scenario in SCENARIOS}


def _scenario(name: str) -> Scenario:
    """The declared scenario called *name*."""
    return next(scenario for scenario in SCENARIOS if scenario.name == name)


def _golden_path(scenario: Scenario, key: str) -> Path:
    return _GOLDEN_DIR / scenario.name / f"{key}.yml"


def _golden_text(scenario: Scenario, key: str) -> str:
    return _golden_path(scenario, key).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# The baselines
# ---------------------------------------------------------------------------


def test_committed_goldens_are_exactly_the_declared_cases() -> None:
    """The files on disk and the declared cases are the same set, both ways.

    A golden nothing declares is a file no test would ever fail on; a declared
    case with no golden is a shape nothing pins. Either way the suite is
    quietly smaller than it reads, which is the failure mode a golden suite is
    least able to notice about itself.
    """
    committed = {f"{path.parent.name}/{path.name}" for path in _GOLDEN_DIR.glob("*/*.yml")}
    declared = {f"{scenario.name}/{key}.yml" for scenario, key in CASES}

    assert committed == declared, (
        "axis-shape goldens and declared scenarios have drifted apart — "
        "regenerate with: PYTHONPATH=src ./.venv/bin/python "
        "tests/templates/test_render_axis_shapes.py"
    )


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_scenario_render_matches_golden(
    case: tuple[Scenario, str], rendered: dict[str, dict[str, str]]
) -> None:
    """The shape renders byte-identically to its committed baseline."""
    scenario, key = case
    golden = _golden_path(scenario, key)
    assert golden.is_file(), f"missing golden for {scenario.name}/{key}"

    assert rendered[scenario.name][f"{key}.yml"] == golden.read_text(encoding="utf-8"), (
        f"{key} no longer renders the {scenario.name} shape. Every changed byte "
        "must trace to a deliberate template edit — see this module's update "
        "discipline before regenerating."
    )


def test_goldens_are_parseable_compose_documents() -> None:
    """The baselines are real YAML, so a truncated one cannot pass as a match.

    Byte equality alone would hold two identically broken files to each other.
    """
    for path in sorted(_GOLDEN_DIR.glob("*/*.yml")):
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(document, dict) and document, (
            f"{path.parent.name}/{path.name} is not a compose document"
        )


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_every_shape_differs_from_the_default_render(case: tuple[Scenario, str]) -> None:
    """A scenario whose delta changed nothing is pinning the default twice.

    Cheap to write by accident — a key spelled onto the wrong service, or onto
    a template that reads a different one — and invisible without this: the
    golden would still match its render, forever, while covering nothing.
    """
    scenario, key = case
    assert _golden_text(scenario, key) != (_DEFAULTS_DIR / f"{key}.yml").read_text(
        encoding="utf-8"
    ), (
        f"{scenario.name}/{key} renders the default shape — its delta reaches this template not at all"
    )


# ---------------------------------------------------------------------------
# What each shape must actually contain
#
# Byte equality pins the render against whatever was committed; these pin what
# was committed against the deployment's requirements. Without them a
# regeneration that dropped an address rewrite would sail through as "the new
# baseline" — the goldens would agree with the templates, and both would be
# wrong.
# ---------------------------------------------------------------------------


def test_host_mode_suppresses_every_compose_network_construct() -> None:
    """Under ``host`` the render carries no network, no ports, no stanza.

    Not three cosmetic omissions: a ``ports:`` block alongside
    ``network_mode: host`` is rejected outright by some runtimes and silently
    ignored by others, and a file-level ``networks:`` stanza declares a network
    no service in the file joins.
    """
    for scenario, key in CASES:
        if scenario.services.get(key, {}).get("network") != "host":
            continue
        document = yaml.safe_load(_golden_text(scenario, key))
        label = f"{scenario.name}/{key}"

        assert "networks" not in document, f"{label} declares a network nothing joins"
        for name, service in document["services"].items():
            assert service.get("network_mode") == "host", f"{label}: {name} is not on the host"
            assert "networks" not in service, f"{label}: {name} still joins a network"
            assert "ports" not in service, f"{label}: {name} still publishes ports"


def test_pair_on_host_narrows_both_bind_addresses() -> None:
    """Both halves bind loopback on the host namespace, not every interface.

    On the compose bridge the network is the boundary and binding ``0.0.0.0``
    is what makes a service reachable by name at all. On the host there is no
    such envelope: the same default would put an unauthenticated-until-token
    MCP server on every interface the machine has.
    """
    dispatcher = _service_env("pair-on-host", "event_dispatcher", "event-dispatcher")
    worker = _service_env("pair-on-host", "dispatch_worker", "dispatch-worker-1")

    assert dispatcher["FASTMCP_HOST"] == "127.0.0.1"
    assert worker["DISPATCH_WORKER_BIND"] == "127.0.0.1"


def test_pair_on_host_reaches_both_stores_where_they_publish() -> None:
    """The worker's store addresses follow the topology, not the config block.

    ``openobserve`` and ``archiver-mongodb`` are compose DNS names that resolve
    to nothing from the host namespace. Both stores publish on the host, so
    from there they are reached at ``localhost`` — mongodb on its published
    ``port_host``, never the 27017 inside its container.
    """
    worker = _service_env("pair-on-host", "dispatch_worker", "dispatch-worker-1")

    assert worker["OSPREY_OTEL_OPENOBSERVE_HOST"] == "localhost"
    assert worker["OSPREY_ARCHIVER_MONGODB_HOST"] == "localhost"
    assert worker["OSPREY_ARCHIVER_MONGODB_PORT"] == "27017"


def test_host_mode_worker_port_is_the_address_the_build_routes_to() -> None:
    """The pinned worker-1 port IS what the build writes as ``dispatch_target``.

    The dispatcher's route lives in the copied ``triggers.yml``, not in any
    compose file, so no golden here can contain it — but the two are one
    decision. ``_inject_dispatch`` rewrites the target to
    ``http://localhost:<worker 1's port>`` under host mode, and worker 1's port
    is what this golden pins. Deriving both from ``_worker_port`` is what makes
    a change to the port rule fail here rather than at the first dispatch,
    where the symptom is a run that queues and never routes.
    """
    from osprey.cli.build_injectors import _worker_port

    worker = _service_env("pair-on-host", "dispatch_worker", "dispatch-worker-1")
    expected = _worker_port(_dispatch_defaults().worker_port_base, 1)

    assert worker["DISPATCH_WORKER_PORT"] == str(expected)
    assert f"http://localhost:{expected}" == f"http://localhost:{worker['DISPATCH_WORKER_PORT']}"


def test_bridges_on_host_address_the_pair_at_the_host() -> None:
    """Both bridges reach the co-deployed pair at ``localhost``, same ports.

    The bridge follows ITS OWN axis here, and the substitution assumes the pair
    is on the host too — which the build's parity check guarantees and this
    scenario's same-mode context reproduces. The worker address is worker 1's,
    derived by the same walk the worker template uses.
    """
    dispatch = _dispatch_defaults()
    for key, compose_name in (
        ("gchat_bridge", "gchat-bridge"),
        ("nextcloud_bridge", "nextcloud-bridge"),
    ):
        environment = _service_env("bridge-on-host", key, compose_name)
        assert environment["DISPATCHER_URL"] == f"http://localhost:{dispatch.dispatcher_port}"
        assert environment["WORKER_URL"] == f"http://localhost:{dispatch.worker_port_base}"


def test_two_workers_on_host_fan_out_across_ports() -> None:
    """Worker ``i`` binds ``base + (i - 1) * stride``, and its probe follows it.

    A healthcheck left on the base port would pass for worker 1 and fail
    forever for every worker after it — the container restarts in a loop while
    the process inside is healthy.
    """
    from osprey.cli.build_injectors import _worker_port

    base = _dispatch_defaults().worker_port_base
    document = yaml.safe_load(_golden_text(_scenario("two-workers-on-host"), "dispatch_worker"))
    services = document["services"]

    assert sorted(services) == ["dispatch-worker-1", "dispatch-worker-2"]
    for index in (1, 2):
        port = _worker_port(base, index)
        service = services[f"dispatch-worker-{index}"]
        assert service["environment"]["DISPATCH_WORKER_PORT"] == str(port)
        probe = " ".join(service["healthcheck"]["test"])
        assert f"http://localhost:{port}/health" in probe, (
            f"worker {index}'s probe does not follow its own port"
        )


def test_shared_chain_is_listed_in_ascending_precedence() -> None:
    """``.env.shared`` first, ``.env`` last — the order that makes local win.

    Compose lets a later ``env_file:`` entry win on any key an earlier one also
    sets, so this order is the whole mechanism by which a machine's local
    secrets override committed defaults. Reversed, every deployment sharing a
    ``.env.shared`` would silently run on the shared values.
    """
    scenario = _scenario("env-shared-chain")
    document = yaml.safe_load(_golden_text(scenario, "dispatch_worker"))

    assert document["services"]["dispatch-worker-1"]["env_file"] == list(scenario.expected_chain)


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_every_shape_lists_the_chain_its_repo_carried(case: tuple[Scenario, str]) -> None:
    """No shape drops or invents a chain member.

    ``env_file:`` naming a path that is not there fails ``compose up`` outright,
    so membership is fixed at render time from the files the repo actually had.
    The bridges do not render a chain; only the worker does.
    """
    scenario, key = case
    if key != "dispatch_worker":
        return
    document = yaml.safe_load(_golden_text(scenario, key))
    for name, service in document["services"].items():
        assert service["env_file"] == list(scenario.expected_chain), (
            f"{scenario.name}/{name} lists a chain its repo did not carry"
        )


def _service_env(scenario_name: str, key: str, compose_name: str) -> dict[str, str]:
    """The ``environment:`` mapping of one service in one scenario's golden.

    Values are stringified because compose env values are strings either way —
    YAML reads an unquoted port as an int, and an assertion that had to know
    which is which would be an assertion about quoting, not about the address.
    """
    document = yaml.safe_load(_golden_text(_scenario(scenario_name), key))
    return {
        name: str(value)
        for name, value in document["services"][compose_name]["environment"].items()
    }


def _regenerate() -> None:
    """Overwrite every scenario golden. See this module's update discipline."""
    for scenario in SCENARIOS:
        directory = _GOLDEN_DIR / scenario.name
        directory.mkdir(parents=True, exist_ok=True)
        for name, text in sorted(_render_scenario(scenario).items()):
            (directory / name).write_text(text, encoding="utf-8")
            print(f"wrote {directory / name}")


if __name__ == "__main__":
    _regenerate()
