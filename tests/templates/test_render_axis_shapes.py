"""Byte-for-byte baselines for the deployment shapes the axes actually produce.

``test_render_defaults_golden.py`` pins every bundled template with the axes
UNSET — the shape a deployment that never heard of them gets. This suite pins
the shapes that appear once they are set: the dispatch pair on the host
namespace, the chat bridges beside it, a repo carrying a shared env-chain file,
a two-worker stack fanning out across host ports, two services passing named
host variables through to their containers, and every OSPREY-built image moved
onto a registry and a released tag.

Each scenario is a small, named delta over that same default context, rendered
through the same context builder and the same Environment (both imported from
the defaults module, so a scenario golden and a default golden can differ only
where the scenario differs). A scenario pins only the templates its delta
reaches: four templates honor the network axis
(``event_dispatcher``, ``dispatch_worker``, and the two bridges) and one renders
the env chain, so rendering the other eight per scenario would commit eight
copies of a file the defaults already pin. The ``env:`` axis is honored by every
service template but declared per service, so its scenarios pin exactly the
templates that declare it: two single-container ones, plus ``bluesky`` on its
own, because that file renders FOUR containers from one service block and the
shape worth pinning there is that all four carry the names. The image axes are
the exception — they are declared once for
the whole stack, so that scenario reaches every template carrying an image this
repo builds and pins all nine.

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
    TEMPLATES,
    _pinned_context,
    _render_templates,
    _service_keys,
)

_GOLDEN_DIR = Path(__file__).parent / "render_axis_shapes"

#: The chain file a repo carries when nothing shared it — the default probe
#: shape, and the one every scenario but ``env-shared-chain`` renders against.
_LOCAL_ENV_CHAIN = ("./.env",)

#: The image axes the ``images-on-a-registry`` scenario declares. A registry
#: with a project path under it and a tag that is obviously not ``local``, so a
#: golden that quietly fell back to the unset shape reads as wrong at a glance.
_AXIS_REGISTRY = "registry.example.org/physics/osprey"
_AXIS_TAG = "v1.2.3"

#: The host variables the ``service-env-passthrough`` scenario declares. Proxy
#: settings are the case the axis exists for: non-secret, set on the host by
#: whoever runs the machine, and meaningless for OSPREY to guess a value for.
#: Three of them on one service so author order is observable in the golden,
#: one on another so the single-name shape is pinned too.
_ENV_PASSTHROUGH = ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY")


@contextmanager
def _axes_unset() -> Iterator[None]:
    """Render from the scenario's config alone, never from the ambient shell.

    The image axes resolve ``OSPREY_IMAGE_REGISTRY`` / ``OSPREY_IMAGE_TAG``
    ahead of the config keys, deliberately — that is how one shell overrides a
    committed pin. It also means a developer who exported either variable would
    render different bytes than CI does, and a byte-exact suite that the
    ambient environment can move is pinning the machine, not the templates.

    Wrapped around the regeneration entry point as well as the suite, because
    the two must produce the same bytes and only one of them runs under pytest.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.delenv("OSPREY_IMAGE_REGISTRY", raising=False)
        patch.delenv("OSPREY_IMAGE_TAG", raising=False)
        yield


@pytest.fixture(scope="module", autouse=True)
def _neutral_axis_environment() -> Iterator[None]:
    """Hold :func:`_axes_unset` open for the whole module.

    Module-scoped so it is in place before ``rendered`` builds its renders — a
    function-scoped fixture would be set up after the module-scoped one it is
    meant to protect.
    """
    with _axes_unset():
        yield


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
        worker["worker_port_stride"] = dispatch.worker_port_stride
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

    overrides: dict = field(default_factory=dict)
    """Top-level config keys overlaid on the default context, for axes that
    live outside ``services.<key>`` — the image axes are declared for the whole
    stack, not per service."""


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
    # Two services handed named host variables. The two are pinned together
    # because their `environment:` blocks end differently — one on the shared
    # `TZ:` line, one on a service-specific variable — and the macro appends to
    # both, so a whitespace contract that only worked for the common shape would
    # show here as a swallowed or duplicated line. Nothing else is declared:
    # `deployed_services` stays empty, exactly as the default context leaves it,
    # so the ONLY thing separating these goldens from the default ones is the
    # passthrough itself.
    Scenario(
        name="service-env-passthrough",
        services={
            "virtual_accelerator": {"env": list(_ENV_PASSTHROUGH)},
            "bluesky_web": {"env": [_ENV_PASSTHROUGH[1]]},
        },
        deployed=(),
        templates=("virtual_accelerator", "bluesky_web"),
    ),
    # The same axis on the one template that renders MORE THAN ONE container
    # from a single service block: the bridge, the queueserver that executes
    # plans, Redis, and Tiled. `services.bluesky.env` is one declaration and the
    # profile has no per-container spelling to narrow it with, so a name that
    # reached some containers and not others would be silently partial — the
    # shape this scenario exists to hold still. `tiled_enabled` is on so the
    # optional fourth container is IN the render; it is the only container here
    # whose presence the config decides.
    Scenario(
        name="service-env-multi-container",
        services={"bluesky": {"env": list(_ENV_PASSTHROUGH), "tiled_enabled": True}},
        deployed=("bluesky",),
        templates=("bluesky",),
    ),
    # Every OSPREY-built image moved onto a registry and a released tag — the
    # shape a deployment gets once CI's images and the compose documents are
    # the same images. All eight are pinned in one scenario because the axes are
    # stack-wide: a template left behind renders a tag nothing pushed, and the
    # deploy fails on that one service alone. ``bluesky`` earns its place twice
    # over — it is the only file where an axis-derived default and two
    # third-party pins sit side by side.
    Scenario(
        name="images-on-a-registry",
        services=_pair_blocks(on_host=False),
        deployed=(
            "event_dispatcher",
            "dispatch_worker",
            "qmd",
            "virtual_accelerator",
            "archiver_recorder",
            "bluesky",
            "bluesky_web",
            "gchat_bridge",
            "nextcloud_bridge",
            "mongodb",
        ),
        templates=(
            "event_dispatcher",
            "dispatch_worker",
            "qmd",
            "virtual_accelerator",
            "archiver_recorder",
            "bluesky",
            "bluesky_web",
            "gchat_bridge",
            "nextcloud_bridge",
        ),
        overrides={"images": {"registry": _AXIS_REGISTRY, "tag": _AXIS_TAG}},
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
        | scenario.overrides
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
    assert worker["OSPREY_ARCHIVER_MONGODB_PORT"] == "10801"


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

    dispatch = _dispatch_defaults()
    worker = _service_env("pair-on-host", "dispatch_worker", "dispatch-worker-1")
    expected = _worker_port(dispatch.worker_port_base, 1, dispatch.worker_port_stride)

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

    dispatch = _dispatch_defaults()
    base = dispatch.worker_port_base
    document = yaml.safe_load(_golden_text(_scenario("two-workers-on-host"), "dispatch_worker"))
    services = document["services"]

    assert sorted(services) == ["dispatch-worker-1", "dispatch-worker-2"]
    for index in (1, 2):
        port = _worker_port(base, index, dispatch.worker_port_stride)
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


#: Every (scenario, service key, declared names) this suite pins a passthrough
#: for. Derived from the scenarios rather than listed, so a template added to an
#: ``env:``-declaring scenario arrives in these assertions instead of being
#: pinned byte-for-byte and checked for nothing.
ENV_CASES: tuple[tuple[Scenario, str, tuple[str, ...]], ...] = tuple(
    (scenario, key, tuple(scenario.services[key]["env"]))
    for scenario, key in CASES
    if scenario.services.get(key, {}).get("env")
)


def _env_case_id(case: tuple[Scenario, str, tuple[str, ...]]) -> str:
    scenario, key, _ = case
    return f"{scenario.name}/{key}"


#: Service templates that carry no ``env:`` axis at all. The dispatch pair is
#: configured through the profile's ``dispatch:`` block, whose two
#: ``services.<half>`` blocks are written wholesale by ``_inject_dispatch`` —
#: there is no author-declared ``env:`` for the macro to read there, so the call
#: is absent by intent rather than by oversight. Named here so that intent is
#: asserted rather than assumed: a template that quietly stopped honoring the
#: axis would otherwise just leave this set unchanged.
_AXIS_FREE_TEMPLATES = frozenset({"event_dispatcher", "dispatch_worker"})


@pytest.mark.parametrize("case", ENV_CASES, ids=_env_case_id)
def test_declared_env_names_render_as_bare_interpolations(
    case: tuple[Scenario, str, tuple[str, ...]],
) -> None:
    """Each declared name arrives as ``NAME: ${NAME}`` — no default, no refusal.

    A ``:-`` fallback would mean the render inventing a value for a name it
    knows nothing about, and the service would then start and look healthy while
    running on the invention. A ``:?`` would let one unset variable abort the
    whole compose document, taking down every other service in the deployment —
    none of which asked for that variable. Empty-when-unset is what is left, and
    it is also what keeps the value rotatable by editing the env chain alone.
    """
    scenario, key, declared = case
    text = _golden_text(scenario, key)
    for name in declared:
        assert f"      {name}: ${{{name}}}\n" in text, (
            f"{key} does not hand {name} to its container as a bare interpolation"
        )


@pytest.mark.parametrize("case", ENV_CASES, ids=_env_case_id)
def test_every_container_of_a_declaring_service_carries_the_names(
    case: tuple[Scenario, str, tuple[str, ...]],
) -> None:
    """EVERY container the block renders ends on the names, in author order.

    Two properties in one assertion, because they are one decision.

    *Order and position*: the names are the author's order because that is the
    only order the rendered block can show, and they close the block because
    that is where the macro is called from — the property that makes "add a
    passthrough" a purely additive edit to a block whose existing lines a
    reviewer can then skip past.

    *Every container*: ``env:`` is declared once, on the service, and the
    profile has no per-container spelling. So a file that renders several
    containers from one block — ``bluesky`` renders four — must hand the names
    to all of them or to none. Handing them to some is the failure this
    enumerates the golden's own services to catch: an author declaring
    ``HTTPS_PROXY`` for ``bluesky`` gets it in the bridge and, without this,
    silently not in the queueserver that actually executes plans. Reading the
    container list off the document rather than naming it here is what makes a
    container ADDED to a template later fail this test instead of joining the
    file un-covered.
    """
    scenario, key, declared = case
    document = yaml.safe_load(_golden_text(scenario, key))

    for name, service in document["services"].items():
        rendered = [str(entry) for entry in service.get("environment", {})]
        assert tuple(rendered[-len(declared) :]) == declared, (
            f"{scenario.name}/{key}: container {name} does not close its "
            f"environment block on the declared names in author order: {rendered}"
        )


def test_nothing_passes_a_variable_through_undeclared() -> None:
    """No template hands over a name no service declared.

    The macro is called from all ten service templates, and the default goldens
    are rendered with the axis unset everywhere — so a call that read the wrong
    service's block, or a macro that fell back to something other than nothing,
    would show up as one of the scenarios' variables appearing in a render that
    declared none of them.
    """
    declared = {name for _, _, names in ENV_CASES for name in names}

    for path in sorted(_DEFAULTS_DIR.glob("*.yml")):
        text = path.read_text(encoding="utf-8")
        for name in declared:
            assert f"{name}: ${{{name}}}" not in text, (
                f"{path.name} renders {name} with nothing declaring it"
            )


def test_every_service_template_hands_the_axis_to_every_container_it_renders() -> None:
    """The whole bundled set, not just the templates a scenario happens to pin.

    The goldens above prove the shape for the templates they name. This proves
    the rule for all of them at once, by declaring the same names on EVERY
    service block and rendering every bundled template: each container in each
    render either carries the names or belongs to the dispatch pair, which
    honors no ``env:`` axis at all.

    Rendered rather than pinned deliberately — committing ten more goldens whose
    only content is "the axis reached this file" would pin ten copies of what
    the defaults already pin, and the property here is not what the bytes are
    but that no container was skipped. A template that grows a second container
    without a macro call fails here on the day it is written, which is the whole
    reason this exists: that is exactly how ``bluesky``'s queueserver came to be
    handed nothing while its bridge was handed everything.
    """
    services = {key: {"env": list(_ENV_PASSTHROUGH)} for key in _service_keys()}
    # The one optional container in the bundled set; off by default, so without
    # this the sweep would never look at Tiled.
    services["bluesky"]["tiled_enabled"] = True

    with _probe_repo(_LOCAL_ENV_CHAIN) as repo_root:
        context = _pinned_context(
            {
                "project_name": _PROJECT_NAME,
                "project_root": str(repo_root),
                "services": services,
                "deployment": {},
                "system": {"timezone": "UTC"},
                "deployed_services": list(services),
            }
        )
        renders = _render_templates(context, TEMPLATES)

    for filename, text in sorted(renders.items()):
        key = filename.removesuffix(".yml")
        document = yaml.safe_load(text)
        # The bundled set includes the services-root document, which declares
        # the shared network and no container at all — nothing there for a
        # per-service axis to reach.
        if "services" not in document:
            continue
        for name, service in document["services"].items():
            rendered = [str(entry) for entry in service.get("environment", {})]
            carried = tuple(rendered[-len(_ENV_PASSTHROUGH) :]) == _ENV_PASSTHROUGH
            if key in _AXIS_FREE_TEMPLATES:
                assert not carried, (
                    f"{key}: {name} now honors the env axis — it is declared here as "
                    "axis-free, so either the exclusion or this list is out of date"
                )
                continue
            assert carried, (
                f"{key}: container {name} is rendered from a service block that "
                "declares an env passthrough and receives none of it"
            )


def test_image_axes_move_every_osprey_built_image() -> None:
    """All eight OSPREY-built images land on the declared registry and tag.

    Enumerated from the shipped suffix map rather than listed here, so an image
    added to the stack arrives in this assertion instead of being quietly left
    on ``:local`` — which is the whole failure this scenario exists to catch: a
    deploy that pulls seven services from the registry and tries to run the
    eighth from a tag the host never built.
    """
    from osprey.deployment.compose_generator import _OSPREY_IMAGE_SUFFIXES

    scenario = _scenario("images-on-a-registry")
    expected = {
        f"${{OSPREY_{key.upper()}_IMAGE:-{_AXIS_REGISTRY}/{_PROJECT_NAME}{suffix}:{_AXIS_TAG}}}"
        for key, suffix in _OSPREY_IMAGE_SUFFIXES.items()
    }

    rendered_images = set()
    for key in scenario.templates:
        document = yaml.safe_load(_golden_text(scenario, key))
        for service in document["services"].values():
            image = service.get("image")
            if image and _AXIS_REGISTRY in image:
                rendered_images.add(image)

    assert rendered_images == expected, (
        "the axis-set goldens do not carry exactly the eight built images — "
        f"missing {sorted(expected - rendered_images)}, "
        f"unexpected {sorted(rendered_images - expected)}"
    )


def test_image_axes_leave_the_third_party_pins_alone() -> None:
    """Redis keeps its upstream coordinates on the registry shape.

    A registry prefix is not a mirror. Applied to an upstream name it produces
    a reference no registry serves, and the failure surfaces as a pull error at
    deploy time rather than anywhere near the config that caused it. The
    bluesky file is where the two kinds of image sit closest together — its
    bridge is built here, its store is pulled from upstream. The set-equality
    above is the general form of this: no image outside the eight acquired the
    prefix anywhere in the nine templates.
    """
    axis = yaml.safe_load(_golden_text(_scenario("images-on-a-registry"), "bluesky"))
    default = yaml.safe_load((_DEFAULTS_DIR / "bluesky.yml").read_text(encoding="utf-8"))

    assert (
        axis["services"]["bluesky-redis"]["image"] == default["services"]["bluesky-redis"]["image"]
    ), "redis's upstream pin was rewritten by the image axes"


def test_image_axes_leave_every_build_block_untouched() -> None:
    """What a service BUILDS is unmoved by where its image is PULLED from.

    The axes name a published image; they say nothing about how a host builds
    one for itself. A build block that drifted with them would mean the same
    ``osprey build`` produced different contexts depending on a registry
    setting — and on the local shape, where the axes are unset, nothing would
    ever show it.
    """
    scenario = _scenario("images-on-a-registry")
    for key in scenario.templates:
        axis = yaml.safe_load(_golden_text(scenario, key))["services"]
        default = yaml.safe_load((_DEFAULTS_DIR / f"{key}.yml").read_text(encoding="utf-8"))[
            "services"
        ]
        for name, service in axis.items():
            if "build" not in service and name not in default:
                continue
            assert service.get("build") == default.get(name, {}).get("build"), (
                f"{key}: {name}'s build block moved with the image axes"
            )


def test_the_required_va_image_variant_survives_the_axes() -> None:
    """A recorder without a co-deployed VA still DEMANDS an explicit image.

    That line is the one image reference the axes must not supply a default
    for: the recorder runs the VA's image, and when the VA is not in this
    deployment there is no local build to fall back to. Turning it into an
    axis-derived default would replace a startup refusal naming the variable to
    set with a pull of an image that was never pushed.
    """
    recorder = (_DEFAULTS_DIR / "archiver_recorder.yml").read_text(encoding="utf-8")

    assert "${OSPREY_VA_IMAGE:?" in recorder, (
        "the recorder's no-VA image is no longer the required-variable form"
    )


def test_the_environment_outranks_a_configured_axis(monkeypatch: pytest.MonkeyPatch) -> None:
    """One shell can override a committed pin, on either axis independently.

    The goldens above can only pin the config layer — a byte-exact baseline that
    read the ambient environment would pin the machine it ran on. This covers
    the layer above it, which is what an operator uses to run a stack against a
    staging registry or a candidate tag without editing a committed file.
    """
    from osprey.deployment.compose_generator import resolve_image_defaults

    config = {
        "project_name": _PROJECT_NAME,
        "images": {"registry": _AXIS_REGISTRY, "tag": _AXIS_TAG},
    }
    monkeypatch.setenv("OSPREY_IMAGE_TAG", "rc-9")

    assert resolve_image_defaults(config)["worker"] == (f"{_AXIS_REGISTRY}/{_PROJECT_NAME}:rc-9"), (
        "the configured tag outranked the environment"
    )

    monkeypatch.setenv("OSPREY_IMAGE_REGISTRY", "staging.example.org/")
    assert resolve_image_defaults(config)["qmd"] == f"staging.example.org/{_PROJECT_NAME}-qmd:rc-9"


def test_a_blank_environment_axis_cannot_clear_a_configured_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exported-but-empty variable leaves the committed pin standing.

    The other half of the precedence rule, and the half that is easy to get
    backwards: the environment replaces a configured axis, it does not clear
    one. Blank counts as unset on both layers, so a stray ``OSPREY_IMAGE_TAG=``
    in a wrapper script cannot silently move a registry deployment back onto the
    ``local`` tag it has no build for.
    """
    from osprey.deployment.compose_generator import resolve_image_defaults

    config = {
        "project_name": _PROJECT_NAME,
        "images": {"registry": _AXIS_REGISTRY, "tag": _AXIS_TAG},
    }
    monkeypatch.setenv("OSPREY_IMAGE_TAG", "")
    monkeypatch.setenv("OSPREY_IMAGE_REGISTRY", "   ")

    assert resolve_image_defaults(config)["worker"] == (
        f"{_AXIS_REGISTRY}/{_PROJECT_NAME}:{_AXIS_TAG}"
    ), "a blank environment axis cleared the configured one"


@pytest.mark.parametrize(
    ("configured_tag", "expected_tag"),
    [
        pytest.param(2024, "2024", id="int"),
        pytest.param(2025.10, "2025.1", id="float"),
    ],
)
def test_a_numeric_configured_tag_pins_that_tag(configured_tag: object, expected_tag: str) -> None:
    """YAML reading a tag as a number does not silently drop the pin.

    ``tag: 2024`` unquoted is an ``int`` and ``tag: 2025.10`` is a ``float``,
    and both name a tag the operator plainly meant. Ignoring the key would
    render the ``:local`` default, which on a host that only pulls fails as
    "No such image" with nothing pointing back at the setting that was dropped.

    The float case is pinned *as it renders*, trailing zero and all: ``2025.10``
    is the number 2025.1, so it pins ``:2025.1``. That is the argument for
    quoting a dotted tag, not a rounding this function may quietly correct.
    """
    from osprey.deployment.compose_generator import resolve_image_defaults

    defaults = resolve_image_defaults(
        {"project_name": _PROJECT_NAME, "images": {"tag": configured_tag}}
    )

    assert defaults["worker"] == f"{_PROJECT_NAME}:{expected_tag}"


@pytest.mark.parametrize(
    ("axis_key", "configured"),
    [
        pytest.param("tag", ["v1", "v2"], id="tag-list"),
        pytest.param("tag", {"value": "v1"}, id="tag-mapping"),
        pytest.param("tag", True, id="tag-bool"),
        pytest.param("registry", ["registry.example.org"], id="registry-list"),
    ],
)
def test_an_unusable_configured_axis_fails_loudly(axis_key: str, configured: object) -> None:
    """A shape no image reference can carry is refused, naming key and type.

    The refusal is the whole point: falling back to the packaged default here
    would render ``<project>:local`` on a deployment that asked for a registry,
    and the operator would meet that decision as a pull failure on a remote
    host. A boolean is refused with the containers rather than coerced with the
    numbers — YAML reads ``tag: yes`` as ``True``, which names no tag anyone
    intended.
    """
    from osprey.deployment.compose_generator import resolve_image_defaults

    with pytest.raises(ValueError) as excinfo:
        resolve_image_defaults({"project_name": _PROJECT_NAME, "images": {axis_key: configured}})

    message = str(excinfo.value)
    assert f"images.{axis_key}" in message, "the refusal does not name the offending key"
    assert type(configured).__name__ in message, "the refusal does not name the received type"


def test_an_images_block_that_is_not_a_block_fails_loudly() -> None:
    """``images:`` written as a bare value is refused rather than ignored.

    The same silent drop one level up: a config that spells the registry as
    ``images: registry.example.org`` instead of ``images.registry`` has said
    exactly what it wants, and rendering ``:local`` past it moves the failure to
    a pull on a host that has no local build.
    """
    from osprey.deployment.compose_generator import resolve_image_defaults

    with pytest.raises(ValueError) as excinfo:
        resolve_image_defaults({"project_name": _PROJECT_NAME, "images": "registry.example.org"})

    assert "images" in str(excinfo.value)
    assert "str" in str(excinfo.value), "the refusal does not name the received type"


def test_an_unset_axis_renders_the_locally_built_name() -> None:
    """With neither layer set the images are exactly what they always were.

    The property every default golden in the sibling suite depends on: adding
    the axes moved no deployment that has not asked for them.
    """
    from osprey.deployment.compose_generator import resolve_image_defaults

    defaults = resolve_image_defaults({"project_name": _PROJECT_NAME})

    assert defaults["worker"] == f"{_PROJECT_NAME}:local"
    assert defaults["bluesky_bridge"] == f"{_PROJECT_NAME}-bluesky-bridge:local"


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
    with _axes_unset():
        for scenario in SCENARIOS:
            directory = _GOLDEN_DIR / scenario.name
            directory.mkdir(parents=True, exist_ok=True)
            for name, text in sorted(_render_scenario(scenario).items()):
                (directory / name).write_text(text, encoding="utf-8")
                print(f"wrote {directory / name}")


if __name__ == "__main__":
    _regenerate()
