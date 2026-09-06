"""What a render says about services: injected for a deployment, nothing of its own attached.

Two writers fill a rendered ``config.yml``'s ``services:`` and
``deployed_services:``, and which one runs is decided by one profile field.

A **deploying** render (``deploy_services: true``) states its stack: the
profile's resolved ``config:`` names the services it declares, and
:func:`~osprey.cli.build_cmd._inject_services` appends every component the
build scaffolds — the dispatch pair, the chat bridges, the virtual accelerator
and its recorder — to ``deployed_services`` in injection order.

An **attached** render (``deploy_services: false``) states none. It deploys
nothing: it connects to the stack another OSPREY deployment runs on the same
host, and what it may dial is copied in afterwards from that host's render, key
by gated key, by the Reach Contract (:mod:`osprey.deployment.reach`). Its own
service statements are therefore stripped on the override path by
:func:`~osprey.cli.build_cmd._attached_service_overrides` — the build-side
successor of the app template's ``{% if deploy_services %} … {% else %}``
branch, which this feature deletes. The suppression has to happen in the build
rather than in a template because a profile flips ``deploy_services``
independently of the preset it extends: every preset spells the deploying
shape, and the five control-assistant personas are attached renders of exactly
those presets.

Leaving that out is not a cosmetic difference. A non-empty ``deployed_services``
is what tells :func:`osprey.deployment.reach.reach_errors` it is reading a
deploying render, so an attached render carrying the preset's list is refused
for every consumer that list omits — on the one render whose whole point is
that it dials someone else's ports.

The control-assistant preset is the only shipped one whose build injects
anything, so the end-to-end cells here use it. Each is a real ``osprey init`` +
``osprey build`` (``slow``, tens of seconds), built once per session and shared;
the rest are unit tests of the two merges and cost nothing.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.cli.build_cmd import _attached_service_overrides
from osprey.cli.build_profile_merge import merge_persona_delta
from osprey.cli.build_profile_ports import layout_port_fill
from osprey.deployment.reach import REACH_CONTRACTS
from osprey.port_layout import DEFAULT_PORT_BASE
from tests.fixtures.explicit_config.freeze import (
    FIXTURE_ROOT,
    PROJECT_NAME,
    _cli_env,
    _failure_reason,
    _run_cli,
    collect_rendered_configs,
)

#: The cell built end to end. ``graph`` is the frozen control-assistant mode
#: whose baseline ``deployed_services`` this module pins against.
PRESET = "control-assistant"
MODE = "graph"

#: The external graph store the attached-profile cell names. Unroutable on
#: purpose: nothing in a build dials it.
EXTERNAL_GRAPH_URI = "bolt://graph.example.invalid:7687"

#: Every ``services.<name>.<leaf>`` the Reach Contract projects into an attached
#: render. Read from the registry rather than listed, so a contract that gains a
#: key does not turn this into a stale allowlist.
PROJECTED_SERVICE_KEYS = frozenset(
    projected.key
    for contract in REACH_CONTRACTS.values()
    for projected in contract.projected
    if projected.key.startswith("services.")
)


def _baseline_root() -> dict[str, Any]:
    """The frozen baseline root render of the cell this module builds.

    Returns:
        The captured document, as ``freeze.py`` wrote it.
    """
    path = FIXTURE_ROOT / PRESET / MODE / "root.yml"
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _preset_config() -> dict[str, Any]:
    """The packaged preset's own ``config:`` block, unresolved.

    Read off the packaged file rather than through the resolver: what this
    module needs is what the preset itself states, which is the half of the
    rendered list the injectors do not write.

    Returns:
        The preset's ``config:`` mapping.
    """
    import osprey.profiles  # noqa: PLC0415

    path = Path(osprey.profiles.__file__).parent / "presets" / f"{PRESET}.yml"
    return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("config") or {}


def _service_leaves(document: dict[str, Any]) -> list[str]:
    """Every dotted ``services.*`` leaf of one rendered document.

    Args:
        document: A rendered ``config.yml``, loaded.

    Returns:
        The dotted path of each leaf under ``services:``, sorted. An empty or
        absent block contributes nothing.
    """
    leaves: list[str] = []

    def walk(node: Any, prefix: str) -> None:
        if isinstance(node, dict) and node:
            for key, value in node.items():
                walk(value, f"{prefix}.{key}")
            return
        leaves.append(prefix)

    for name, block in (document.get("services") or {}).items():
        walk(block, f"services.{name}")
    return sorted(leaves)


@pytest.fixture(scope="session")
def rendered(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict[str, Any]]:
    """One real ``osprey init`` + ``osprey build`` of the cell, shared by the session.

    Invoked exactly as the fixture freeze invoked it — same verbs, same fixed
    project name, same ``*_API_KEY``-stripped environment — so what it renders
    is comparable to what was frozen. No source directories are prepended to
    ``PYTHONPATH``: the CLI imports OSPREY as installed, which is the tree under
    test.

    Returns:
        The build's rendered documents, keyed by render name: ``root`` for the
        deployment, one per persona for the attached renders.
    """
    scratch = tmp_path_factory.mktemp("deployed-services-injection")
    env = _cli_env([])

    result = _run_cli(
        [
            "init",
            PROJECT_NAME,
            "--preset",
            PRESET,
            "--no-git",
            "--set",
            f"channel_finder_mode={MODE}",
        ],
        cwd=scratch,
        env=env,
    )
    assert result.returncode == 0, f"osprey init refused the cell:\n{_failure_reason(result)}"

    project: Path = scratch / PROJECT_NAME
    result = _run_cli(["build"], cwd=project, env=env)
    assert result.returncode == 0, f"osprey build refused the cell:\n{_failure_reason(result)}"

    return {name: document or {} for name, document in collect_rendered_configs(project).items()}


@pytest.fixture(scope="session")
def external_store_render(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """An attached project naming an external graph store nothing here deploys.

    ``channel-finder-standalone`` deploys no store of its own
    (``deployed_services: []``), so ``channel_finder_mode: graph`` is refused
    outright unless the profile names one the facility already runs — the
    refusal's own instruction. Flipped to ``deploy_services: false``, that
    profile is the case the claimed-stack drop must not swallow: the only
    ``services.graphdb`` statement in it is an address, not a claim.

    Returns:
        The attached project's rendered ``config.yml``.
    """
    scratch = tmp_path_factory.mktemp("external-graph-store")
    env = _cli_env([])

    result = _run_cli(
        [
            "init",
            PROJECT_NAME,
            "--preset",
            "channel-finder-standalone",
            "--no-git",
            "--set",
            "channel_finder_mode=graph",
            "--set",
            "deploy_services=false",
            "--set",
            f"config.services.graphdb.uri={EXTERNAL_GRAPH_URI}",
        ],
        cwd=scratch,
        env=env,
    )
    assert result.returncode == 0, f"osprey init refused the profile:\n{_failure_reason(result)}"

    project: Path = scratch / PROJECT_NAME
    result = _run_cli(["build"], cwd=project, env=env)
    assert result.returncode == 0, (
        "osprey build refused an attached profile that names an external graph store:\n"
        f"{_failure_reason(result)}"
    )

    return collect_rendered_configs(project)["root"] or {}


# ─────────────────────────────────────────────────────────────────────────────
# The deploying render
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
def test_deploying_render_lists_the_baseline_services_in_order(
    rendered: dict[str, dict[str, Any]],
) -> None:
    """The deployment's ``deployed_services`` still equals the frozen baseline's.

    Order and all: it is the order the injectors ran in, and two of them read it
    back — the chat bridges gate their dispatcher URLs on the dispatch pair
    already being listed, the recorder gates its image and addressing on the
    virtual accelerator being listed — so a reordering is a behaviour change,
    not a formatting one.
    """
    assert rendered["root"]["deployed_services"] == _baseline_root()["deployed_services"]


@pytest.mark.slow
def test_injection_appends_to_what_the_profile_declared(
    rendered: dict[str, dict[str, Any]],
) -> None:
    """The profile's own names come first, the injected ones follow, none is lost.

    The two halves are separately owned — the preset's ``config:`` states the
    declared stack, the injectors state what the build scaffolds — and this is
    the seam between them. It is what a config overlay that replaced the list
    instead of preceding it would break.
    """
    declared = list(_preset_config()["deployed_services"])
    deployed = list(rendered["root"]["deployed_services"])

    assert deployed[: len(declared)] == declared
    injected = deployed[len(declared) :]
    assert injected, "the control-assistant build injects services; none reached deployed_services"
    assert len(set(deployed)) == len(deployed), f"a service is listed twice: {deployed}"


@pytest.mark.slow
def test_deploying_render_keeps_the_services_its_profile_declares(
    rendered: dict[str, dict[str, Any]],
) -> None:
    """Suppression is scoped to attached renders and does not touch a deployment.

    The preset states its stack's shape — image tags, heap sizes, data paths —
    under ``services.*``, and those are exactly the keys the attached side
    strips. A deployment keeps every one of them.
    """
    leaves = _service_leaves(rendered["root"])
    unprojected = [leaf for leaf in leaves if leaf not in PROJECTED_SERVICE_KEYS]
    assert unprojected, "the deployment render carries none of the preset's own service keys"


# ─────────────────────────────────────────────────────────────────────────────
# The attached renders
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
def test_attached_renders_deploy_nothing(rendered: dict[str, dict[str, Any]]) -> None:
    """Every persona render carries an empty ``deployed_services``.

    The personas are attached renders of a preset that spells four deployed
    services, so an empty list here is the suppression's doing, not the
    profile's.
    """
    personas = {name: document for name, document in rendered.items() if name != "root"}
    assert personas, "the control-assistant build renders personas; none were captured"
    assert {name: document.get("deployed_services") for name, document in personas.items()} == {
        name: [] for name in personas
    }


@pytest.mark.slow
def test_attached_renders_carry_only_what_their_host_told_them(
    rendered: dict[str, dict[str, Any]],
) -> None:
    """No persona states a service key of its own.

    Every ``services.*`` leaf a persona ends up with is one the Reach Contract
    projects from the hosting deployment's render. The preset's own service
    keys — the image tags and heap sizes of a stack this render does not run —
    are not among them.
    """
    strayed = {
        name: [leaf for leaf in _service_leaves(document) if leaf not in PROJECTED_SERVICE_KEYS]
        for name, document in rendered.items()
        if name != "root"
    }
    assert {name: leaves for name, leaves in strayed.items() if leaves} == {}


@pytest.mark.slow
def test_an_attached_render_keeps_the_external_store_it_names(
    external_store_render: dict[str, Any],
) -> None:
    """An endpoint the profile does not claim to deploy survives into the render.

    Requirement 5's external-store clause: ``services.graphdb`` keeps
    ``port_host`` and ``http_port_host`` beside its ``uri``, exactly as a
    deploying render would. ``uri`` is the profile's own; the two ports are the
    layout fill's, and a drop scoped to the whole ``services.*`` surface took
    all three — leaving the graph consumer with the address the Reach Contract
    happens to project back for a host of its own, and nothing at all beside a
    host that runs no store.

    The render still says it deploys nothing, which is the other half: naming
    where a store is and claiming to run one are different statements.
    """
    fill = layout_port_fill({"services.graphdb.uri": EXTERNAL_GRAPH_URI}, DEFAULT_PORT_BASE)
    expected = {"uri": EXTERNAL_GRAPH_URI} | {
        key.rpartition(".")[2]: value for key, value in fill.items()
    }

    assert set(expected) == {"uri", "port_host", "http_port_host"}, (
        f"the layout fill no longer supplies the external store's two ports: {fill}"
    )
    assert external_store_render["deployed_services"] == []
    assert external_store_render["services"]["graphdb"] == expected


# ─────────────────────────────────────────────────────────────────────────────
# The two merges, on their own
# ─────────────────────────────────────────────────────────────────────────────


def test_attached_overrides_strip_the_stack_the_profile_claims_to_deploy() -> None:
    """The claimed stack goes, in both spellings, and nothing else does."""
    overrides = _attached_service_overrides(
        {
            "services": {"graphdb": {"image": "graphdb:11"}},
            "services.postgresql.username": "osprey",
            "deployed_services": ["postgresql", "graphdb"],
            "approval.enabled": True,
            "archiver.mongodb_archiver.port": 10061,
        }
    )

    assert overrides == {
        "deployed_services": [],
        "approval.enabled": True,
        "archiver.mongodb_archiver.port": 10061,
    }


def test_attached_overrides_keep_an_endpoint_the_profile_does_not_claim_to_deploy() -> None:
    """A service absent from ``deployed_services`` is an EXTERNAL endpoint, and it stays.

    The two statements look alike and mean opposite things. ``services.qmd.*``
    beside ``qmd`` in the list is "I run this", which an attached render must
    not say. ``services.graphdb.uri`` with ``graphdb`` NOT in the list is "the
    facility already runs one over there" — the spelling
    :func:`osprey.deployment.reach.reach_errors` itself prescribes when a
    consumer has nothing to dial, so dropping it would delete the answer to the
    refusal it prevents.
    """
    overrides = _attached_service_overrides(
        {
            "services.qmd.path": "./services/qmd",
            "services.graphdb.uri": "bolt://graph.example:7687",
            "services.graphdb.http_port_host": 10803,
            "deployed_services": ["qmd"],
        }
    )

    assert overrides == {
        "deployed_services": [],
        "services.graphdb.uri": "bolt://graph.example:7687",
        "services.graphdb.http_port_host": 10803,
    }


def test_attached_overrides_state_an_empty_list_when_the_profile_stated_none() -> None:
    """``deployed_services: []`` is written even for a profile that never spelled it.

    The rewritten framework template does not write ``deployed_services`` at
    all, so on an attached render this override is its only writer — and the
    empty list is what tells the reach check it is not looking at a deployment.
    """
    assert _attached_service_overrides({}) == {"deployed_services": []}


def test_a_persona_override_of_deployed_services_reaches_nothing_else() -> None:
    """A persona spelling ``deployed_services`` moves that key and no other.

    Requirement 6's first clause: a persona delta merges as it always has. It
    names one ``config:`` key and leaves every sibling — the service blocks,
    the approval posture — exactly as the root resolved them, and does not
    mutate the root it was merged over.

    The list itself merges by **union**
    (:func:`~osprey.cli.build_profile_merge._merge_lists` dedups
    ``base + child`` for a list of strings), so a persona spelling a shorter
    list adds nothing and removes nothing. That is deliberate and stays: a
    persona subtracts nothing from ``config:`` — the proposal puts a
    persona-level ``exclude:`` of a ``config:`` key out of scope for the same
    reason. Dropping a container is the root profile's move, made by omitting
    that service's ``services.<name>.*`` keys and leaving its name out of
    ``deployed_services``. The union is pinned below so the ruled behaviour is
    stated, not merely tolerated: the delta omits ``graphdb`` and ``graphdb``
    survives.
    """
    root = {
        "config": {
            "deployed_services": ["postgresql", "openobserve", "qmd", "graphdb"],
            "services.graphdb.image": "graphdb:11",
            "services.postgresql.username": "osprey",
            "approval.enabled": True,
        }
    }
    before = copy.deepcopy(root)

    merged = merge_persona_delta(
        root, {"config": {"deployed_services": ["postgresql", "openobserve", "qmd"]}}
    )

    assert merged["config"]["deployed_services"] == [
        "postgresql",
        "openobserve",
        "qmd",
        "graphdb",
    ]
    assert {
        key: value for key, value in merged["config"].items() if key != "deployed_services"
    } == {key: value for key, value in root["config"].items() if key != "deployed_services"}
    assert root == before, "merge_persona_delta mutated the root profile"
