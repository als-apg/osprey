"""The shipped control assistant, built, against every row of its end state.

The control assistant is one root profile plus tier deltas and complete app
files. Built, it is a host render and one render per tier and app; every
consumer downstream reads ONE model of it:

    cards      = roster_entries(): the people, then one card per shared app
    served set = every card's render ∪ default_tier ∪ every role's tier
                 ∪ dispatch.tier
    reach      = per served render, a predicate over its rendered config.yml;
                 the render's kind is never asked
    privilege  = read off the composed render; an app render holds none, even
                 on a host that opens every deployment-editing door
    images     = one build per served render
    old words  = no retired key survives into any rendered config

Every row here reads the repo ``osprey init --preset control-assistant`` and
``osprey build`` actually write — never a hand-built config — so a consumer
that drifts from the model fails here, on the deployment people run.
"""

from __future__ import annotations

import contextlib
import copy
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

from osprey.deployment.compose_generator import _dispatch_worker_corpus_mounts
from osprey.deployment.web_terminals import env_production, render_images, roster
from osprey.deployment.web_terminals.artifacts import resolve_render_inputs
from osprey.deployment.web_terminals.env_production import ensure_env_production
from osprey.deployment.web_terminals.lint import lint_app_render, lint_web_terminals
from osprey.deployment.web_terminals.roster import (
    ALL_PRIVILEGES,
    config_archiver_ca_bundles,
    config_archiver_credential_envs,
    config_needs_ariel_mirror,
    config_needs_ariel_password,
    config_needs_dispatcher_token,
    config_needs_facility_bundle,
    config_needs_graphdb_password,
    config_needs_launch_token_for,
    config_needs_phoebus_handles,
    resolve_cards,
    roster_entries,
    served_renders,
    tier_privileges,
)
from tests._builds import copy_build
from tests.cli.test_tier_app_presets import _build_layered_stack

#: Each fixture reads a real build of the flagship preset — seconds, not
#: milliseconds — because the end state is a property of what a build writes.
pytestmark = pytest.mark.slow

#: The three deployment-editing doors a host may open, as ``osprey init --set``
#: spells them: the setup tool (an empty deny list), the Config panel and the
#: gallery's writes.
OPEN_DOORS = (
    "config.claude_code.permissions.deny=[]",
    "config.web.config_panel.enabled=true",
    "config.web.scaffold_gallery.write_enabled=true",
)

#: The served set of the shipped control assistant: its three tiers and its
#: three shared apps.
SERVED = {"readwrite", "readonly", "admin", "logbook", "knowledge", "channel-finder"}
APPS = ("logbook", "knowledge", "channel-finder")

#: Keys no rendered config may carry once the old vocabulary is retired.
RETIRED_KEYS = frozenset(
    {
        "personas",  # retired-spelling
        "default_persona",  # retired-spelling
        "persona",  # retired-spelling
    }
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _init(repo: Path, *sets: str) -> Any:
    """``osprey init`` the flagship preset into *repo* with ``--set`` overrides."""
    from osprey.cli.init_cmd import init

    args = [str(repo), "--preset", "control-assistant", "--no-git"]
    for assignment in sets:
        args += ["--set", assignment]
    return CliRunner().invoke(init, args)


def _host(repo: Path) -> dict:
    """The deployment's own render — what every host-side reader runs against."""
    return _load(repo / "build" / "config.yml")


def _render_config(repo: Path, render: roster.ServedRender) -> dict:
    """A served render's rendered ``config.yml``, joined through its projection entry."""
    assert render.project_path is not None, render
    return _load(repo / render.project_path / "config.yml")


def _render_hooks(repo: Path, render: roster.ServedRender) -> list[str]:
    """The hook selection a render was built with, read off its installed hooks."""
    hooks_dir = repo / render.project_path / ".claude" / "hooks"
    return sorted(
        path.stem.removeprefix("osprey_").replace("_", "-")
        for path in hooks_dir.glob("osprey_*.py")
    )


def _keys(node: Any) -> Iterator[str]:
    """Every mapping key anywhere under *node*, lists included."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield key
            yield from _keys(value)
    elif isinstance(node, list):
        for item in node:
            yield from _keys(item)


def _grants(config: dict, repo: Path) -> dict[str, Any]:
    """The per-render grant sets of :func:`resolve_render_inputs`."""
    inputs = resolve_render_inputs(config, repo)
    return {key: value for key, value in inputs.items() if key.endswith("_renders")}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def stack(built_control_assistant) -> Path:
    """The flagship preset as shipped: the run's shared, read-only build (``tests/_builds.py``)."""
    return built_control_assistant.repo


@pytest.fixture(scope="module")
def config(stack: Path) -> dict:
    return _host(stack)


@pytest.fixture(scope="module")
def served(config: dict) -> dict[str, roster.ServedRender]:
    renders = served_renders(config, strict=True)
    assert None not in renders.values(), renders
    return renders


@pytest.fixture(scope="module")
def open_stack(tmp_path_factory) -> Path:
    """The flagship preset with every deployment-editing door opened on the host.

    The headless dispatch worker has no approver, so a build refuses a
    ``dispatch.tier`` whose render holds an open door — and the bundled
    ``readonly`` tier is a delta over the host, so it inherits the doors. This
    deployment therefore runs its worker on its own render (``dispatch.tier``
    unset); the refusal itself is pinned by
    :func:`test_open_doors_refuse_a_headless_tier`.
    """
    repo = tmp_path_factory.mktemp("end-state-open") / "my-facility"
    created = _init(repo, *OPEN_DOORS, "dispatch.tier=null")
    assert created.exit_code == 0, created.output
    return _build_layered_stack(repo)


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


def test_cards(config: dict) -> None:
    """The people in roster order, then one card per shared app in ``shared:`` order."""
    cards = roster_entries(config["modules"]["web_terminals"])
    assert [card["name"] for card in cards] == [
        "alice",
        "bob",
        "carol",
        "logbook",
        "knowledge",
        "channel-finder",
    ]
    assert [card["name"] for card in cards if card.get("kind") == "app"] == list(APPS)


def test_served_set(config: dict, served: dict[str, roster.ServedRender]) -> None:
    """The served set is every card's render, the default tier, every role's
    tier and the dispatch tier — each read straight off the built config."""
    assert set(served) == SERVED

    web_terminals = config["modules"]["web_terminals"]
    cards = resolve_cards(web_terminals, config.get("registry") or {}, strict=True)
    roles = ((web_terminals.get("authorization") or {}).get("roles")) or {}
    expected = (
        {card["render"] for card in cards}
        | {web_terminals["default_tier"]}
        | {role["tier"] for role in roles.values()}
        | {config["services"]["dispatch_worker"]["tier"]}
    )
    assert set(served) == expected


def test_one_list(
    stack: Path,
    served: dict[str, roster.ServedRender],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Grants, the env write, the image build and lint each walk exactly the served set."""
    # The env write adds files to the repo, so it runs on a copy of the shared build.
    stack = copy_build(stack, tmp_path)
    config = _host(stack)
    paths = {render.project_path: name for name, render in served.items()}

    roster_seen: list[str] = []
    real_rendered_config_file = roster.rendered_config_file

    def record_roster(project_root, project_path, layer_root=None):
        roster_seen.append(paths.get(project_path, project_path))
        return real_rendered_config_file(project_root, project_path, layer_root)

    env_seen: list[str] = []
    real_render_config_yml = env_production._render_config_yml

    def record_env(project_root, render):
        env_seen.append(render.name)
        return real_render_config_yml(project_root, render)

    built_tags: list[str] = []
    monkeypatch.setattr(roster, "rendered_config_file", record_roster)
    monkeypatch.setattr(env_production, "_render_config_yml", record_env)
    monkeypatch.setattr(render_images, "get_runtime_command", lambda config: ["docker"])
    monkeypatch.setattr(
        render_images,
        "_render_image_build_cmd",
        lambda runtime, context, tag, *rest: ["docker", "build", tag],
    )
    monkeypatch.setattr(render_images, "run_captured", lambda cmd, **kw: built_tags.append(cmd[-1]))

    # The grant walk.
    resolve_render_inputs(config, stack)
    assert set(roster_seen) == SERVED

    # The env write. A placeholder provider key lets the env chain pass the
    # deploy's required-variable check; no value is ever asserted on.
    with (stack / ".env").open("a", encoding="utf-8") as fh:
        fh.write("\nANTHROPIC_API_KEY=placeholder-not-a-key\n")
    ensure_env_production(config, stack)
    assert set(env_seen) == SERVED

    # The image build, dry: one local image per served render.
    cards = resolve_cards(config["modules"]["web_terminals"], config.get("registry") or {})
    monkeypatch.chdir(stack)
    render_images.build_render_images(config, cards, False, dict(os.environ))
    projects = {render.project for render in served.values()}
    assert sorted(built_tags) == sorted(f"{project}:local" for project in projects)

    # Lint's per-render reader at the rendered altitude.
    roster_seen.clear()
    findings = lint_web_terminals(config, project_root=stack)
    assert set(roster_seen) == SERVED
    assert [f for f in findings if f.severity == "error"] == []


def test_reach(stack: Path, config: dict, served: dict[str, roster.ServedRender]) -> None:
    """Every grant is its predicate over each served render's rendered config."""
    grants = _grants(config, stack)
    configs = {name: _render_config(stack, render) for name, render in served.items()}

    predicates = {
        "dispatcher_renders": config_needs_dispatcher_token,
        "ariel_renders": config_needs_ariel_password,
        "graphdb_renders": config_needs_graphdb_password,
        "facility_bundle_renders": config_needs_facility_bundle,
        "ariel_mirror_renders": config_needs_ariel_mirror,
        "phoebus_handle_renders": config_needs_phoebus_handles,
        "archiver_credential_renders": lambda cfg: bool(config_archiver_credential_envs(cfg)),
        "archiver_ca_bundle_renders": lambda cfg: bool(config_archiver_ca_bundles(cfg)),
    }
    assert set(predicates) | {"launch_token_renders"} == set(grants)
    for grant, predicate in predicates.items():
        for name, render_config in configs.items():
            assert (name in grants[grant]) == bool(predicate(render_config)), (grant, name)
    for lane in roster.LANE_KEYS:
        entitled = grants["launch_token_renders"].get(lane, set())
        for name, render_config in configs.items():
            assert (name in entitled) == config_needs_launch_token_for(render_config, lane), (
                lane,
                name,
            )

    # The kind is never asked: the projection with every ``kind`` removed
    # grants exactly the same.
    unkinded = copy.deepcopy(config)
    for entry in unkinded["modules"]["web_terminals"]["renders"].values():
        entry.pop("kind", None)
    assert _grants(unkinded, stack) == grants

    # An app reaches the corpora it reads: the logbook holds the graph store,
    # the facility bundle and the ARIEL mirror.
    for grant in ("graphdb_renders", "facility_bundle_renders", "ariel_mirror_renders"):
        assert "logbook" in grants[grant], grant

    # The dispatch worker mounts the bundle and the mirror exactly when its
    # tier is granted them; the flagship's readonly worker needs both.
    worker_tier = config["services"]["dispatch_worker"]["tier"]
    assert worker_tier == "readonly"
    with contextlib.chdir(stack):
        mounts = _dispatch_worker_corpus_mounts(config, "services/dispatch_worker")
    assert mounts is not None
    assert (mounts["bundle"] is not None) == (worker_tier in grants["facility_bundle_renders"])
    assert (mounts["mirror"] is not None) == (worker_tier in grants["ariel_mirror_renders"])
    assert mounts["bundle"] is not None and mounts["mirror"] is not None

    compose = _load(stack / "build" / "services" / "dispatch_worker" / "docker-compose.yml")
    volumes = compose["services"]["dispatch-worker-1"]["volumes"]
    for mount in (mounts["bundle"], mounts["mirror"]):
        assert f"{mount['source']}:{mount['target']}" in volumes, mount


def test_open_doors_refuse_a_headless_tier(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Opening the host's doors is accepted at init, and a build then refuses
    to run the headless worker on a tier that inherited them."""
    from osprey.cli.build_cmd import build

    repo = tmp_path / "my-facility"
    created = _init(repo, *OPEN_DOORS)
    assert created.exit_code == 0, created.output
    built = CliRunner().invoke(build, ["--repo", str(repo), "--skip-deps", "--skip-lifecycle"])
    assert built.exit_code != 0
    assert "dispatch.tier 'readonly'" in caplog.text
    assert "no approver" in caplog.text


def test_privilege(stack: Path, served: dict[str, roster.ServedRender], open_stack: Path) -> None:
    """An app render holds no deployment-editing door, even on a host that
    opens them all; the admin tier keeps what it held."""
    open_served = served_renders(_host(open_stack), strict=True)
    assert set(open_served) == SERVED
    assert None not in open_served.values()

    for app in APPS:
        render = open_served[app]
        assert render.kind == roster.RENDER_KIND_APP, app
        app_config = _render_config(open_stack, render)
        assert tier_privileges(app_config) == (), app
        document = {"hooks": _render_hooks(open_stack, render), "config": app_config}
        assert lint_app_render(document) == [], app

    # The doors really are open on the host's tiers, so the app result above is
    # the composer closing them, not a host that never opened them.
    for tier in ("readwrite", "readonly", "admin"):
        assert tier_privileges(_render_config(open_stack, open_served[tier])) == ALL_PRIVILEGES

    shipped_admin = tier_privileges(_render_config(stack, served["admin"]))
    assert shipped_admin == ALL_PRIVILEGES
    assert tier_privileges(_render_config(open_stack, open_served["admin"])) == shipped_admin


def test_no_retired_key(stack: Path, config: dict, served: dict[str, roster.ServedRender]) -> None:
    """No retired key survives into the host render or any served render."""
    documents = {"build/config.yml": config}
    documents.update(
        {render.project_path: _render_config(stack, render) for render in served.values()}
    )
    for where, document in documents.items():
        assert RETIRED_KEYS.isdisjoint(_keys(document)), where
