"""Every staged login→tier map, checked against the roster from first principles.

A shared container that binds a control-context tree (it sets
``OSPREY_CONTROL_CONTEXT_TREE``: the Bluesky queue, the dispatch worker) runs one
render for every login whose job it carries, so the build stages beside that
render a ``control_system.login_tiers`` map. This module checks every such map
on disk without calling the map's producer: the expected map is restated here
from the roster in the built ``config.yml`` and from each card's own render, read
from disk.

The rules restated (the C-LT contract):

* a roster card is in the map unless single sign-on decides its tier — ``auth.method``
  is ``oidc``, ``authorization`` names a claim, and the card names no ``role`` —
  or it resolves to no tier at all;
* its value is ``tier_writes`` of its own render's ``control_system``, where its
  render is its role's tier, else its ``tier`` pin, else ``default_tier``;
* the build account is in the map at the root render's tier only when it names
  no roster card (a person, or an app ``shared:`` derives); when it does, the
  roster rule alone decides it, and an app card is never a key. ``'unknown'``
  is never a key;
* the staged document minus the map is the root flatten (Bluesky, the untiered
  worker) or the container-rooted tier render (the tiered worker);
* no root, terminal-served or image render carries a map.

Each case is checked on the build and again after ``prepare_compose_files``
alone, the redeploy path ``deploy up/down/restart`` takes. Every build pins
``OSPREY_AUDIT_IDENTITY``, so the build account does not depend on who runs it.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.fixtures.lifecycle_repo import (
    EXEMPLAR_DIRNAME,
    build_exemplar_repo,
    preserved_environ,
)

pytestmark = pytest.mark.slow

#: The env var a tree-bound shared container sets in its compose service.
TREE_ENV = "OSPREY_CONTROL_CONTEXT_TREE"

#: The key the map is rendered under, inside ``control_system``.
MAP_KEY = "login_tiers"

#: The identity no login may be mapped as.
UNKNOWN = "unknown"

#: A pinned build account that is roster-shaped and names no roster card.
BUILDER = "builder"


@dataclass(frozen=True)
class Case:
    """One roster shape, as exact edits to the exemplar's ``profile.yml``."""

    id: str
    build_account: str
    #: The dispatch worker's tier, or ``None`` for an untiered worker.
    worker_tier: str | None = "readonly"
    edits: tuple[tuple[str, str], ...] = ()
    #: Exact edits to ``tiers/readwrite.yml``, the render alice's card lands on.
    readwrite_edits: tuple[tuple[str, str], ...] = ()
    #: Hand-checked entries the map must carry, so the oracle is not vacuous.
    must_map: dict[str, str] = field(default_factory=dict)
    #: Logins the map must leave out.
    must_omit: tuple[str, ...] = ()
    #: ``must_map`` is the whole map: nothing beyond it may be keyed.
    exact: bool = False


_DISPATCH_TIER_LINE = ("\n  tier: readonly\n", "\n")

#: The readwrite tier spelling ``allowed`` over its shipped ``asked``; the root
#: stays ``asked`` and the admin tier ``allowed``.
_READWRITE_ALLOWS = ("\n  control_system.writes: asked\n", "\n  control_system.writes: allowed\n")

_OIDC_AUTH = (
    "    auth:\n      method: password\n",
    "    auth:\n      method: oidc\n      oidc:\n"
    "        issuer: https://idp.example.org\n        claim: preferred_username\n",
)

_AUTHORIZATION_WITH_CLAIM = (
    "    landing:\n",
    "    authorization:\n"
    "      roles: {operator: {tier: readwrite}, viewer: {tier: readonly}}\n"
    "      claims: {claim: groups, map: {ca-operators: operator, ca-viewers: viewer}}\n"
    "    landing:\n",
)

_ALICE_BY_ROLE = (
    '        tier: readwrite\n        display_name: "Control Room (Alice)"\n',
    '        role: operator\n        display_name: "Control Room (Alice)"\n',
)

#: Roles bind under password sign-in too; no claim, so sign-in decides nothing.
_AUTHORIZATION_WITHOUT_CLAIM = (
    "    landing:\n",
    "    authorization:\n"
    "      roles: {operator: {tier: readwrite}, viewer: {tier: readonly}}\n"
    "    landing:\n",
)

#: alice binds by role alone, and the role is readonly.
_ALICE_BY_VIEWER_ROLE = (
    '        tier: readwrite\n        display_name: "Control Room (Alice)"\n',
    '        role: viewer\n        display_name: "Control Room (Alice)"\n',
)

#: dave carries no pin and no role: ``default_tier`` (readonly) is his tier.
_DAVE_ON_DEFAULT_TIER = (
    '        display_name: "Deployment Admin (Carol)"\n',
    '        display_name: "Deployment Admin (Carol)"\n      - name: dave\n        index: 3\n',
)

CASES = (
    # A tier word of its own: the map carries all three, verbatim.
    Case(
        id="allowed-readwrite-tier",
        build_account=BUILDER,
        readwrite_edits=(_READWRITE_ALLOWS,),
        must_map={"alice": "allowed", "bob": "never", "carol": "allowed", BUILDER: "asked"},
        exact=True,
    ),
    # The exemplar as shipped: three pinned cards, a tiered worker on readonly.
    Case(
        id="pinned-cards-tiered-worker",
        build_account=BUILDER,
        must_map={"alice": "asked", "bob": "never", "carol": "allowed", BUILDER: "asked"},
    ),
    Case(
        id="pinned-cards-untiered-worker",
        build_account=BUILDER,
        worker_tier=None,
        edits=(_DISPATCH_TIER_LINE,),
        must_map={"alice": "asked", "bob": "never", "carol": "allowed", BUILDER: "asked"},
    ),
    # Single sign-on picks the tier: only alice's explicit role binds statically;
    # bob's and carol's pins and dave's default_tier do not.
    Case(
        id="oidc-claim-with-default-tier",
        build_account=BUILDER,
        edits=(_OIDC_AUTH, _AUTHORIZATION_WITH_CLAIM, _ALICE_BY_ROLE, _DAVE_ON_DEFAULT_TIER),
        must_map={"alice": "asked", BUILDER: "asked"},
        must_omit=("bob", "carol", "dave"),
    ),
    # The build account is a readonly roster login: it keeps its roster tier,
    # never the root's.
    Case(
        id="build-account-is-readonly-login",
        build_account="bob",
        must_map={"alice": "asked", "bob": "never", "carol": "allowed"},
        must_omit=(BUILDER,),
    ),
    # The build account names a card whose tier sign-in decides: absent stays
    # absent, it is not filled in at the root tier.
    Case(
        id="build-account-is-sign-in-decided-card",
        build_account="bob",
        worker_tier=None,
        edits=(_DISPATCH_TIER_LINE, _OIDC_AUTH, _AUTHORIZATION_WITH_CLAIM, _ALICE_BY_ROLE),
        must_map={"alice": "asked"},
        must_omit=("bob", "carol"),
    ),
    # The build account names a shared app card, not a person: it is a roster
    # card, so it is not filled in at the root tier, and no person rule maps it.
    Case(
        id="build-account-is-app-card-tiered-worker",
        build_account="logbook",
        must_map={"alice": "asked", "bob": "never", "carol": "allowed"},
        must_omit=("logbook",),
        exact=True,
    ),
    Case(
        id="build-account-is-app-card-untiered-worker",
        build_account="logbook",
        worker_tier=None,
        edits=(_DISPATCH_TIER_LINE,),
        must_map={"alice": "asked", "bob": "never", "carol": "allowed"},
        must_omit=("logbook",),
        exact=True,
    ),
    # An unresolved build account: ``'unknown'`` is never a key, so the map is
    # the roster's alone.
    Case(
        id="build-account-is-unknown",
        build_account=UNKNOWN,
        must_map={"alice": "asked", "bob": "never", "carol": "allowed"},
        must_omit=(UNKNOWN,),
        exact=True,
    ),
    # Password sign-in with roles: no claim, so sign-in decides nothing and an
    # explicit role binds statically. alice's role (viewer) gives her readonly;
    # dave, with no pin and no role, takes default_tier. (A card with both role
    # and pin, or a roster with no default_tier, fails lint before any build.)
    Case(
        id="password-sign-in-explicit-role",
        build_account=BUILDER,
        edits=(_AUTHORIZATION_WITHOUT_CLAIM, _ALICE_BY_VIEWER_ROLE, _DAVE_ON_DEFAULT_TIER),
        must_map={
            "alice": "never",
            "bob": "never",
            "carol": "allowed",
            "dave": "never",
            BUILDER: "asked",
        },
        exact=True,
    ),
)


# -- building -----------------------------------------------------------------


@contextmanager
def _pinned_env(build_account: str) -> Iterator[None]:
    """A confined environment with the build account pinned and no tree bound."""
    with preserved_environ():
        os.environ["OSPREY_AUDIT_IDENTITY"] = build_account
        os.environ.pop("OSPREY_TERMINAL_USER", None)
        os.environ.pop(TREE_ENV, None)
        yield


@contextmanager
def _cwd(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _build(base: Path, case: Case) -> Path:
    """The exemplar repo, edited per *case*, after one real ``osprey build``."""
    from click.testing import CliRunner

    from osprey.cli.build_cmd import build as build_command

    base.mkdir(parents=True, exist_ok=True)
    with _pinned_env(case.build_account):
        repo = build_exemplar_repo(base / EXEMPLAR_DIRNAME, seed_env=True)
        profile = repo / "profile.yml"
        text = profile.read_text(encoding="utf-8")
        for old, new in case.edits:
            assert text.count(old) == 1, old
            text = text.replace(old, new)
        profile.write_text(text, encoding="utf-8")
        tier_file = repo / "tiers" / "readwrite.yml"
        tier_text = tier_file.read_text(encoding="utf-8")
        for old, new in case.readwrite_edits:
            assert tier_text.count(old) == 1, old
            tier_text = tier_text.replace(old, new)
        tier_file.write_text(tier_text, encoding="utf-8")
        with _cwd(repo):
            result = CliRunner().invoke(build_command, ["--skip-deps", "--skip-lifecycle"])
    assert result.exit_code == 0, result.output
    return repo


def _redeploy(repo: Path, case: Case, **kwargs: Any) -> None:
    """``prepare_compose_files`` over the built config, with no ``osprey build`` around it."""
    from osprey.deployment.compose_generator import prepare_compose_files

    build = repo / "build"
    with _pinned_env(case.build_account), _cwd(build):
        prepare_compose_files(str(build / "config.yml"), **kwargs)


def _load(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    return document if isinstance(document, dict) else {}


def _carries_map(document: dict[str, Any]) -> bool:
    section = document.get("control_system")
    return isinstance(section, dict) and MAP_KEY in section


def _without_map(document: dict[str, Any]) -> dict[str, Any]:
    staged = dict(document)
    section = dict(staged["control_system"])
    section.pop(MAP_KEY)
    staged["control_system"] = section
    return staged


# -- the oracle, from first principles ------------------------------------------


def _service_envs(compose: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
    for name, service in (compose.get("services") or {}).items():
        env = (service or {}).get("environment") or {}
        if isinstance(env, list):
            env = dict(item.split("=", 1) if "=" in item else (item, None) for item in env)
        yield name, env


def _tree_bound_services(services_dir: Path) -> list[str]:
    """Each service directory whose compose file runs a container that binds a tree."""
    bound = []
    for compose_file in sorted(services_dir.glob("*/docker-compose.yml")):
        compose = _load(compose_file)
        if any(TREE_ENV in env for _, env in _service_envs(compose)):
            bound.append(compose_file.parent.name)
    return bound


def _roster_name(entry: Any) -> str | None:
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict) and isinstance(entry.get("name"), str):
        return entry["name"]
    return None


def _shared_app_cards(web_terminals: dict[str, Any]) -> set[str]:
    """The app cards the roster derives from ``shared:``, read by hand."""
    return {name for name in web_terminals.get("shared") or [] if isinstance(name, str) and name}


def _own_render_writes(repo: Path, web_terminals: dict[str, Any]) -> dict[str, str]:
    """Each person's own render's ``tier_writes``, whatever sign-in later decides."""
    from osprey_connectors.types import tier_writes

    roles = (web_terminals.get("authorization") or {}).get("roles") or {}
    renders = web_terminals["renders"]
    writes: dict[str, str] = {}
    for entry in web_terminals.get("users") or []:
        card = entry if isinstance(entry, dict) else {}
        role = card.get("role")
        tier = (
            roles[role]["tier"] if role else card.get("tier") or web_terminals.get("default_tier")
        )
        if tier is None:
            continue
        own_render = _load(repo / renders[tier]["project_path"] / "config.yml")
        writes[_roster_name(entry)] = tier_writes(own_render.get("control_system"))
    return writes


def _expected_map(repo: Path, build_account: str) -> dict[str, str]:
    """The map the roster in ``build/config.yml`` and the renders on disk call for."""
    from osprey_connectors.types import tier_writes

    root = _load(repo / "build" / "config.yml")
    web_terminals = root["modules"]["web_terminals"]
    auth = web_terminals.get("auth") or {}
    authorization = web_terminals.get("authorization") or {}
    claim = (authorization.get("claims") or {}).get("claim")
    sign_in_picks = auth.get("method") == "oidc" and bool(claim)
    roles = authorization.get("roles") or {}
    renders = web_terminals["renders"]

    expected: dict[str, str] = {}
    # Every roster card: the people, then the apps ``shared:`` derives.
    cards: set[str] = _shared_app_cards(web_terminals)
    for entry in web_terminals.get("users") or []:
        name = _roster_name(entry)
        assert name, entry
        cards.add(name)
        card = entry if isinstance(entry, dict) else {}
        role = card.get("role")
        if sign_in_picks and not role:
            continue
        tier = (
            roles[role]["tier"] if role else card.get("tier") or web_terminals.get("default_tier")
        )
        if tier is None:
            continue
        own_render = _load(repo / renders[tier]["project_path"] / "config.yml")
        expected[name] = tier_writes(own_render.get("control_system"))

    if build_account and build_account != UNKNOWN and build_account not in cards:
        expected[build_account] = tier_writes(root.get("control_system"))
    return expected


def _container_render(repo: Path, service: str, case: Case) -> Path:
    """The render a staged config must equal once its map is taken out."""
    build = repo / "build"
    if service == "dispatch_worker" and case.worker_tier is not None:
        return build / ".image" / f"{EXEMPLAR_DIRNAME}-{case.worker_tier}" / "build" / "config.yml"
    return build / "config.yml"


def _assert_staged_maps(repo: Path, staged_root: Path, case: Case) -> None:
    """Every tree-bound container's staged config under *staged_root* obeys the contract."""
    from osprey_connectors.types import tier_writes

    build = repo / "build"
    services = _tree_bound_services(build / "services")
    # The control-assistant build runs both tree-bound containers; a test that
    # found neither would pass on nothing.
    assert {"bluesky", "dispatch_worker"} <= set(services), services

    expected = _expected_map(repo, case.build_account)
    for login, tier in case.must_map.items():
        assert expected.get(login) == tier, (login, expected)
    for login in case.must_omit:
        assert login not in expected, (login, expected)
    if case.exact:
        assert expected == case.must_map, expected
    web_terminals = _load(build / "config.yml")["modules"]["web_terminals"]
    people = {_roster_name(entry) for entry in web_terminals.get("users") or []}
    apps = _shared_app_cards(web_terminals)
    # The exemplar's apps, so a case naming one exercises the app-card rule.
    assert {"logbook", "knowledge", "channel-finder"} <= apps, apps
    own_writes = _own_render_writes(repo, web_terminals)
    root_writes = tier_writes(_load(build / "config.yml").get("control_system"))

    for service in services:
        staged_path = staged_root / "services" / service / "config.yml"
        staged = _load(staged_path)
        assert _carries_map(staged), staged_path
        login_tiers = staged["control_system"][MAP_KEY]

        assert UNKNOWN not in login_tiers, staged_path
        assert set(login_tiers.values()) <= {"never", "asked", "allowed"}, staged_path
        assert login_tiers == expected, (staged_path, login_tiers, expected)

        # Restated apart from `expected`: no person whose own render is
        # `never` is mapped an arming word, whatever sign-in decides.
        for login, writes in own_writes.items():
            if writes == "never":
                assert login_tiers.get(login, "never") == "never", (staged_path, login)
        # No app card is a key: an app's container never acts as a login.
        assert not apps & set(login_tiers), (staged_path, login_tiers)
        # Every key is a person or the build account; nothing else.
        assert set(login_tiers) <= people | {case.build_account}, (staged_path, login_tiers)
        if case.build_account in people:
            # A build account that is a roster login has its roster tier or none,
            # never the root's by way of being the build account.
            if case.build_account in login_tiers:
                assert login_tiers[case.build_account] == own_writes[case.build_account]
        elif case.build_account not in apps and case.build_account != UNKNOWN:
            assert login_tiers.get(case.build_account) == root_writes, staged_path
        else:
            assert case.build_account not in login_tiers, staged_path

        render = _load(_container_render(repo, service, case))
        assert _without_map(staged) == render, staged_path
        assert staged["control_system"].get("writes") == render["control_system"].get("writes")
        assert tier_writes(staged["control_system"]) == tier_writes(render["control_system"])

    if case.worker_tier is not None:
        # The tiered worker runs its tier's posture, not the root's.
        worker = _load(staged_root / "services" / "dispatch_worker" / "config.yml")
        tier_render = _load(_container_render(repo, "dispatch_worker", case))
        assert tier_writes(worker["control_system"]) == tier_writes(tier_render["control_system"])
        compose = _load(staged_root / "services" / "dispatch_worker" / "docker-compose.yml")
        target = f"/app/{EXEMPLAR_DIRNAME}-{case.worker_tier}/build/config.yml"
        workers = [env for name, env in _service_envs(compose) if TREE_ENV in env]
        assert workers
        for env in workers:
            assert env.get("CONFIG_FILE") == target


def _assert_no_other_render_carries_a_map(repo: Path, *output_trees: Path) -> None:
    """No config under ``build/`` carries a map but a tree-bound service's own.

    *output_trees* are the staging roots that hold tree-bound service configs:
    the build itself, and any redeploy output tree (which sits inside
    ``build/``, so this sweep covers that redeploy's other outputs too).
    """
    build = repo / "build"
    staged = {
        tree / "services" / name / "config.yml"
        for name in _tree_bound_services(build / "services")
        for tree in {build, *output_trees}
    }
    configs = [path for path in build.rglob("config.yml") if path not in staged]

    root = _load(build / "config.yml")
    renders = root["modules"]["web_terminals"]["renders"]
    must_check = {build / "config.yml"}
    must_check |= {repo / entry["project_path"] / "config.yml" for entry in renders.values()}
    image_configs = {p for p in configs if ".image" in p.relative_to(build).parts}
    assert image_configs, "no image render found"
    assert must_check <= set(configs), must_check - set(configs)

    for path in configs:
        assert not _carries_map(_load(path)), path


# -- the cases ------------------------------------------------------------------------


@pytest.fixture(scope="module", params=CASES, ids=[case.id for case in CASES])
def built(request, tmp_path_factory) -> tuple[Path, Case]:
    case: Case = request.param
    return _build(tmp_path_factory.mktemp(f"lt-{case.id}"), case), case


def test_staged_maps_on_the_build(built: tuple[Path, Case]) -> None:
    repo, case = built
    _assert_staged_maps(repo, repo / "build", case)
    _assert_no_other_render_carries_a_map(repo, repo / "build")


@pytest.mark.parametrize("case", CASES, ids=[case.id for case in CASES])
def test_staged_maps_after_redeploy(case: Case, tmp_path: Path) -> None:
    """``prepare_compose_files`` alone, into its own output tree and in place."""
    repo = _build(tmp_path, case)
    build = repo / "build"

    # Into a fresh output tree (``deploy restart``'s call, no output root).
    _redeploy(repo, case)
    _assert_staged_maps(repo, build / "build", case)
    _assert_no_other_render_carries_a_map(repo, build / "build")

    # In place, after removing what the shared staging writes. The tiered
    # worker's copy is the build's to write, so it stays where the build put it,
    # byte for byte: a redeploy must not rewrite it with another posture.
    kept: dict[Path, bytes] = {}
    for service in _tree_bound_services(build / "services"):
        staged_path = build / "services" / service / "config.yml"
        if service == "dispatch_worker" and case.worker_tier is not None:
            kept[staged_path] = staged_path.read_bytes()
            continue
        staged_path.unlink()
    _redeploy(repo, case, output_root=".")
    for staged_path, content in kept.items():
        assert staged_path.read_bytes() == content, staged_path
    _assert_staged_maps(repo, build, case)
    # The fresh tree from the first redeploy is still there; its own staged
    # configs keep their maps, and nothing else anywhere may carry one.
    _assert_no_other_render_carries_a_map(repo, build, build / "build")


# -- a single-user build ----------------------------------------------------------------


def test_single_user_hello_world_build_stages_no_map(tmp_path: Path) -> None:
    """With no tree-bound shared container, no render anywhere carries a map."""
    from click.testing import CliRunner

    from osprey.cli.build_cmd import build as build_command
    from osprey.cli.init_cmd import init

    repo = tmp_path / "solo"
    with _pinned_env(BUILDER):
        runner = CliRunner()
        created = runner.invoke(init, [str(repo), "--preset", "hello-world", "--no-git"])
        assert created.exit_code == 0, created.output
        result = runner.invoke(
            build_command, ["--repo", str(repo), "--skip-deps", "--skip-lifecycle"]
        )
    assert result.exit_code == 0, result.output

    build = repo / "build"
    assert _tree_bound_services(build / "services") == []
    configs = list(build.rglob("config.yml"))
    assert build / "config.yml" in configs
    for path in configs:
        assert not _carries_map(_load(path)), path
