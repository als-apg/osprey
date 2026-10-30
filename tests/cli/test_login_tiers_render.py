"""Where the build stages ``control_system.login_tiers``, and where it does not.

The map is written only into the staged configs of the tree-bound shared
containers: ``build/services/bluesky/config.yml``, the untiered dispatch
worker's ``build/services/dispatch_worker/config.yml``, and the tiered worker's
staged copy — the tier's container-rooted render plus the map. Every other
render (the root, each terminal-served render, each image render) carries none.

Every build here runs with ``OSPREY_AUDIT_IDENTITY`` pinned, so the build
account the map names does not depend on who runs the tests.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
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

#: The pinned build account (BC-3): roster-shaped, and no roster card's name.
BUILD_ACCOUNT = "builder"

#: The exemplar profile's dispatch tier line (the only two-space ``tier:``).
_DISPATCH_TIER_LINE = "\n  tier: readonly\n"

#: The readwrite tier's write line, spelled in ``tiers/readwrite.yml``; the root
#: profile's, spelled the same way, is the render the shared containers run.
_ASKED_LINE = "\n  control_system.writes: asked\n"

#: The exemplar's roster, each person's own render's tier.
_ROSTER_TIERS = {"alice": "asked", "bob": "never", "carol": "allowed"}

#: Where the two shared-container configs are staged, under ``build/``.
_BLUESKY = Path("services/bluesky/config.yml")
_WORKER = Path("services/dispatch_worker/config.yml")


@contextmanager
def _pinned_env() -> Iterator[None]:
    """A confined environment with the build account pinned and no tree bound."""
    with preserved_environ():
        os.environ["OSPREY_AUDIT_IDENTITY"] = BUILD_ACCOUNT
        os.environ.pop("OSPREY_TERMINAL_USER", None)
        os.environ.pop("OSPREY_CONTROL_CONTEXT_TREE", None)
        yield


@contextmanager
def _cwd(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _ask_in(path: Path) -> None:
    """*path* authors ``asked``, the one write line it has.

    The shipped stance (FR4) puts the control-assistant root and the readwrite
    tier at ``asked``, so the exemplar already authors the word; this proves it
    rather than writing it.
    """
    text = path.read_text(encoding="utf-8")
    assert text.count(_ASKED_LINE) == 1, path
    assert text.count("\n  control_system.writes:") == 1, path


def _build(base: Path, *, tiered: bool = True, asked: bool = False) -> Path:
    """The exemplar repo after one real ``osprey build`` (no venv, no hooks).

    *asked* checks that the root and the readwrite tier author ``asked``; the
    admin tier keeps ``allowed``, so such a build carries all three words.
    """
    from click.testing import CliRunner

    from osprey.cli.build_cmd import build as build_command

    base.mkdir(parents=True, exist_ok=True)
    with _pinned_env():
        repo = build_exemplar_repo(base / EXEMPLAR_DIRNAME, seed_env=True)
        if not tiered:
            profile = repo / "profile.yml"
            text = profile.read_text(encoding="utf-8")
            assert text.count(_DISPATCH_TIER_LINE) == 1
            profile.write_text(text.replace(_DISPATCH_TIER_LINE, "\n"), encoding="utf-8")
        if asked:
            _ask_in(repo / "profile.yml")
            _ask_in(repo / "tiers" / "readwrite.yml")
        with _cwd(repo):
            result = CliRunner().invoke(build_command, ["--skip-deps", "--skip-lifecycle"])
    assert result.exit_code == 0, result.output
    return repo


def _redeploy(repo: Path, **kwargs: Any) -> None:
    """``prepare_compose_files`` over the built config, with no ``osprey build`` around it."""
    from osprey.deployment.compose_generator import prepare_compose_files

    build = repo / "build"
    with _pinned_env(), _cwd(build):
        prepare_compose_files(str(build / "config.yml"), **kwargs)


def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _without_map(document: dict[str, Any]) -> dict[str, Any]:
    staged = dict(document)
    section = dict(staged["control_system"])
    section.pop("login_tiers")
    staged["control_system"] = section
    return staged


def _expected_map() -> dict[str, str]:
    """From first principles: each roster card at its render's tier, the build account at root."""
    return {**_ROSTER_TIERS, BUILD_ACCOUNT: "asked"}


@pytest.fixture(scope="module")
def tiered(tmp_path_factory) -> Path:
    return _build(tmp_path_factory.mktemp("login-tiers-tiered"))


@pytest.fixture(scope="module")
def untiered(tmp_path_factory) -> Path:
    return _build(tmp_path_factory.mktemp("login-tiers-untiered"), tiered=False)


@pytest.fixture(scope="module")
def asked(tmp_path_factory) -> Path:
    return _build(tmp_path_factory.mktemp("login-tiers-asked"), asked=True)


# -- an authored `asked` -------------------------------------------------------


def test_an_asked_render_stages_the_word_verbatim(asked: Path) -> None:
    build = asked / "build"
    expected = {"alice": "asked", "bob": "never", "carol": "allowed", BUILD_ACCOUNT: "asked"}

    assert _load(build / _BLUESKY)["control_system"]["login_tiers"] == expected
    assert _load(build / _WORKER)["control_system"]["login_tiers"] == expected


def test_an_asked_render_holds_the_queue_token_and_a_never_render_does_not(asked: Path) -> None:
    from osprey.deployment.web_terminals.roster import config_needs_launch_token

    build = asked / "build"
    project = f"{EXEMPLAR_DIRNAME}-"

    assert config_needs_launch_token(_load(build / "config.yml"))
    assert config_needs_launch_token(_load(build / f"{project}readwrite" / "config.yml"))
    assert not config_needs_launch_token(_load(build / f"{project}readonly" / "config.yml"))


# -- the tiered dispatch copy (BE-1) ------------------------------------------


def test_tiered_worker_copy_is_the_tier_render_plus_the_map(tiered: Path) -> None:
    build = tiered / "build"
    root = _load(build / "config.yml")
    image = _load(build / ".image" / f"{EXEMPLAR_DIRNAME}-readonly" / "build" / "config.yml")
    staged = _load(build / _WORKER)

    assert staged["control_system"]["login_tiers"] == _expected_map()
    assert _without_map(staged) == image
    assert staged["control_system"]["writes"] == "never"
    # A copy of the root would not pass: the control-assistant root arms writes.
    assert root["control_system"]["writes"] == "asked"


def test_tiered_worker_compose_mounts_the_staged_copy(tiered: Path) -> None:
    compose = _load(tiered / "build" / "services" / "dispatch_worker" / "docker-compose.yml")
    target = f"/app/{EXEMPLAR_DIRNAME}-readonly/build/config.yml"
    workers = [s for k, s in compose["services"].items() if k.startswith("dispatch-worker")]
    assert workers
    for service in workers:
        assert f"./build/{_WORKER.as_posix()}:{target}:ro" in service["volumes"]
        assert service["environment"]["CONFIG_FILE"] == target


# -- the shared staging (BE-6) -------------------------------------------------


def test_bluesky_config_carries_the_map_and_is_otherwise_the_root_flatten(tiered: Path) -> None:
    build = tiered / "build"
    staged = _load(build / _BLUESKY)
    assert staged["control_system"]["login_tiers"] == _expected_map()
    assert _without_map(staged) == _load(build / "config.yml")


def test_untiered_worker_and_bluesky_carry_the_same_map_over_the_root(untiered: Path) -> None:
    build = untiered / "build"
    root = _load(build / "config.yml")
    for staged_path in (_BLUESKY, _WORKER):
        staged = _load(build / staged_path)
        assert staged["control_system"]["login_tiers"] == _expected_map(), staged_path
        assert _without_map(staged) == root, staged_path


def test_no_other_render_carries_the_map(tiered: Path) -> None:
    build = tiered / "build"
    staged = {build / _BLUESKY, build / _WORKER}
    configs = [p for p in build.rglob("config.yml") if p not in staged]
    # The root, every terminal-served render, and every image render are among them.
    assert build / "config.yml" in configs
    assert build / f"{EXEMPLAR_DIRNAME}-readwrite" / "config.yml" in configs
    assert build / ".image" / f"{EXEMPLAR_DIRNAME}-readonly" / "build" / "config.yml" in configs
    for path in configs:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        section = document.get("control_system") if isinstance(document, dict) else None
        assert not (isinstance(section, dict) and "login_tiers" in section), path


# -- redeploy paths --------------------------------------------------------------


def test_redeploy_in_place_stages_the_same_map_and_keeps_the_tiered_copy(tmp_path: Path) -> None:
    repo = _build(tmp_path)
    build = repo / "build"
    tiered_copy = (build / _WORKER).read_bytes()
    built_map = _load(build / _BLUESKY)["control_system"]["login_tiers"]

    _redeploy(repo, output_root=".")

    assert _load(build / _BLUESKY)["control_system"]["login_tiers"] == built_map
    assert (build / _WORKER).read_bytes() == tiered_copy


def test_redeploy_without_build_stages_the_same_map(tmp_path: Path) -> None:
    """``prepare_compose_files(config_path)`` alone — the ``deploy restart`` call."""
    repo = _build(tmp_path)
    build = repo / "build"
    tiered_copy = (build / _WORKER).read_bytes()
    built_map = _load(build / _BLUESKY)["control_system"]["login_tiers"]

    _redeploy(repo)

    out = build / "build"
    assert _load(out / _BLUESKY)["control_system"]["login_tiers"] == built_map
    # The root flatten never writes the tiered worker's config: the context
    # carries the copy the build staged, byte for byte ...
    assert (out / _WORKER).read_bytes() == tiered_copy
    # ... and that copy is untouched.
    assert (build / _WORKER).read_bytes() == tiered_copy

    # A redeploy that clears a persistent output directory keeps the copy it
    # finds there, even when the source context no longer carries one.
    (build / _WORKER).unlink()
    _redeploy(repo)
    assert (out / _WORKER).read_bytes() == tiered_copy


def test_flatten_failure_fallback_still_stages_the_map(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import osprey.deployment.compose_generator as compose_generator

    repo = _build(tmp_path, tiered=False)
    build = repo / "build"

    class _Broken:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("flatten unavailable")

    monkeypatch.setattr(compose_generator, "ConfigBuilder", _Broken)
    for staged_path in (_BLUESKY, _WORKER):
        (build / staged_path).unlink()

    _redeploy(repo, output_root=".")

    root = _load(build / "config.yml")
    for staged_path in (_BLUESKY, _WORKER):
        staged = _load(build / staged_path)
        assert staged["control_system"]["login_tiers"] == _expected_map(), staged_path
        assert _without_map(staged) == root, staged_path


# -- who is in the map (BB-1, BB-8) -----------------------------------------------


def test_sign_in_decided_card_is_absent_and_its_own_terminal_is_not_refused(
    tmp_path: Path,
) -> None:
    from osprey_connectors.posture_store import owner_tier_term
    from osprey_connectors.types import tier_writes

    repo = _build(tmp_path)
    build = repo / "build"
    config = _load(build / "config.yml")
    web_terminals = config["modules"]["web_terminals"]
    web_terminals["auth"] = {**web_terminals["auth"], "method": "oidc"}
    web_terminals["authorization"] = {
        "roles": {"operator": {"tier": "readwrite"}},
        "claims": {"claim": "groups", "map": {"ops": "operator"}},
    }
    # alice is bound statically by her role; dave's tier is decided at sign-in,
    # whatever his pin says.
    for user in web_terminals["users"]:
        if user["name"] == "alice":
            user.pop("tier")
            user["role"] = "operator"
    web_terminals["users"].append({"name": "dave", "index": 3, "tier": "readwrite"})
    (build / "config.yml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    _redeploy(repo, output_root=".")

    shared_map = _load(build / _BLUESKY)["control_system"]["login_tiers"]
    assert shared_map["alice"] == "asked"
    assert "dave" not in shared_map
    assert "dave" not in _load(build / _WORKER)["control_system"]["login_tiers"]

    # dave's own terminal runs the readwrite render, which carries no map: with no
    # tree bound the owner rung does not apply, and the render's tier governs.
    terminal = _load(build / f"{EXEMPLAR_DIRNAME}-readwrite" / "config.yml")["control_system"]
    with _pinned_env():
        assert owner_tier_term(terminal, "dave") == "skip"
    assert tier_writes(terminal) == "asked"


def test_single_user_host_process_as_the_build_account_resolves(tiered: Path) -> None:
    from osprey_connectors.posture_store import owner_tier_term
    from osprey_connectors.types import tier_writes

    build = tiered / "build"
    root = _load(build / "config.yml")["control_system"]
    shared = _load(build / _BLUESKY)["control_system"]
    with _pinned_env():
        # The host process reads the root render: no map, no tree, its own tier.
        assert owner_tier_term(root, BUILD_ACCOUNT) == "skip"
        # A shared container carrying that account's job finds it at the root tier.
        assert owner_tier_term(shared, BUILD_ACCOUNT) == tier_writes(root) == "asked"


# -- a profile never spells the key --------------------------------------------------


@pytest.mark.parametrize(
    "config",
    [
        {"control_system.login_tiers": {"mallory": "allowed"}},
        {"control_system.login_tiers.mallory": "allowed"},
        {"control_system": {"login_tiers": {"mallory": "allowed"}}},
        {"control_system": {"writes": "never", "login_tiers": {}}},
    ],
)
def test_a_profile_spelling_the_map_is_refused(config: dict[str, Any]) -> None:
    from osprey.cli.build_profile_deploy import deploy_aware_config_errors
    from osprey.cli.build_profile_schema import login_tiers_key_errors

    (error,) = login_tiers_key_errors(config)
    assert "control_system.login_tiers" in error
    assert error in deploy_aware_config_errors(None, config)


def test_a_profile_without_the_map_is_not_refused() -> None:
    from osprey.cli.build_profile_schema import login_tiers_key_errors

    assert login_tiers_key_errors({"control_system.writes": "allowed"}) == []
    assert login_tiers_key_errors({"control_system": {"login_tier": "x"}}) == []
    assert login_tiers_key_errors(None) == []
