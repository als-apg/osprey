"""``dispatch.tier``: the image half.

The renders built and verified are the served set — every card's render,
``default_tier``, every role's tier and the dispatch tier; :func:`resolve_cards` names every image in both
image modes; ``osprey up`` builds the tier image beside the project image and
before the worker; and ``OSPREY_WORKER_IMAGE`` is refused once a tier is set.
An unset ``dispatch.tier`` leaves every answer as it was.
"""

from __future__ import annotations

import copy

import pytest

from osprey.deployment import container_lifecycle
from osprey.deployment.compose_generator import _inject_project_metadata
from osprey.deployment.web_terminals import render_images
from osprey.deployment.web_terminals.render_images import (
    _referenced_renders,
    dispatch_tier_name,
    render_build_session,
    resolve_dispatch_tier,
)
from osprey.deployment.web_terminals.roster import resolve_cards


def _renders(root: str = "/repo/build") -> dict:
    return {
        name: {"kind": "tier", "project": f"demo-{name}", "project_path": f"{root}/demo-{name}"}
        for name in ("readonly", "readwrite")
    }


def _config(
    *,
    tier: str | None = "readonly",
    image_source: str = "local",
    users: list | None = None,
    default_tier: str = "readwrite",
    worker: dict | None = None,
) -> dict:
    worker_cfg = dict(worker or {})
    if tier is not None:
        worker_cfg["tier"] = tier
    return {
        "project_name": "demo",
        "registry": {"url": "reg.example/osprey"},
        "deployed_services": ["event_dispatcher", "dispatch_worker"],
        "services": {"dispatch_worker": worker_cfg},
        "modules": {
            "web_terminals": {
                "enabled": True,
                "image_source": image_source,
                "image_tag": "v1",
                "default_tier": default_tier,
                "users": list(users or []),
                "renders": _renders(),
            }
        },
    }


def _roster(config: dict) -> list[dict]:
    return resolve_cards(
        config["modules"]["web_terminals"], config.get("registry") or {}, strict=True
    )


# ── the dispatch tier and its one producer ──────────────────────────────────


def test_dispatch_tier_name_reads_the_worker_block() -> None:
    assert dispatch_tier_name(_config()) == "readonly"
    assert dispatch_tier_name(_config(tier=None)) is None
    assert dispatch_tier_name({}) is None
    assert dispatch_tier_name({"services": {"dispatch_worker": None}}) is None


def test_unset_tier_resolves_to_nothing() -> None:
    assert resolve_dispatch_tier(_config(tier=None)) is None


def test_local_mode_tier_image_is_the_render_tag() -> None:
    entry = resolve_dispatch_tier(_config())
    assert entry is not None
    assert entry["render"] == "readonly"
    assert entry["image"] == "demo-readonly:local"
    assert entry["project"] == "demo-readonly"
    assert entry["container_project_dir"] == "/app/demo-readonly"


def test_registry_mode_pulls_the_suffixed_tier_image() -> None:
    entry = resolve_dispatch_tier(_config(image_source="registry"))
    assert entry is not None
    assert entry["image"] == "reg.example/osprey/web-terminal-readonly:v1"


@pytest.mark.parametrize("image_source", ["local", "registry"])
@pytest.mark.parametrize("tier", ["readonly", "readwrite"])
def test_tier_image_is_the_one_a_roster_user_on_that_tier_runs(image_source, tier) -> None:
    """One producer: the worker's image is exactly what a person on the tier gets."""
    config = _config(
        tier=tier, image_source=image_source, users=[{"name": "bob", "index": 0, "tier": tier}]
    )
    (bob,) = _roster(config)
    entry = resolve_dispatch_tier(config)
    assert entry is not None
    assert entry["image"] == bob["image"]
    assert entry["project"] == bob["project"]


def test_tier_without_a_render_is_refused_by_name() -> None:
    config = _config(tier="operator")
    with pytest.raises(ValueError, match=r"dispatch\.tier.*'operator'.*osprey build"):
        resolve_dispatch_tier(config)


def test_tier_without_a_render_is_none_when_lenient() -> None:
    assert resolve_dispatch_tier(_config(tier="operator"), strict=False) is None


def test_resolving_never_mutates_the_config() -> None:
    config = _config(users=[{"name": "bob", "index": 0, "tier": "readwrite"}])
    before = copy.deepcopy(config)
    resolve_dispatch_tier(config)
    _referenced_renders(config, _roster(config))
    assert config == before


# ── the reference set ────────────────────────────────────────────────────────


def test_reference_set_holds_the_dispatch_tier_with_no_user_on_it() -> None:
    config = _config()
    units = _referenced_renders(config, [])
    assert [u["render"] for u in units] == ["readwrite", "readonly"]
    (dispatch,) = (u for u in units if u["render"] == "readonly")
    assert dispatch["image"] == "demo-readonly:local"
    assert dispatch["project_path"] == "/repo/build/demo-readonly"
    assert dispatch["build_profile"] == "tiers/readonly.yml"


def test_reference_set_is_roster_tiers_union_the_dispatch_tier() -> None:
    config = _config(users=[{"name": "alice", "index": 0, "tier": "readwrite"}])
    units = _referenced_renders(config, _roster(config))
    assert [u["render"] for u in units] == ["readwrite", "readonly"]


def test_reference_set_builds_a_shared_tier_once() -> None:
    config = _config(users=[{"name": "bob", "index": 0, "tier": "readonly"}])
    units = _referenced_renders(config, _roster(config))
    assert [u["render"] for u in units] == ["readonly", "readwrite"]


def test_reference_set_builds_a_role_tier_nobody_holds() -> None:
    """Single sign-on may hand out any role's tier, so it is built before anyone
    holds it, its image named exactly as a person on it would pull it."""
    config = _config(tier=None, users=[{"name": "alice", "index": 0, "tier": "readwrite"}])
    web_terminals = config["modules"]["web_terminals"]
    web_terminals["renders"]["operator"] = {
        "kind": "tier",
        "project": "demo-operator",
        "project_path": "/repo/build/demo-operator",
    }
    web_terminals["authorization"] = {"roles": {"op": {"tier": "operator"}}}
    units = _referenced_renders(config, _roster(config))
    assert [u["render"] for u in units] == ["readwrite", "operator"]
    (operator,) = (u for u in units if u["render"] == "operator")
    assert operator["image"] == "demo-operator:local"
    assert operator["build_profile"] == "tiers/operator.yml"


def test_reference_set_builds_an_app_under_its_own_file() -> None:
    config = _config(tier=None)
    web_terminals = config["modules"]["web_terminals"]
    web_terminals["renders"]["logbook"] = {
        "kind": "app",
        "project": "demo-logbook",
        "project_path": "/repo/build/demo-logbook",
    }
    web_terminals["shared"] = ["logbook"]
    units = _referenced_renders(config, _roster(config))
    assert [u["render"] for u in units] == ["logbook", "readwrite"]
    (app,) = (u for u in units if u["render"] == "logbook")
    assert app["build_profile"] == "apps/logbook.yml"


def test_reference_set_without_a_tier_is_the_roster_alone() -> None:
    config = _config(tier=None, users=[{"name": "alice", "index": 0, "tier": "readwrite"}])
    units = _referenced_renders(config, _roster(config))
    assert [u["render"] for u in units] == ["readwrite"]


def test_verify_checks_the_dispatch_tier_render(tmp_path) -> None:
    """A missing tier render is refused before the build, as for a roster tier.

    The dispatch tier is the only served render here (it is also the default),
    so the refusal is the dispatch tier's."""
    config = _config(default_tier="readonly")
    config["modules"]["web_terminals"]["renders"] = _renders(str(tmp_path / "build"))
    (tmp_path / "tiers").mkdir()
    with pytest.raises(ValueError, match="readonly"):
        render_images.verify_renders(config, [], repo_root=tmp_path)


# ── the local build ──────────────────────────────────────────────────────────


class _Builds:
    def __init__(self) -> None:
        self.tags: list[str] = []

    def __call__(self, cmd, **kwargs) -> None:
        self.tags.append(cmd[-1])


@pytest.fixture
def builds(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    recorder = _Builds()
    monkeypatch.setattr(render_images, "get_runtime_command", lambda config: ["docker"])
    monkeypatch.setattr(
        render_images,
        "_render_image_build_cmd",
        lambda runtime, context, tag, *rest: ["docker", "build", tag],
    )
    monkeypatch.setattr(render_images, "run_captured", recorder)
    return recorder


def test_local_build_builds_the_tier_image_with_no_user_on_it(builds) -> None:
    """The dispatch tier is built with nobody on it, beside the default tier,
    which the next person added inherits."""
    render_images.build_render_images(_config(), [], False, {})
    assert builds.tags == ["demo-readwrite:local", "demo-readonly:local"]


def test_without_web_terminals_only_the_dispatch_tier_is_built(builds) -> None:
    """No terminal is deployed, so neither the default tier nor any roster tier
    is; the workers' tier is."""
    config = _config(users=[{"name": "alice", "index": 0, "tier": "readwrite"}])
    config["modules"]["web_terminals"]["enabled"] = False

    assert [u["render"] for u in _referenced_renders(config, [], serves_roster=False)] == [
        "readonly"
    ]
    render_images.build_render_images(config, [], False, {}, serves_roster=False)
    assert builds.tags == ["demo-readonly:local"]


def test_registry_mode_builds_nothing(builds) -> None:
    render_images.build_render_images(_config(image_source="registry"), [], False, {})
    assert builds.tags == []


def test_one_session_builds_each_render_once(builds) -> None:
    """``osprey up`` builds beside the project image; the web stack's later call is a no-op."""
    config = _config(users=[{"name": "alice", "index": 0, "tier": "readwrite"}])
    roster = _roster(config)
    with render_build_session():
        render_images.build_render_images(config, roster, False, {})
        render_images.build_render_images(config, roster, False, {})
    assert builds.tags == ["demo-readwrite:local", "demo-readonly:local"]


def test_outside_a_session_each_call_builds(builds) -> None:
    config = _config()
    render_images.build_render_images(config, [], False, {})
    render_images.build_render_images(config, [], False, {})
    assert builds.tags == ["demo-readwrite:local", "demo-readonly:local"] * 2


# ── the worker image and its env override ───────────────────────────────────


def test_worker_runs_the_tier_image() -> None:
    assert container_lifecycle._worker_image_target(_config(), {}) == "demo-readonly:local"


def test_worker_env_override_is_refused_when_a_tier_is_set() -> None:
    with pytest.raises(ValueError, match=r"OSPREY_WORKER_IMAGE.*dispatch\.tier"):
        container_lifecycle._worker_image_target(_config(), {"OSPREY_WORKER_IMAGE": "x:1"})


def test_worker_env_override_still_wins_without_a_tier() -> None:
    config = _config(tier=None)
    assert container_lifecycle._worker_image_target(config, {"OSPREY_WORKER_IMAGE": "x:1"}) == "x:1"


def test_worker_pinned_image_contradicting_the_tier_is_refused() -> None:
    config = _config(worker={"image": "pinned:1"})
    with pytest.raises(ValueError, match=r"services\.dispatch_worker\.image.*dispatch\.tier"):
        container_lifecycle._worker_image_target(config, {})


def test_tiered_worker_adds_no_project_image_build_of_its_own() -> None:
    """The project image is built for the services that run it: a tiered worker
    is not one, and agent-record, which every deployment runs, is."""
    tiered = container_lifecycle._project_image_build_target(_config(), {})
    untiered = container_lifecycle._project_image_build_target(_config(tier=None), {})
    assert tiered == untiered == "demo:local"


# ── the compose render ───────────────────────────────────────────────────────


def test_compose_render_names_the_tier_image() -> None:
    config = _config()
    rendered = _inject_project_metadata(config)
    assert rendered["services"]["dispatch_worker"]["image"] == "demo-readonly:local"
    assert "image" not in config["services"]["dispatch_worker"]


def test_compose_render_registry_mode_names_the_suffixed_image() -> None:
    rendered = _inject_project_metadata(_config(image_source="registry"))
    assert (
        rendered["services"]["dispatch_worker"]["image"]
        == "reg.example/osprey/web-terminal-readonly:v1"
    )


def test_compose_render_without_a_tier_is_unchanged() -> None:
    rendered = _inject_project_metadata(_config(tier=None))
    worker_image = rendered["services"]["dispatch_worker"]["image"]
    assert worker_image == rendered["osprey_images"]["worker"]


# ── osprey up: the tier image before the worker ─────────────────────────────


def test_up_step_does_nothing_without_a_tier(monkeypatch, tmp_path) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        container_lifecycle, "verify_renders", lambda *a, **k: calls.append("verify")
    )
    monkeypatch.setattr(
        container_lifecycle, "build_render_images", lambda *a, **k: calls.append("build")
    )
    container_lifecycle._build_referenced_renders(
        _config(tier=None), tmp_path, False, {}, web_terminals_enabled=True
    )
    assert calls == []


@pytest.mark.parametrize("web_terminals_enabled", [True, False])
def test_up_step_verifies_then_builds_the_whole_reference_set(
    monkeypatch, tmp_path, web_terminals_enabled
) -> None:
    config = _config(users=[{"name": "alice", "index": 0, "tier": "readwrite"}])
    seen: list[tuple[str, list[str], bool]] = []
    monkeypatch.setattr(
        container_lifecycle,
        "verify_renders",
        lambda cfg, users, repo_root, serves_roster: seen.append(
            ("verify", [u["name"] for u in users], serves_roster)
        ),
    )
    monkeypatch.setattr(
        container_lifecycle,
        "build_render_images",
        lambda cfg, users, dev, env, serves_roster: seen.append(
            ("build", [u["name"] for u in users], serves_roster)
        ),
    )
    container_lifecycle._build_referenced_renders(
        config, tmp_path, False, {}, web_terminals_enabled=web_terminals_enabled
    )
    roster = ["alice"] if web_terminals_enabled else []
    assert seen == [
        ("verify", roster, web_terminals_enabled),
        ("build", roster, web_terminals_enabled),
    ]


def test_up_step_refuses_the_worker_env_override(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(container_lifecycle, "verify_renders", lambda *a, **k: None)
    monkeypatch.setattr(container_lifecycle, "build_render_images", lambda *a, **k: None)
    with pytest.raises(ValueError, match="OSPREY_WORKER_IMAGE"):
        container_lifecycle._build_referenced_renders(
            _config(), tmp_path, False, {"OSPREY_WORKER_IMAGE": "x:1"}, web_terminals_enabled=True
        )
