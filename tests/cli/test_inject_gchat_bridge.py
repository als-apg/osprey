"""Tests for the ``_inject_gchat_bridge`` build step in ``osprey.cli.build_cmd``.

Covers the three responsibilities of the gchat-bridge-injection step: copying
the bundled ``templates/services/gchat_bridge/`` compose template, writing the
``services.gchat_bridge`` config + registering it in ``deployed_services``
(additively and idempotently), and — the part a config-level assertion cannot
reach — that the config block it writes is *sufficient to render the template*.
The compose template reads ``services.gchat_bridge.trigger`` with no ``|
default``, so a missing key would render an empty ``DISPATCH_TRIGGER`` and
produce a bridge that either crash-loops or fires a trigger nobody declared. The
render tests therefore assert on the rendered compose text, not merely on a zero
exit code.

Mirrors ``test_inject_nextcloud_bridge.py``, the other chat channel, which the
injector is a structural copy of. The assertions that genuinely differ are the
Google credential group (``GCHAT_SA_KEY``/``GCHAT_SUBSCRIPTION``/``GCHAT_APP_ID``
stay bare, and the SA key doubles as a volume source) and the absence of any
published port or healthcheck — this service is a Pub/Sub subscriber, not a
server.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner
from jinja2 import Environment, FileSystemLoader
from ruamel.yaml import YAML

from osprey.cli.build_cmd import _inject_gchat_bridge, build
from osprey.cli.build_profile import GChatBridgeProfileConfig
from osprey.deployment.compose_generator import _inject_project_metadata

_TRIGGERS_YAML = """\
dispatcher:
  dispatch_target: http://dispatch-worker-1:10011
triggers:
  - name: gchat-question
    source: webhook
    action:
      prompt: Answer the operator's question.
  - name: other-trigger
    source: webhook
    action:
      prompt: Something else.
"""

_PROFILE_YAML = """\
name: GChatBridgeTest
data_bundle: hello_world
provider: anthropic
model: haiku
dispatch:
  triggers: triggers.yml
{gchat_bridge}
"""


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _write_config(project_path: Path, *, deployed: list[str] | None = None) -> None:
    """Write a minimal config.yml with a pre-existing deployed service."""
    yaml_rt = YAML()
    config: dict = {
        "project_name": "gcfixture",
        "deployed_services": ["postgresql"] if deployed is None else deployed,
        "services": {"postgresql": {}},
        "system": {"timezone": "UTC"},
    }
    with open(project_path / "config.yml", "w") as fh:
        yaml_rt.dump(config, fh)


def _read_config(project_path: Path) -> dict:
    """Reload config.yml as a plain dict."""
    yaml_rt = YAML()
    with open(project_path / "config.yml") as fh:
        return yaml_rt.load(fh)


def _render_bridge_compose(project_path: Path) -> str:
    """Render the project's copy of the bridge compose template.

    Mirrors ``compose_generator.render_template``'s production context exactly:
    the project's own ``config.yml`` passed through ``_inject_project_metadata``
    (which is what supplies ``osprey_labels``/``osprey_version``/
    ``osprey_env_present``). Rendering the *project's* copy rather than the
    packaged template is deliberate — it is the copy ``osprey up``
    renders, so this is the artifact the injector is responsible for.
    """
    config = yaml.safe_load((project_path / "config.yml").read_text(encoding="utf-8"))
    env = Environment(loader=FileSystemLoader(str(project_path)))
    template = env.get_template("services/gchat_bridge/docker-compose.yml.j2")
    return template.render(**_inject_project_metadata(config))


def _walk_scalars(node: object, path: str = "") -> list[tuple[str, object]]:
    """Flatten a parsed-YAML subtree to ``(dotted-path, scalar)`` pairs."""
    if isinstance(node, dict):
        return [p for k, v in node.items() for p in _walk_scalars(v, f"{path}.{k}")]
    if isinstance(node, list):
        return [p for i, v in enumerate(node) for p in _walk_scalars(v, f"{path}[{i}]")]
    return [(path, node)]


def _build_bridge_project(
    runner: CliRunner, tmp_path: Path, *, gchat_bridge: str = "gchat_bridge: {}"
) -> tuple[Path, object]:
    """Run a full ``osprey build`` from a bridge-declaring profile.

    Returns the built project path (which may not exist, for the failure cases)
    and the click result.
    """
    repo_dir = tmp_path / "gcproj"
    repo_dir.mkdir()
    (repo_dir / "triggers.yml").write_text(_TRIGGERS_YAML, encoding="utf-8")
    (repo_dir / "profile.yml").write_text(
        _PROFILE_YAML.format(gchat_bridge=gchat_bridge), encoding="utf-8"
    )
    result = runner.invoke(
        build,
        ["--repo", str(repo_dir), "--skip-deps", "--skip-lifecycle"],
    )
    return repo_dir / "build", result


# ── injector unit behavior ───────────────────────────────────────────────────


def test_inject_gchat_bridge_copies_template_dir(tmp_path: Path) -> None:
    """The bundled compose template dir is copied into services/gchat_bridge."""
    project_path = tmp_path / "project"
    project_path.mkdir()
    _write_config(project_path)

    _inject_gchat_bridge(GChatBridgeProfileConfig(), project_path=project_path)

    dest = project_path / "services" / "gchat_bridge"
    assert (dest / "docker-compose.yml.j2").is_file()
    assert (dest / "Dockerfile").is_file()
    assert (dest / ".dockerignore").is_file()


def test_inject_gchat_bridge_writes_service_config(tmp_path: Path) -> None:
    """services.gchat_bridge is written with path + trigger, and
    deployed_services is additive — keeps existing services, appends the bridge."""
    project_path = tmp_path / "project"
    project_path.mkdir()
    _write_config(project_path)

    _inject_gchat_bridge(GChatBridgeProfileConfig(), project_path=project_path)

    config = _read_config(project_path)
    svc = config["services"]["gchat_bridge"]
    assert svc["path"] == "./services/gchat_bridge"
    assert svc["trigger"] == "gchat-question"
    # No pinned image: the template's own `| default` supplies the local build
    # tag, matching the sibling injectors (_inject_nextcloud_bridge, _inject_bluesky).
    assert "image" not in svc

    deployed = [str(s) for s in config["deployed_services"]]
    assert "postgresql" in deployed
    assert "gchat_bridge" in deployed


def test_inject_gchat_bridge_writes_custom_trigger(tmp_path: Path) -> None:
    """A non-default profile trigger reaches the service config verbatim."""
    project_path = tmp_path / "project"
    project_path.mkdir()
    _write_config(project_path)

    _inject_gchat_bridge(
        GChatBridgeProfileConfig(trigger="other-trigger"), project_path=project_path
    )

    assert _read_config(project_path)["services"]["gchat_bridge"]["trigger"] == "other-trigger"


def test_inject_gchat_bridge_deployed_services_idempotent(tmp_path: Path) -> None:
    """Re-running the injector does not duplicate the deployed_services entry."""
    project_path = tmp_path / "project"
    project_path.mkdir()
    _write_config(project_path)

    _inject_gchat_bridge(GChatBridgeProfileConfig(), project_path=project_path)
    _inject_gchat_bridge(GChatBridgeProfileConfig(), project_path=project_path)

    config = _read_config(project_path)
    deployed = [str(s) for s in config["deployed_services"]]
    assert deployed.count("gchat_bridge") == 1
    # And the second run left one well-formed service block, not a merged mess.
    assert config["services"]["gchat_bridge"]["path"] == "./services/gchat_bridge"


def test_inject_gchat_bridge_missing_config_yml_is_noop(tmp_path: Path) -> None:
    """Missing config.yml is a warned no-op (mirrors _inject_bluesky), not a crash."""
    project_path = tmp_path / "project"
    project_path.mkdir()
    # No config.yml written.

    _inject_gchat_bridge(GChatBridgeProfileConfig(), project_path=project_path)

    assert not (project_path / "config.yml").exists()
    # Template is still copied before the config.yml check.
    assert (project_path / "services" / "gchat_bridge" / "Dockerfile").is_file()


def test_inject_gchat_bridge_ships_the_partials_its_template_imports(tmp_path: Path) -> None:
    """The injector lands the shared macro file the compose template imports.

    The template imports ``services/_network_axis.j2`` by a path relative to the
    PROJECT root, which is where the deploy-time renderer's loader is rooted. A
    project that received the template but not the partial therefore renders
    nothing at all: ``osprey up`` fails with ``TemplateNotFound``, one deploy too
    late. The injector runs alone here, with no build behind it, because that is
    exactly the path on which the partial would otherwise never arrive.
    """
    project_path = tmp_path / "project"
    project_path.mkdir()
    _write_config(project_path, deployed=[])

    _inject_gchat_bridge(GChatBridgeProfileConfig(), project_path=project_path)

    assert (project_path / "services" / "_network_axis.j2").is_file(), (
        "the compose template's own import target must travel with it"
    )


def test_inject_gchat_bridge_skips_user_owned_service(tmp_path: Path) -> None:
    """A `scaffold claim services/gchat_bridge` leaves the project copy untouched.

    The claim is the operator's edit surface for the compose file, so the
    injector must not refresh it back to the packaged version. Config
    registration still runs — the claim covers the template, not the service's
    presence in `deployed_services`.
    """
    project_path = tmp_path / "project"
    project_path.mkdir()
    yaml_rt = YAML()
    with open(project_path / "config.yml", "w") as fh:
        yaml_rt.dump(
            {
                "project_name": "gcfixture",
                "deployed_services": [],
                "services": {},
                "system": {"timezone": "UTC"},
                "scaffold": {"user_owned": ["services/gchat_bridge"]},
            },
            fh,
        )
    dest = project_path / "services" / "gchat_bridge"
    dest.mkdir(parents=True)
    (dest / "docker-compose.yml.j2").write_text("# operator's own\n", encoding="utf-8")

    _inject_gchat_bridge(GChatBridgeProfileConfig(), project_path=project_path)

    assert (dest / "docker-compose.yml.j2").read_text(encoding="utf-8") == "# operator's own\n"
    assert not (dest / "Dockerfile").exists()
    assert "gchat_bridge" in [str(s) for s in _read_config(project_path)["deployed_services"]]


def test_inject_gchat_bridge_renders_without_dispatch_pair_deployed(tmp_path: Path) -> None:
    """The config block alone renders a valid compose for an EXTERNAL dispatcher.

    With neither event_dispatcher nor dispatch_worker in deployed_services the
    template takes its `else` branches: no `depends_on` (compose errors on a
    depends_on naming an undefined service) and passthrough URLs. This is the
    render path that proves the injector supplies every variable the template
    needs on its own, without help from _inject_dispatch.
    """
    project_path = tmp_path / "project"
    project_path.mkdir()
    _write_config(project_path, deployed=[])

    _inject_gchat_bridge(GChatBridgeProfileConfig(), project_path=project_path)

    rendered = _render_bridge_compose(project_path)
    assert "{{" not in rendered and "{%" not in rendered
    doc = yaml.safe_load(rendered)
    svc = doc["services"]["gchat-bridge"]
    assert "depends_on" not in svc
    env = svc["environment"]
    # Both URLs must be REQUIRED-variable references (`${VAR:?...}`). Compose
    # renders an unset *bare* `${VAR}` as an empty string rather than an absent
    # key, so CoreConfig.from_env's default would never fire: the stack would
    # boot and then fail every dispatch against a protocol-less URL, and because
    # a raising handle_event leaves the Pub/Sub message unacknowledged, the
    # subscription would redeliver the same event forever. The `:?` guard is
    # what makes that a startup failure instead.
    #
    # Asserted as "names the var and carries a :? guard" rather than by string
    # equality on purpose: the guard's PRESENCE is the contract, the operator
    # message after `:?` is free wording. Do not tighten this back into an
    # equality check against the current message — that couples the test to
    # prose and breaks on every rewording.
    for var in ("DISPATCHER_URL", "WORKER_URL"):
        value = env[var]
        assert value.startswith(f"${{{var}"), f"{var} must reference its own env var: {value!r}"
        assert ":?" in value, f"{var} must fail closed when unset (no :? guard): {value!r}"
    assert env["DISPATCH_TRIGGER"] == "gchat-question"


# ── full `osprey build` ──────────────────────────────────────────────────────


def test_full_build_with_bridge_renders_service_dir(runner: CliRunner, tmp_path: Path) -> None:
    """A full build from a bridge-declaring profile scaffolds the service dir and
    registers the bridge in the project's config.yml."""
    project_path, result = _build_bridge_project(runner, tmp_path)

    assert result.exit_code == 0, result.output

    dest = project_path / "services" / "gchat_bridge"
    assert (dest / "docker-compose.yml.j2").is_file()
    assert (dest / "Dockerfile").is_file()

    config = yaml.safe_load((project_path / "config.yml").read_text(encoding="utf-8"))
    assert config["services"]["gchat_bridge"] == {
        "path": "./services/gchat_bridge",
        "trigger": "gchat-question",
    }
    deployed = config["deployed_services"]
    assert "gchat_bridge" in deployed
    # Step 10b3 runs after 10b, so the dispatch pair the template gates on is
    # already registered by the time the bridge is injected.
    assert "event_dispatcher" in deployed
    assert "dispatch_worker" in deployed


def test_full_build_compose_renders_with_no_unrendered_jinja(
    runner: CliRunner, tmp_path: Path
) -> None:
    """The built project's bridge compose renders clean: no leftover Jinja, and
    every template-substituted value is non-empty.

    This is the assertion that actually guards the injector's contract with the
    template. A variable the injector forgot to supply does not fail the build —
    it renders an empty value (or, for `services.gchat_bridge`, an undefined
    attribute error at deploy time), so exit code 0 proves nothing on its own.
    """
    project_path, result = _build_bridge_project(runner, tmp_path)
    assert result.exit_code == 0, result.output

    rendered = _render_bridge_compose(project_path)

    assert "{{" not in rendered, f"unrendered Jinja expression in compose:\n{rendered}"
    assert "{%" not in rendered, f"unrendered Jinja statement in compose:\n{rendered}"

    svc = yaml.safe_load(rendered)["services"]["gchat-bridge"]
    # A Jinja variable the injector failed to supply resolves to Undefined, which
    # renders as the empty string — `TZ: ` parses back as None, `DISPATCH_TRIGGER:
    # ""` as "". Neither is ever legitimate inside this service block, so walking
    # it for empty scalars catches a dropped variable that the render itself
    # swallows. (Scoped to the service: the top-level `volumes:`/`networks:`
    # declarations are valueless by design.)
    for path, value in _walk_scalars(svc):
        # OSPREY_VERSION is the one legitimately-empty value: the template
        # defaults it to '' for installs where the version cannot be resolved.
        if path.endswith("OSPREY_VERSION"):
            continue
        assert value not in (None, ""), f"empty rendered value at {path}"
    # Image falls back to the per-project local build tag (no pinned image).
    assert svc["image"] == "${OSPREY_GCHAT_BRIDGE_IMAGE:-gcproj-gchat-bridge:local}"
    assert svc["container_name"] == "gcproj-gchat-bridge"
    assert svc["build"]["context"] == "./build/services/gchat_bridge"
    assert svc["build"]["args"]["OSPREY_PROJECT_NAME"] == "gcproj"
    env = svc["environment"]
    # The one variable with no template-side default.
    assert env["DISPATCH_TRIGGER"] == "gchat-question"
    # deployed_services carries the dispatch pair, so both URLs are resolved
    # in-network from the injected ports. This branch deliberately gets NO `${...:?}`
    # required-variable guard (unlike the external-dispatcher branch): there is no
    # host env var to demand, so a guard here would break a co-deployed stack that
    # correctly sets nothing.
    assert env["DISPATCHER_URL"] == "http://event-dispatcher:10010"
    assert env["WORKER_URL"] == "http://dispatch-worker-1:10011"
    assert ":?" not in env["DISPATCHER_URL"] and ":?" not in env["WORKER_URL"]
    assert svc["depends_on"] == {"event-dispatcher": {"condition": "service_healthy"}}
    # Google credentials stay BARE — deliberately, and for a different reason than
    # the URLs above. A bare `${VAR}` is not fail-closed at the compose layer (unset
    # renders an empty string, not an absent key); these fail closed one layer up,
    # in the bridge's require_startup(), whose falsy check treats "" as missing and
    # aborts naming every missing var at once — a better operator message than
    # compose's per-variable `:?`. What must never appear here is a `:-` default,
    # which would boot the bridge authenticating as nobody or pulling some other
    # project's queue. Do not "harmonise" these with the `:?` form above without
    # moving the require_startup contract too.
    for var in ("GCHAT_SA_KEY", "GCHAT_SUBSCRIPTION", "GCHAT_APP_ID"):
        assert env[var] == f"${{{var}}}"
    # The dispatch tokens are bare for the same reason, and `osprey up` mints them.
    for var in ("EVENT_DISPATCHER_TOKEN", "DISPATCH_WORKER_TOKEN"):
        assert env[var] == f"${{{var}}}"
    assert env["TZ"] == "UTC"
    # A Pub/Sub subscriber opens no listening socket: no published ports, and no
    # healthcheck to probe one with.
    assert "ports" not in svc
    assert "healthcheck" not in svc
    # State volume plus the read-only SA key, mounted at the SAME host path it
    # carries in the container so GCHAT_SA_KEY names one file on both sides. The
    # `:-/dev/null` sentinel is what keeps an unset key a boot-time abort (named
    # by require_startup) rather than a compose-level "empty spec ::ro" error
    # that never mentions the variable.
    assert "gchat_bridge_data:/data" in svc["volumes"]
    assert "${GCHAT_SA_KEY:-/dev/null}:${GCHAT_SA_KEY:-/dev/null}:ro" in svc["volumes"]


def test_full_build_custom_trigger_reaches_rendered_compose(
    runner: CliRunner, tmp_path: Path
) -> None:
    """A profile-declared trigger flows profile → config.yml → DISPATCH_TRIGGER."""
    project_path, result = _build_bridge_project(
        runner, tmp_path, gchat_bridge="gchat_bridge:\n  trigger: other-trigger"
    )
    assert result.exit_code == 0, result.output

    svc = yaml.safe_load(_render_bridge_compose(project_path))["services"]["gchat-bridge"]
    assert svc["environment"]["DISPATCH_TRIGGER"] == "other-trigger"


def test_full_build_without_bridge_scaffolds_no_bridge_service(
    runner: CliRunner, tmp_path: Path
) -> None:
    """The block is opt-in: a profile without it deploys no bridge (step 10b3 skipped)."""
    project_path, result = _build_bridge_project(runner, tmp_path, gchat_bridge="")
    assert result.exit_code == 0, result.output

    config = yaml.safe_load((project_path / "config.yml").read_text(encoding="utf-8"))
    assert "gchat_bridge" not in config["deployed_services"]
    assert "gchat_bridge" not in config.get("services", {})
    assert not (project_path / "services" / "gchat_bridge").exists()


def test_full_build_both_bridges_coexist(runner: CliRunner, tmp_path: Path) -> None:
    """Declaring both chat bridges deploys both, each with its own trigger.

    The two channels are independent injectors writing disjoint config keys, so
    a project may answer on Talk and Chat at once. Guards against a copy-paste
    slip in either injector clobbering the other's service block.
    """
    repo_dir = tmp_path / "gcproj"
    repo_dir.mkdir()
    (repo_dir / "triggers.yml").write_text(
        _TRIGGERS_YAML + "  - name: nextcloud-question\n    source: webhook\n"
        "    action:\n      prompt: Answer on Talk.\n",
        encoding="utf-8",
    )
    (repo_dir / "profile.yml").write_text(
        _PROFILE_YAML.format(gchat_bridge="gchat_bridge: {}") + "nextcloud_bridge: {}\n",
        encoding="utf-8",
    )
    result = runner.invoke(
        build,
        ["--repo", str(repo_dir), "--skip-deps", "--skip-lifecycle"],
    )
    assert result.exit_code == 0, result.output

    project_path = repo_dir / "build"
    config = yaml.safe_load((project_path / "config.yml").read_text(encoding="utf-8"))
    assert config["services"]["gchat_bridge"]["trigger"] == "gchat-question"
    assert config["services"]["nextcloud_bridge"]["trigger"] == "nextcloud-question"
    deployed = config["deployed_services"]
    assert "gchat_bridge" in deployed and "nextcloud_bridge" in deployed
    assert (project_path / "services" / "gchat_bridge" / "Dockerfile").is_file()
    assert (project_path / "services" / "nextcloud_bridge" / "Dockerfile").is_file()


# ── validation failures surface through the build ────────────────────────────


def _build_errors(caplog: pytest.LogCaptureFixture) -> str:
    """The build's logged error text, unwrapped.

    ``result.output`` carries rich's ANSI codes and terminal-width line wrapping,
    so substring assertions against it are width-dependent; the log records hold
    the raw message (the idiom ``test_build_preset`` uses for its warning
    assertions).
    """
    return " ".join(r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR)


def test_full_build_bridge_without_dispatch_block_aborts(
    runner: CliRunner, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A bridge declared with no dispatch: block aborts the build with the fix named.

    The check itself lives in BuildProfile.validate; this asserts the
    BuildProfileError it raises actually reaches an operator running
    `osprey build`, rather than being swallowed into a bare non-zero exit.
    """
    repo_dir = tmp_path / "gcproj"
    repo_dir.mkdir()
    (repo_dir / "profile.yml").write_text(
        "name: GcNoDispatch\ndata_bundle: hello_world\nprovider: anthropic\n"
        "model: haiku\ngchat_bridge: {}\n",
        encoding="utf-8",
    )
    with caplog.at_level(logging.ERROR):
        result = runner.invoke(
            build,
            ["--repo", str(repo_dir), "--skip-deps", "--skip-lifecycle"],
        )

    assert result.exit_code != 0
    errors = _build_errors(caplog)
    assert "Build profile validation failed" in errors
    assert "gchat_bridge requires a 'dispatch:' block" in errors
    assert not (repo_dir / "build" / "services" / "gchat_bridge").exists()


def test_full_build_unknown_trigger_aborts(
    runner: CliRunner, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A trigger absent from the resolved triggers file aborts the build, listing
    the names that ARE declared."""
    with caplog.at_level(logging.ERROR):
        project_path, result = _build_bridge_project(
            runner, tmp_path, gchat_bridge="gchat_bridge:\n  trigger: ghost"
        )

    assert result.exit_code != 0
    errors = _build_errors(caplog)
    assert "Build profile validation failed" in errors
    assert "gchat_bridge.trigger 'ghost' is not declared in" in errors
    # The declared alternatives are named so the operator can fix it in one pass.
    assert "gchat-question" in errors
    assert not (project_path / "services" / "gchat_bridge").exists()
