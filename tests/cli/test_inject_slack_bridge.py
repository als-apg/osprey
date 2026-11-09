"""Tests for the ``_inject_slack_bridge`` build step in ``osprey.cli.build_cmd``.

Covers the three responsibilities of the slack-bridge-injection step: copying
the bundled ``templates/services/slack_bridge/`` compose template, writing the
``services.slack_bridge`` config + registering it in ``deployed_services``
(additively and idempotently), and — the part a config-level assertion cannot
reach — that the config block it writes is *sufficient to render the template*.
The compose template reads ``services.slack_bridge.trigger`` with no ``|
default``, so a missing key would render an empty ``DISPATCH_TRIGGER`` and
produce a bridge that either refuses to boot or fires a trigger nobody declared.
The render tests therefore assert on the rendered compose text, not merely on a
zero exit code.

Mirrors ``test_inject_teams_bridge.py``, which the injector is a structural copy
of. Two things genuinely differ. The credential group is Slack's two tokens.
And there is no relay: Socket Mode is an outbound connection the bridge opens
itself, so the template directory is flat and nothing travels beside the
compose file but its image context.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest
import yaml
from jinja2 import Environment, FileSystemLoader
from ruamel.yaml import YAML

from osprey.bridges.core import CoreConfig
from osprey.bridges.slack.config import SlackBridgeConfig
from osprey.cli.build_cmd import _inject_slack_bridge
from osprey.cli.build_profile import SlackBridgeProfileConfig
from osprey.deployment.compose_generator import _inject_project_metadata
from tests.cli._lifecycle_build import invoke_build

_TRIGGERS_YAML = """\
dispatcher:
  dispatch_target: http://dispatch-worker-1:10011
triggers:
  - name: slack-question
    source: webhook
    action:
      prompt: Answer the operator's question.
  - name: other-trigger
    source: webhook
    action:
      prompt: Something else.
"""

_PROFILE_YAML = """\
extends: hello-world
name: SlackBridgeTest
project_name: tmproj
data: data
provider: anthropic
model: claude-haiku-4-5
dispatch:
  triggers: triggers.yml
{slack_bridge}
"""

#: The two credentials the operator supplies by hand. Each stays a BARE
#: ``${VAR}``: the bridge's own ``require_startup()`` is what fails closed on
#: them, naming every missing one at once.
_SLACK_CREDENTIALS = ("SLACK_BOT_TOKEN", "SLACK_APP_TOKEN")

#: ``${NAME}`` / ``${NAME:-default}`` / ``${NAME:?message}``.
_COMPOSE_VAR_RE = re.compile(r"^\$\{(?P<name>[A-Z0-9_]+)(?::-(?P<default>[^}]*))?(?::\?[^}]*)?\}$")


def _write_config(project_path: Path, *, deployed: list[str] | None = None) -> None:
    """Write a minimal config.yml with a pre-existing deployed service."""
    yaml_rt = YAML()
    config: dict = {
        "project_name": "tmfixture",
        "deployed_services": ["postgresql"] if deployed is None else deployed,
        "services": {"postgresql": {}},
        "system": {"timezone": "UTC"},
    }
    yaml_rt.dump(config, project_path / "config.yml")


def _read_config(project_path: Path) -> dict:
    """Reload config.yml as a plain dict."""
    yaml_rt = YAML()
    return yaml_rt.load(project_path / "config.yml")


def _render_bridge_compose(project_path: Path) -> str:
    """Render the project's copy of the bridge compose template.

    Mirrors ``compose_generator.render_template``'s production context: the
    project's own ``config.yml`` passed through ``_inject_project_metadata``.
    Rendering the *project's* copy rather than the packaged template is
    deliberate — it is the copy ``osprey up`` renders.
    """
    config = yaml.safe_load((project_path / "config.yml").read_text(encoding="utf-8"))
    env = Environment(loader=FileSystemLoader(str(project_path)))
    template = env.get_template("services/slack_bridge/docker-compose.yml.j2")
    return template.render(**_inject_project_metadata(config))


def _walk_scalars(node: object, path: str = "") -> list[tuple[str, object]]:
    """Flatten a parsed-YAML subtree to ``(dotted-path, scalar)`` pairs."""
    if isinstance(node, dict):
        return [p for k, v in node.items() for p in _walk_scalars(v, f"{path}.{k}")]
    if isinstance(node, list):
        return [p for i, v in enumerate(node) for p in _walk_scalars(v, f"{path}[{i}]")]
    return [(path, node)]


def _build_bridge_project(
    tmp_path: Path, *, slack_bridge: str = "slack_bridge: {}"
) -> tuple[Path, object]:
    """Run a full ``osprey build`` from a bridge-declaring profile."""
    repo_dir = tmp_path / "tmproj"
    repo_dir.mkdir()
    (repo_dir / "data").mkdir(exist_ok=True)
    (repo_dir / "triggers.yml").write_text(_TRIGGERS_YAML, encoding="utf-8")
    (repo_dir / "profile.yml").write_text(
        _PROFILE_YAML.format(slack_bridge=slack_bridge), encoding="utf-8"
    )
    result = invoke_build(repo_dir)
    return repo_dir / "build", result


def _resolve_compose_env(
    environment: dict[str, object], host_env: dict[str, str] | None = None
) -> dict[str, str]:
    """Resolve each ``${VAR...}`` entry the way compose would against ``host_env``."""
    host_env = host_env or {}
    resolved: dict[str, str] = {}
    for key, raw in environment.items():
        value = str(raw)
        match = _COMPOSE_VAR_RE.match(value)
        if match is None:
            resolved[key] = value
            continue
        name = match.group("name")
        if host_env.get(name):
            resolved[key] = host_env[name]
        elif match.group("default") is not None:
            resolved[key] = match.group("default")
        else:
            resolved[key] = ""
    return resolved


# ── injector unit behavior ───────────────────────────────────────────────────


def test_inject_slack_bridge_copies_template_dir(tmp_path: Path) -> None:
    """The bundled compose template dir is copied into services/slack_bridge."""
    project_path = tmp_path / "project"
    project_path.mkdir()
    _write_config(project_path)

    _inject_slack_bridge(SlackBridgeProfileConfig(), project_path=project_path)

    dest = project_path / "services" / "slack_bridge"
    assert (dest / "docker-compose.yml.j2").is_file()
    assert (dest / "Dockerfile").is_file()
    assert (dest / ".dockerignore").is_file()
    assert (dest / ".osprey-layer-anchor").is_file()


def test_slack_bridge_layer_anchor_is_byte_identical_to_every_other_context() -> None:
    """The anchor file is the same bytes as every other service context's.

    BuildKit shares the framework deps layer across service images only when
    every instruction up to it — and every file those instructions COPY — is
    identical. A drifted anchor quietly rebuilds the whole framework install
    for this one image.
    """
    services = Path(__file__).resolve().parents[2] / "src" / "osprey" / "templates" / "services"
    ours = (services / "slack_bridge" / ".osprey-layer-anchor").read_bytes()
    assert ours == (services / "nextcloud_bridge" / ".osprey-layer-anchor").read_bytes()
    assert ours == (services / "teams_bridge" / ".osprey-layer-anchor").read_bytes()


def test_slack_bridge_dockerfile_is_the_no_extra_bridge_recipe() -> None:
    """The Dockerfile follows the no-extra bridge precedent (nextcloud) below its header.

    The Slack bridge needs only the framework's core dependencies, so the image
    carries no extras layer: everything from ``FROM`` down is the Nextcloud
    bridge's recipe, instruction for instruction.
    """
    services = Path(__file__).resolve().parents[2] / "src" / "osprey" / "templates" / "services"

    def body(name: str) -> str:
        text = (services / name / "Dockerfile").read_text(encoding="utf-8")
        return text[text.index("FROM ") :]

    assert body("slack_bridge") == body("nextcloud_bridge")
    header = (services / "slack_bridge" / "Dockerfile").read_text(encoding="utf-8")
    assert "OSPREY_SLACK_BRIDGE_IMAGE" in header
    assert "CMD" not in header


def test_inject_slack_bridge_writes_service_config(tmp_path: Path) -> None:
    """services.slack_bridge is written with path + trigger + mentions, and
    deployed_services is additive — keeps existing services, appends the bridge."""
    project_path = tmp_path / "project"
    project_path.mkdir()
    _write_config(project_path)

    _inject_slack_bridge(SlackBridgeProfileConfig(), project_path=project_path)

    config = _read_config(project_path)
    svc = config["services"]["slack_bridge"]
    assert svc["path"] == "./services/slack_bridge"
    assert svc["trigger"] == "slack-question"
    assert svc["mentions"] is True
    assert "image" not in svc

    deployed = [str(s) for s in config["deployed_services"]]
    assert "postgresql" in deployed
    assert "slack_bridge" in deployed


def test_inject_slack_bridge_writes_mentions_off(tmp_path: Path) -> None:
    """``mentions: false`` in the profile reaches the service config and the render."""
    project_path = tmp_path / "project"
    project_path.mkdir()
    _write_config(project_path, deployed=[])

    _inject_slack_bridge(SlackBridgeProfileConfig(mentions=False), project_path=project_path)

    assert _read_config(project_path)["services"]["slack_bridge"]["mentions"] is False
    env = yaml.safe_load(_render_bridge_compose(project_path))["services"]["slack-bridge"][
        "environment"
    ]
    assert env["SLACK_MENTIONS"] == "false"


def test_inject_slack_bridge_writes_custom_trigger(tmp_path: Path) -> None:
    """A non-default profile trigger reaches the service config verbatim."""
    project_path = tmp_path / "project"
    project_path.mkdir()
    _write_config(project_path)

    _inject_slack_bridge(
        SlackBridgeProfileConfig(trigger="other-trigger"), project_path=project_path
    )

    assert _read_config(project_path)["services"]["slack_bridge"]["trigger"] == "other-trigger"


def test_inject_slack_bridge_deployed_services_idempotent(tmp_path: Path) -> None:
    """Re-running the injector does not duplicate the deployed_services entry."""
    project_path = tmp_path / "project"
    project_path.mkdir()
    _write_config(project_path)

    _inject_slack_bridge(SlackBridgeProfileConfig(), project_path=project_path)
    _inject_slack_bridge(SlackBridgeProfileConfig(), project_path=project_path)

    config = _read_config(project_path)
    deployed = [str(s) for s in config["deployed_services"]]
    assert deployed.count("slack_bridge") == 1
    assert config["services"]["slack_bridge"]["path"] == "./services/slack_bridge"


def test_inject_slack_bridge_missing_config_yml_is_noop(tmp_path: Path) -> None:
    """Missing config.yml is a warned no-op, not a crash."""
    project_path = tmp_path / "project"
    project_path.mkdir()

    _inject_slack_bridge(SlackBridgeProfileConfig(), project_path=project_path)

    assert not (project_path / "config.yml").exists()
    # Template is still copied before the config.yml check.
    assert (project_path / "services" / "slack_bridge" / "Dockerfile").is_file()


def test_inject_slack_bridge_ships_the_partials_its_template_imports(tmp_path: Path) -> None:
    """The injector lands the shared macro files the compose template imports.

    Without them ``osprey up`` fails with ``TemplateNotFound``, one deploy too
    late.
    """
    project_path = tmp_path / "project"
    project_path.mkdir()
    _write_config(project_path, deployed=[])

    _inject_slack_bridge(SlackBridgeProfileConfig(), project_path=project_path)

    for partial in ("_network_axis.j2", "_env_axis.j2", "_site_image_args.j2"):
        assert (project_path / "services" / partial).is_file(), (
            f"the compose template's own import target {partial} must travel with it"
        )


def test_inject_slack_bridge_skips_user_owned_service(tmp_path: Path) -> None:
    """A `scaffold claim services/slack_bridge` leaves the project copy untouched,
    while config registration still runs."""
    project_path = tmp_path / "project"
    project_path.mkdir()
    yaml_rt = YAML()
    with open(project_path / "config.yml", "w") as fh:
        yaml_rt.dump(
            {
                "project_name": "tmfixture",
                "deployed_services": [],
                "services": {},
                "system": {"timezone": "UTC"},
                "scaffold": {"user_owned": ["services/slack_bridge"]},
            },
            fh,
        )
    dest = project_path / "services" / "slack_bridge"
    dest.mkdir(parents=True)
    (dest / "docker-compose.yml.j2").write_text("# operator's own\n", encoding="utf-8")

    _inject_slack_bridge(SlackBridgeProfileConfig(), project_path=project_path)

    assert (dest / "docker-compose.yml.j2").read_text(encoding="utf-8") == "# operator's own\n"
    assert not (dest / "Dockerfile").exists()
    assert "slack_bridge" in [str(s) for s in _read_config(project_path)["deployed_services"]]


def test_inject_slack_bridge_carries_operator_authored_keys(tmp_path: Path) -> None:
    """An author-written axis on the service block (an env passthrough) survives
    the injector's whole-block replacement and reaches the render."""
    project_path = tmp_path / "project"
    project_path.mkdir()
    yaml_rt = YAML()
    with open(project_path / "config.yml", "w") as fh:
        yaml_rt.dump(
            {
                "project_name": "tmfixture",
                "deployed_services": [],
                "services": {"slack_bridge": {"env": ["SITE_EGRESS_HINT"]}},
                "system": {"timezone": "UTC"},
            },
            fh,
        )

    _inject_slack_bridge(SlackBridgeProfileConfig(), project_path=project_path)

    svc = _read_config(project_path)["services"]["slack_bridge"]
    assert list(svc["env"]) == ["SITE_EGRESS_HINT"]
    assert svc["trigger"] == "slack-question"
    rendered = yaml.safe_load(_render_bridge_compose(project_path))["services"]["slack-bridge"]
    assert rendered["environment"]["SITE_EGRESS_HINT"] == "${SITE_EGRESS_HINT}"


def test_slack_bridge_foreign_image_is_run_as_named_never_built(tmp_path: Path) -> None:
    """An image other than the OSPREY-built tag is pulled and run, never built."""
    project_path = tmp_path / "project"
    project_path.mkdir()
    _write_config(project_path, deployed=[])
    _inject_slack_bridge(SlackBridgeProfileConfig(), project_path=project_path)

    config = _read_config(project_path)
    config["services"]["slack_bridge"]["image"] = "registry.example/slack:1"
    YAML().dump(config, project_path / "config.yml")

    rendered = yaml.safe_load(_render_bridge_compose(project_path))["services"]["slack-bridge"]
    assert "build" not in rendered
    assert rendered["image"] == "${OSPREY_SLACK_BRIDGE_IMAGE:-registry.example/slack:1}"


def test_inject_slack_bridge_renders_without_dispatch_pair_deployed(tmp_path: Path) -> None:
    """The config block alone renders a valid compose for an EXTERNAL dispatcher:
    no depends_on, and both URLs are required (`:?`) references."""
    project_path = tmp_path / "project"
    project_path.mkdir()
    _write_config(project_path, deployed=[])

    _inject_slack_bridge(SlackBridgeProfileConfig(), project_path=project_path)

    rendered = _render_bridge_compose(project_path)
    assert "{{" not in rendered and "{%" not in rendered
    svc = yaml.safe_load(rendered)["services"]["slack-bridge"]
    assert "depends_on" not in svc
    env = svc["environment"]
    for var in ("DISPATCHER_URL", "WORKER_URL"):
        value = env[var]
        assert value.startswith(f"${{{var}"), f"{var} must reference its own env var: {value!r}"
        assert ":?" in value, f"{var} must fail closed when unset (no :? guard): {value!r}"
    assert env["DISPATCH_TRIGGER"] == "slack-question"


def test_slack_bridge_state_paths_match_their_code_defaults(tmp_path: Path) -> None:
    """``DEDUP_PATH``'s compose default equals the config dataclass's default.

    Its reader is a plain ``e.get(NAME, default)``, for which "" is an accepted
    value, so the compose file restates the code default as a literal ``:-``
    fallback. The default is read back OUT of the real config class rather
    than restated here, so changing either side alone fails. The inbox lives
    beside it (``default_inbox_dir``), so it lands on the same volume.
    """
    project_path = tmp_path / "project"
    project_path.mkdir()
    _write_config(project_path, deployed=[])
    _inject_slack_bridge(SlackBridgeProfileConfig(), project_path=project_path)

    svc = yaml.safe_load(_render_bridge_compose(project_path))["services"]["slack-bridge"]
    environment = svc["environment"]
    code_default = CoreConfig.from_env({}).dedup_path

    match = _COMPOSE_VAR_RE.match(str(environment["DEDUP_PATH"]))
    assert match is not None and match.group("default") == code_default
    assert _resolve_compose_env(environment)["DEDUP_PATH"] == code_default
    # The state root the default sits under is the named volume's mount point.
    assert code_default.startswith("/data/")
    assert "slack_bridge_data:/data" in svc["volumes"]
    overridden = _resolve_compose_env(environment, host_env={"DEDUP_PATH": "/srv/x.json"})
    assert overridden["DEDUP_PATH"] == "/srv/x.json"


def test_slack_bridge_rendered_env_boots_the_real_config(tmp_path: Path) -> None:
    """The rendered environment, with the operator's .env supplied, satisfies the
    bridge's own startup check — and without the tokens it names both."""
    project_path = tmp_path / "project"
    project_path.mkdir()
    _write_config(project_path, deployed=[])
    _inject_slack_bridge(SlackBridgeProfileConfig(), project_path=project_path)

    environment = yaml.safe_load(_render_bridge_compose(project_path))["services"]["slack-bridge"][
        "environment"
    ]
    host_env = {
        "SLACK_BOT_TOKEN": "xoxb-test",
        "SLACK_APP_TOKEN": "xapp-test",
        "EVENT_DISPATCHER_TOKEN": "d",
        "DISPATCH_WORKER_TOKEN": "w",
        "DISPATCHER_URL": "http://dispatcher:1",
        "WORKER_URL": "http://worker:2",
    }
    cfg = SlackBridgeConfig.from_env(_resolve_compose_env(environment, host_env))
    cfg.require_startup()
    assert cfg.bot_token == "xoxb-test"
    assert cfg.app_token == "xapp-test"
    assert cfg.mentions is True
    assert cfg.core.trigger == "slack-question"

    no_tokens = {k: v for k, v in host_env.items() if not k.startswith("SLACK_")}
    bare = SlackBridgeConfig.from_env(_resolve_compose_env(environment, no_tokens))
    with pytest.raises(ValueError, match="SLACK_BOT_TOKEN, SLACK_APP_TOKEN"):
        bare.require_startup()


# ── full `osprey build` ──────────────────────────────────────────────────────


def test_full_build_with_bridge_renders_service_dir(tmp_path: Path) -> None:
    """A full build from a bridge-declaring profile scaffolds the service dir and
    registers the bridge in the project's config.yml."""
    project_path, result = _build_bridge_project(tmp_path)

    assert result.exit_code == 0, result.output

    dest = project_path / "services" / "slack_bridge"
    assert (dest / "docker-compose.yml.j2").is_file()
    assert (dest / "Dockerfile").is_file()

    config = yaml.safe_load((project_path / "config.yml").read_text(encoding="utf-8"))
    assert config["services"]["slack_bridge"] == {
        "path": "./services/slack_bridge",
        "trigger": "slack-question",
        "mentions": True,
    }
    deployed = config["deployed_services"]
    assert "slack_bridge" in deployed
    assert "event_dispatcher" in deployed
    assert "dispatch_worker" in deployed


def test_full_build_compose_renders_with_no_unrendered_jinja(tmp_path: Path) -> None:
    """The built project's bridge compose renders clean: no leftover Jinja, and
    every template-substituted value is non-empty."""
    project_path, result = _build_bridge_project(tmp_path)
    assert result.exit_code == 0, result.output

    rendered = _render_bridge_compose(project_path)

    assert "{{" not in rendered, f"unrendered Jinja expression in compose:\n{rendered}"
    assert "{%" not in rendered, f"unrendered Jinja statement in compose:\n{rendered}"

    svc = yaml.safe_load(rendered)["services"]["slack-bridge"]
    for path, value in _walk_scalars(svc):
        if path.endswith("OSPREY_VERSION"):
            continue
        assert value not in (None, ""), f"empty rendered value at {path}"
    assert svc["image"] == "${OSPREY_SLACK_BRIDGE_IMAGE:-tmproj-slack-bridge:local}"
    assert svc["container_name"] == "tmproj-slack-bridge"
    assert svc["build"]["context"] == "./build/services/slack_bridge"
    assert svc["build"]["args"]["OSPREY_PROJECT_NAME"] == "tmproj"
    assert svc["command"] == ["python", "-m", "osprey.bridges.slack"]
    assert svc["depends_on"] == {"event-dispatcher": {"condition": "service_healthy"}}
    env = svc["environment"]
    assert env["DISPATCH_TRIGGER"] == "slack-question"
    assert env["SLACK_MENTIONS"] == "true"
    assert env["DISPATCHER_URL"] == "http://event-dispatcher:10010"
    assert env["WORKER_URL"] == "http://dispatch-worker-1:10011"
    # Slack tokens stay BARE: require_startup() fails closed on them, naming
    # both at once; a `:-` default would boot the bridge as nobody.
    for var in _SLACK_CREDENTIALS:
        assert env[var] == f"${{{var}}}"
    for var in ("EVENT_DISPATCHER_TOKEN", "DISPATCH_WORKER_TOKEN"):
        assert env[var] == f"${{{var}}}"
    # Optional settings pass through empty, so CoreConfig/SlackBridgeConfig
    # own their defaults.
    for var in ("APP_VERSION_DISPLAY", "BRIDGE_TRUST_ENV", "POLL_INTERVAL", "HISTORY_ANSWER_LIMIT"):
        assert env[var] == f"${{{var}:-}}"
    assert env["OSPREY_SPOOL_DIR"].endswith("/slack-bridge")
    assert env["OSPREY_STATE_DIR"].endswith("/slack-bridge")
    assert env["TZ"] == "UTC"
    # No Teams/Azure/Chat leftovers from the template it was copied from.
    assert not [k for k in env if k.startswith(("TEAMS_", "GCHAT_", "NEXTCLOUD_"))]
    # A Socket Mode client opens no listening socket.
    assert "ports" not in svc
    assert "healthcheck" not in svc
    assert svc["volumes"] == [
        "slack_bridge_data:/data",
        "./var/spool/slack-bridge:/app/var/spool/slack-bridge",
        "./var/state/slack-bridge:/app/var/state/slack-bridge",
    ]
    assert "slack_bridge_data" in yaml.safe_load(rendered)["volumes"]


def test_full_build_custom_trigger_reaches_rendered_compose(tmp_path: Path) -> None:
    """A profile-declared trigger flows profile → config.yml → DISPATCH_TRIGGER."""
    project_path, result = _build_bridge_project(
        tmp_path, slack_bridge="slack_bridge:\n  trigger: other-trigger"
    )
    assert result.exit_code == 0, result.output

    svc = yaml.safe_load(_render_bridge_compose(project_path))["services"]["slack-bridge"]
    assert svc["environment"]["DISPATCH_TRIGGER"] == "other-trigger"


def test_full_build_without_bridge_scaffolds_no_bridge_service(tmp_path: Path) -> None:
    """The block is opt-in: a profile without it deploys no bridge."""
    project_path, result = _build_bridge_project(tmp_path, slack_bridge="")
    assert result.exit_code == 0, result.output

    config = yaml.safe_load((project_path / "config.yml").read_text(encoding="utf-8"))
    assert "slack_bridge" not in config["deployed_services"]
    assert "slack_bridge" not in config.get("services", {})
    assert not (project_path / "services" / "slack_bridge").exists()


def test_full_build_all_four_bridges_coexist(tmp_path: Path) -> None:
    """Declaring every chat bridge deploys all four, each with its own trigger.

    Guards against a copy-paste slip in one injector clobbering another's
    service block.
    """
    repo_dir = tmp_path / "tmproj"
    repo_dir.mkdir()
    (repo_dir / "data").mkdir(exist_ok=True)
    (repo_dir / "triggers.yml").write_text(
        _TRIGGERS_YAML + "  - name: nextcloud-question\n    source: webhook\n"
        "    action:\n      prompt: Answer on Talk.\n"
        "  - name: gchat-question\n    source: webhook\n"
        "    action:\n      prompt: Answer on Chat.\n"
        "  - name: teams-question\n    source: webhook\n"
        "    action:\n      prompt: Answer on Teams.\n",
        encoding="utf-8",
    )
    (repo_dir / "profile.yml").write_text(
        _PROFILE_YAML.format(slack_bridge="slack_bridge: {}")
        + "nextcloud_bridge: {}\ngchat_bridge: {}\nteams_bridge: {}\n",
        encoding="utf-8",
    )
    result = invoke_build(repo_dir)
    assert result.exit_code == 0, result.output

    project_path = repo_dir / "build"
    config = yaml.safe_load((project_path / "config.yml").read_text(encoding="utf-8"))
    assert config["services"]["slack_bridge"]["trigger"] == "slack-question"
    assert config["services"]["teams_bridge"]["trigger"] == "teams-question"
    assert config["services"]["gchat_bridge"]["trigger"] == "gchat-question"
    assert config["services"]["nextcloud_bridge"]["trigger"] == "nextcloud-question"
    deployed = config["deployed_services"]
    for name in ("slack_bridge", "teams_bridge", "gchat_bridge", "nextcloud_bridge"):
        assert name in deployed
        assert (project_path / "services" / name / "Dockerfile").is_file()


# ── validation failures surface through the build ────────────────────────────


def _build_errors(caplog: pytest.LogCaptureFixture) -> str:
    """The build's logged error text, unwrapped."""
    return " ".join(r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR)


def test_full_build_bridge_without_dispatch_block_aborts(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A bridge declared with no dispatch: block aborts the build with the fix named."""
    repo_dir = tmp_path / "tmproj"
    repo_dir.mkdir()
    (repo_dir / "data").mkdir(exist_ok=True)
    (repo_dir / "profile.yml").write_text(
        "extends: hello-world\nname: SlackNoDispatch\nproject_name: tmproj\ndata: data\n"
        "provider: anthropic\nmodel: claude-haiku-4-5\nslack_bridge: {}\n",
        encoding="utf-8",
    )
    with caplog.at_level(logging.ERROR):
        result = invoke_build(repo_dir)

    assert result.exit_code != 0
    errors = _build_errors(caplog)
    assert "Build profile validation failed" in errors
    assert "slack_bridge requires a 'dispatch:' block" in errors
    assert not (repo_dir / "build" / "services" / "slack_bridge").exists()


def test_full_build_unknown_trigger_aborts(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A trigger absent from the resolved triggers file aborts the build, listing
    the names that ARE declared."""
    with caplog.at_level(logging.ERROR):
        project_path, result = _build_bridge_project(
            tmp_path, slack_bridge="slack_bridge:\n  trigger: ghost"
        )

    assert result.exit_code != 0
    errors = _build_errors(caplog)
    assert "Build profile validation failed" in errors
    assert "slack_bridge.trigger 'ghost' is not declared in" in errors
    assert "slack-question" in errors
    assert not (project_path / "services" / "slack_bridge").exists()
