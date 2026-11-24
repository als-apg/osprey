"""Every path a profile can name is protected from executed code, end to end.

The chain runs from what an operator writes to what a child process may do:
a profile states a path under a key — in the base profile, a file it extends,
a login file or the host variant; ``osprey build`` writes the protected-path
set into every render; the python executor reads it, refuses a literal write
there before the run, and the child's filesystem guard refuses the write at
run time while reads stay allowed.

These tests run real builds of a deployment that states every key of the
emitter's three feeds, imported rather than retyped, and nothing else is
mocked on the way from the profile to the child. The table :data:`STATED`
records what is stated, where, and the location its reader opens — taken from
the product's own resolver for the key where one exists, from the reader the
table cites otherwise. A key of the feeds that no buildable profile could state
would be a finding against the table, not a case to skip.

Then the two refusals the chain depends on: a code-review prompt pointed at a
profile-named directory is still refused at build — being protected does not
make a location a source file — and a copy of a real build with the set
removed is refused by each python-executor tool and by ``osprey up``.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import stat
import subprocess
import sys
import textwrap
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
import yaml
from click.testing import Result
from fastmcp.exceptions import ToolError

import osprey
import osprey.mcp_server.python_executor.executor as executor
from osprey.cli.application_material import dependency_package_dirs
from osprey.cli.build_posture_check import CODE_REVIEW_HOOKS
from osprey.cli.build_profile_merge import profile_triggers_material
from osprey.cli.build_profile_presets import resolve_service_template
from osprey.cli.build_protected_paths import (
    CONFIG_PATH_KEYS,
    INPUT_ENUMERATION_KEYS,
    PROFILE_PATH_FIELDS,
)
from osprey.mcp_server.python_executor.executor import resolve_protected_roots
from osprey.mcp_server.python_executor.tools import _execution_gates as gates
from osprey.registry.manager import resolve_registry_path
from osprey.services.channel_directory.demo import resolve_replay_file
from osprey.services.python_executor.execution.path_policy import path_policy_issues
from osprey.services.python_executor.execution.wrapper import ExecutionWrapper
from osprey.utils.config_paths import resolve_config_relative_path, resolve_render_relative_path
from osprey.utils.workspace import (
    PROTECTED_PATHS_CONFIG_KEY,
    ProtectedPathsState,
    anchor_protected_entry,
    read_protected_paths,
)
from osprey_connectors.workspace import anchored_path, reset_config_cache
from osprey_facility.roots import DEPLOYMENT_CONTENT_DIRS
from tests.cli._lifecycle_build import invoke_build_in
from tests.deployment.test_up_as_built import run_up, started  # noqa: F401 - a fixture
from tests.fixtures.lifecycle_repo import EXEMPLAR_DIRNAME, build_exemplar_repo, preserved_environ
from tests.mcp_server.conftest import get_tool_fn

pytestmark = pytest.mark.slow

#: The bundled facility the machine root is written from.
EXAMPLE_FACILITY = Path(osprey.__file__).parent / "templates" / "facilities" / "example"

# ── where a key is stated ─────────────────────────────────────────────────────

BASE = "base profile"
EXTENDED = "extended file"
LOGIN = "login file"
VARIANT = "host variant"

#: The file the base profile extends.
EXTENDED_FILE = "shared/parent.yml"

#: The login file that states keys; a second login states none.
LOGIN_NAME = "readwrite"
QUIET_LOGIN = "readonly"

#: The host variant the deployment selects.
VARIANT_NAME = "chainhost"

#: The environment variable the two placeholder-stated values name. It is set
#: to the directory outside the deployment, at build and at run time.
PLACEHOLDER_VAR = "CHAIN_OUTSIDE"

#: The interpreters ``environment.python`` and ``python_env`` name: files
#: outside the deployment, executable, never run by a build without dependencies.
INTERPRETER = "interp/bin/python3"
RUNTIME_INTERPRETER = "interp/bin/python3-runtime"

#: The application the deployment installs.
APP_NAME = "knob"
APP: dict[str, Any] = {
    "name": APP_NAME,
    "containers": {
        APP_NAME: {
            "runtime": "osprey",
            "command": ["python", "-m", "knob_app.server"],
            "health": "/health",
        },
    },
}

#: The watchdog the deployment enables, from those the exemplar states.
WATCHDOG = "guard-circumvention"

#: The local service whose template the profile names.
SERVICE_NAME = "chainsvc"
SERVICE_TEMPLATE = "templates/chainsvc"


@dataclass(frozen=True)
class Ctx:
    """The deployment's root and the directory outside it, for one build."""

    repo: Path
    outside: Path

    @property
    def render(self) -> Path:
        return self.repo / "build"


@dataclass(frozen=True)
class Stated:
    """One key stated by the deployment.

    Attributes:
        key: The key as the emitter's feed spells it.
        spelling: The key as the profile spells it.
        where: Which file states it.
        value: What it states; ``{outside}`` stands for the outside directory.
        kind: ``file`` or ``dir``: what the reader opens.
        reader: The resolver the location comes from, or the reader cited.
        locate: The location the reader opens, from the context.
    """

    key: str
    spelling: str
    where: str
    value: Any
    kind: str
    reader: str
    locate: Callable[[Ctx], Path]

    def stated(self, ctx: Ctx) -> Any:
        return _fill(self.value, ctx)


def _fill(value: Any, ctx: Ctx) -> Any:
    if isinstance(value, str):
        return value.replace("{outside}", str(ctx.outside))
    if isinstance(value, list):
        return [_fill(item, ctx) for item in value]
    if isinstance(value, dict):
        return {key: _fill(item, ctx) for key, item in value.items()}
    return value


def _rooted(relative: str) -> Callable[[Ctx], Path]:
    """A location the reader anchors at the project root (``anchored_path``)."""
    return lambda ctx: anchored_path(_fill(relative, ctx), ctx.repo).resolve()


def _config_relative(relative: str) -> Callable[[Ctx], Path]:
    """A location as ``resolve_config_relative_path`` reads it from the root render."""
    return lambda ctx: resolve_config_relative_path(
        os.path.expandvars(_fill(relative, ctx)), ctx.render
    ).resolve()


def _trigger_file(ctx: Ctx) -> Path:
    """The trigger file as the build's material resolver finds it."""
    found = profile_triggers_material({"dispatch": {"triggers": "triggers.yml"}}, ctx.repo)
    assert found is not None
    return found[1].resolve()


def _outside(relative: str) -> Callable[[Ctx], Path]:
    return lambda ctx: (ctx.outside / relative).resolve()


# ── the table of what is stated ───────────────────────────────────────────────

STATED: tuple[Stated, ...] = (
    # Feed 1: the build's own input enumeration.
    Stated(
        "extends",
        "extends",
        BASE,
        EXTENDED_FILE,
        "file",
        "ResolvedProfileDocument.extends_files",
        _rooted(EXTENDED_FILE),
    ),
    Stated("data", "data", BASE, "data", "dir", "BuildProfile.resolved_data_root", _rooted("data")),
    Stated(
        "machine",
        "machine",
        BASE,
        "machine",
        "dir",
        "profile_inputs (machine root)",
        _rooted("machine"),
    ),
    Stated(
        "services.<name>.template",
        f"services.{SERVICE_NAME}.template",
        BASE,
        SERVICE_TEMPLATE,
        "dir",
        "resolve_service_template",
        lambda ctx: resolve_service_template(ctx.repo, SERVICE_TEMPLATE).path.resolve(),
    ),
    Stated(
        "dispatch.triggers",
        "dispatch.triggers",
        BASE,
        "triggers.yml",
        "file",
        "profile_triggers_material",
        lambda ctx: _trigger_file(ctx),
    ),
    Stated(
        "dependencies",
        "dependencies",
        BASE,
        ["./packages/chainpkg"],
        "dir",
        "dependency_package_dirs",
        lambda ctx: dependency_package_dirs(["./packages/chainpkg"], ctx.repo)[0].resolve(),
    ),
    Stated("hooks", "hooks/", BASE, "hooks", "dir", "CONVENTION_SOURCES", _rooted("hooks")),
    Stated("rules", "rules/", BASE, "rules", "dir", "CONVENTION_SOURCES", _rooted("rules")),
    Stated("skills", "skills/", BASE, "skills", "dir", "CONVENTION_SOURCES", _rooted("skills")),
    Stated("agents", "agents/", BASE, "agents", "dir", "CONVENTION_SOURCES", _rooted("agents")),
    Stated(
        "output_styles",
        "output-styles/",
        BASE,
        "output-styles",
        "dir",
        "CONVENTION_SOURCES",
        _rooted("output-styles"),
    ),
    Stated(
        "watchdogs.<name>",
        f"watchdogs.{WATCHDOG}",
        BASE,
        {"enabled": True},
        "dir",
        "enabled_watchdogs",
        _rooted(f"watchdogs/{WATCHDOG}"),
    ),
    Stated(
        "apps.<name>",
        f"apps.{APP_NAME}",
        BASE,
        {},
        "dir",
        "application_dirs",
        _rooted(f"apps/{APP_NAME}"),
    ),
    # Feed 2: profile fields the enumeration does not list.
    Stated(
        "environment.python",
        "environment.python",
        BASE,
        f"{{outside}}/{INTERPRETER}",
        "file",
        "EnvironmentConfig.resolved_python",
        _outside(INTERPRETER),
    ),
    Stated(
        "python_env",
        "python_env",
        VARIANT,
        f"{{outside}}/{RUNTIME_INTERPRETER}",
        "file",
        "_render_context current_python_env",
        _outside(RUNTIME_INTERPRETER),
    ),
    Stated(
        "lifecycle.pre_build[].cwd",
        "lifecycle.pre_build[].cwd",
        BASE,
        "{project_root}/tools",
        "dir",
        "lifecycle step cwd ({project_root} substituted)",
        _rooted("tools"),
    ),
    Stated(
        "bluesky.plan_dir",
        "bluesky.plan_dir",
        BASE,
        "plans/",
        "dir",
        "_inject_bluesky plan mount",
        _rooted("plans"),
    ),
    Stated(
        "bluesky.external.parameter_schemas",
        "bluesky.external.parameter_schemas",
        BASE,
        {"scan.motor": "schemas/scan.json"},
        "file",
        "BuildProfile validation of parameter_schemas",
        _rooted("schemas/scan.json"),
    ),
    Stated(
        "optimizer.environments_dir",
        "optimizer.environments_dir",
        BASE,
        f"${{{PLACEHOLDER_VAR}}}/optimizer-envs",
        "dir",
        "_inject_optimizer environments mount",
        _outside("optimizer-envs"),
    ),
    Stated(
        "deploy.host.project_path",
        "deploy.host.project_path",
        BASE,
        "{outside}/deployed",
        "dir",
        "deploy host project path",
        _outside("deployed"),
    ),
    # Feed 3: configuration keys.
    Stated(
        "services.<name>.path",
        "services.archive.path",
        BASE,
        "{outside}/compose/archive",
        "dir",
        "resolve_render_relative_path",
        lambda ctx: resolve_render_relative_path(
            str(ctx.outside / "compose" / "archive"), ctx.render
        ).resolve(),
    ),
    Stated(
        "services.qmd.models_dir",
        "services.qmd.models_dir",
        BASE,
        "{outside}/qmd-models",
        "dir",
        "resolve_config_relative_path",
        _config_relative("{outside}/qmd-models"),
    ),
    Stated(
        "services.qmd.corpora",
        "services.qmd.corpora[].source",
        BASE,
        [{"name": "docs", "source": "corpus/docs"}],
        "dir",
        "resolve_config_relative_path",
        _config_relative("corpus/docs"),
    ),
    Stated(
        "services.graphdb.ttl_path",
        "services.graphdb.ttl_path",
        BASE,
        "../seeds/graph.ttl",
        "file",
        "resolve_render_relative_path",
        lambda ctx: resolve_render_relative_path("../seeds/graph.ttl", ctx.render).resolve(),
    ),
    Stated(
        "services.bluesky.plan_dir",
        "services.bluesky.plan_dir",
        BASE,
        "svc-plans",
        "dir",
        "resolve_config_relative_path",
        _config_relative("svc-plans"),
    ),
    Stated(
        "services.bluesky.external.parameter_schema_mounts",
        "services.bluesky.external.parameter_schema_mounts[].source",
        BASE,
        [{"source": "schemas/mount.json", "target": "/app/project/plan_schemas/mount.json"}],
        "file",
        "resolve_config_relative_path",
        _config_relative("schemas/mount.json"),
    ),
    Stated(
        "services.optimizer.environments_dir",
        "services.optimizer.environments_dir",
        BASE,
        "svc-envs",
        "dir",
        "resolve_config_relative_path",
        _config_relative("svc-envs"),
    ),
    Stated(
        "services.event_dispatcher.additional_dirs",
        "services.event_dispatcher.additional_dirs",
        BASE,
        ["dispatch-extra"],
        "dir",
        "resolve_config_relative_path",
        _config_relative("dispatch-extra"),
    ),
    Stated(
        "registry_path",
        "registry_path",
        EXTENDED,
        "src/facility/registry.py",
        "file",
        "resolve_registry_path",
        lambda ctx: Path(
            resolve_registry_path({"registry_path": "src/facility/registry.py"}, base_path=ctx.repo)
            or ""
        ).resolve(),
    ),
    Stated(
        "application.registry_path",
        "application.registry_path",
        EXTENDED,
        "src/app/registry.py",
        "file",
        "resolve_registry_path",
        lambda ctx: Path(
            resolve_registry_path(
                {"application": {"registry_path": "src/app/registry.py"}}, base_path=ctx.repo
            )
            or ""
        ).resolve(),
    ),
    Stated(
        "health.plugins",
        "health.plugins",
        LOGIN,
        ["facility.health.module", "plugins/check.py"],
        "file",
        "resolve_config_relative_path",
        _config_relative("plugins/check.py"),
    ),
    Stated(
        "facility_knowledge.bundle_path",
        "facility_knowledge.bundle_path",
        BASE,
        f"${{{PLACEHOLDER_VAR}}}/knowledge",
        "dir",
        "resolve_config_relative_path",
        _config_relative(f"${{{PLACEHOLDER_VAR}}}/knowledge"),
    ),
    Stated(
        "ariel.vocabulary.path",
        "ariel.vocabulary.path",
        LOGIN,
        "vocab/terms.yaml",
        "file",
        "resolve_config_relative_path",
        _config_relative("vocab/terms.yaml"),
    ),
    Stated(
        "channel_directory.alarm_state.replay_file",
        "channel_directory.alarm_state.replay_file",
        VARIANT,
        "replays/alarms.jsonl",
        "file",
        "resolve_replay_file",
        lambda ctx: resolve_replay_file("replays/alarms.jsonl", ctx.repo).resolve(),
    ),
    Stated(
        "ariel.ingestion.source_url",
        "ariel.ingestion.source_url",
        LOGIN,
        "logbook/export.json",
        "file",
        "resolve_config_relative_path",
        _config_relative("logbook/export.json"),
    ),
    Stated(
        "phoebus.panels",
        "phoebus.panels",
        BASE,
        {"main": {"file": "displays/main.bob"}},
        "file",
        "resolve_config_relative_path",
        _config_relative("displays/main.bob"),
    ),
    Stated(
        "web.panels.canopy.theme_file",
        "web.panels.canopy.theme_file",
        BASE,
        "theme/site.json",
        "file",
        "resolve_config_relative_path",
        _config_relative("theme/site.json"),
    ),
    Stated(
        "images.site_ca",
        "images.site_ca",
        EXTENDED,
        "certs/site-ca.pem",
        "file",
        "resolve_config_relative_path",
        _config_relative("certs/site-ca.pem"),
    ),
    Stated(
        "archiver.epics_archiver.tls.ca_bundle",
        "archiver.epics_archiver.tls.ca_bundle",
        VARIANT,
        "certs/epics-ca.pem",
        "file",
        "resolve_config_relative_path",
        _config_relative("certs/epics-ca.pem"),
    ),
    Stated(
        "archiver.mongodb_archiver.tls.ca_bundle",
        "archiver.mongodb_archiver.tls.ca_bundle",
        VARIANT,
        "certs/mongo-ca.pem",
        "file",
        "resolve_config_relative_path",
        _config_relative("certs/mongo-ca.pem"),
    ),
    Stated(
        "channel_directory.settings.tls.ca_bundle",
        "channel_directory.settings.tls.ca_bundle",
        BASE,
        "certs/cd-ca.pem",
        "file",
        "resolve_config_relative_path",
        _config_relative("certs/cd-ca.pem"),
    ),
    Stated(
        "channel_directory.alarm_state.settings.tls.ca_bundle",
        "channel_directory.alarm_state.settings.tls.ca_bundle",
        BASE,
        "certs/alarm-ca.pem",
        "file",
        "resolve_config_relative_path",
        _config_relative("certs/alarm-ca.pem"),
    ),
    Stated(
        "ariel.ingestion.ca_bundle",
        "ariel.ingestion.ca_bundle",
        LOGIN,
        "certs/ariel-ca.pem",
        "file",
        "resolve_config_relative_path",
        _config_relative("certs/ariel-ca.pem"),
    ),
    Stated(
        "modules.web_terminals.tls.host_cert_dir",
        "modules.web_terminals.tls.host_cert_dir",
        BASE,
        "{outside}/host-certs",
        "dir",
        "resolve_config_relative_path",
        _config_relative("{outside}/host-certs"),
    ),
    Stated(
        "bluesky.plan_dirs",
        "bluesky.plan_dirs",
        LOGIN,
        ["login-plans"],
        "dir",
        "resolve_config_relative_path",
        _config_relative("login-plans"),
    ),
    Stated(
        "bluesky.plan_module",
        "bluesky.plan_module",
        BASE,
        "plan-module/legacy.py",
        "file",
        "resolve_config_relative_path",
        _config_relative("plan-module/legacy.py"),
    ),
    Stated(
        "ariel.enhancement_modules.qmd_export.settings.mirror_path",
        "ariel.enhancement_modules.qmd_export.settings.mirror_path",
        BASE,
        "mirror/nested",
        "dir",
        "resolve_config_relative_path",
        _config_relative("mirror/nested"),
    ),
    Stated(
        "ariel.enhancement_modules.qmd_export.mirror_path",
        "ariel.enhancement_modules.qmd_export.mirror_path",
        BASE,
        "mirror/flat",
        "dir",
        "resolve_config_relative_path",
        _config_relative("mirror/flat"),
    ),
    Stated(
        "modules.web_terminals.landing.notices",
        "modules.web_terminals.landing.notices",
        LOGIN,
        ["notices/one.md"],
        "file",
        "resolve_config_relative_path",
        _config_relative("notices/one.md"),
    ),
)

#: The one profile field and the one configuration key stated through a placeholder.
PLACEHOLDER_KEYS = ("optimizer.environments_dir", "facility_knowledge.bundle_path")

#: Profile fields with a top-level block of their own; everything else of feed 3
#: is stated under ``config:``.
_PROFILE_FIELD_KEYS = INPUT_ENUMERATION_KEYS | {row.key for row in PROFILE_PATH_FIELDS}


def _by_key(key: str) -> Stated:
    (found,) = [stated for stated in STATED if stated.key == key]
    return found


# ── the deployment ────────────────────────────────────────────────────────────


def _load(path: Path) -> dict[str, Any]:
    document: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return document


def _dump(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def _config_key(stated: Stated) -> str:
    """The dotted key a ``config:`` block states *stated* under."""
    return stated.spelling.split("[")[0]


def _config_block(ctx: Ctx, where: str) -> dict[str, Any]:
    return {
        _config_key(stated): stated.stated(ctx)
        for stated in STATED
        if stated.where == where and stated.key not in _PROFILE_FIELD_KEYS
    }


#: Content for a stated file, by suffix; every reader that parses one gets a valid one.
_FILE_CONTENT = {
    ".py": "VALUE = 1\n",
    ".json": "{}\n",
    ".yml": "{}\n",
    ".yaml": "terms: []\n",
    ".jsonl": "",
    ".md": "# Notice\n",
    ".pem": "-----BEGIN CERTIFICATE-----\nAAAA\n-----END CERTIFICATE-----\n",
    ".bob": "<display/>\n",
    ".ttl": "",
}


def _materialise(ctx: Ctx, stated: Stated) -> None:
    location = stated.locate(ctx)
    if stated.kind == "dir":
        location.mkdir(parents=True, exist_ok=True)
        return
    if location.exists():
        return
    location.parent.mkdir(parents=True, exist_ok=True)
    location.write_text(_FILE_CONTENT.get(location.suffix, "stated\n"), encoding="utf-8")


def _mount_machine(repo: Path) -> None:
    """Mount a machine root: the bundled example facility's machine facts.

    The deployment's ``data/facility`` keeps only its content directories, as
    a deployment with a machine root does.
    """
    machine = repo / _by_key("machine").value
    shutil.copytree(
        EXAMPLE_FACILITY,
        machine,
        ignore=lambda _dir, names: [n for n in names if n in DEPLOYMENT_CONTENT_DIRS],
        dirs_exist_ok=True,
    )
    facility = repo / "data" / "facility"
    for entry in sorted(facility.iterdir()):
        if entry.name not in DEPLOYMENT_CONTENT_DIRS:
            shutil.rmtree(entry) if entry.is_dir() else entry.unlink()


def make_deployment(ctx: Ctx) -> None:
    """The exemplar, stating every key of :data:`STATED` where the table says."""
    repo = build_exemplar_repo(ctx.repo)
    ctx.outside.mkdir(parents=True, exist_ok=True)
    for name in (INTERPRETER, RUNTIME_INTERPRETER):
        interpreter = ctx.outside / name
        interpreter.parent.mkdir(parents=True, exist_ok=True)
        interpreter.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        interpreter.chmod(interpreter.stat().st_mode | stat.S_IXUSR)

    profile = _load(repo / "profile.yml")
    profile["extends"] = EXTENDED_FILE
    profile["machine"] = _by_key("machine").value
    profile.setdefault("services", {})[SERVICE_NAME] = {"template": SERVICE_TEMPLATE}
    profile["dependencies"] = _by_key("dependencies").value
    profile["watchdogs"][WATCHDOG].update(_by_key("watchdogs.<name>").value)
    profile["apps"] = {APP_NAME: {}}
    # The archiver block is the home of the archiver's TLS keys, so the
    # configuration key is stated with the block absent.
    profile.pop("va_archiver", None)
    profile["environment"] = {"python": _by_key("environment.python").stated(ctx)}
    profile["lifecycle"]["pre_build"].append(
        {"name": "chain-tools", "run": "true", "timeout": 30, "cwd": "{project_root}/tools"}
    )
    profile["bluesky"] = {
        "plan_dir": _by_key("bluesky.plan_dir").value,
        "external": {
            "zmq_control_addr": "tcp://qserver:60615",
            "insecure_plaintext": True,
            "parameter_schemas": _by_key("bluesky.external.parameter_schemas").value,
        },
    }
    profile["optimizer"] = {"environments_dir": _by_key("optimizer.environments_dir").value}
    # The deploy block is the one home of the image source.
    profile["config"]["modules.web_terminals"].pop("image_source", None)
    profile["deploy"] = {
        "ci": "gitlab",
        "image_source": "local",
        "host": {
            "name": socket.gethostname(),
            "user": "ops",
            "project_path": _by_key("deploy.host.project_path").stated(ctx),
        },
    }
    profile["config"].update(_config_block(ctx, BASE))
    # The machine root marks its own start set; the bundle's is cancelled.
    profile["config"]["simulation.default_scenarios"] = None
    _dump(repo / "profile.yml", profile)

    _dump(repo / EXTENDED_FILE, {"config": _config_block(ctx, EXTENDED)})

    login_path = repo / "tiers" / f"{LOGIN_NAME}.yml"
    login = _load(login_path)
    login.setdefault("config", {}).update(_config_block(ctx, LOGIN))
    _dump(login_path, login)

    _dump(
        repo / "profiles" / f"{VARIANT_NAME}.yml",
        {
            "python_env": _by_key("python_env").stated(ctx),
            "config": _config_block(ctx, VARIANT),
        },
    )
    (repo / ".env.variant").write_text(f"OSPREY_PROFILE_VARIANT={VARIANT_NAME}\n", encoding="utf-8")

    _dump(repo / "apps" / APP_NAME / "app.yml", APP)
    template = repo / SERVICE_TEMPLATE
    template.mkdir(parents=True, exist_ok=True)
    (template / "docker-compose.yml.j2").write_text("services: {}\n", encoding="utf-8")
    package = repo / "packages" / "chainpkg"
    package.mkdir(parents=True, exist_ok=True)
    (package / "pyproject.toml").write_text(
        '[project]\nname = "chainpkg"\nversion = "0"\n', encoding="utf-8"
    )
    _mount_machine(repo)
    for stated in STATED:
        _materialise(ctx, stated)
    (repo / "watchdogs" / WATCHDOG / "instructions.md").write_text(
        "Flag attempts to work around the guard.\n", encoding="utf-8"
    )


def build(repo: Path, outside: Path):
    """A real ``osprey build`` of *repo*, with the placeholder variable set."""
    with preserved_environ():
        os.environ[PLACEHOLDER_VAR] = str(outside)
        return invoke_build_in(repo)


# ── the one real build ────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Ctx:
    """The deployment of :data:`STATED`, built once for the module."""
    base = tmp_path_factory.mktemp("protected-chain").resolve()
    ctx = Ctx(base / EXEMPLAR_DIRNAME, base / "outside")
    make_deployment(ctx)
    result = build(ctx.repo, ctx.outside)
    assert result.exit_code == 0, result.output
    assert "OSPREY does not know as a path" not in result.output, result.output
    return ctx


@pytest.fixture
def placeholder(built: Ctx, monkeypatch: pytest.MonkeyPatch) -> Ctx:
    """The run-time environment: the placeholder variable set, as at build."""
    monkeypatch.setenv(PLACEHOLDER_VAR, str(built.outside))
    return built


def _agent_data(ctx: Ctx) -> Path:
    return ctx.repo / "var" / "agent_data"


#: The renders whose sets are read: the root render, the login that states
#: keys and a login that states none.
RENDERS = (
    "config.yml",
    f"{EXEMPLAR_DIRNAME}-{LOGIN_NAME}/config.yml",
    f"{EXEMPLAR_DIRNAME}-{QUIET_LOGIN}/config.yml",
)


def _written_set(ctx: Ctx, render: str) -> tuple[str, ...]:
    reading = read_protected_paths(_load(ctx.render / render))
    assert reading.state is ProtectedPathsState.PRESENT, f"{render}: {reading.state}"
    return reading.entries


_KEYS = pytest.mark.parametrize("stated", STATED, ids=[s.spelling for s in STATED])


# ── the table ─────────────────────────────────────────────────────────────────


def test_the_stated_keys_are_exactly_the_emitters_three_feeds() -> None:
    feeds = (
        INPUT_ENUMERATION_KEYS
        | {row.key for row in PROFILE_PATH_FIELDS}
        | {row.key for row in CONFIG_PATH_KEYS}
    )
    keys = [stated.key for stated in STATED]

    assert len(keys) == len(set(keys)), sorted(k for k in keys if keys.count(k) > 1)
    assert set(keys) == feeds


def test_keys_are_stated_in_every_kind_of_profile_file() -> None:
    assert {stated.where for stated in STATED} == {BASE, EXTENDED, LOGIN, VARIANT}


def test_one_profile_field_and_one_configuration_key_are_stated_through_a_placeholder() -> None:
    field, config_key = (_by_key(key) for key in PLACEHOLDER_KEYS)

    assert field.key in {row.key for row in PROFILE_PATH_FIELDS}
    assert config_key.key in {row.key for row in CONFIG_PATH_KEYS}
    for stated in (field, config_key):
        assert f"${{{PLACEHOLDER_VAR}}}" in stated.value


def test_the_placeholders_are_written_unexpanded(built: Ctx) -> None:
    entries = _written_set(built, "config.yml")

    for key in PLACEHOLDER_KEYS:
        assert _by_key(key).value in entries


# ── every stated location ─────────────────────────────────────────────────────


def _covers(ctx: Ctx, entries: tuple[str, ...], location: Path) -> list[str]:
    """The entries whose run-time location is *location* or one of its parents."""
    covering: list[str] = []
    for entry in entries:
        anchored = anchor_protected_entry(
            entry, ctx.repo, (_agent_data(ctx),), environ=dict(os.environ)
        )
        if anchored.path is None:
            continue
        protected = Path(anchored.path).resolve()
        if protected == location or protected in location.parents:
            covering.append(entry)
    return covering


@_KEYS
@pytest.mark.parametrize("render", RENDERS)
def test_the_location_is_covered_by_the_written_set(
    placeholder: Ctx, stated: Stated, render: str
) -> None:
    location = stated.locate(placeholder)
    assert location.exists(), location

    assert _covers(placeholder, _written_set(placeholder, render), location), (
        f"{stated.spelling} ({stated.where}) opens {location} via {stated.reader}; "
        f"{render} does not cover it"
    )


def _write_target(ctx: Ctx, stated: Stated) -> Path:
    """What a literal write aims at: the file, or a new file inside the directory."""
    location = stated.locate(ctx)
    return location if stated.kind == "file" else location / "chain-probe.txt"


def _literal(ctx: Ctx, target: Path) -> str:
    """The path as agent code would spell it: relative inside the root, else absolute."""
    try:
        return target.relative_to(ctx.repo.resolve()).as_posix()
    except ValueError:
        return str(target)


def _run_roots(ctx: Ctx) -> tuple[tuple[str, ...], tuple[Path, ...]]:
    """The executor's own read of the set, and the roots it resolves to."""
    entries = executor.read_run_protected_entries(ctx.repo)
    roots = resolve_protected_roots(ctx.repo, entries=entries, permitted_roots=(_agent_data(ctx),))
    return entries, roots


@pytest.fixture
def executor_on(placeholder: Ctx, monkeypatch: pytest.MonkeyPatch) -> Iterator[Ctx]:
    """The executor's view of the built deployment: its root and its render."""
    monkeypatch.setenv("OSPREY_CONFIG", str(placeholder.render / "config.yml"))
    monkeypatch.setattr(executor, "_resolve_project_root", lambda: placeholder.repo)
    monkeypatch.chdir(placeholder.repo)
    reset_config_cache()
    yield placeholder
    reset_config_cache()


@_KEYS
def test_the_pre_run_check_refuses_a_literal_write(executor_on: Ctx, stated: Stated) -> None:
    _entries, roots = _run_roots(executor_on)
    literal = _literal(executor_on, _write_target(executor_on, stated))

    issues = path_policy_issues(
        f"open({literal!r}, 'w').write('x')\n",
        protected_roots=roots,
        working_directory=executor_on.repo,
        permitted_roots=(_agent_data(executor_on),),
    )

    assert issues, f"{stated.spelling}: a write to {literal!r} passed the pre-run check"


#: The child: one write, one read and, for a file key, one neighbour write per
#: stated location, each reported as ``REFUSED: <error>`` or ``ALLOWED``.
_CHILD = textwrap.dedent(
    """
    import json
    import os

    def attempt(action):
        try:
            action()
        except PermissionError as exc:
            return "REFUSED: " + str(exc)
        return "ALLOWED"

    def write(path):
        def action():
            with open(path, "w") as handle:
                handle.write("written by executed code")
        return action

    def read(path, kind):
        def action():
            if kind == "dir":
                os.listdir(path)
            else:
                with open(path, "rb") as handle:
                    handle.read()
        return action

    results = {}
    for spelling, kind, target, location, neighbour in json.loads(PROBES):
        results[spelling] = {
            "write": attempt(write(target)),
            "read": attempt(read(location, kind)),
            "neighbour": attempt(write(neighbour)) if neighbour else None,
        }
    print("RESULTS " + json.dumps(results))
    """
)


def _neighbour(location: Path) -> Path:
    return location.with_name(f"{location.name}.neighbour")


def _child_results(ctx: Ctx, mode: str) -> dict[str, dict[str, str | None]]:
    """Run the executor's guard and :data:`_CHILD` in a real child process."""
    _entries, roots = _run_roots(ctx)
    probes = [
        [
            stated.spelling,
            stated.kind,
            _literal(ctx, _write_target(ctx, stated)),
            _literal(ctx, stated.locate(ctx)),
            _literal(ctx, _neighbour(stated.locate(ctx))) if stated.kind == "file" else None,
        ]
        for stated in STATED
    ]
    execution_folder = _agent_data(ctx) / "data" / "python_executions" / f"chain_{mode}"
    execution_folder.mkdir(parents=True, exist_ok=True)
    wrapper = ExecutionWrapper(
        execution_mode=mode, protected_roots=roots, permitted_roots=(_agent_data(ctx),)
    )
    script = execution_folder / "probe.py"
    script.write_text(
        wrapper._get_filesystem_guard(execution_folder)
        + f"\nPROBES = {json.dumps(probes)!r}\n"
        + _CHILD,
        encoding="utf-8",
    )
    proc = subprocess.run(  # fixed argv, test-authored script
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        cwd=str(ctx.repo),
        timeout=120,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    (line,) = [line for line in proc.stdout.splitlines() if line.startswith("RESULTS ")]
    results: dict[str, dict[str, str | None]] = json.loads(line.removeprefix("RESULTS "))
    return results


def _refused(outcome: str | None) -> bool:
    return outcome is not None and outcome.startswith("REFUSED:") and "write denied for" in outcome


@pytest.mark.parametrize("mode", ["readonly", "readwrite"])
def test_a_real_child_is_refused_on_write_and_allowed_on_read(executor_on: Ctx, mode: str) -> None:
    contents = {
        stated.spelling: stated.locate(executor_on).read_bytes()
        for stated in STATED
        if stated.kind == "file"
    }

    results = _child_results(executor_on, mode)

    assert set(results) == {stated.spelling for stated in STATED}
    not_refused = {k: r["write"] for k, r in results.items() if not _refused(r["write"])}
    not_read = {k: r["read"] for k, r in results.items() if r["read"] != "ALLOWED"}
    assert not not_refused, f"{mode}: writes not refused: {not_refused}"
    assert not not_read, f"{mode}: reads refused: {not_read}"
    for stated in STATED:
        location = stated.locate(executor_on)
        if stated.kind == "file":
            assert location.read_bytes() == contents[stated.spelling], stated.spelling
        else:
            assert not (location / "chain-probe.txt").exists(), stated.spelling


@pytest.mark.parametrize("mode", ["readonly", "readwrite"])
def test_a_file_keys_neighbour_stays_writable(executor_on: Ctx, mode: str) -> None:
    results = _child_results(executor_on, mode)

    files = [stated for stated in STATED if stated.kind == "file"]
    assert files
    blocked = {
        stated.spelling: results[stated.spelling]["neighbour"]
        for stated in files
        if results[stated.spelling]["neighbour"] != "ALLOWED"
    }
    assert not blocked, f"{mode}: neighbours refused: {blocked}"
    for stated in files:
        neighbour = _neighbour(stated.locate(executor_on))
        assert neighbour.exists(), neighbour
        neighbour.unlink()


# ── a protected location is not a source file ────────────────────────────────


def test_a_review_prompt_in_the_plan_directory_is_refused_at_build(tmp_path: Path) -> None:
    repo = build_exemplar_repo(tmp_path / EXEMPLAR_DIRNAME)
    plan_dir = _by_key("bluesky.plan_dir").value
    prompt = f"{plan_dir.rstrip('/')}/review.md"
    (repo / prompt).parent.mkdir(parents=True, exist_ok=True)
    (repo / prompt).write_text("Review every change.\n", encoding="utf-8")
    profile = _load(repo / "profile.yml")
    profile["bluesky"]["plan_dir"] = plan_dir
    profile["hooks"] += [hook for hook in CODE_REVIEW_HOOKS if hook not in profile["hooks"]]
    profile["config"]["code_review.enabled"] = True
    profile["config"]["code_review.prompt"] = prompt
    _dump(repo / "profile.yml", profile)

    with preserved_environ():
        result = invoke_build_in(repo)

    assert result.exit_code != 0
    assert "code_review.prompt" in result.output, result.output
    assert prompt in result.output, result.output
    assert not (repo / "build" / "config.yml").exists()


# ── a build without the set ──────────────────────────────────────────────────


def _copy_of_the_build(ctx: Ctx, dest: Path, *, remove_the_set: bool) -> Path:
    """A copy of the real build, optionally with the protected-path key removed."""
    copy = dest / EXEMPLAR_DIRNAME
    shutil.copytree(ctx.repo, copy, symlinks=True)
    (copy / ".env").write_text("ANTHROPIC_API_KEY=x\n", encoding="utf-8")
    if not remove_the_set:
        return copy
    removed = 0
    for path in (copy / "build").rglob("config.yml"):
        if ".venv" in path.parts:
            continue
        document = _load(path)
        if document.pop(PROTECTED_PATHS_CONFIG_KEY, None) is not None:
            removed += 1
            _dump(path, document)
    assert removed
    return copy


@pytest.fixture
def without_the_set(built: Ctx, tmp_path: Path) -> Path:
    """A copy of the real build with the protected-path key removed everywhere."""
    return _copy_of_the_build(built, tmp_path, remove_the_set=True)


@pytest.fixture
def executor_without_the_set(
    without_the_set: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    repo = without_the_set
    monkeypatch.setenv("OSPREY_CONFIG", str(repo / "build" / "config.yml"))
    monkeypatch.setattr(executor, "_resolve_project_root", lambda: repo)
    monkeypatch.chdir(repo)
    launch = AsyncMock()
    monkeypatch.setattr(executor, "execute_code", launch)
    monkeypatch.setattr(gates, "notify_agent_activity_async", AsyncMock())

    async def _spawn(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("a child process was started")

    monkeypatch.setattr(executor.asyncio, "create_subprocess_exec", _spawn)
    reset_config_cache()
    yield repo
    reset_config_cache()
    launch.assert_not_awaited()


async def _execute() -> Any:
    from osprey.mcp_server.python_executor.tools.python_execute import execute

    return await get_tool_fn(execute)(
        code="print(1)\n", description="chain probe", execution_mode="readonly", save_output=False
    )


async def _execute_file(repo: Path) -> Any:
    from osprey.mcp_server.python_executor.tools.python_execute_file import execute_file

    script = repo / "var" / "agent_data" / "chain_probe.py"
    script.write_text("print(1)\n", encoding="utf-8")
    return await get_tool_fn(execute_file)(
        file_path=str(script),
        description="chain probe",
        execution_mode="readonly",
        save_output=False,
    )


class _Stop(Exception):
    """Raised by a stubbed step the refusal must come before."""


async def _model_response_matrix(monkeypatch: pytest.MonkeyPatch) -> Any:
    from osprey.mcp_server.python_executor.tools import model_response_matrix as tool

    def _stop(*_args: Any, **_kwargs: Any) -> None:
        raise _Stop

    monkeypatch.setattr(tool, "resolve_model_request", _stop)
    fn = tool.model_response_matrix
    return await getattr(fn, "fn", fn)(model="line")


async def _pyaml_measure(monkeypatch: pytest.MonkeyPatch) -> Any:
    from osprey.mcp_server.python_executor.tools import pyaml_measure as tool

    def _stop(*_args: Any, **_kwargs: Any) -> None:
        raise _Stop

    monkeypatch.setattr(tool, "resolve_measure_args", _stop)
    # The deployment's posture refuses this tool on its own; the refusal under
    # test is the one about the set, so the clamp is let through.
    monkeypatch.setattr(tool, "enforce_posture_clamp", lambda mode, *, tool: None)
    monkeypatch.setattr(tool, "run_gated_execution", AsyncMock(side_effect=_Stop))
    fn = tool.pyaml_measure
    return await getattr(fn, "fn", fn)(kind="orm")


_TOOL_DRIVERS: dict[str, Callable[[Path, pytest.MonkeyPatch], Any]] = {
    "execute": lambda _repo, _monkeypatch: _execute(),
    "execute_file": lambda repo, _monkeypatch: _execute_file(repo),
    "model_response_matrix": lambda _repo, monkeypatch: _model_response_matrix(monkeypatch),
    "pyaml_measure": lambda _repo, monkeypatch: _pyaml_measure(monkeypatch),
}


@pytest.mark.parametrize("tool", sorted(_TOOL_DRIVERS))
async def test_each_python_executor_tool_refuses_a_build_without_the_set(
    executor_without_the_set: Path, tool: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ToolError) as exc_info:
        await _TOOL_DRIVERS[tool](executor_without_the_set, monkeypatch)

    envelope = json.loads(str(exc_info.value))
    assert envelope["error_type"] == gates.NO_PROTECTED_PATHS_ERROR_TYPE
    assert "`osprey build`" in envelope["error_message"]


def _up(repo: Path) -> Result:
    result = run_up(repo, "-d")
    assert isinstance(result, Result)
    return result


def test_osprey_up_refuses_a_build_without_the_set(
    without_the_set: Path, request: pytest.FixtureRequest
) -> None:
    record: dict = request.getfixturevalue("started")

    result = _up(without_the_set)

    assert result.exit_code == 1, result.output
    assert "without the protected-path list" in result.stderr
    assert "osprey build" in result.stderr
    assert not record, "a start reached the container runtime"


@pytest.mark.usefixtures("started")
def test_osprey_up_does_not_refuse_the_same_build_with_the_set(built: Ctx, tmp_path: Path) -> None:
    """The control: the copy with the set intact is not refused for it."""
    result = _up(_copy_of_the_build(built, tmp_path, remove_the_set=False))

    assert "without the protected-path list" not in result.output
