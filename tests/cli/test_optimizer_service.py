"""The optimizer service's build step and its rendered compose file.

The injector writes ``services.optimizer`` and registers it for deployment; the
compose template turns that block into two containers whose safety properties
live entirely in how they are wired. The job store is reachable from nowhere
but the optimizer, the routes that move hardware are armed by a token with no
default, every write is made under the optimizer's own control identity, and
the limits database comes with the container whenever the deployment's write
tier allows writes. Those are properties of the rendered file, so they are asserted
against it, rendered through the generator's own context injection.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml
from jinja2 import Environment, FileSystemLoader
from ruamel.yaml import YAML

from osprey.cli.build_cmd import _inject_optimizer
from osprey.cli.build_injectors import _compose_bind_source
from osprey.cli.build_profile import OptimizerConfig
from osprey.deployment.compose_generator import (
    LOGIN_TIERS_SERVICES,
    OPTIMIZER_CONTROL_IDENTITY,
    WRITE_LEDGER_CONTAINER_DIR,
    _inject_project_metadata,
    _optimizer_write_posture,
    _stage_login_tiers,
    write_ledger_relpath,
)
from osprey.deployment.errors import DeploymentPreconditionError
from osprey.port_layout import default_port
from osprey_connectors.write_authorization import LEDGER_DIR_ENV_VAR

_TEMPLATES_ROOT = Path(__file__).resolve().parents[2] / "src" / "osprey" / "templates"
_TEMPLATE = "services/optimizer/docker-compose.yml.j2"
_PORT = default_port("optimizer")

#: A deployment whose write tier allows writes. The tier is target-blind: it is
#: a ceiling, and the owner's switch picks the machine.
_CONTROL_SYSTEM: dict[str, Any] = {
    "type": "virtual_accelerator",
    "writes": "allowed",
    "connector": {
        "virtual_accelerator": {},
        "live_standin": {"host": "live-standin", "port": 5070},
    },
}

#: The same deployment with a tier that allows no writes.
_READ_ONLY_CONTROL_SYSTEM: dict[str, Any] = {**_CONTROL_SYSTEM, "writes": "never"}

_LIMITS_MOUNT = {
    "source": "./data/channel_limits.json",
    "target": "/app/project/data/channel_limits.json",
}


# ── injector ────────────────────────────────────────────────────────────────


def _write_config(project: Path) -> None:
    with open(project / "config.yml", "w") as fh:
        YAML().dump({"deployed_services": ["postgresql"], "services": {"postgresql": {}}}, fh)


def _read_config(project: Path) -> dict:
    with open(project / "config.yml") as fh:
        return YAML().load(fh)


def test_the_injector_copies_the_template_and_registers_the_service(tmp_path: Path) -> None:
    _write_config(tmp_path)

    _inject_optimizer(OptimizerConfig(targets=["va", "standin"]), project_path=tmp_path)

    dest = tmp_path / "services" / "optimizer"
    assert (dest / "docker-compose.yml.j2").is_file()
    assert (dest / "Dockerfile").is_file()
    config = _read_config(tmp_path)
    block = config["services"]["optimizer"]
    assert block["path"] == "./services/optimizer"
    assert block["port"] == _PORT
    assert list(block["targets"]) == ["va", "standin"]
    assert "environments_dir" not in block  # omit-when-unset: no mount
    assert "image" not in block
    assert [str(s) for s in config["deployed_services"]] == ["postgresql", "optimizer"]


def test_the_injector_writes_an_environments_dir_only_when_set(tmp_path: Path) -> None:
    _write_config(tmp_path)
    _inject_optimizer(OptimizerConfig(environments_dir="optimizer/envs"), project_path=tmp_path)
    # Stored `./`-prefixed: compose reads a bare relative source as a named volume.
    assert _read_config(tmp_path)["services"]["optimizer"]["environments_dir"] == "./optimizer/envs"


@pytest.mark.parametrize(
    ("authored", "source"),
    [
        # A bare relative path is a compose named volume unless it is written
        # `./`-prefixed, and an undefined named volume fails `osprey up`.
        ("optimizer/envs", "./optimizer/envs"),
        ("./optimizer/envs", "./optimizer/envs"),
        ("../shared/envs", "../shared/envs"),
        ("/srv/optimizer/envs", "/srv/optimizer/envs"),
        ("~/envs", "~/envs"),
    ],
)
def test_a_bind_source_is_spelled_so_compose_reads_a_host_path(authored: str, source: str) -> None:
    assert _compose_bind_source(authored) == source


def test_the_injector_is_idempotent(tmp_path: Path) -> None:
    _write_config(tmp_path)
    _inject_optimizer(OptimizerConfig(), project_path=tmp_path)
    _inject_optimizer(OptimizerConfig(), project_path=tmp_path)
    deployed = [str(s) for s in _read_config(tmp_path)["deployed_services"]]
    assert deployed.count("optimizer") == 1


def test_the_injector_is_a_noop_without_a_config(tmp_path: Path) -> None:
    _inject_optimizer(OptimizerConfig(), project_path=tmp_path)
    assert not (tmp_path / "config.yml").exists()


# ── write posture ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("tier", "armed"),
    [
        pytest.param("allowed", True, id="allowed"),
        pytest.param("asked", True, id="asked"),
        pytest.param("never", False, id="never"),
        pytest.param(None, False, id="no_tier"),
    ],
)
@pytest.mark.parametrize("targets", [["va"], ["standin"], ["standin", "va"]])
def test_the_write_posture_is_the_deployment_tier(
    tier: str | None, armed: bool, targets: list[str]
) -> None:
    section = {key: value for key, value in _CONTROL_SYSTEM.items() if key != "writes"}
    if tier is not None:
        section["writes"] = tier
    assert _optimizer_write_posture({"targets": targets}, section) is armed


# ── rendered compose ────────────────────────────────────────────────────────


def _render(
    *,
    targets: list[str],
    deployed: tuple[str, ...] = (),
    limits: bool = True,
    control_system: dict[str, Any] = _CONTROL_SYSTEM,
    **block: Any,
) -> dict:
    config: dict[str, Any] = {
        "project_name": "demo",
        "project_root": "/tmp/demo",
        "services": {"optimizer": {"port": _PORT, "targets": targets, **block}},
        "deployment": {},
        "system": {"timezone": "UTC"},
        "deployed_services": ["optimizer", *deployed],
        "control_system": control_system,
    }
    context = _inject_project_metadata(config)
    if limits:
        context["limits_mount"] = _LIMITS_MOUNT
    environment = Environment(loader=FileSystemLoader(str(_TEMPLATES_ROOT)), autoescape=False)
    return yaml.safe_load(environment.get_template(_TEMPLATE).render(**context))


def test_the_job_store_is_internal_and_unpublished() -> None:
    doc = _render(targets=["va"])
    redis = doc["services"]["optimizer-redis"]
    assert redis["networks"] == ["optimizer-internal"]
    assert "ports" not in redis
    assert doc["networks"]["optimizer-internal"] == {"internal": True}
    optimizer = doc["services"]["optimizer"]
    assert set(optimizer["networks"]) == {"osprey-network", "optimizer-internal"}
    assert optimizer["environment"]["REDIS_URL"] == "redis://optimizer-redis:6379/0"
    assert optimizer["environment"]["OPTIMIZER_RESULTS_DIR"] == "/app/results"
    assert "optimizer_results:/app/results" in optimizer["volumes"]
    assert optimizer["depends_on"]["optimizer-redis"] == {"condition": "service_healthy"}


def test_the_service_publishes_its_port_on_loopback() -> None:
    optimizer = _render(targets=["va"])["services"]["optimizer"]
    assert optimizer["ports"] == [f"127.0.0.1:{_PORT}:{_PORT}"]
    assert f"--port {_PORT}" in optimizer["command"][-1]


def test_the_launch_token_has_no_default() -> None:
    env = _render(targets=["va"])["services"]["optimizer"]["environment"]
    assert env["OPTIMIZER_LAUNCH_TOKEN"] == "${OPTIMIZER_LAUNCH_TOKEN}"


def test_the_targets_reach_the_service() -> None:
    env = _render(targets=["va", "standin"])["services"]["optimizer"]["environment"]
    assert env["OPTIMIZER_TARGETS"] == "va,standin"


def test_writes_are_made_under_the_optimizer_identity() -> None:
    optimizer = _render(targets=["va"])["services"]["optimizer"]
    command = optimizer["command"][-1]
    assert f"--name {OPTIMIZER_CONTROL_IDENTITY}" in command
    assert "exec uvicorn osprey_optimizer.api.main:app" in command
    assert optimizer["environment"]["OSPREY_AUDIT_IDENTITY"] == "optimizer"


def test_a_writing_tier_brings_the_limits_database_and_the_control_tree() -> None:
    optimizer = _render(targets=["va"])["services"]["optimizer"]
    mounts = optimizer["volumes"]
    assert f"{_LIMITS_MOUNT['source']}:{_LIMITS_MOUNT['target']}:ro" in mounts
    assert "OSPREY_CONTROL_CONTEXT_TREE" in optimizer["environment"]
    tree = optimizer["environment"]["OSPREY_CONTROL_CONTEXT_TREE"]
    assert any(m.endswith(f":{tree}:ro") for m in mounts)


def test_a_read_only_tier_mounts_neither() -> None:
    optimizer = _render(targets=["va"], control_system=_READ_ONLY_CONTROL_SYSTEM)["services"][
        "optimizer"
    ]
    assert not any("channel_limits" in m for m in optimizer["volumes"])
    assert "OSPREY_CONTROL_CONTEXT_TREE" not in optimizer["environment"]
    assert "OSPREY_WRITE_LEDGER_DIR" not in optimizer["environment"]
    assert not any(write_ledger_relpath() in m for m in optimizer["volumes"])


# ── write ledger ────────────────────────────────────────────────────────────


def test_a_writing_tier_binds_the_write_ledger_read_write_outside_var() -> None:
    """Under the tree bind a writer with no ledger directory has every write refused
    ``auth_unreadable``: the ledger comes with the tree, read-write, at a path
    outside the container's ``var/``."""
    optimizer = _render(targets=["va"])["services"]["optimizer"]
    environment = optimizer["environment"]
    assert environment[LEDGER_DIR_ENV_VAR] == WRITE_LEDGER_CONTAINER_DIR
    assert "/var/" not in WRITE_LEDGER_CONTAINER_DIR
    assert f"./{write_ledger_relpath()}:{WRITE_LEDGER_CONTAINER_DIR}" in optimizer["volumes"]
    tree = environment["OSPREY_CONTROL_CONTEXT_TREE"]
    # The tree stays read-only; only the ledger is writable.
    assert f":{tree}:ro" in " ".join(optimizer["volumes"])
    assert not any(
        m.startswith(f"./{write_ledger_relpath()}:") and m.endswith(":ro")
        for m in optimizer["volumes"]
    )


def test_the_optimizer_is_the_holder_its_authorizations_name() -> None:
    environment = _render(targets=["va"])["services"]["optimizer"]["environment"]
    assert environment["OSPREY_AUDIT_IDENTITY"] == "optimizer"


def _probe(command: str, ledger: Path) -> subprocess.CompletedProcess[str]:
    """The rendered command's ledger probe, run by sh as compose would hand it over."""
    start = command.index("probe=")
    end = command.index("exec uvicorn")
    script = command[start:end].replace("$$", "$")
    return subprocess.run(
        ["sh", "-c", script],
        env={"PATH": os.environ.get("PATH", ""), LEDGER_DIR_ENV_VAR: str(ledger)},
        capture_output=True,
        text=True,
        check=True,
    )


def test_the_command_probes_the_ledger_and_warns_when_it_cannot_be_written(
    tmp_path: Path,
) -> None:
    command = _render(targets=["va"])["services"]["optimizer"]["command"][-1]
    assert command.index("probe=") < command.index("exec uvicorn")
    writable = tmp_path / "ledger"
    writable.mkdir()
    assert _probe(command, writable).stderr == ""
    assert list(writable.iterdir()) == []
    if os.geteuid() == 0:
        pytest.skip("root writes a read-only directory; the unwritable case needs a user")
    locked = tmp_path / "locked"
    locked.mkdir(mode=0o500)
    try:
        assert "cannot be written" in _probe(command, locked).stderr
    finally:
        locked.chmod(0o700)


def test_a_read_only_tier_runs_no_probe() -> None:
    command = _render(targets=["va"], control_system=_READ_ONLY_CONTROL_SYSTEM)["services"][
        "optimizer"
    ]["command"][-1]
    assert "probe=" not in command


def test_the_environments_dir_is_mounted_read_only_when_set() -> None:
    # Rendered verbatim: the injector stores it compose-ready.
    mounts = _render(targets=["va"], environments_dir="./optimizer/envs")["services"]["optimizer"][
        "volumes"
    ]
    assert "./optimizer/envs:/app/environments:ro" in mounts


def test_no_environments_dir_means_no_mount() -> None:
    mounts = _render(targets=["va"])["services"]["optimizer"]["volumes"]
    assert not any(m.endswith(":/app/environments:ro") for m in mounts)


@pytest.mark.parametrize(
    ("targets", "deployed", "waits_on"),
    [
        pytest.param(["va"], ("virtual_accelerator",), {"virtual-accelerator"}, id="va"),
        pytest.param(["standin"], ("live_standin",), {"live-standin"}, id="standin"),
        pytest.param(["va"], ("live_standin",), set(), id="standin_deployed_not_driven"),
        pytest.param(["va"], (), set(), id="nothing_co_deployed"),
    ],
)
def test_it_waits_only_on_the_machines_it_drives_and_that_are_deployed(
    targets: list[str], deployed: tuple[str, ...], waits_on: set[str]
) -> None:
    depends = _render(targets=targets, deployed=deployed)["services"]["optimizer"]["depends_on"]
    assert set(depends) - {"optimizer-redis"} == waits_on


@pytest.mark.parametrize("targets", [["live"], ["va", "live"], ["sim"]])
def test_a_target_the_optimizer_may_not_drive_is_refused_at_render(targets: list[str]) -> None:
    """The profile refuses them first; an edited config.yml must not slip past."""
    with pytest.raises(DeploymentPreconditionError, match="services.optimizer.targets"):
        _render(targets=targets)


# ── owner under the tree bind ───────────────────────────────────────────────


def test_the_optimizer_is_a_login_tiers_service() -> None:
    """Its container holds the control-context tree, so a write is judged by its
    owner's rung: with no ``login_tiers`` map every owner is ``owner_unknown``."""
    assert "optimizer" in LOGIN_TIERS_SERVICES


def test_the_staged_optimizer_config_carries_the_login_tiers_map(tmp_path: Path) -> None:
    out = tmp_path / "build" / "services" / "optimizer"
    out.mkdir(parents=True)
    staged = out / "config.yml"
    staged.write_text(yaml.safe_dump({"control_system": dict(_CONTROL_SYSTEM)}))
    tiers = {"alice": "allowed", "bob": "never"}

    _stage_login_tiers({}, str(tmp_path / "services" / "optimizer"), str(out), tiers)

    section = yaml.safe_load(staged.read_text())["control_system"]
    assert section["login_tiers"] == tiers
    assert {k: v for k, v in section.items() if k != "login_tiers"} == _CONTROL_SYSTEM


def test_a_writing_tier_binds_the_tree_the_owner_rung_reads() -> None:
    """The map is read only under a bound tree; the tree is bound whenever the
    tier allows writes, which is when an owner is consulted at all."""
    env = _render(targets=["va"])["services"]["optimizer"]["environment"]
    assert "OSPREY_CONTROL_CONTEXT_TREE" in env
