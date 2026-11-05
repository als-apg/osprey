"""The binds a program's write authorization needs, and the directories behind them.

A program (the optimizer) writes its requests and ledgers into the
deployment's write ledger, ``var/write_ledger/``; every web terminal reads the
requests through one read-only bind of ``var/write_ledger/requests``, and the
holders' self-issued grants and ledgers through one read-only bind of the
whole ``var/write_ledger``. No terminal sees another login's grants. Both
sources are provisioned before compose runs, because a bind whose source the
runtime creates is root-owned.
"""

from __future__ import annotations

import copy
import stat
from pathlib import Path
from typing import Any

import yaml

from osprey.deployment.compose_generator import (
    WRITE_LEDGER_VIEW_CONTAINER_DIR,
    WRITE_REQUESTS_CONTAINER_DIR,
    _ensure_agent_data_structure,
    ensure_write_ledger_dirs,
    write_ledger_relpath,
    write_requests_relpath,
)
from osprey.deployment.web_terminals.render import render_web_terminals
from osprey_connectors.write_authorization import (
    AUTHORIZATIONS_DIR_NAME,
    LEDGER_DIR_ENV_VAR,
    LEDGER_VIEW_DIR_ENV_VAR,
    REQUESTS_DIR_ENV_VAR,
)

from .web_terminals.test_golden_render import EXAMPLE_CONFIG


def _environment(service: dict[str, Any]) -> dict[str, str]:
    environment = service.get("environment") or {}
    if isinstance(environment, dict):
        return {str(key): str(value) for key, value in environment.items()}
    return dict(str(entry).split("=", 1) for entry in environment)


def _terminals() -> dict[str, dict[str, Any]]:
    rendered = render_web_terminals(copy.deepcopy(EXAMPLE_CONFIG))["docker-compose.web.yml"]
    services = yaml.safe_load(rendered)["services"]
    return {
        name: service
        for name, service in services.items()
        if "OSPREY_CONTROL_CONTEXT_DIR" in _environment(service)
    }


def test_every_terminal_reads_the_requests_read_only():
    terminals = _terminals()
    assert len(terminals) >= 2
    for name, service in terminals.items():
        assert _environment(service)[REQUESTS_DIR_ENV_VAR] == WRITE_REQUESTS_CONTAINER_DIR, name
        assert (
            f"./{write_requests_relpath()}:{WRITE_REQUESTS_CONTAINER_DIR}:ro" in service["volumes"]
        ), name


def test_every_terminal_sees_the_holders_self_issued_grants_read_only():
    terminals = _terminals()
    assert len(terminals) >= 2
    for name, service in terminals.items():
        environment = _environment(service)
        assert environment[LEDGER_VIEW_DIR_ENV_VAR] == WRITE_LEDGER_VIEW_CONTAINER_DIR, name
        # A view, never the holder's own ledger: a terminal spends nothing there.
        assert LEDGER_DIR_ENV_VAR not in environment, name
        assert (
            f"./{write_ledger_relpath()}:{WRITE_LEDGER_VIEW_CONTAINER_DIR}:ro" in service["volumes"]
        ), name


def test_no_terminal_binds_another_logins_grants():
    for name, service in _terminals().items():
        assert "OSPREY_WRITE_AUTHORIZATION_PEERS_DIR" not in _environment(service), name
        grant_binds = [
            volume
            for volume in service["volumes"]
            if isinstance(volume, str)
            and volume.split(":")[0].endswith(f"/{AUTHORIZATIONS_DIR_NAME}")
        ]
        assert grant_binds == [], name


def _shared(path: Path) -> None:
    mode = path.stat().st_mode
    assert path.is_dir()
    assert mode & stat.S_ISGID and mode & stat.S_IWGRP


def test_the_write_ledger_and_its_requests_are_provisioned(tmp_path: Path):
    gid = ensure_write_ledger_dirs(tmp_path)
    _shared(tmp_path / "var" / "write_ledger")
    _shared(tmp_path / write_requests_relpath())
    assert gid == (tmp_path / "var" / "write_ledger").stat().st_gid


def test_the_build_path_provisions_the_write_ledger(tmp_path: Path):
    _ensure_agent_data_structure({"project_root": str(tmp_path), "deployed_services": []})
    _shared(tmp_path / write_requests_relpath())
