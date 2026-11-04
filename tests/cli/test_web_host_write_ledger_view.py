"""A host web terminal sees the deployment's write ledger, as a container's binds show it.

A terminal in a container is handed the holders' requests and their ledger
read-only. On the host there are no binds, so ``osprey web`` points the server
at the same directories of the repo, or a run in this deployment's containers
waiting on an approval could be approved only from the CLI.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from osprey.cli.web_cmd import _export_write_ledger_view, _resolve_render
from osprey.deployment.compose_generator import write_ledger_relpath, write_requests_relpath
from osprey_connectors import write_authorization as wa


def _provisioned(repo: Path) -> Path:
    (repo / write_requests_relpath()).mkdir(parents=True)
    return repo


def test_a_provisioned_ledger_is_exported_to_the_host_terminal(tmp_path: Path) -> None:
    repo = _provisioned(tmp_path)
    environ: dict[str, str] = {}
    assert _export_write_ledger_view(environ, repo) == [
        wa.REQUESTS_DIR_ENV_VAR,
        wa.LEDGER_VIEW_DIR_ENV_VAR,
    ]
    assert environ[wa.REQUESTS_DIR_ENV_VAR] == str(repo / write_requests_relpath())
    assert environ[wa.LEDGER_VIEW_DIR_ENV_VAR] == str(repo / write_ledger_relpath())


def test_a_containers_binds_are_left_as_they_are(tmp_path: Path) -> None:
    repo = _provisioned(tmp_path)
    environ = {wa.REQUESTS_DIR_ENV_VAR: "/bound/requests", wa.LEDGER_VIEW_DIR_ENV_VAR: "/bound"}
    assert _export_write_ledger_view(environ, repo) == []
    assert environ == {
        wa.REQUESTS_DIR_ENV_VAR: "/bound/requests",
        wa.LEDGER_VIEW_DIR_ENV_VAR: "/bound",
    }


def test_a_ledger_the_deploy_did_not_provision_is_not_named(tmp_path: Path) -> None:
    environ: dict[str, str] = {}
    assert _export_write_ledger_view(environ, tmp_path) == []
    assert environ == {}


def test_osprey_web_exports_the_ledger_when_it_resolves_its_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _provisioned(tmp_path)
    (repo / "profile.yml").write_text("project_name: p\n")
    (repo / "build").mkdir()
    (repo / "build" / "config.yml").write_text("project_name: p\n")
    monkeypatch.chdir(repo)  # _resolve_render moves into build/; this moves back
    with patch.dict(os.environ, clear=False):
        os.environ.pop(wa.REQUESTS_DIR_ENV_VAR, None)
        _resolve_render(repo)
        assert os.environ[wa.REQUESTS_DIR_ENV_VAR] == str(repo / write_requests_relpath())
