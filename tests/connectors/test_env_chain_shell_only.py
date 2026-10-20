"""The env chain never sets a shell-only launch variable.

:func:`osprey_connectors.config.apply_env_chain` is the one code path that
applies the chain to a process environment. A launch variable a chain file sets
leaves the environment exactly as the shell gave it, and the operator is told
which names were ignored, never their values.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

from osprey_connectors import config
from osprey_connectors.config import (
    SHELL_ONLY_ENV_PREFIXES,
    SHELL_ONLY_ENV_VARS,
    apply_env_chain,
    dotenv_shell_overrides,
    is_shell_only_env_var,
    load_project_dotenv,
)
from osprey_connectors.dotenv import ENV_CHAIN_APPLIED_ENV

#: Every shell-only name, plus one name per loader prefix.
PROBE_NAMES = sorted(SHELL_ONLY_ENV_VARS) + [
    f"{prefix}OSPREY_PROBE" for prefix in SHELL_ONLY_ENV_PREFIXES
]


def _file_value(name: str) -> str:
    return f"from-chain-file-{name.lower()}"


def _shell_value(name: str) -> str:
    return f"from-the-shell-{name.lower()}"


def _write_chain(directory: Path, filename: str, extra: dict[str, str]) -> Path:
    lines = [f"{name}={_file_value(name)}" for name in PROBE_NAMES]
    lines += [f"{key}={value}" for key, value in extra.items()]
    path = directory / filename
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _fresh_report_record(monkeypatch):
    monkeypatch.setattr(config, "_shell_only_reported", set())


@pytest.fixture
def chain(tmp_path) -> list[Path]:
    shared = _write_chain(
        tmp_path, ".env.shared", {"OSPREY_ORDINARY": "shared", "OSPREY_BOTH": "s"}
    )
    local = _write_chain(tmp_path, ".env", {"OSPREY_BOTH": "local"})
    return [shared, local]


def test_every_probe_name_is_shell_only():
    for name in PROBE_NAMES:
        assert is_shell_only_env_var(name), name
    assert not is_shell_only_env_var("EPICS_CA_ADDR_LIST")


@pytest.mark.parametrize("override", [True, False])
@pytest.mark.parametrize("name", PROBE_NAMES)
def test_shell_value_is_kept(chain, name, override):
    environ = {name: _shell_value(name)}
    apply_env_chain(environ, chain, override=override)
    assert environ[name] == _shell_value(name)


@pytest.mark.parametrize("override", [True, False])
@pytest.mark.parametrize("name", PROBE_NAMES)
def test_absent_name_stays_absent(chain, name, override):
    environ: dict[str, str] = {}
    apply_env_chain(environ, chain, override=override)
    assert name not in environ


@pytest.mark.parametrize("override", [True, False])
def test_ordinary_keys_apply_and_local_wins(chain, override):
    environ: dict[str, str] = {}
    written = apply_env_chain(environ, chain, override=override)
    assert environ["OSPREY_ORDINARY"] == "shared"
    assert environ["OSPREY_BOTH"] == "local"
    assert written == ["OSPREY_BOTH", "OSPREY_ORDINARY"]


def test_override_false_keeps_a_preexisting_value(chain):
    environ = {"OSPREY_BOTH": "shell"}
    written = apply_env_chain(environ, chain, override=False)
    assert environ["OSPREY_BOTH"] == "shell"
    assert "OSPREY_BOTH" not in written


def test_override_true_replaces_a_preexisting_value(chain):
    environ = {"OSPREY_BOTH": "shell"}
    apply_env_chain(environ, chain)
    assert environ["OSPREY_BOTH"] == "local"


def test_missing_path_is_skipped(tmp_path):
    environ: dict[str, str] = {}
    assert apply_env_chain(environ, [tmp_path / ".env"]) == []
    assert environ == {}


def test_warning_names_file_and_every_ignored_name_and_no_value(chain, caplog):
    caplog.set_level(logging.WARNING, logger="CONFIG")
    apply_env_chain({}, chain)
    records = [record for record in caplog.records if record.name == "CONFIG"]
    assert len(records) == len(chain)
    for record, path in zip(records, chain, strict=True):
        message = record.getMessage()
        assert str(path) in message
        for name in PROBE_NAMES:
            assert name in message
    for name in PROBE_NAMES:
        assert _file_value(name) not in caplog.text


def test_second_call_on_the_same_file_does_not_warn_again(chain, caplog):
    caplog.set_level(logging.WARNING, logger="CONFIG")
    apply_env_chain({}, chain)
    caplog.clear()
    apply_env_chain({}, chain)
    assert [record for record in caplog.records if record.name == "CONFIG"] == []


def test_load_project_dotenv_skips_and_does_not_record_shell_only(chain, monkeypatch):
    monkeypatch.chdir(chain[0].parent)
    monkeypatch.delenv(ENV_CHAIN_APPLIED_ENV, raising=False)
    monkeypatch.setattr(config, "_dotenv_shell_overrides", {})
    for name in PROBE_NAMES:
        monkeypatch.setenv(name, _shell_value(name))
    monkeypatch.setenv("OSPREY_BOTH", "shell")
    monkeypatch.delenv("OSPREY_ORDINARY", raising=False)

    load_project_dotenv()

    for name in PROBE_NAMES:
        assert os.environ[name] == _shell_value(name)
    assert os.environ["OSPREY_BOTH"] == "local"
    assert os.environ["OSPREY_ORDINARY"] == "shared"
    recorded = dotenv_shell_overrides()
    assert recorded == {"OSPREY_BOTH": "shell"}
