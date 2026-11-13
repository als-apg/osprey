"""The host identity a build records, and what it does with the login's own spelling.

Pinned: a login that is already an identity is recorded as it is; any other is
recorded as the identity derived from it and the build says so; a derived name
another login holds, or a login nothing can be derived from, is recorded as
nothing and said, and the build goes on; a name the profile states is kept as
written. Then the directories and the login file a build leaves for a
capital-letter login: compared by what the directory lists, on whichever kind
of filesystem the test's temporary directory is.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.agent_record import credentials as cr
from osprey.agent_record.identities import host_identity
from osprey.cli import build_cmd, output
from osprey.deployment.compose_generator import ensure_spool_dir
from tests.deployment.web_terminals.test_golden_render import EXAMPLE_CONFIG


@pytest.fixture
def said(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every line the build says, warnings by their summary."""
    lines: list[str] = []
    monkeypatch.setattr(output, "note", lines.append)
    monkeypatch.setattr(output, "warn_fact", lambda _logger, summary, *_rest: lines.append(summary))
    return lines


def _roster(*users: str) -> dict[str, Any]:
    """A rendered deployment with web terminals for *users*."""
    config: dict[str, Any] = copy.deepcopy(EXAMPLE_CONFIG)
    config["modules"]["web_terminals"]["users"] = list(users)
    config["deployed_services"] = []
    config["services"] = {}
    return config


def _stamp(tmp_path: Path, monkeypatch, account: str, config: dict | None = None):
    """Record the host identity *account* builds as; return (changed, stamped config)."""
    monkeypatch.setattr("getpass.getuser", lambda: account)
    render = tmp_path / "build"
    render.mkdir(exist_ok=True)
    path = render / "config.yml"
    path.write_text(yaml.safe_dump(config or {"deployed_services": []}), encoding="utf-8")
    changed = build_cmd._record_host_identity(render, yaml.safe_load(path.read_text()))
    return changed, yaml.safe_load(path.read_text(encoding="utf-8"))


# --- the stamp ---------------------------------------------------------------------


def test_a_login_that_is_an_identity_is_recorded_silently(tmp_path, monkeypatch, said):
    changed, stamped = _stamp(tmp_path, monkeypatch, "builder")
    assert changed is True
    assert host_identity(stamped) == "builder"
    assert said == []


def test_a_capital_letter_login_is_recorded_lowercase_and_said(tmp_path, monkeypatch, said):
    changed, stamped = _stamp(tmp_path, monkeypatch, "GMartino")
    assert changed is True
    assert host_identity(stamped) == "gmartino"
    assert said == ["Host identity: gmartino (login GMartino)"]


def test_a_dotted_login_is_recorded_with_a_dash(tmp_path, monkeypatch, said):
    _changed, stamped = _stamp(tmp_path, monkeypatch, "first.last")
    assert host_identity(stamped) == "first-last"
    assert said == ["Host identity: first-last (login first.last)"]


@pytest.mark.parametrize("stated", ["operator-1", "GMartino"])
def test_a_name_the_profile_states_is_kept_as_written(tmp_path, monkeypatch, said, stated):
    config = {"deployed_services": [], "deployment": {"host_identity": stated}}
    changed, stamped = _stamp(tmp_path, monkeypatch, "Someone.Else", config)
    assert changed is False
    assert host_identity(stamped) == stated
    if stated == "GMartino":
        (line,) = said
        assert "'GMartino' is not a usable host identity" in line
    else:
        assert said == []


@pytest.mark.parametrize(
    ("account", "config"),
    [
        ("GMartino", _roster("alice", "gmartino")),
        ("Operator", None),
        (
            "Dispatch-Worker-1",
            {
                "deployed_services": ["dispatch_worker"],
                "services": {"dispatch_worker": {"worker_count": 1}},
            },
        ),
    ],
    ids=["roster-user", "fixed-login", "dispatch-worker"],
)
def test_a_derived_name_another_login_holds_is_not_recorded(
    tmp_path, monkeypatch, said, account, config
):
    changed, stamped = _stamp(tmp_path, monkeypatch, account, config)
    assert changed is False
    assert host_identity(stamped) is None
    (line,) = said
    assert f"the login {account!r} would be recorded as {account.lower()!r}" in line
    # And the build goes on: the record's credentials are minted for the rest.
    cr.ensure_agent_record_credentials(tmp_path, stamped, reader_keys=False)


def test_a_login_that_is_an_identity_and_is_claimed_is_refused_as_before(
    tmp_path, monkeypatch, said
):
    changed, stamped = _stamp(tmp_path, monkeypatch, "builder", _roster("alice", "builder"))
    assert changed is True and host_identity(stamped) == "builder"
    assert said == []
    with pytest.raises(ValueError, match="claimed twice"):
        cr.ensure_agent_record_credentials(tmp_path, stamped, reader_keys=False)


def test_a_login_nothing_can_be_derived_from_is_not_recorded(tmp_path, monkeypatch, said):
    changed, stamped = _stamp(tmp_path, monkeypatch, "___")
    assert changed is False
    assert host_identity(stamped) is None
    (line,) = said
    assert "no host identity can be made from the login '___'" in line


def test_an_account_with_no_name_records_nothing_and_says_nothing(tmp_path, monkeypatch, said):
    render = tmp_path / "build"
    render.mkdir()
    (render / "config.yml").write_text("deployed_services: []\n", encoding="utf-8")

    def nameless() -> str:
        raise KeyError("uid 4242")

    monkeypatch.setattr("getpass.getuser", nameless)
    assert build_cmd._record_host_identity(render, {"deployed_services": []}) is False
    assert said == []


def test_the_deployments_second_render_says_nothing_again(tmp_path, monkeypatch, said):
    monkeypatch.setattr("getpass.getuser", lambda: "GMartino")
    already: set[str] = set()
    for name in ("host", "image"):
        render = tmp_path / name
        render.mkdir()
        (render / "config.yml").write_text("deployed_services: []\n", encoding="utf-8")
        assert build_cmd._record_host_identity(render, {"deployed_services": []}, said=already)
    assert said == ["Host identity: gmartino (login GMartino)"]


# --- the directories and the login file ----------------------------------------------


def _ignores_case(directory: Path) -> bool:
    """Whether *directory*'s filesystem treats two spellings that differ in case as one."""
    probe = directory / "CaseProbe"
    probe.mkdir()
    try:
        return (directory / "caseprobe").exists()
    finally:
        probe.rmdir()


def _build(repo: Path, monkeypatch, account: str) -> dict[str, Any]:
    """The build's host-identity steps, in its order: stamp, credentials, carry, provision."""
    _changed, stamped = _stamp(repo, monkeypatch, account)
    cr.ensure_agent_record_credentials(repo, stamped, reader_keys=False)
    build_cmd._carry_host_identity_dirs(repo, stamped)
    ensure_spool_dir(repo, host_identity(stamped), relative_to=repo)
    return stamped


def _login_files(repo: Path) -> list[str]:
    return sorted(
        name[len(cr.LOGIN_FILE_PREFIX) :]
        for name in os.listdir(repo)
        if name.startswith(cr.LOGIN_FILE_PREFIX)
    )


def _seed(repo: Path) -> Path:
    """What a session run under the login's own spelling left: a spool segment and a login."""
    old = repo / "var" / "spool" / "GMartino"
    old.mkdir(parents=True)
    (old / "segment-0001.jsonl").write_text('{"kind": "audit"}\n', encoding="utf-8")
    login = repo / cr.login_file("GMartino")
    login.write_text(f"{cr.LOGIN_ENV}=GMartino:stale-token\n", encoding="utf-8")
    login.chmod(0o600)
    return old


@pytest.mark.usefixtures("said")
def test_a_capital_letter_login_leaves_only_the_identity_listed(tmp_path, monkeypatch):
    _build(tmp_path, monkeypatch, "GMartino")
    assert sorted(os.listdir(tmp_path / "var" / "spool")) == ["gmartino"]
    logins = _login_files(tmp_path)
    assert "gmartino" in logins and "GMartino" not in logins


def test_a_stale_login_file_in_another_case_does_not_cost_the_new_one(tmp_path):
    """Purged before minting, so a filesystem that ignores case still gets the new file."""
    stale = tmp_path / cr.login_file("GMartino")
    stale.write_text(f"{cr.LOGIN_ENV}=GMartino:stale-token\n", encoding="utf-8")
    config = {"deployed_services": [], "deployment": {"host_identity": "gmartino"}}

    report = cr.ensure_agent_record_credentials(tmp_path, config, reader_keys=False)

    assert cr.login_file("GMartino") in report.purged
    assert "GMartino" not in _login_files(tmp_path)
    assert cr.read_login(tmp_path / cr.login_file("gmartino"), "gmartino") is not None


def test_on_a_case_sensitive_filesystem_the_old_directory_stays_beside_the_new(
    tmp_path, monkeypatch, said
):
    if _ignores_case(tmp_path):
        pytest.skip("the temporary directory's filesystem ignores case")
    old = _seed(tmp_path)

    _build(tmp_path, monkeypatch, "GMartino")
    _build(tmp_path, monkeypatch, "GMartino")

    assert sorted(os.listdir(tmp_path / "var" / "spool")) == ["GMartino", "gmartino"]
    assert (old / "segment-0001.jsonl").read_text(encoding="utf-8") == '{"kind": "audit"}\n'
    notices = [line for line in said if "holds entries written before this build" in line]
    assert notices == [
        "var/spool/GMartino holds entries written before this build; "
        "new host entries go to var/spool/gmartino."
    ]
    assert _login_files(tmp_path) == ["gmartino", "history", "operator"]
    assert cr.read_login(tmp_path / cr.login_file("gmartino"), "gmartino") is not None


def test_on_a_case_insensitive_filesystem_one_build_renames_to_the_identity(
    tmp_path, monkeypatch, said
):
    if not _ignores_case(tmp_path):
        pytest.skip("the temporary directory's filesystem is case-sensitive")
    _seed(tmp_path)

    _build(tmp_path, monkeypatch, "GMartino")

    spool = tmp_path / "var" / "spool"
    assert sorted(os.listdir(spool)) == ["gmartino"]
    assert os.listdir(spool / "gmartino") == ["segment-0001.jsonl"]
    assert not [line for line in said if "holds entries written before this build" in line]
    assert _login_files(tmp_path) == ["gmartino", "history", "operator"]
    token = cr.read_env_value(tmp_path / cr.login_file("gmartino"), cr.LOGIN_ENV)
    assert token is not None and token.startswith("gmartino:")


def test_a_login_that_is_an_identity_moves_nothing(tmp_path, monkeypatch, said):
    old = tmp_path / "var" / "spool" / "Builder"
    old.mkdir(parents=True)
    _build(tmp_path, monkeypatch, "builder")
    assert "Builder" in os.listdir(tmp_path / "var" / "spool")
    assert said == []
