"""The agent record's credential files.

Pinned: every identity of the credential table gets a login (readers a key,
ClickHouse accounts with ``backend: clickhouse``); a roster user, worker or
reader named like a fixed login is refused naming both; a secret holding
``$``, a quote or a newline survives ``.env.agent-record.service`` as compose
reads it; login files are written once while the verifier and the secret set
follow the config and the env chain; the verifier matches its tokens; and the
secret set holds every minted token.
"""

from __future__ import annotations

import copy
import hashlib
import os
import stat
from pathlib import Path

import pytest

from osprey.agent_record import credentials as cr
from osprey_connectors.dotenv import parse_dotenv_file
from tests.deployment.web_terminals.test_golden_render import EXAMPLE_CONFIG


def _config(
    *,
    users: list[str] | None = None,
    workers: int = 0,
    shared: bool = False,
    bridges: tuple[str, ...] = (),
    host: str | None = "builder",
    readers: list[str] | None = None,
    backend: str = "embedded",
) -> dict:
    config: dict = copy.deepcopy(EXAMPLE_CONFIG)
    if users is not None:
        config["modules"]["web_terminals"]["users"] = users
    deployed = list(bridges)
    services: dict = {}
    if workers:
        deployed.append("dispatch_worker")
        services["dispatch_worker"] = {
            "worker_count": workers,
            "workspace_mode": "shared" if shared else "isolated",
        }
    config["deployed_services"] = deployed
    config["services"] = services
    if host is not None:
        config["deployment"] = {"host_identity": host}
    config["agent_record"] = {
        "backend": backend,
        "readers": [{"name": name} for name in readers or []],
    }
    return config


def _service(root: Path) -> tuple[dict[str, str], list[str]]:
    env = parse_dotenv_file(root / cr.SERVICE_FILE)
    return cr.decode_verifier(env[cr.VERIFIER_ENV]), cr.decode_secret_set(env[cr.SECRETS_ENV])


def _token(root: Path, name: str) -> str:
    login, _, token = parse_dotenv_file(root / name)[cr.LOGIN_ENV].partition(":")
    assert login and token
    return token


def test_every_identity_gets_a_login(tmp_path):
    config = _config(workers=2, bridges=("teams_bridge", "gchat_bridge", "nextcloud_bridge"))
    report = cr.ensure_agent_record_credentials(tmp_path, config, reader_keys=False)
    expected = {
        "alice",
        "bob",
        "dispatch-worker-1",
        "dispatch-worker-2",
        "teams-bridge",
        "gchat-bridge",
        "nextcloud-bridge",
        "sidecar",
        "builder",
        "history",
        "operator",
    }
    assert set(report.minted) == {cr.login_file(login) for login in expected}
    for login in expected:
        path = tmp_path / cr.login_file(login)
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert parse_dotenv_file(path)[cr.LOGIN_ENV].startswith(f"{login}:")
    assert stat.S_IMODE((tmp_path / cr.SERVICE_FILE).stat().st_mode) == 0o600
    # No ClickHouse accounts on the embedded backend.
    assert not (tmp_path / cr.CLICKHOUSE_FILE).exists()


def test_shared_mode_mints_one_dispatch_login(tmp_path):
    cr.ensure_agent_record_credentials(tmp_path, _config(workers=3, shared=True), reader_keys=False)
    logins = {name for name in (p.name for p in tmp_path.iterdir()) if "dispatch" in name}
    assert logins == {cr.login_file("dispatch-worker-shared")}
    env = cr.digest_env(tmp_path, _config(workers=3, shared=True))
    shared = cr.files_digest(tmp_path, [cr.login_file("dispatch-worker-shared")])
    for i in (1, 2, 3):
        assert env[cr.digest_env_name(f"dispatch-worker-{i}")] == shared


def test_readers_get_a_key_only_at_up(tmp_path):
    config = _config(readers=["ops"])
    cr.ensure_agent_record_credentials(tmp_path, config, reader_keys=False)
    assert not (tmp_path / cr.reader_file("ops")).exists()
    report = cr.ensure_agent_record_credentials(tmp_path, config, reader_keys=True)
    assert report.reader_keys == ["ops"]
    value = parse_dotenv_file(tmp_path / cr.reader_file("ops"))[cr.LOGIN_ENV]
    assert value.startswith("reader.ops:")
    verifier, _ = _service(tmp_path)
    assert "reader.ops" in verifier


def test_clickhouse_backend_mints_both_accounts(tmp_path):
    config = _config(backend="clickhouse")
    report = cr.ensure_agent_record_credentials(tmp_path, config, reader_keys=False)
    assert {cr.CLICKHOUSE_FILE, cr.CLICKHOUSE_ADMIN_FILE} <= set(report.minted)
    account = parse_dotenv_file(tmp_path / cr.CLICKHOUSE_FILE)
    admin = parse_dotenv_file(tmp_path / cr.CLICKHOUSE_ADMIN_FILE)
    _, secret_set = _service(tmp_path)
    assert account[cr.CLICKHOUSE_PASSWORD_ENV] in secret_set
    assert admin["CLICKHOUSE_PASSWORD"] in secret_set
    env = cr.digest_env(tmp_path, config)
    assert env[cr.digest_env_name(cr.CLICKHOUSE_SERVICE)] == cr.files_digest(
        tmp_path, [cr.CLICKHOUSE_ADMIN_FILE]
    )
    # agent-record holds its own account, never the admin file.
    assert cr.CLICKHOUSE_ADMIN_FILE not in cr.owner_files(config)[cr.RECORD_SERVICE]


@pytest.mark.parametrize(
    ("config", "claimants"),
    [
        (_config(users=["alice", "operator"]), ("roster user 'operator'", "operator")),
        (_config(users=["alice", "history"]), ("roster user 'history'", "history")),
        (_config(users=["sidecar"]), ("roster user 'sidecar'", "sidecar")),
        # A bridge identity is fixed whether or not that bridge is deployed.
        (_config(users=["teams-bridge"]), ("roster user 'teams-bridge'", "teams-bridge")),
        (_config(users=["builder"]), ("roster user 'builder'", "deployment.host_identity")),
        (_config(readers=["operator"]), ("readers name 'operator'", "fixed login 'operator'")),
        (_config(readers=["sidecar"]), ("readers name 'sidecar'", "fixed login 'sidecar'")),
        (
            _config(readers=["gchat-bridge"]),
            ("readers name 'gchat-bridge'", "fixed login 'gchat-bridge'"),
        ),
        (_config(readers=["builder"]), ("readers name 'builder'", "deployment.host_identity")),
        (
            _config(workers=1, host="dispatch-worker-1"),
            ("dispatch worker 'dispatch-worker-1'", "deployment.host_identity"),
        ),
    ],
)
def test_minting_refuses_a_fixed_login_naming_both(tmp_path, config, claimants):
    with pytest.raises(ValueError) as excinfo:
        cr.ensure_agent_record_credentials(tmp_path, config, reader_keys=True)
    message = str(excinfo.value)
    assert "claimed twice" in message
    for part in claimants:
        assert part in message
    assert not any(cr.is_credential_file(p.name) for p in tmp_path.iterdir())


@pytest.mark.parametrize(
    "secret", ["ab$cd${HOME}efgh", 'it\'s "quoted" 123456', "line1\nline2-xyz"]
)
def test_a_hostile_secret_round_trips_as_compose_reads_it(tmp_path, secret):
    text = cr.render_service_env(["alice:" + "0" * 64], [secret])
    path = tmp_path / cr.SERVICE_FILE
    path.write_text(text, encoding="utf-8")
    # One KEY=value line per variable, nothing compose would interpolate.
    assert len(text.splitlines()) == 2
    assert "$" not in text and "'" not in text and '"' not in text
    env = parse_dotenv_file(path)
    assert cr.decode_secret_set(env[cr.SECRETS_ENV]) == [secret]
    assert cr.decode_verifier(env[cr.VERIFIER_ENV]) == {"alice": "0" * 64}


def test_a_dollar_and_quote_secret_from_the_env_chain_round_trips(tmp_path):
    (tmp_path / ".env").write_text("PROVIDER_API_KEY='sk-$abc\"def\"ghi'\n", encoding="utf-8")
    cr.ensure_agent_record_credentials(tmp_path, _config(), reader_keys=False)
    _, secret_set = _service(tmp_path)
    assert 'sk-$abc"def"ghi' in secret_set


def test_login_files_are_never_rewritten_and_the_service_file_follows(tmp_path):
    config = _config(workers=1)
    cr.ensure_agent_record_credentials(tmp_path, config, reader_keys=True)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.name != cr.SERVICE_FILE}
    service_before = (tmp_path / cr.SERVICE_FILE).read_bytes()

    again = cr.ensure_agent_record_credentials(tmp_path, config, reader_keys=True)
    assert again.minted == [] and not again.service_changed
    assert (tmp_path / cr.SERVICE_FILE).read_bytes() == service_before

    # A new user, a new worker, a new reader and a new env-chain secret.
    grown = _config(users=["alice", "bob", "carol"], workers=2, readers=["ops"])
    (tmp_path / ".env").write_text(
        "DISPATCH_WORKER_TOKEN=short-bearer\nSOME_SECRET=abcdefghijklmnop\nSHORT_TOKEN=abc\n",
        encoding="utf-8",
    )
    report = cr.ensure_agent_record_credentials(tmp_path, grown, reader_keys=True)
    assert set(report.minted) == {
        cr.login_file("carol"),
        cr.login_file("dispatch-worker-2"),
        cr.reader_file("ops"),
    }
    for name, content in before.items():
        assert (tmp_path / name).read_bytes() == content, name
    verifier, secret_set = _service(tmp_path)
    assert {"carol", "dispatch-worker-2", "reader.ops"} <= set(verifier)
    assert _token(tmp_path, cr.login_file("carol")) in secret_set
    assert "short-bearer" in secret_set  # a worker bearer, whatever its length
    assert "abcdefghijklmnop" in secret_set
    assert "abc" not in secret_set  # under the rule's length

    # A user who leaves takes their login out of the files and the verifier.
    cr.ensure_agent_record_credentials(tmp_path, _config(workers=2), reader_keys=False)
    assert not (tmp_path / cr.login_file("carol")).exists()
    verifier, _ = _service(tmp_path)
    assert "carol" not in verifier


def test_the_verifier_matches_its_tokens(tmp_path):
    config = _config(workers=1, bridges=("teams_bridge",), readers=["ops"])
    cr.ensure_agent_record_credentials(tmp_path, config, reader_keys=True)
    verifier, _ = _service(tmp_path)
    assert set(verifier) == {
        "alice",
        "bob",
        "dispatch-worker-1",
        "teams-bridge",
        "sidecar",
        "builder",
        "history",
        "operator",
        "reader.ops",
    }
    for login, digest in verifier.items():
        name = cr.reader_file("ops") if login == "reader.ops" else cr.login_file(login)
        token = _token(tmp_path, name)
        assert digest == hashlib.sha256(token.encode()).hexdigest()
        assert cr.verifier_line(login, token) == f"{login}:{digest}"


def test_the_secret_set_holds_every_minted_token(tmp_path):
    config = _config(workers=2, bridges=("teams_bridge",), readers=["ops"], backend="clickhouse")
    cr.ensure_agent_record_credentials(tmp_path, config, reader_keys=True)
    _, secret_set = _service(tmp_path)
    tokens = [
        _token(tmp_path, p.name)
        for p in tmp_path.iterdir()
        if p.name.startswith((cr.LOGIN_FILE_PREFIX, cr.READER_FILE_PREFIX))
    ]
    assert len(tokens) == 10
    assert set(tokens) <= set(secret_set)
    assert secret_set == sorted(secret_set, key=lambda v: (-len(v), v))


def test_digest_env_follows_a_changed_file_and_never_raises(tmp_path):
    config = _config(workers=1, bridges=("teams_bridge",))
    empty = cr.digest_env(tmp_path, config)
    cr.ensure_agent_record_credentials(tmp_path, config, reader_keys=False)
    minted = cr.digest_env(tmp_path, config)
    for service in ("agent-record", "history", "dispatch-worker-1", "teams_bridge"):
        name = cr.digest_env_name(service)
        assert name.startswith("OSPREY_AGENT_RECORD_DIGEST_")
        assert empty[name] != minted[name]
    assert cr.digest_env(tmp_path, {"deployed_services": 7}).keys() >= {
        cr.digest_env_name(cr.RECORD_SERVICE),
        cr.digest_env_name(cr.CLICKHOUSE_SERVICE),
    }
    assert set(cr.login_digests(tmp_path, config)) == {"alice", "bob", "sidecar"}


def test_existing_files_are_listed_for_reset(tmp_path):
    cr.ensure_agent_record_credentials(tmp_path, _config(readers=["ops"]), reader_keys=True)
    (tmp_path / ".env.unrelated").write_text("x\n", encoding="utf-8")
    listed = dict(cr.existing_credential_files(tmp_path))
    assert ".env.unrelated" not in listed
    assert cr.SERVICE_FILE in listed
    assert listed[cr.login_file("alice")] == "the agent record login of alice"
    assert listed[cr.reader_file("ops")] == "the key of the agent record reader ops"
    assert all(cr.is_credential_file(name) for name in listed)


def _history_config() -> dict:
    """:func:`_config` with a sign-in, so the history page is deployed."""
    config = _config()
    config["modules"]["web_terminals"]["auth"] = {"method": "password"}
    return config


def test_history_gate_follows_the_page(tmp_path):
    config = _history_config()
    report = cr.ensure_agent_record_credentials(tmp_path, config, reader_keys=False)
    gate = tmp_path / cr.HISTORY_GATE_FILE
    assert cr.HISTORY_GATE_FILE in report.minted
    assert cr.is_credential_file(cr.HISTORY_GATE_FILE)
    assert stat.S_IMODE(gate.stat().st_mode) == 0o600
    secret = parse_dotenv_file(gate)[cr.HISTORY_GATE_ENV]
    assert len(secret) >= 32
    assert cr.HISTORY_GATE_ENV.startswith("OSPREY_TERMINAL_SECRET_")

    # Written once: a second call leaves the secret as it is.
    again = cr.ensure_agent_record_credentials(tmp_path, config, reader_keys=False)
    assert cr.HISTORY_GATE_FILE not in again.minted
    assert parse_dotenv_file(gate)[cr.HISTORY_GATE_ENV] == secret

    # The page leaves the deployment: so does its gate secret.
    gone = cr.ensure_agent_record_credentials(tmp_path, _config(), reader_keys=False)
    assert gone.purged == [cr.HISTORY_GATE_FILE]
    assert not gate.exists()


def test_no_history_page_mints_no_gate(tmp_path):
    report = cr.ensure_agent_record_credentials(tmp_path, _config(), reader_keys=False)
    assert cr.HISTORY_GATE_FILE not in report.minted
    assert not (tmp_path / cr.HISTORY_GATE_FILE).exists()


def test_history_gate_has_owners_and_one_digest_label(tmp_path):
    config = _history_config()
    owners = cr.owner_files(config)
    assert owners["history"] == (cr.login_file("history"), cr.HISTORY_GATE_FILE)
    assert owners[cr.PROXY_SERVICE] == (cr.HISTORY_GATE_FILE,)
    # Fixed owners: listed whether or not the page is deployed.
    assert cr.owner_files(_config())[cr.PROXY_SERVICE] == (cr.HISTORY_GATE_FILE,)
    before = cr.digest_env(tmp_path, config)
    cr.ensure_agent_record_credentials(tmp_path, config, reader_keys=False)
    after = cr.digest_env(tmp_path, config)
    for service in ("history", cr.PROXY_SERVICE):
        assert before[cr.digest_env_name(service)] != after[cr.digest_env_name(service)]


def test_history_gate_is_listed_for_reset(tmp_path):
    cr.ensure_agent_record_credentials(tmp_path, _history_config(), reader_keys=False)
    listed = dict(cr.existing_credential_files(tmp_path))
    assert listed[cr.HISTORY_GATE_FILE] == "the secret the history page is gated on"


def test_file_check_names_a_missing_or_open_gate(tmp_path):
    config = _history_config()
    problems = cr.credential_file_problems(tmp_path, config)
    assert [p for p, _ in problems] == [f"{cr.HISTORY_GATE_FILE} is missing"]
    assert cr.credential_file_problems(tmp_path, _config()) == []

    cr.ensure_agent_record_credentials(tmp_path, config, reader_keys=False)
    assert cr.credential_file_problems(tmp_path, config) == []

    gate = tmp_path / cr.HISTORY_GATE_FILE
    gate.chmod(0o644)
    ((problem, remedy),) = cr.credential_file_problems(tmp_path, config)
    assert "0644" in problem and remedy == f"chmod 600 {cr.HISTORY_GATE_FILE}"

    gate.unlink()
    gate.mkdir()
    ((problem, _),) = cr.credential_file_problems(tmp_path, config)
    assert problem == f"{cr.HISTORY_GATE_FILE} is not a regular file"


def _record_credential_names(root: Path, config: dict) -> set[str]:
    """Every ``OSPREY_AGENT_RECORD_*`` name the module defines or the files carry."""
    names = {
        value
        for value in vars(cr).values()
        if isinstance(value, str) and value.startswith("OSPREY_AGENT_RECORD_")
    }
    names |= set(cr.digest_env(root, config))
    for path in root.glob(f"{cr.CREDENTIAL_PREFIX}*"):
        names |= {
            name for name in parse_dotenv_file(path) if name.startswith("OSPREY_AGENT_RECORD_")
        }
    return names


def test_no_record_credential_name_is_stripped_as_sensitive(tmp_path):
    from osprey.utils.sensitive_env import is_sensitive

    config = _config(backend="clickhouse", workers=1, readers=["ops"])
    cr.ensure_agent_record_credentials(tmp_path, config, reader_keys=True)
    names = _record_credential_names(tmp_path, config)

    assert {
        cr.LOGIN_ENV,
        cr.VERIFIER_ENV,
        cr.SECRETS_ENV,
        cr.CLICKHOUSE_USER_ENV,
        cr.CLICKHOUSE_PASSWORD_ENV,
    } <= names
    assert any(name.startswith(cr.DIGEST_ENV_PREFIX) for name in names)
    assert [name for name in sorted(names) if is_sensitive(name)] == []


@pytest.mark.parametrize("include_sensitive", [False, True], ids=["stripped", "terminal"])
def test_a_host_launch_reads_each_credential_from_its_own_file(
    tmp_path, monkeypatch, include_sensitive
):
    from osprey.agent_record import host as hr
    from osprey.mcp_env import load_dotenv_from_project
    from osprey_connectors.dotenv import merge_chain

    config = _config(backend="clickhouse")
    cr.ensure_agent_record_credentials(tmp_path, config, reader_keys=False)
    (tmp_path / ".env").write_text("OSPREY_TERMINAL_SECRET=terminal-secret-value\n")
    (tmp_path / ".env.shared").write_text("RECORD_TEST_SHARED=shared\n")
    for name in _record_credential_names(tmp_path, config):
        monkeypatch.delenv(name, raising=False)
    for name in ("OSPREY_TERMINAL_SECRET", "RECORD_TEST_SHARED"):
        monkeypatch.setenv(name, "")
    monkeypatch.delenv("OSPREY_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)

    chain = merge_chain(tmp_path)
    assert not [name for name in chain if name.startswith("OSPREY_AGENT_RECORD_")]
    load_dotenv_from_project(include_sensitive=include_sensitive)
    assert os.environ["RECORD_TEST_SHARED"] == "shared"
    assert not [name for name in os.environ if name.startswith("OSPREY_AGENT_RECORD_")]

    # The host login, from its login file.
    monkeypatch.setattr(hr, "_probe", lambda *_args: (hr._READY, ""))
    site = hr._host_site(tmp_path, config)
    assert (site.login, site.token) == ("builder", _token(tmp_path, cr.login_file("builder")))

    # The verifier and the secret set, from the service file.
    service = parse_dotenv_file(tmp_path / cr.SERVICE_FILE)
    child = hr._child_env(tmp_path, config, tmp_path / "state", tmp_path / "config.yml")
    assert child[cr.VERIFIER_ENV] == service[cr.VERIFIER_ENV]
    assert child[cr.SECRETS_ENV] == service[cr.SECRETS_ENV]
    assert cr.LOGIN_ENV not in child

    # The ClickHouse password, from the account file the record service owns.
    account = parse_dotenv_file(tmp_path / cr.CLICKHOUSE_FILE)
    password = cr.read_env_value(tmp_path / cr.CLICKHOUSE_FILE, cr.CLICKHOUSE_PASSWORD_ENV)
    assert password and password == account[cr.CLICKHOUSE_PASSWORD_ENV]
    assert cr.CLICKHOUSE_FILE in cr.owner_files(config)[cr.RECORD_SERVICE]
