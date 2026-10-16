"""The telemetry store's credential files: what is minted, for whom, and when.

Pins :mod:`osprey.deployment.telemetry_credentials` and its call sites:

* the file set a build mints and the one a start adds (reader keys), each file
  0600, written once, holding no ``$``;
* the sender set — who emits, and therefore who holds a login;
* the sender list the gateway reads, byte-equal across renders;
* the recreate triggers: each file moves the digest of its owners and of
  nothing else, and the web overlay hands each terminal its own;
* host launches: the identity the ledger files under is the login's name;
* the reporting pass over the files, which writes nothing;
* what ``osprey up`` and ``osprey reset`` say about them — names, never values.
"""

from __future__ import annotations

import base64
import copy
import datetime
import hashlib
import stat
from pathlib import Path

import pytest
import yaml

from osprey.deployment import telemetry_credentials as creds
from osprey.deployment.web_terminals.render import render_web_terminals

from .web_terminals.test_golden_render import EXAMPLE_CONFIG

SERVICE_FILES = {creds.ADMIN_FILE, creds.INGEST_FILE, creds.ARCHIVE_FILE}
HISTORY_FILES = {creds.HISTORY_FILE, creds.HISTORY_GATE_FILE}


def _config(
    *,
    store: bool = True,
    gateway: bool = True,
    roster: bool = False,
    auth: str | None = None,
    workers: int = 0,
    host: str | None = "builder",
    readers: tuple[str, ...] = (),
) -> dict:
    """A rendered config with the telemetry halves a test asks for."""
    config: dict = copy.deepcopy(EXAMPLE_CONFIG) if roster else {"project_name": "p"}
    if not roster:
        config["modules"] = {}
    deployed = []
    if store:
        deployed.append("telemetry_store")
    if gateway:
        deployed.append("telemetry_gateway")
    services: dict = {"telemetry_store": {"readers": [{"name": n} for n in readers]}}
    if workers:
        deployed.append("dispatch_worker")
        services["dispatch_worker"] = {"worker_count": workers}
    config["deployed_services"] = deployed
    config["services"] = services
    if host is not None:
        config["deployment"] = {"host_identity": host}
    if auth is not None:
        config["modules"]["web_terminals"]["auth"] = {
            "method": auth,
            "allow_insecure_http": True,
        }
    return config


def _files(root: Path) -> set[str]:
    return {path.name for path in root.iterdir() if path.name.startswith(".env.telemetry-")}


def _snapshot(root: Path) -> dict[str, tuple[bytes, int, int]]:
    return {
        path.name: (path.read_bytes(), path.stat().st_mode, path.stat().st_mtime_ns)
        for path in sorted(root.iterdir())
        if path.is_file()
    }


# --- the file set ------------------------------------------------------------


def test_a_build_mints_every_service_and_sender_file(tmp_path: Path) -> None:
    config = _config(roster=True, workers=2)

    report = creds.ensure_telemetry_credentials(tmp_path, config, reader_keys=False)

    senders = {"alice", "bob", "dispatch-worker-1", "dispatch-worker-2", "builder"}
    expected = SERVICE_FILES | {creds.HTPASSWD_FILE} | {creds.sender_file(name) for name in senders}
    assert _files(tmp_path) == expected
    assert set(report.minted) == expected - {creds.HTPASSWD_FILE}
    assert set(creds.expected_files(config)) == expected


def test_every_file_is_owner_only_and_holds_no_dollar(tmp_path: Path) -> None:
    config = _config(roster=True, workers=1, auth="password", readers=("ops",))

    creds.ensure_telemetry_credentials(tmp_path, config, reader_keys=True)

    assert HISTORY_FILES <= _files(tmp_path)
    for name in _files(tmp_path):
        path = tmp_path / name
        assert stat.S_IMODE(path.stat().st_mode) == 0o600, name
        assert "$" not in path.read_text(encoding="utf-8"), name


def test_the_admin_file_is_the_stores_bootstrap_account(tmp_path: Path) -> None:
    creds.ensure_telemetry_credentials(tmp_path, _config(), reader_keys=False)

    lines = dict(
        line.split("=", 1) for line in (tmp_path / creds.ADMIN_FILE).read_text().splitlines()
    )
    assert lines["CLICKHOUSE_USER"] == "osprey_admin"
    assert lines["CLICKHOUSE_DEFAULT_ACCESS_MANAGEMENT"] == "1"
    assert len(lines["CLICKHOUSE_PASSWORD"]) >= 32


def test_a_second_call_changes_nothing(tmp_path: Path) -> None:
    config = _config(roster=True, workers=1, readers=("ops",))
    creds.ensure_telemetry_credentials(tmp_path, config, reader_keys=True)
    before = _snapshot(tmp_path)

    report = creds.ensure_telemetry_credentials(tmp_path, config, reader_keys=True)

    assert _snapshot(tmp_path) == before
    assert report == creds.CredentialReport()


def test_a_deleted_file_is_minted_again_with_a_new_secret(tmp_path: Path) -> None:
    config = _config()
    creds.ensure_telemetry_credentials(tmp_path, config, reader_keys=False)
    old = (tmp_path / creds.INGEST_FILE).read_text()
    (tmp_path / creds.INGEST_FILE).unlink()

    report = creds.ensure_telemetry_credentials(tmp_path, config, reader_keys=False)

    assert report.minted == [creds.INGEST_FILE]
    assert (tmp_path / creds.INGEST_FILE).read_text() != old


def test_a_deployment_without_the_store_is_left_exactly_as_it_is(tmp_path: Path) -> None:
    (tmp_path / creds.ADMIN_FILE).write_text("CLICKHOUSE_PASSWORD=kept\n")
    before = _snapshot(tmp_path)
    config = _config(store=False, gateway=False)

    report = creds.ensure_telemetry_credentials(tmp_path, config, reader_keys=True)

    assert report == creds.CredentialReport()
    assert _snapshot(tmp_path) == before
    assert creds.expected_files(config, reader_keys=True) == []
    assert creds.credential_file_problems(tmp_path, config) == []


# --- the sender set ----------------------------------------------------------


def test_the_sender_set_is_the_emitters_only() -> None:
    config = _config(roster=True, workers=2)
    # Recording services that emit nothing: no login.
    config["deployed_services"] += ["bluesky_web", "bluesky"]

    assert creds.sender_identities(config) == [
        "alice",
        "bob",
        "dispatch-worker-1",
        "dispatch-worker-2",
        "builder",
    ]


def test_no_gateway_no_senders() -> None:
    assert creds.sender_identities(_config(gateway=False, roster=True, workers=1)) == []


@pytest.mark.parametrize("name", ["a/b", "a:b", "a b", "", "..", "a$b"])
def test_a_host_identity_that_cannot_be_a_login_is_refused(name: str, tmp_path: Path) -> None:
    config = _config(host="x")
    config["deployment"]["host_identity"] = name or " "
    if not name:
        # A blank value reads as "none recorded", which is not an identity at all.
        assert creds.sender_identities(config) == []
        return
    with pytest.raises(ValueError, match="deployment.host_identity"):
        creds.ensure_telemetry_credentials(tmp_path, config, reader_keys=False)


def test_a_sender_that_left_the_config_loses_its_file(tmp_path: Path) -> None:
    creds.ensure_telemetry_credentials(tmp_path, _config(workers=2), reader_keys=False)

    report = creds.ensure_telemetry_credentials(tmp_path, _config(workers=1), reader_keys=False)

    gone = creds.sender_file("dispatch-worker-2")
    assert report.purged == [gone]
    assert gone not in _files(tmp_path)
    assert "dispatch-worker-2:" not in (tmp_path / creds.HTPASSWD_FILE).read_text()


# --- the history page's files ------------------------------------------------


@pytest.mark.parametrize(
    ("auth", "present"), [("password", True), ("oidc", True), ("none", False), (None, False)]
)
def test_history_files_follow_the_history_page(
    auth: str | None, present: bool, tmp_path: Path
) -> None:
    config = _config(roster=True, auth=auth)

    creds.ensure_telemetry_credentials(tmp_path, config, reader_keys=False)

    assert (HISTORY_FILES <= _files(tmp_path)) is present
    assert not (HISTORY_FILES & _files(tmp_path)) or present


def test_history_files_are_removed_when_the_page_goes(tmp_path: Path) -> None:
    creds.ensure_telemetry_credentials(
        tmp_path, _config(roster=True, auth="password"), reader_keys=False
    )
    gate = (tmp_path / creds.HISTORY_GATE_FILE).read_text()
    assert gate.startswith("OSPREY_TERMINAL_SECRET_HISTORY=")

    report = creds.ensure_telemetry_credentials(tmp_path, _config(roster=True), reader_keys=False)

    assert set(report.purged) == HISTORY_FILES
    assert not HISTORY_FILES & _files(tmp_path)


# --- the sender list ---------------------------------------------------------


def test_the_sender_list_verifies_each_login_as_sha(tmp_path: Path) -> None:
    config = _config(workers=1)
    creds.ensure_telemetry_credentials(tmp_path, config, reader_keys=False)

    lines = (tmp_path / creds.HTPASSWD_FILE).read_text().splitlines()

    assert [line.split(":", 1)[0] for line in lines] == ["builder", "dispatch-worker-1"]
    for line in lines:
        identity, digest = line.split(":", 1)
        login = creds.read_sender_login(tmp_path, identity)
        assert login is not None
        password = login.split(":", 1)[1]
        expected = base64.b64encode(hashlib.sha1(password.encode()).digest()).decode()
        assert digest == "{SHA}" + expected


def test_two_renders_of_the_sender_list_are_byte_equal(tmp_path: Path) -> None:
    config = _config(roster=True, workers=2)
    creds.ensure_telemetry_credentials(tmp_path, config, reader_keys=False)
    senders = creds.sender_identities(config)
    first = (tmp_path / creds.HTPASSWD_FILE).read_bytes()
    stamp = (tmp_path / creds.HTPASSWD_FILE).stat().st_mtime_ns

    creds.ensure_telemetry_credentials(tmp_path, config, reader_keys=False)

    assert (tmp_path / creds.HTPASSWD_FILE).read_bytes() == first
    assert (tmp_path / creds.HTPASSWD_FILE).stat().st_mtime_ns == stamp
    assert creds.render_htpasswd(tmp_path, senders) == first.decode()
    assert creds.render_htpasswd(tmp_path, reversed(senders)) == first.decode()


# --- reader keys -------------------------------------------------------------


def test_a_build_mints_no_reader_key(tmp_path: Path) -> None:
    report = creds.ensure_telemetry_credentials(
        tmp_path, _config(readers=("ops",)), reader_keys=False
    )

    assert creds.reader_file("ops") not in _files(tmp_path)
    assert report.reader_keys == []


def test_a_start_mints_each_readers_key_once(tmp_path: Path) -> None:
    config = _config(readers=("ops", "jean-luc"))

    report = creds.ensure_telemetry_credentials(tmp_path, config, reader_keys=True)

    assert report.reader_keys == ["ops", "jean-luc"]
    assert (tmp_path / creds.reader_file("ops")).read_text().startswith("password=")
    again = creds.ensure_telemetry_credentials(tmp_path, config, reader_keys=True)
    assert again.reader_keys == []


def test_a_withdrawn_readers_key_is_set_aside_and_never_revived(tmp_path: Path) -> None:
    day = datetime.date(2026, 3, 4)
    creds.ensure_telemetry_credentials(tmp_path, _config(readers=("ops",)), reader_keys=True)
    old = (tmp_path / creds.reader_file("ops")).read_text()

    report = creds.ensure_telemetry_credentials(tmp_path, _config(), reader_keys=True, today=day)

    aside = f"{creds.reader_file('ops')}.withdrawn-2026-03-04"
    assert report.withdrawn == [("ops", aside)]
    assert (tmp_path / aside).read_text() == old
    assert creds.reader_file("ops") not in _files(tmp_path)

    # The name comes back: a new key, and the one set aside stays where it is.
    back = creds.ensure_telemetry_credentials(
        tmp_path, _config(readers=("ops",)), reader_keys=True, today=day
    )
    assert back.reader_keys == ["ops"] and back.withdrawn == []
    assert (tmp_path / creds.reader_file("ops")).read_text() != old
    assert (tmp_path / aside).read_text() == old

    # Withdrawn again on the same day: a second dated name, the first untouched.
    twice = creds.ensure_telemetry_credentials(tmp_path, _config(), reader_keys=True, today=day)
    assert twice.withdrawn == [("ops", f"{aside}.2")]
    assert (tmp_path / aside).read_text() == old


def test_a_build_never_withdraws_a_reader(tmp_path: Path) -> None:
    creds.ensure_telemetry_credentials(tmp_path, _config(readers=("ops",)), reader_keys=True)

    report = creds.ensure_telemetry_credentials(tmp_path, _config(), reader_keys=False)

    assert report.withdrawn == []
    assert creds.reader_file("ops") in _files(tmp_path)


# --- recreate triggers -------------------------------------------------------


def test_each_file_moves_the_digest_of_its_owners_and_of_nothing_else(tmp_path: Path) -> None:
    config = _config(roster=True, workers=2, auth="password")
    creds.ensure_telemetry_credentials(tmp_path, config, reader_keys=False)
    owners = creds.owner_files(config)
    before = creds.digest_env(tmp_path, config)
    assert set(before) == {creds.digest_env_name(service) for service in owners}

    every_file = {name for files in owners.values() for name in files}
    assert every_file == _files(tmp_path) - {
        creds.sender_file(name) for name in ("alice", "bob", "builder")
    }
    for name in sorted(every_file):
        path = tmp_path / name
        original = path.read_bytes()
        path.write_bytes(original + b"#\n")
        after = creds.digest_env(tmp_path, config)
        path.write_bytes(original)

        moved = {var for var in before if before[var] != after[var]}
        expected = {creds.digest_env_name(svc) for svc, files in owners.items() if name in files}
        assert moved == expected, name

    assert creds.digest_env(tmp_path, config) == before


def test_the_owners_are_the_ones_the_design_names() -> None:
    owners = creds.owner_files(_config(workers=1))

    assert owners == {
        "telemetry_store": (creds.ADMIN_FILE,),
        "telemetry_gateway": (creds.INGEST_FILE, creds.HTPASSWD_FILE),
        "archive": (creds.ARCHIVE_FILE,),
        "history": (creds.HISTORY_FILE, creds.HISTORY_GATE_FILE),
        "nginx": (creds.HISTORY_GATE_FILE,),
        "dispatch-worker-1": (creds.sender_file("dispatch-worker-1"),),
    }
    assert creds.digest_env_name("dispatch-worker-1") == "OSPREY_TELEMETRY_DIGEST_DISPATCH_WORKER_1"


def test_a_missing_file_digests_as_empty_content(tmp_path: Path) -> None:
    empty = hashlib.sha256(b"").hexdigest()

    assert set(creds.digest_env(tmp_path, None).values()) == {empty}
    assert creds.files_digest(tmp_path, [creds.ADMIN_FILE]) == empty


def _web_services(config: dict, root: Path | None) -> dict:
    digests = creds.sender_digests(root, config) if root is not None else None
    rendered = render_web_terminals(config, telemetry_cred_digests=digests)
    return yaml.safe_load(rendered["docker-compose.web.yml"])["services"]


def test_each_terminal_is_handed_its_own_login_file_and_label(tmp_path: Path) -> None:
    config = _config(roster=True)
    creds.ensure_telemetry_credentials(tmp_path, config, reader_keys=False)

    services = _web_services(config, tmp_path)

    for user in ("alice", "bob"):
        service = services[f"web-{user}"]
        assert service["env_file"] == [".env.users", creds.sender_file(user)]
        assert service["labels"][creds.DIGEST_LABEL] == creds.files_digest(
            tmp_path, [creds.sender_file(user)]
        )
    carriers = [
        name
        for name, service in services.items()
        if any("telemetry" in str(entry) for entry in _as_list(service.get("env_file")))
    ]
    assert carriers == ["web-alice", "web-bob"]


def test_one_terminals_login_moves_only_that_terminals_label(tmp_path: Path) -> None:
    config = _config(roster=True)
    creds.ensure_telemetry_credentials(tmp_path, config, reader_keys=False)
    before = _web_services(config, tmp_path)

    (tmp_path / creds.sender_file("alice")).unlink()
    creds.ensure_telemetry_credentials(tmp_path, config, reader_keys=False)
    after = _web_services(config, tmp_path)

    changed = [name for name in before if before[name] != after[name]]
    assert changed == ["web-alice"]


def test_the_login_file_is_listed_whether_or_not_it_exists() -> None:
    """`status`, `down` and `logs` resolve the document on any built repo."""
    services = _web_services(_config(roster=True), None)

    assert services["web-alice"]["env_file"] == [".env.users", creds.sender_file("alice")]
    assert creds.DIGEST_LABEL not in services["web-alice"]["labels"]


def test_without_a_gateway_a_terminal_renders_as_before() -> None:
    services = _web_services(_config(roster=True, gateway=False), None)

    assert services["web-alice"]["env_file"] == ".env.users"
    assert creds.DIGEST_LABEL not in services["web-alice"]["labels"]


def _as_list(value: object) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


# --- host launches -----------------------------------------------------------


def test_a_host_launch_records_under_the_logins_name(tmp_path: Path) -> None:
    config = _config(host="builder")
    creds.ensure_telemetry_credentials(tmp_path, config, reader_keys=False)
    environ = {"OSPREY_AUDIT_IDENTITY": "someone-else"}

    names = creds.export_host_launch_identity(environ, tmp_path, config)

    assert names == ["OSPREY_AUDIT_IDENTITY", "OSPREY_TELEMETRY_LOGIN"]
    login = environ["OSPREY_TELEMETRY_LOGIN"]
    assert login and login.split(":", 1)[0] == environ["OSPREY_AUDIT_IDENTITY"] == "builder"


@pytest.mark.parametrize(
    "declared", [{"OSPREY_OTEL_GATEWAY_HOST": "telemetry_gateway"}, {"OSPREY_TERMINAL_USER": "al"}]
)
def test_a_rendered_container_keeps_what_compose_gave_it(declared: dict, tmp_path: Path) -> None:
    config = _config(host="builder")
    creds.ensure_telemetry_credentials(tmp_path, config, reader_keys=False)
    environ = {"OSPREY_AUDIT_IDENTITY": "dispatch-worker-1", **declared}
    before = dict(environ)

    assert creds.export_host_launch_identity(environ, tmp_path, config) == []
    assert environ == before


def test_a_config_no_build_stamped_exports_nothing(tmp_path: Path) -> None:
    environ: dict[str, str] = {}

    assert creds.export_host_launch_identity(environ, tmp_path, _config(host=None)) == []
    assert environ == {}


def test_a_login_filed_under_another_name_is_not_used(tmp_path: Path) -> None:
    (tmp_path / creds.sender_file("builder")).write_text("OSPREY_TELEMETRY_LOGIN=mallory:x\n")

    assert creds.read_sender_login(tmp_path, "builder") is None
    assert creds.host_launch_env(tmp_path, _config()) == {"OSPREY_AUDIT_IDENTITY": "builder"}


def _built_repo(root: Path, config: dict) -> None:
    (root / "profile.yml").write_text("project_name: p\n")
    (root / "build").mkdir()
    (root / "build" / "config.yml").write_text(yaml.safe_dump(config))
    creds.ensure_telemetry_credentials(root, config, reader_keys=False)


def test_the_built_render_is_where_a_launch_reads_its_identity(tmp_path: Path) -> None:
    _built_repo(tmp_path, _config(host="builder"))
    environ: dict[str, str] = {}

    creds.export_built_host_identity(environ, tmp_path)

    assert environ["OSPREY_AUDIT_IDENTITY"] == "builder"
    assert environ["OSPREY_TELEMETRY_LOGIN"].startswith("builder:")


def test_a_repo_with_no_render_exports_nothing(tmp_path: Path) -> None:
    environ: dict[str, str] = {}

    assert creds.export_built_host_identity(environ, tmp_path) == []
    assert environ == {}


@pytest.mark.parametrize(
    ("module", "anchor"),
    [
        ("chat_cmd", "_overlay_repo_env(repo_root)\n"),
        ("query_cmd", "_overlay_repo_env(repo_root)\n"),
        ("web_cmd", "load_dotenv_from_project()\n"),
    ],
)
def test_every_host_launch_builder_exports_the_identity(module: str, anchor: str) -> None:
    """Each launcher makes the one call, after the env chain is loaded.

    After, because the chain is loaded with override: an identity exported
    first could be replaced by a stale line in ``.env``.
    """
    import importlib
    import inspect

    source = inspect.getsource(importlib.import_module(f"osprey.cli.{module}"))
    call = "export_built_host_identity(os.environ, repo_root)"

    assert source.count(call) == 1
    assert source.rindex(anchor) < source.index(call)


def test_web_resolves_its_render_with_the_identity_exported(tmp_path: Path, monkeypatch) -> None:
    from osprey.cli import web_cmd

    _built_repo(tmp_path, _config(host="builder"))
    monkeypatch.chdir(tmp_path)
    # Set, not deleted: the call under test writes both, and only a variable
    # monkeypatch has touched is put back afterwards.
    for name in ("OSPREY_AUDIT_IDENTITY", "OSPREY_TELEMETRY_LOGIN"):
        monkeypatch.setenv(name, "")
    for name in ("OSPREY_TERMINAL_USER", "OSPREY_OTEL_GATEWAY_HOST"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OSPREY_CONFIG", "")

    web_cmd._resolve_render(tmp_path)

    import os

    login = creds.read_sender_login(tmp_path, "builder")
    assert os.environ["OSPREY_AUDIT_IDENTITY"] == "builder"
    assert os.environ["OSPREY_TELEMETRY_LOGIN"] == login


# --- the reporting pass ------------------------------------------------------


def test_the_probe_reports_a_missing_and_a_wrong_mode_file(tmp_path: Path) -> None:
    config = _config(workers=1, readers=("ops",))
    creds.ensure_telemetry_credentials(tmp_path, config, reader_keys=True)
    assert creds.credential_file_problems(tmp_path, config) == []

    (tmp_path / creds.INGEST_FILE).unlink()
    (tmp_path / creds.ADMIN_FILE).chmod(0o644)
    (tmp_path / creds.sender_file("builder")).write_text("")
    before = _snapshot(tmp_path)

    problems = [problem for problem, _ in creds.credential_file_problems(tmp_path, config)]

    assert _snapshot(tmp_path) == before
    assert any(p.startswith(f"{creds.INGEST_FILE} is missing") for p in problems)
    assert any(p.startswith(f"{creds.ADMIN_FILE} has mode 0644") for p in problems)
    assert any(p.startswith(f"{creds.sender_file('builder')} holds no") for p in problems)
    assert len(problems) == 3


def test_the_start_pass_reports_the_files_and_writes_nothing(tmp_path: Path) -> None:
    from osprey.deployment import container_lifecycle as lifecycle
    from osprey.deployment.errors import UnmetPreconditionsError

    config = _config(workers=1)
    config["project_root"] = str(tmp_path)
    creds.ensure_telemetry_credentials(tmp_path, config, reader_keys=True)
    lifecycle._collect_unmet_preconditions(
        config, tmp_path, {}, dev_mode=False, web_terminals_enabled=False
    )

    (tmp_path / creds.ARCHIVE_FILE).unlink()
    before = _snapshot(tmp_path)
    with pytest.raises(UnmetPreconditionsError) as raised:
        lifecycle._collect_unmet_preconditions(
            config, tmp_path, {}, dev_mode=False, web_terminals_enabled=False
        )

    assert f"{creds.ARCHIVE_FILE} is missing" in str(raised.value)
    assert _snapshot(tmp_path) == before


# --- what the commands say ---------------------------------------------------


def _secrets_in(root: Path) -> set[str]:
    """Every secret value a credential file under *root* holds."""
    values = set()
    for name in _files(root):
        for line in (root / name).read_text().splitlines():
            value = line.split("=", 1)[-1]
            values.add(value.split(":", 1)[-1])
    return {value for value in values if len(value) >= 20}


def test_a_start_names_files_and_never_a_value(tmp_path: Path, monkeypatch) -> None:
    from osprey.deployment import container_lifecycle as lifecycle

    said: list[str] = []
    monkeypatch.setattr(lifecycle, "_report_fact", lambda message, **_: said.append(message))
    config = _config(workers=1, readers=("ops",))

    lifecycle._ensure_telemetry_credentials(config, tmp_path)

    text = "\n".join(said)
    assert f"reader ops: key minted → {creds.reader_file('ops')}" in said
    assert any(line.startswith("reader ops: signs in at http://127.0.0.1:") for line in said)
    assert "/play?user=ops" in text
    secrets_minted = _secrets_in(tmp_path)
    assert len(secrets_minted) >= 5
    for value in secrets_minted:
        assert value not in text

    said.clear()
    lifecycle._ensure_telemetry_credentials(_config(workers=1), tmp_path)
    assert len(said) == 1 and said[0].startswith("reader ops: withdrawn, key file set aside → ")
    assert ".withdrawn-" in said[0]

    said.clear()
    lifecycle._ensure_telemetry_credentials(_config(workers=1), tmp_path)
    assert said == []


def test_reset_keeps_every_credential_file_and_lists_it(tmp_path: Path) -> None:
    config = _config(workers=1, readers=("ops",))
    creds.ensure_telemetry_credentials(tmp_path, config, reader_keys=True)
    creds.ensure_telemetry_credentials(
        tmp_path, _config(workers=1), reader_keys=True, today=datetime.date(2026, 3, 4)
    )

    listed = dict(creds.existing_credential_files(tmp_path))

    assert set(listed) == _files(tmp_path)
    assert listed[creds.sender_file("builder")] == "the telemetry login of builder"
    withdrawn = f"{creds.reader_file('ops')}.withdrawn-2026-03-04"
    assert "withdrawn telemetry reader ops" in listed[withdrawn]
    for value in _secrets_in(tmp_path):
        assert all(value not in holds for holds in listed.values())


def test_the_reset_plan_discloses_the_kept_files(tmp_path: Path) -> None:
    from osprey.deployment.reset import plan_reset

    config = _config(workers=1)
    config["project_root"] = str(tmp_path)
    (tmp_path / "profile.yml").write_text("project_name: p\n")
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "config.yml").write_text(yaml.safe_dump(config))
    creds.ensure_telemetry_credentials(tmp_path, config, reader_keys=False)
    before = _snapshot(tmp_path)

    plan = plan_reset(tmp_path, probe=_NoRuntime())

    kept = "\n".join(plan._kept_lines())
    for name in _files(tmp_path):
        assert name in kept
        assert tmp_path / name not in plan.paths
    assert "creates it again with a new secret" in kept
    assert _snapshot(tmp_path) == before


class _NoRuntime:
    """A container runtime that holds nothing of this project's."""

    def __getattr__(self, name: str):
        return lambda *args, **kwargs: []


def test_a_start_provisions_the_files_above_the_pass_that_reads_them() -> None:
    """Its own step: after the web tier's secrets are carried over, and above
    the reporting pass, whose probes only read."""
    import inspect

    from osprey.deployment import container_lifecycle as lifecycle

    source = inspect.getsource(lifecycle._start_stack)
    migrate = source.index("migrate_users_env(repo_root)")
    provision = source.index("_ensure_telemetry_credentials(config, repo_root)")
    report = source.index("_collect_unmet_preconditions(\n")

    assert migrate < provision < report
    assert "ensure_telemetry_credentials" not in inspect.getsource(
        lifecycle._collect_unmet_preconditions
    )
