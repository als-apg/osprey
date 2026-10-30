"""Named readers read the agent record under ``Reader(name, kinds, days_back)``.

On the deploy host a reader reads the read API with the credential the deploy
minted for them (``.env.agent-record.reader.<name>``); from anywhere else they
read the history page, whose ``history`` login names the ``reader:<name>``
viewer the sidecar mapped their sign-in to. Both routes reach one scope: the
reader's kinds, within ``days_back`` of arrival, and nothing else.

Proven against a real agent-record service over an embedded backend, with the
logins and verifier the deploy mints from ``agent_record.readers``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from osprey.agent_record import contract as c
from osprey.agent_record import credentials as cr
from osprey.agent_record import identities as ids
from osprey.agent_record import service
from osprey.agent_record.backends.embedded import EmbeddedBackend
from osprey.agent_record.client import URL_ENV, RecordClient
from osprey.interfaces._serving import run_app_server
from osprey_connectors.dotenv import parse_dotenv_file
from tests.agent_record.test_identities import _config

#: Fixed at import, so every entry's age is whole days from the service's now.
NOW = datetime.now(UTC)
SESSION = "sess-named-readers"

#: Feedback and audit, the last two days only.
OPS = {"name": "ops", "kinds": [c.KIND_FEEDBACK, c.KIND_AUDIT], "days_back": 2}
#: No kinds and no window: every kind, as far back as the record keeps.
WIDE = {"name": "wide"}
#: Prompts only, no window.
PROMPTS = {"name": "prompts", "kinds": [c.KIND_PROMPT]}


def _entry(key: str, kind: str, days_ago: float, **fields) -> c.Entry:
    at = NOW - timedelta(days=days_ago)
    return c.Entry(
        key=key,
        kind=kind,
        ts=at,
        identity="alice",
        body={"n": key},
        inserted_at=at,
        **fields,
    )


ENTRIES = [
    _entry("sp:alice:fb-new", c.KIND_FEEDBACK, 1, session_id=SESSION),
    _entry("sp:alice:fb-edge", c.KIND_FEEDBACK, 3, session_id=SESSION),
    _entry("sp:alice:fb-old", c.KIND_FEEDBACK, 6, session_id=SESSION),
    _entry("sp:alice:au-new", c.KIND_AUDIT, 0.5, session_id=SESSION),
    _entry("sp:alice:au-old", c.KIND_AUDIT, 4, session_id=SESSION),
    _entry(f"cc:alice:{SESSION}:u1:0", c.KIND_PROMPT, 1, session_id=SESSION),
    _entry(f"cc:alice:{SESSION}:u2:0", c.KIND_ANSWER, 1, session_id=SESSION),
]


@pytest.fixture(scope="module")
def deployed(tmp_path_factory) -> dict:
    """The minted credentials and a running service reading them."""
    repo = tmp_path_factory.mktemp("repo")
    config = _config(readers=[OPS, WIDE, PROMPTS])
    config["project_root"] = str(repo)
    report = cr.ensure_agent_record_credentials(repo, config, reader_keys=True)
    assert set(report.reader_keys) == {"ops", "wide", "prompts"}

    service_env = parse_dotenv_file(repo / cr.SERVICE_FILE)
    root = tmp_path_factory.mktemp("service")
    env = {
        cr.VERIFIER_ENV: service_env[cr.VERIFIER_ENV],
        cr.SECRETS_ENV: service_env[cr.SECRETS_ENV],
        service.SPOOL_ROOT_ENV: str(root / "spool"),
        service.RECORD_DIR_ENV: str(root / "record"),
        service.TRANSCRIPTS_DIR_ENV: str(root / "transcripts"),
    }
    with EmbeddedBackend(root / "record.duckdb") as backend:
        backend.append(ENTRIES)
        app = service.create_app(config=config, backend=backend, env=env)
        with run_app_server(app) as url:
            yield {"repo": repo, "url": url}


def _on_host(deployed: dict, name: str) -> RecordClient:
    """The client a reader builds on the deploy host from their minted file."""
    login_pair = parse_dotenv_file(deployed["repo"] / cr.reader_file(name))[cr.LOGIN_ENV]
    return RecordClient.from_env({URL_ENV: deployed["url"], cr.LOGIN_ENV: login_pair})


def _off_host(deployed: dict, name: str) -> RecordClient:
    """The history page's client, naming the viewer the sidecar mapped the sign-in to."""
    login = ids.HISTORY_IDENTITY
    token = cr.read_login(deployed["repo"] / cr.login_file(login), login)
    assert token
    return RecordClient(deployed["url"], login, token, viewer=ids.reader_identity(name))


def _keys(answer) -> list[str]:
    assert answer.unavailable is None, answer.unavailable
    return sorted(item["key"] for item in answer.data)


_ROUTES = [pytest.param(_on_host, id="on-host"), pytest.param(_off_host, id="history-page")]


def test_the_reader_logs_in_with_the_key_the_deploy_minted(deployed: dict) -> None:
    pair = parse_dotenv_file(deployed["repo"] / cr.reader_file("ops"))[cr.LOGIN_ENV]
    assert pair.startswith(f"{ids.login_for(ids.reader_identity('ops'))}:")
    assert ids.credential_table(_config(readers=[OPS]))["reader.ops"].scope == c.Reader(
        "ops", frozenset({c.KIND_FEEDBACK, c.KIND_AUDIT}), 2
    )


@pytest.mark.parametrize("route", _ROUTES)
def test_a_reader_sees_its_kinds_within_days_back(deployed: dict, route) -> None:
    ops = route(deployed, "ops")
    assert _keys(ops.entries(c.KIND_FEEDBACK)) == ["sp:alice:fb-new"]
    assert _keys(ops.entries(c.KIND_AUDIT)) == ["sp:alice:au-new"]


@pytest.mark.parametrize("route", _ROUTES)
def test_a_reader_sees_nothing_outside_its_kinds(deployed: dict, route) -> None:
    ops = route(deployed, "ops")
    for kind in (c.KIND_PROMPT, c.KIND_ANSWER, c.KIND_DEBUG):
        assert _keys(ops.entries(kind)) == [], kind
    timeline = ops.timeline(SESSION)
    assert timeline.unavailable is None, timeline.unavailable
    flat = repr(timeline.data)
    assert "u1:0" not in flat and "u2:0" not in flat


@pytest.mark.parametrize("route", _ROUTES)
def test_a_reader_without_a_window_sees_everything_the_record_keeps(deployed: dict, route) -> None:
    wide = route(deployed, "wide")
    assert _keys(wide.entries(c.KIND_FEEDBACK)) == [
        "sp:alice:fb-edge",
        "sp:alice:fb-new",
        "sp:alice:fb-old",
    ]
    assert _keys(wide.entries(c.KIND_PROMPT)) == [f"cc:alice:{SESSION}:u1:0"]


@pytest.mark.parametrize("route", _ROUTES)
def test_a_reader_with_one_kind_reads_that_kind_alone(deployed: dict, route) -> None:
    prompts = route(deployed, "prompts")
    assert _keys(prompts.entries(c.KIND_PROMPT)) == [f"cc:alice:{SESSION}:u1:0"]
    assert _keys(prompts.entries(c.KIND_FEEDBACK)) == []
    assert _keys(prompts.entries(c.KIND_ANSWER)) == []


def test_one_readers_key_is_not_anothers(deployed: dict) -> None:
    """The ``ops`` key under ``wide``'s login is refused, not widened."""
    pair = parse_dotenv_file(deployed["repo"] / cr.reader_file("ops"))[cr.LOGIN_ENV]
    _login, _, token = pair.partition(":")
    forged = RecordClient(deployed["url"], "reader.wide", token)
    assert forged.entries(c.KIND_FEEDBACK).unavailable is not None


def test_a_reader_login_cannot_name_a_viewer(deployed: dict) -> None:
    """Only the history page asserts a viewer; a reader reads its own scope."""
    pair = parse_dotenv_file(deployed["repo"] / cr.reader_file("ops"))[cr.LOGIN_ENV]
    login, _, token = pair.partition(":")
    widened = RecordClient(deployed["url"], login, token, viewer=ids.reader_identity("wide"))
    assert widened.entries(c.KIND_PROMPT).unavailable is not None


def test_the_reader_scope_is_the_configured_one_and_no_wider() -> None:
    """``kinds`` left out is every kind; ``days_back`` left out is no window."""
    table = ids.credential_table(_config(readers=[OPS, WIDE, PROMPTS]))
    assert table["reader.wide"].scope == c.Reader("wide", frozenset(c.KINDS), None)
    assert table["reader.prompts"].scope == c.Reader("prompts", frozenset({c.KIND_PROMPT}), None)
