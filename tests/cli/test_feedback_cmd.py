"""Tests for ``osprey feedback`` (``osprey.cli.feedback_cmd``).

The verbs read ``feedback`` entries from the agent record with the operator
login. Two layers are exercised:

* **A real host-local record.** A deployment repo with no stack and no host
  session: the verb starts the record itself, drains every spool (so lines
  still pending are in the answer), reads every user's submissions, and leaves
  no process behind (q-0002).
* **A fake record client** for how each answer is rendered and how each
  failure is reported, without a service.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from click.testing import CliRunner

from osprey.agent_record import credentials as creds
from osprey.agent_record import host as hr
from osprey.agent_record import writer
from osprey.agent_record.client import URL_ENV, ClientAnswer
from osprey.agent_record.contract import KIND_AUDIT, KIND_FEEDBACK, Drain, spool_key
from osprey.cli import feedback_cmd
from osprey.cli.feedback_cmd import feedback
from osprey_connectors_tests._harness._ports import reserved_port
from tests.agent_record.fakes import FakeRecordClient

HOST = "builder"
SESSION_ID = "3f2504e0-4f89-41d3-9a0c-0305e82c3301"


#: The port this test's record would bind, reserved for the test by
#: :func:`_record_port`: the config names it before any record starts.
_RECORD_PORT: list[int] = []


@pytest.fixture(autouse=True)
def _record_port() -> Iterator[None]:
    with reserved_port() as port:
        _RECORD_PORT[:] = [port]
        yield


def _config(root: Path, port: int) -> dict[str, Any]:
    return {
        "project_name": "fbcli",
        "project_root": str(root),
        "build_dir": "./build",
        "deployment": {"host_identity": HOST},
        "agent_record": {"port": port, "backend": "embedded"},
        "modules": {"web_terminals": {"enabled": True, "users": ["alice", "bob"]}},
    }


def _make_repo(tmp_path: Path, *, port: int | None = None) -> tuple[Path, dict[str, Any]]:
    """A deployment repo: a profile at the root, a rendered build, minted logins."""
    root = (tmp_path / "deployment").resolve()
    (root / "build").mkdir(parents=True)
    (root / "profile.yml").write_text("preset: hello-world\n", encoding="utf-8")
    config = _config(root, port or _RECORD_PORT[0])
    (root / "build" / "config.yml").write_text(yaml.safe_dump(config), encoding="utf-8")
    creds.ensure_agent_record_credentials(root, config, reader_keys=False)
    return root, config


def _body(entry_id: str, user: str, excerpt: str, **extra: Any) -> dict[str, Any]:
    return {
        "id": entry_id,
        "identity": user,
        "created_at": "2026-10-25T09:00:00+00:00",
        "channel": "local",
        "session_id": SESSION_ID,
        "excerpt": excerpt,
        "text": excerpt,
        "context": "## deployment\n",
        "scrollback": "",
        "refs": [],
        **extra,
    }


def _spool_as(monkeypatch: pytest.MonkeyPatch, root: Path, identity: str) -> None:
    monkeypatch.setenv("OSPREY_SPOOL_DIR", str(root / "var" / "spool" / identity))


@pytest.fixture
def no_stack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Nothing runs: no record URL in the environment, a Claude dir of the test's own."""
    claude = tmp_path / "claude"
    claude.mkdir()
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(claude))
    monkeypatch.delenv("OSPREY_CONFIG", raising=False)
    monkeypatch.delenv(URL_ENV, raising=False)
    yield


def _no_process_left(root: Path, config: dict[str, Any]) -> None:
    state_dir = hr.host_state_dir(root, config)
    try:
        assert hr._running_process(state_dir) is None, "an agent-record process was left"
        port = config["agent_record"]["port"]
        with pytest.raises(httpx.ConnectError):
            httpx.get(f"http://127.0.0.1:{port}/health", trust_env=False, timeout=2)
    finally:
        if state_dir.exists():
            hr.stop_host_process(state_dir, wait_s=10)


# ---------------------------------------------------------------------------
# A real host-local record
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("no_stack")
def test_list_on_a_no_stack_project_sees_every_users_pending_lines_and_leaves_no_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, config = _make_repo(tmp_path)
    _spool_as(monkeypatch, root, "alice")
    assert writer.write(KIND_FEEDBACK, _body("a1", "alice", "alice saw the orbit drift"))
    _spool_as(monkeypatch, root, "bob")
    assert writer.write(KIND_FEEDBACK, _body("b1", "bob", "bob found a typo"))
    # Not sealed and not consumed: the lines are still pending in open segments.
    monkeypatch.delenv("OSPREY_SPOOL_DIR")

    result = CliRunner().invoke(feedback, ["list", "--repo", str(root)])

    assert result.exit_code == 0, result.output
    assert "alice saw the orbit drift" in result.output
    assert "bob found a typo" in result.output
    assert "alice" in result.output and "bob" in result.output
    _no_process_left(root, config)


@pytest.mark.usefixtures("no_stack")
def test_export_resolves_the_referenced_steps_and_leaves_no_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, config = _make_repo(tmp_path)
    _spool_as(monkeypatch, root, "alice")
    step = writer.write(KIND_AUDIT, {"surface": "test", "note": "a step"}, session_id=SESSION_ID)
    assert step
    present = spool_key("alice", step)
    absent = spool_key("alice", "0" * 32)
    assert writer.write(KIND_FEEDBACK, _body("a1", "alice", "see the step", refs=[present, absent]))
    writer.seal_all()
    monkeypatch.delenv("OSPREY_SPOOL_DIR")
    out = tmp_path / "export.json"

    result = CliRunner().invoke(feedback, ["export", "--repo", str(root), "--output", str(out)])

    assert result.exit_code == 0, result.output
    (exported,) = json.loads(out.read_text(encoding="utf-8"))
    assert exported["id"] == "a1"
    assert exported["text"] == "see the step"
    resolved, missing = exported["entries"]
    assert resolved["key"] == present
    assert resolved["body"]["note"] == "a step"
    assert missing == {"key": absent, "missing": True}
    assert "1 referenced session step could not be resolved" in result.output
    _no_process_left(root, config)


# ---------------------------------------------------------------------------
# A fake record client
# ---------------------------------------------------------------------------


def _item(entry_id: str, user: str, excerpt: str, created_at: str, **extra: Any) -> dict:
    body = _body(entry_id, user, excerpt, **extra)
    body["created_at"] = created_at
    return {"key": f"sp:{user}:{entry_id}", "kind": KIND_FEEDBACK, "person": user, "body": body}


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    """Point the verbs at a fake client; ``state["client"]`` is what they read."""
    root, _config_ = _make_repo(tmp_path, port=1)
    state: dict[str, Any] = {"root": root, "client": FakeRecordClient()}

    @contextmanager
    def _client(deployment):
        state["deployment"] = deployment
        yield state["client"]

    monkeypatch.setattr(feedback_cmd, "_record_client", _client)
    yield state


def test_list_reads_feedback_entries_after_a_drain_of_every_spool(fake) -> None:
    fake["client"] = FakeRecordClient(
        {
            "entries": ClientAnswer(
                data=[
                    _item("old", "alice", "the older one", "2026-10-20T08:00:00+00:00"),
                    _item("new", "bob", "the newer one", "2026-10-25T08:00:00+00:00"),
                ],
                complete=True,
            )
        }
    )

    result = CliRunner().invoke(feedback, ["list", "--repo", str(fake["root"])])

    assert result.exit_code == 0, result.output
    (call,) = fake["client"].calls_to("entries")
    assert call["kind"] == KIND_FEEDBACK
    assert call["wait"] == Drain()
    assert result.output.index("the newer one") < result.output.index("the older one")
    assert "2026-10-25 08:00" in result.output


def test_list_says_so_when_nothing_was_sent(fake) -> None:
    result = CliRunner().invoke(feedback, ["list", "--repo", str(fake["root"])])

    assert result.exit_code == 0, result.output
    assert "No feedback has been sent" in result.output


def test_list_fails_when_the_record_does_not_answer(fake) -> None:
    fake["client"] = FakeRecordClient(unavailable="connection refused")

    result = CliRunner().invoke(feedback, ["list", "--repo", str(fake["root"])])

    assert result.exit_code != 0
    assert "did not answer" in result.output
    assert "connection refused" in result.output


def test_list_warns_when_the_drain_did_not_finish(fake) -> None:
    fake["client"] = FakeRecordClient(
        {
            "entries": ClientAnswer(
                data=[_item("a", "alice", "one", "2026-10-25T08:00:00+00:00")],
                complete=False,
                reason="ingest has not caught up",
            )
        }
    )

    result = CliRunner().invoke(feedback, ["list", "--repo", str(fake["root"])])

    assert result.exit_code == 0, result.output
    assert "ingest has not caught up" in result.output
    assert "one" in result.output


def test_list_cuts_a_long_excerpt(fake) -> None:
    fake["client"] = FakeRecordClient(
        {
            "entries": ClientAnswer(
                data=[_item("a", "alice", "word " * 40, "2026-10-25T08:00:00+00:00")],
                complete=True,
            )
        }
    )

    result = CliRunner().invoke(feedback, ["list", "--repo", str(fake["root"])])

    assert result.exit_code == 0, result.output
    assert "..." in result.output


def test_export_to_stdout_is_one_json_array(fake) -> None:
    fake["client"] = FakeRecordClient(
        {
            "entries": ClientAnswer(
                data=[_item("a", "alice", "one", "2026-10-25T08:00:00+00:00", refs=["sp:a:1"])],
                complete=True,
            ),
            "full_entry": ClientAnswer(data={"key": "sp:a:1", "body": {"x": 1}}, complete=True),
        }
    )

    result = CliRunner().invoke(feedback, ["export", "--repo", str(fake["root"])])

    assert result.exit_code == 0, result.output
    start = result.output.index("[")
    end = result.output.rindex("]") + 1
    (exported,) = json.loads(result.output[start:end])
    assert exported["key"] == "sp:alice:a"
    assert exported["person"] == "alice"
    assert exported["entries"] == [{"key": "sp:a:1", "body": {"x": 1}}]
    assert [c["key"] for c in fake["client"].calls_to("full_entry")] == ["sp:a:1"]


def test_export_with_nothing_sent_writes_an_empty_array(fake, tmp_path: Path) -> None:
    out = tmp_path / "empty.json"

    result = CliRunner().invoke(
        feedback, ["export", "--repo", str(fake["root"]), "--output", str(out)]
    )

    assert result.exit_code == 0, result.output
    assert json.loads(out.read_text(encoding="utf-8")) == []


def test_export_refuses_an_output_directory(fake, tmp_path: Path) -> None:
    result = CliRunner().invoke(
        feedback, ["export", "--repo", str(fake["root"]), "--output", str(tmp_path)]
    )

    assert result.exit_code != 0
    assert "is a directory" in result.output
    assert fake["client"].calls == []


# ---------------------------------------------------------------------------
# Finding the record
# ---------------------------------------------------------------------------


def test_a_repo_with_no_build_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "bare"
    root.mkdir()
    (root / "profile.yml").write_text("preset: hello-world\n", encoding="utf-8")

    result = CliRunner().invoke(feedback, ["list", "--repo", str(root)])

    assert result.exit_code != 0
    assert "osprey build" in result.output


def test_a_set_record_url_is_read_without_a_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, config = _make_repo(tmp_path)
    monkeypatch.setenv(URL_ENV, "http://agent-record:4318")

    def _never(*_a: Any, **_k: Any):
        raise AssertionError("no lease is taken when the URL is set")

    monkeypatch.setattr(hr, "ensure_host_record", _never)
    deployment = feedback_cmd._Deployment(root=root, config=config)

    with feedback_cmd._record_client(deployment) as client:
        assert client.url == "http://agent-record:4318"
        assert client.login == "operator"


def test_a_refused_host_record_names_the_fix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _config_ = _make_repo(tmp_path)
    monkeypatch.delenv(URL_ENV, raising=False)

    @contextmanager
    def _refuse(*_a: Any, **_k: Any):
        raise hr.HostRecordRefused("no stack answers: run `osprey up`")
        yield  # pragma: no cover

    monkeypatch.setattr(hr, "ensure_host_record", _refuse)

    result = CliRunner().invoke(feedback, ["list", "--repo", str(root)])

    assert result.exit_code != 0
    assert "osprey up" in result.output


def test_a_missing_operator_login_names_the_fix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _config_ = _make_repo(tmp_path)
    (root / creds.login_file("operator")).unlink()
    monkeypatch.setenv(URL_ENV, "http://agent-record:4318")

    result = CliRunner().invoke(feedback, ["list", "--repo", str(root)])

    assert result.exit_code != 0
    assert "operator" in result.output
    assert "osprey build" in result.output
