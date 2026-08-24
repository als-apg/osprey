"""Tests for the readers behind ``osprey feedback``.

The first half covers the concatenated-JSON stream parser. ``osprey feedback``
reads a store by concatenating its record files -- either inside an ephemeral
container mounted on a user's agent-data volume, or off the host filesystem.
What comes back is a run of JSON objects with no framing, delivered in chunks
that respect neither document nor character boundaries. Those tests feed
:func:`iter_json_documents` a chunking a real pipe could produce and pin what
it yields, when it yields it, and which byte offset it names when the stream
does not parse.

The rest covers the three readers built on it: the container reader (which
volumes it will and will not mount, and how a failed read is reported), the
host-filesystem reader for a single-user deployment, and the pairing pass that
joins a header to its session context on the way to an export file.
"""

from __future__ import annotations

import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import click
import pytest
import yaml
from click.testing import CliRunner

from osprey.cli import feedback_cmd
from osprey.cli.feedback_cmd import (
    FeedbackReadError,
    feedback,
    iter_json_documents,
    local_feedback_dir,
    pair_records,
    read_local_headers,
    read_volume_headers,
    stream_local_contexts,
    stream_volume_contexts,
)
from osprey.deployment.web_terminals import lifecycle

_PROJECT = "demo"
_IMAGE = "registry.example.org/physics/web-terminal:latest"


def _config(users: list[str]) -> dict[str, Any]:
    """A rendered config carrying a web-terminal roster and nothing else."""
    return {
        "project_name": _PROJECT,
        "facility": {"prefix": "dls"},
        "registry": {"url": "registry.example.org/physics"},
        "modules": {"web_terminals": {"enabled": True, "users": list(users)}},
    }


def _agent_data(user: str) -> str:
    """The runtime name of ``user``'s agent-data volume in ``_PROJECT``."""
    return f"{_PROJECT}_{user}-agent-data"


def _mounted_volume(argv: list[str]) -> str | None:
    """Return the volume name a ``run`` argv mounts, if it mounts one."""
    for index, token in enumerate(argv):
        if token == "--mount" and index + 1 < len(argv):
            for field_ in argv[index + 1].split(","):
                if field_.startswith("source="):
                    return field_[len("source=") :]
    return None


@dataclass
class _FakeRuntime:
    """A scripted container runtime: what it lists, and what each store holds."""

    calls: list[list[str]] = field(default_factory=list)
    listed: list[str] = field(default_factory=list)
    #: volume name -> (exit code, stdout bytes, stderr bytes) for a store read.
    stores: dict[str, tuple[int, bytes, bytes]] = field(default_factory=dict)
    list_exit: int = 0

    def runs(self) -> list[list[str]]:
        """Every ``<runtime> run`` argv the code under test issued."""
        return [argv for argv in self.calls if argv[1:2] == ["run"]]


@pytest.fixture
def fake_runtime(monkeypatch: pytest.MonkeyPatch) -> _FakeRuntime:
    """Stand a recording double in for docker/podman, daemon reported up."""
    runtime = _FakeRuntime()

    def _fake_run(argv, capture_output=False, text=False, env=None, check=False):
        runtime.calls.append(list(argv))
        if argv[1:3] == ["volume", "ls"]:
            listing = "\n".join(runtime.listed)
            return subprocess.CompletedProcess(argv, runtime.list_exit, listing, "")
        if argv[1:2] == ["run"]:
            code, stdout, stderr = runtime.stores.get(_mounted_volume(argv) or "", (3, b"", b""))
            return subprocess.CompletedProcess(argv, code, stdout, stderr)
        raise AssertionError(f"unexpected subprocess call: {argv}")

    monkeypatch.setattr(feedback_cmd.subprocess, "run", _fake_run)
    monkeypatch.setattr(
        feedback_cmd, "get_runtime_command", lambda config=None: ["docker", "compose"]
    )
    monkeypatch.setattr(feedback_cmd, "runtime_env", lambda config, *args, **kwargs: {"E": "1"})
    monkeypatch.setattr(lifecycle, "verify_runtime_is_running", lambda config=None: (True, ""))
    return runtime


class _FakePopen:
    """A ``Popen`` double whose stdout is a fixed byte string."""

    def __init__(self, argv, stdout=None, stderr=None, env=None):
        self.args = list(argv)
        if argv[1:2] != ["run"]:
            raise AssertionError(f"unexpected subprocess call: {argv}")
        self._sink = stderr
        volume = _mounted_volume(self.args) or ""
        if _FakePopen.start_error is not None:
            raise _FakePopen.start_error
        code, payload, error = _FakePopen.scripted.get(volume, (3, b"", b""))
        self.stdout = io.BytesIO(payload)
        self._code = code
        self._error = error
        self.returncode: int | None = None
        self.killed = False
        _FakePopen.calls.append(list(argv))
        _FakePopen.live.append(self)

    def wait(self, timeout: float | None = None) -> int:
        """Report the scripted exit status, spilling stderr into the sink.

        A child scripted to hang refuses the bounded wait once, the way a
        container ignoring its broken output pipe would, and afterwards
        reports the negative status a killed process really carries.
        """
        if timeout is not None and _FakePopen.hangs and not self.killed:
            raise subprocess.TimeoutExpired(self.args, timeout)
        if self.killed:
            self.returncode = -9
            return self.returncode
        if self._error and self._sink is not None:
            self._sink.write(self._error)
        self.returncode = self._code
        return self._code

    def kill(self) -> None:
        """Record the kill that ends a hung read."""
        self.killed = True

    calls: list[list[str]] = []
    live: list[_FakePopen] = []
    scripted: dict[str, tuple[int, bytes, bytes]] = {}
    start_error: OSError | None = None
    hangs: bool = False


@pytest.fixture
def fake_popen(monkeypatch: pytest.MonkeyPatch) -> type[_FakePopen]:
    """Install the streaming double and reset its script between tests."""
    _FakePopen.calls = []
    _FakePopen.live = []
    _FakePopen.scripted = {}
    _FakePopen.start_error = None
    _FakePopen.hangs = False
    monkeypatch.setattr(feedback_cmd.subprocess, "Popen", _FakePopen)
    return _FakePopen


def _one_byte_chunks(data: bytes) -> list[bytes]:
    """Split ``data`` into single-byte chunks -- the worst chunking possible."""
    return [data[index : index + 1] for index in range(len(data))]


def _flat(output: str) -> str:
    """Collapse Rich's line wrapping so a rendered cell matches as one string."""
    return " ".join(re.sub(r"\x1b\[[0-9;]*m", "", output).split())


def _header_bytes(*headers: dict[str, Any]) -> bytes:
    """Serialize headers the way a store read concatenates them."""
    return "".join(json.dumps(header) for header in headers).encode("utf-8")


def _make_repo(tmp_path: Path, config: dict[str, Any], *, name: str = "deployment") -> Path:
    """Create a deployment repo: a profile at the root and a rendered build."""
    repo_root = tmp_path / name
    (repo_root / "build").mkdir(parents=True)
    (repo_root / "profile.yml").write_text("preset: control-assistant\n", encoding="utf-8")
    (repo_root / "build" / "config.yml").write_text(yaml.safe_dump(config), encoding="utf-8")
    return repo_root.resolve()


def _single_user_config() -> dict[str, Any]:
    """A deployment with no roster: one store, on this machine's disk."""
    return {"project_name": _PROJECT, "agent_data": {"base_dir": "var/agent_data"}}


def _write_store(repo_root: Path, *, headers: dict[str, str], contexts: dict[str, str]) -> Path:
    """Write a host-side feedback store under the repo's agent-data root."""
    store = repo_root / "var" / "agent_data" / "feedback"
    store.mkdir(parents=True, exist_ok=True)
    for name, payload in {**headers, **contexts}.items():
        (store / name).write_text(payload, encoding="utf-8")
    return store


@pytest.fixture
def cli_runner() -> CliRunner:
    """A runner with a wide terminal, so no table cell is wrapped mid-token."""
    return CliRunner(env={"COLUMNS": "200", "TERM": "dumb"})


class TestStreamParsing:
    """Well-formed streams: what comes out, and how soon."""

    def test_stream_yields_two_concatenated_documents(self):
        """The base case: two objects run together with no separator."""
        stream = [b'{"id": "fb-1", "channel": "local"}{"id": "fb-2", "channel": "github"}']

        assert list(iter_json_documents(stream)) == [
            {"id": "fb-1", "channel": "local"},
            {"id": "fb-2", "channel": "github"},
        ]

    def test_stream_joins_a_document_split_across_a_chunk_boundary(self):
        """A chunk boundary inside a document is not a document boundary."""
        stream = [b'{"id": "fb-1", "co', b'unt": 2}{"id": "fb-2"}']

        assert list(iter_json_documents(stream)) == [
            {"id": "fb-1", "count": 2},
            {"id": "fb-2"},
        ]

    def test_stream_survives_a_boundary_at_every_single_byte(self):
        """Byte-at-a-time delivery exercises every partial-document state.

        A truncated ``true``/``null`` literal fails to parse in a way that
        looks nothing like a truncated object, so the fixture carries both.
        """
        documents = [
            {"id": "fb-1", "context": None, "outbound": True},
            {"id": "fb-2", "flags": [1, 2, 3], "excerpt": "}{ not a boundary"},
            {"id": "fb-3", "nested": {"a": {"b": False}}},
        ]
        raw = "".join(json.dumps(document) for document in documents).encode("utf-8")

        assert list(iter_json_documents(_one_byte_chunks(raw))) == documents

    def test_stream_skips_whitespace_between_documents(self):
        """Newlines, tabs and spaces around documents are not documents."""
        stream = [b'\n  {"id": "fb-1"}\r\n\n\t{"id": "fb-2"}  \n']

        assert list(iter_json_documents(stream)) == [{"id": "fb-1"}, {"id": "fb-2"}]

    def test_stream_of_whitespace_only_yields_nothing(self):
        """An empty store directory yields blank output, not an error."""
        assert list(iter_json_documents([b"   \n\t\r\n"])) == []

    def test_stream_of_empty_input_yields_nothing(self):
        """No chunks at all, and a single empty chunk, are both a clean end."""
        assert list(iter_json_documents([])) == []
        assert list(iter_json_documents([b""])) == []

    def test_stream_joins_a_multibyte_character_split_across_chunks(self):
        """A chunk may end mid-character; UTF-8 is decoded incrementally."""
        raw = '{"excerpt": "the µ-beam drifted 3°"}'.encode()
        split = raw.index(b"\xc2\xb5") + 1  # between the two bytes of "µ"

        assert list(iter_json_documents([raw[:split], raw[split:]])) == [
            {"excerpt": "the µ-beam drifted 3°"}
        ]

    def test_stream_yields_a_document_before_the_next_chunk_is_pulled(self):
        """Records are yielded as they complete, not collected at EOF.

        ``export`` writes each record out as it arrives while the container is
        still ``cat``-ing the store, so the parser must not be a two-pass read.
        """
        pulled: list[int] = []

        def chunks() -> Iterator[bytes]:
            for index, chunk in enumerate([b'{"id": "fb-1"}', b'{"id": "fb-2"}']):
                pulled.append(index)
                yield chunk

        stream = iter_json_documents(chunks())

        assert next(stream) == {"id": "fb-1"}
        assert pulled == [0], "the second chunk was pulled before the first record came out"


class TestStreamFailures:
    """Malformed streams: every message names a byte offset."""

    def test_stream_truncated_document_names_the_byte_offset(self):
        """A store cut off mid-record fails loudly, pointing at the record."""
        stream = [b'{"id": "fb-1"}{"id": "fb-2", "excerp']

        with pytest.raises(FeedbackReadError, match=r"byte offset 14") as caught:
            list(iter_json_documents(stream))

        assert "fb-1" not in str(caught.value), "the error should name offsets, not payloads"

    def test_stream_truncation_is_reported_after_the_records_that_did_parse(self):
        """The records before the damage are still delivered."""
        stream = iter_json_documents([b'{"id": "fb-1"}{"id": ', b'"fb-2"'])

        assert next(stream) == {"id": "fb-1"}
        with pytest.raises(FeedbackReadError):
            next(stream)

    def test_stream_malformed_document_names_the_byte_offset(self):
        """Corruption inside a record is reported at the record's offset."""
        stream = [b'{"id": "fb-1"}{"id": , }']

        with pytest.raises(FeedbackReadError, match=r"byte offset 14"):
            list(iter_json_documents(stream))

    def test_stream_trailing_garbage_names_the_byte_offset(self):
        """Bytes after the last record that start no document are an error."""
        stream = [b'{"id": "fb-1"} not json']

        with pytest.raises(FeedbackReadError, match=r"byte offset 15"):
            list(iter_json_documents(stream))

    def test_stream_offsets_count_bytes_not_characters(self):
        """Offsets are stream byte offsets, so a caller can seek on them."""
        first = '{"excerpt": "3µm, 2°C, café"}'.encode()
        assert len(first) > len(first.decode("utf-8")), "fixture must be non-ASCII"

        with pytest.raises(FeedbackReadError, match=rf"byte offset {len(first)}\b"):
            list(iter_json_documents([first + b'{"id": ']))

    def test_stream_rejects_a_non_object_document(self):
        """The store only ever holds objects; anything else is corruption."""
        stream = [b'{"id": "fb-1"}[1, 2, 3]']

        with pytest.raises(FeedbackReadError, match=r"byte offset 14") as caught:
            list(iter_json_documents(stream))

        assert "list" in str(caught.value)

    def test_stream_rejects_a_bare_scalar_document(self):
        """A truncated header must not be mistaken for a valid number."""
        with pytest.raises(FeedbackReadError, match=r"byte offset 0"):
            list(iter_json_documents([b"1234"]))

    def test_stream_rejects_invalid_utf8_at_its_byte_offset(self):
        """A non-UTF-8 byte is named where it sits, not where its record does."""
        stream = [b'{"excerpt": "\xff"}']

        with pytest.raises(FeedbackReadError, match=r"UTF-8 at byte offset 13"):
            list(iter_json_documents(stream))

    def test_stream_rejects_a_multibyte_character_truncated_at_eof(self):
        """A stream ending mid-character is truncation, not a clean end."""
        stream = [b'{"excerpt": "3\xc2']

        with pytest.raises(FeedbackReadError, match=r"UTF-8 at byte offset 14"):
            list(iter_json_documents(stream))


class TestVolumeReader:
    """The container reader: which volumes it mounts, and how reads fail."""

    def test_volume_read_fails_loudly_when_the_runtime_daemon_is_down(
        self, fake_runtime: _FakeRuntime, monkeypatch: pytest.MonkeyPatch
    ):
        """A stopped daemon must not read as "this deployment has no feedback"."""
        monkeypatch.setattr(
            lifecycle,
            "verify_runtime_is_running",
            lambda config=None: (False, "Docker daemon is not running"),
        )

        with pytest.raises(RuntimeError, match="Docker daemon is not running"):
            read_volume_headers(_config(["alice"]))

        assert fake_runtime.calls == [], "nothing may be listed or mounted after the precheck"

    def test_volume_read_asserts_each_roster_volume_before_mounting_it(
        self, fake_runtime: _FakeRuntime
    ):
        """An unlisted volume is a per-user error, never a mount.

        ``--mount type=volume`` creates a missing volume on the spot, so
        mounting an unverified name would fabricate an empty store and litter
        the host instead of reporting that the user has never been brought up.
        """
        fake_runtime.listed = [_agent_data("bob")]

        result = read_volume_headers(_config(["alice", "bob"]))

        assert [error.user for error in result.errors] == ["alice"]
        assert _agent_data("alice") in result.errors[0].message
        # Wording pin: the volume is created by the container start, which
        # `osprey up` performs -- nobody has to open a browser for it. Blaming
        # it on a terminal nobody opened sends an admin looking in the wrong
        # place, and the how-to tells the same story.
        assert "container first starts" in result.errors[0].message
        assert "osprey up" in result.errors[0].message
        assert [_mounted_volume(argv) for argv in fake_runtime.runs()] == [_agent_data("bob")]

    def test_volume_read_mounts_read_only_by_name_with_the_long_form_flag(
        self, fake_runtime: _FakeRuntime
    ):
        """The mount is long-form, read-only, and made with the user's image."""
        fake_runtime.listed = [_agent_data("alice")]

        read_volume_headers(_config(["alice"]))

        argv = fake_runtime.runs()[0]
        assert argv[:3] == ["docker", "run", "--rm"]
        assert "--mount" in argv
        mount = argv[argv.index("--mount") + 1]
        assert mount == f"type=volume,source={_agent_data('alice')},destination=/data,readonly"
        assert _IMAGE in argv
        assert not any(token.startswith("-v") and ":" in token for token in argv)
        script = argv[-1]
        assert script.startswith("[ -d /data/feedback ] || exit 3;")
        assert "find /data/feedback -maxdepth 1" in script
        assert '-name "fb-*.json"' in script
        assert script.endswith("-exec cat {} +")
        assert "ctx-" not in script

    def test_volume_read_names_the_agent_data_volume_the_deployment_derives(self):
        """The suffix this reader filters on is the one compose actually uses."""
        from osprey.deployment.compose_generator import resolve_user_volume_names

        claude_config, agent_data = resolve_user_volume_names(_config(["alice"]), "alice")

        assert agent_data.endswith(feedback_cmd._AGENT_DATA_SUFFIX)
        assert not claude_config.endswith(feedback_cmd._AGENT_DATA_SUFFIX)

    def test_volume_read_never_mounts_a_claude_config_volume(self, fake_runtime: _FakeRuntime):
        """The claude-config volume holds session state and is never read."""
        fake_runtime.listed = [_agent_data("alice"), f"{_PROJECT}_alice-claude-config"]

        read_volume_headers(_config(["alice"]))

        assert not any("claude-config" in " ".join(argv) for argv in fake_runtime.runs())

    def test_volume_read_returns_the_headers_the_store_holds(self, fake_runtime: _FakeRuntime):
        """A clean read yields every header, tagged with the volume it came from."""
        fake_runtime.listed = [_agent_data("alice")]
        fake_runtime.stores[_agent_data("alice")] = (0, b'{"id": "fb-1"}{"id": "fb-2"}', b"")

        result = read_volume_headers(_config(["alice"]))

        assert result.errors == []
        assert [volume.user for volume in result.volumes] == ["alice"]
        assert result.volumes[0].headers == [{"id": "fb-1"}, {"id": "fb-2"}]
        assert [target.volume for target in result.targets] == [_agent_data("alice")]

    def test_volume_read_treats_the_no_store_exit_as_zero_records(self, fake_runtime: _FakeRuntime):
        """Exit 3 means the user never submitted anything -- benign, not an error."""
        fake_runtime.listed = [_agent_data("alice")]
        fake_runtime.stores[_agent_data("alice")] = (3, b"", b"")

        result = read_volume_headers(_config(["alice"]))

        assert result.errors == []
        assert result.volumes[0].headers == []

    def test_volume_read_reports_a_failed_read_with_the_volume_name_and_stderr(
        self, fake_runtime: _FakeRuntime
    ):
        """Any other non-zero exit is a loud per-volume error, never exit 3's silence."""
        fake_runtime.listed = [_agent_data("alice"), _agent_data("bob")]
        fake_runtime.stores[_agent_data("alice")] = (
            1,
            b"",
            b"find: /data/feedback: Permission denied",
        )
        fake_runtime.stores[_agent_data("bob")] = (0, b'{"id": "fb-9"}', b"")

        result = read_volume_headers(_config(["alice", "bob"]))

        assert len(result.errors) == 1
        message = result.errors[0].message
        assert _agent_data("alice") in message
        assert "Permission denied" in message
        assert result.volumes[1].headers == [{"id": "fb-9"}], (
            "one bad volume must not hide the rest"
        )

    def test_volume_read_keeps_the_records_a_failed_read_did_produce(
        self, fake_runtime: _FakeRuntime
    ):
        """A read that dies part-way still wrote real records; they are kept."""
        fake_runtime.listed = [_agent_data("alice")]
        fake_runtime.stores[_agent_data("alice")] = (
            1,
            b'{"id": "fb-1"}{"id": "fb-2"}{"id": ',
            b"find: /data/feedback/fb-3.json: Input/output error",
        )

        result = read_volume_headers(_config(["alice"]))

        assert result.volumes[0].headers == [{"id": "fb-1"}, {"id": "fb-2"}]
        assert len(result.errors) == 1
        assert "Input/output error" in result.errors[0].message

    def test_volume_read_reports_a_corrupt_store_and_keeps_what_parsed(
        self, fake_runtime: _FakeRuntime
    ):
        """Corruption surfaces at the end of the stream; earlier records survive."""
        fake_runtime.listed = [_agent_data("alice")]
        fake_runtime.stores[_agent_data("alice")] = (0, b'{"id": "fb-1"}{"id": ', b"")

        result = read_volume_headers(_config(["alice"]))

        assert result.volumes[0].headers == [{"id": "fb-1"}]
        assert len(result.errors) == 1
        assert "byte offset" in result.errors[0].message

    def test_volume_listing_failure_is_reported_loudly(self, fake_runtime: _FakeRuntime):
        """A failed ``volume ls`` cannot read as "this deployment has no volumes"."""
        fake_runtime.list_exit = 125

        with pytest.raises(RuntimeError, match="volume ls"):
            read_volume_headers(_config(["alice"]))

    def test_volume_read_picks_up_an_orphan_agent_data_volume(self, fake_runtime: _FakeRuntime):
        """A decommissioned user's volume still holds feedback worth reading."""
        fake_runtime.listed = [
            _agent_data("alice"),
            _agent_data("zoe"),
            f"{_PROJECT}_zoe-claude-config",
        ]
        fake_runtime.stores[_agent_data("zoe")] = (0, b'{"id": "fb-z"}', b"")

        result = read_volume_headers(_config(["alice"]))

        orphans = [target for target in result.targets if not target.on_roster]
        assert [target.user for target in orphans] == ["zoe"]
        assert orphans[0].image == _IMAGE, "an orphan is read with the first resolved image"
        assert {volume.user: volume.headers for volume in result.volumes}["zoe"] == [{"id": "fb-z"}]

    def test_volume_module_never_spools_reads_through_the_capture_helper(self):
        """Session context must never reach a host log file.

        The deploy tree's spooling capture helper merges a child's output into
        ``<repo>/var/logs/``. A feedback read carries whole sessions, so this
        module reads through plain ``subprocess`` and must not name that helper
        at all -- not in an import, and not in a call.
        """
        source = Path(feedback_cmd.__file__).read_text(encoding="utf-8")

        assert "run_captured" not in source
        assert "subprocess_capture" not in source


class TestVolumeContextStream:
    """Streaming a volume's session contexts without buffering the store."""

    def test_stream_volume_contexts_yields_each_document_as_it_arrives(
        self, fake_runtime: _FakeRuntime, fake_popen: type[_FakePopen]
    ):
        """Contexts come back one document at a time from a piped read."""
        config = _config(["alice"])
        fake_runtime.listed = [_agent_data("alice")]
        fake_popen.scripted[_agent_data("alice")] = (
            0,
            b'{"id": "fb-1", "context": "a"}{"id": "fb-2", "context": "b"}',
            b"",
        )
        targets = read_volume_headers(config).targets

        documents = list(stream_volume_contexts(config, targets))

        assert documents == [
            {"id": "fb-1", "context": "a"},
            {"id": "fb-2", "context": "b"},
        ]
        script = fake_popen.calls[0][-1]
        assert script.startswith("[ -d /data/feedback ] || exit 3;")
        assert "find /data/feedback -maxdepth 1" in script
        assert '-name "ctx-*.json"' in script
        assert script.endswith("-exec cat {} +")

    def test_stream_volume_contexts_kills_a_container_that_will_not_exit(
        self, fake_runtime: _FakeRuntime, fake_popen: type[_FakePopen]
    ):
        """Closing the pipe ends the read; a child ignoring that is killed.

        Without the bound, one wedged container would hang the export forever.
        The kill is this command's own doing after a complete read, so the
        negative exit status it leaves behind is not reported as a failed read.
        """
        config = _config(["alice"])
        fake_runtime.listed = [_agent_data("alice")]
        fake_popen.scripted[_agent_data("alice")] = (0, b'{"id": "fb-1"}', b"")
        targets = read_volume_headers(config).targets
        fake_popen.hangs = True
        errors: list[feedback_cmd.VolumeError] = []

        documents = list(stream_volume_contexts(config, targets, on_error=errors.append))

        assert documents == [{"id": "fb-1"}]
        assert [child.killed for child in fake_popen.live] == [True]
        assert [child.returncode for child in fake_popen.live] == [-9]
        assert errors == [], "a kill after a complete read is not a read failure"

    def test_stream_volume_contexts_still_reports_a_broken_stream_it_killed(
        self, fake_runtime: _FakeRuntime, fake_popen: type[_FakePopen]
    ):
        """A killed child that also cut the stream short is still an error."""
        config = _config(["alice"])
        fake_runtime.listed = [_agent_data("alice")]
        fake_popen.scripted[_agent_data("alice")] = (0, b'{"id": "fb-1"}{"id": ', b"")
        targets = read_volume_headers(config).targets
        fake_popen.hangs = True
        errors: list[feedback_cmd.VolumeError] = []

        documents = list(stream_volume_contexts(config, targets, on_error=errors.append))

        assert documents == [{"id": "fb-1"}]
        assert [child.killed for child in fake_popen.live] == [True]
        assert len(errors) == 1
        assert "could not be read in full" in errors[0].message
        assert "byte offset" in errors[0].message

    def test_stream_volume_contexts_reports_a_container_that_will_not_start(
        self, fake_runtime: _FakeRuntime, fake_popen: type[_FakePopen]
    ):
        """A runtime that cannot launch the reader is a per-volume error."""
        config = _config(["alice"])
        fake_runtime.listed = [_agent_data("alice")]
        targets = read_volume_headers(config).targets
        fake_popen.start_error = OSError("no such file or directory: docker")
        errors: list[feedback_cmd.VolumeError] = []

        assert list(stream_volume_contexts(config, targets, on_error=errors.append)) == []
        assert len(errors) == 1
        assert _agent_data("alice") in errors[0].message
        assert "no such file or directory" in errors[0].message

    def test_stream_volume_contexts_reports_a_failed_read_to_the_error_sink(
        self, fake_runtime: _FakeRuntime, fake_popen: type[_FakePopen]
    ):
        """A failed context read is a per-volume error, not a silent short read."""
        config = _config(["alice"])
        fake_runtime.listed = [_agent_data("alice")]
        fake_popen.scripted[_agent_data("alice")] = (1, b"", b"cat: cannot open")
        targets = read_volume_headers(config).targets
        errors: list[feedback_cmd.VolumeError] = []

        documents = list(stream_volume_contexts(config, targets, on_error=errors.append))

        assert documents == []
        assert len(errors) == 1
        assert _agent_data("alice") in errors[0].message
        assert "cat: cannot open" in errors[0].message

    def test_stream_volume_contexts_raises_when_no_error_sink_is_given(
        self, fake_runtime: _FakeRuntime, fake_popen: type[_FakePopen]
    ):
        """Without a sink to collect them, read failures raise rather than vanish."""
        config = _config(["alice"])
        fake_runtime.listed = [_agent_data("alice")]
        fake_popen.scripted[_agent_data("alice")] = (1, b"", b"cat: cannot open")
        targets = read_volume_headers(config).targets

        with pytest.raises(FeedbackReadError, match="cat: cannot open"):
            list(stream_volume_contexts(config, targets))

    def test_stream_volume_contexts_accepts_the_no_store_exit(
        self, fake_runtime: _FakeRuntime, fake_popen: type[_FakePopen]
    ):
        """A user with no store yields nothing and reports nothing."""
        config = _config(["alice"])
        fake_runtime.listed = [_agent_data("alice")]
        fake_popen.scripted[_agent_data("alice")] = (3, b"", b"")
        targets = read_volume_headers(config).targets
        errors: list[feedback_cmd.VolumeError] = []

        assert list(stream_volume_contexts(config, targets, on_error=errors.append)) == []
        assert errors == []


class TestLocalReader:
    """The host-filesystem reader a single-user deployment is read with."""

    def test_local_dir_follows_a_relocated_base_dir_from_an_unrelated_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """``base_dir`` is anchored at the repo, never at where the CLI was run."""
        repo_root = tmp_path / "deployment"
        repo_root.mkdir()
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)

        directory = local_feedback_dir({"agent_data": {"base_dir": "state/agents"}}, repo_root)

        assert directory == repo_root.resolve() / "state" / "agents" / "feedback"

    def test_local_dir_follows_an_absolute_base_dir(self, tmp_path: Path):
        """An absolute ``base_dir`` resolves to itself, repo root or not."""
        directory = local_feedback_dir({"agent_data": {"base_dir": str(tmp_path)}}, tmp_path / "r")

        assert directory == tmp_path.resolve() / "feedback"

    def test_local_dir_rejects_a_home_relative_base_dir(self, tmp_path: Path):
        """``~/...`` means the records are somewhere this reader cannot reach.

        A container expands ``~`` against its own ``HOME``, which the compose
        template points at the claude-config volume -- the one volume the
        feedback readers never mount. Reporting that beats reading an empty
        store and calling it "no feedback".
        """
        with pytest.raises(click.ClickException) as caught:
            local_feedback_dir({"agent_data": {"base_dir": "~/agent_data"}}, tmp_path)

        message = str(caught.value)
        assert "agent_data.base_dir" in message
        assert "claude-config" in message

    def test_local_headers_are_empty_when_the_store_does_not_exist(self, tmp_path: Path):
        """A deployment nobody has sent feedback from reads as zero records."""
        assert read_local_headers(tmp_path / "feedback") == []
        assert list(stream_local_contexts(tmp_path / "feedback")) == []

    def test_local_reader_returns_every_record_file_in_the_store(self, tmp_path: Path):
        """Headers and contexts come back from their own file families."""
        store = tmp_path / "feedback"
        store.mkdir()
        (store / "fb-2-b.json").write_text('{"id": "fb-2-b"}', encoding="utf-8")
        (store / "fb-1-a.json").write_text('{"id": "fb-1-a"}', encoding="utf-8")
        (store / "ctx-1-a.json").write_text('{"id": "fb-1-a", "context": "c"}', encoding="utf-8")
        (store / ".fb-3-c.json.tmp").write_text("half a record", encoding="utf-8")

        assert read_local_headers(store) == [{"id": "fb-1-a"}, {"id": "fb-2-b"}]
        assert list(stream_local_contexts(store)) == [{"id": "fb-1-a", "context": "c"}]

    def test_local_reader_survives_a_record_pruned_mid_read(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """A file that vanishes between listing and open is routine, not corruption.

        The store prunes old session contexts as new submissions arrive, so a
        long export races them by design. The survivors still come back, and
        the header whose context went missing is exported flagged.
        """
        store = tmp_path / "feedback"
        store.mkdir()
        pruned = store / "ctx-1-a.json"
        pruned.write_text('{"id": "fb-1-a", "context": "gone"}', encoding="utf-8")
        (store / "ctx-2-b.json").write_text('{"id": "fb-2-b", "context": "here"}', encoding="utf-8")
        real_glob = Path.glob

        def _prune_after_listing(self: Path, pattern: str, **kwargs: Any):
            listing = sorted(real_glob(self, pattern, **kwargs))
            pruned.unlink(missing_ok=True)
            return iter(listing)

        monkeypatch.setattr(Path, "glob", _prune_after_listing)

        written: list[dict[str, Any]] = []
        summary = pair_records(
            [{"id": "fb-1-a"}, {"id": "fb-2-b"}], stream_local_contexts(store), written.append
        )

        assert summary.paired == 1
        assert summary.missing == 1
        assert written == [
            {"id": "fb-2-b", "context": "here"},
            {"id": "fb-1-a", "context_missing": True},
        ]

    @pytest.mark.skipif(os.geteuid() == 0, reason="root reads a mode-000 file regardless")
    def test_local_reader_still_raises_on_an_unreadable_record(self, tmp_path: Path):
        """A permission problem is not a prune, and stays loud."""
        store = tmp_path / "feedback"
        store.mkdir()
        unreadable = store / "fb-1-a.json"
        unreadable.write_text('{"id": "fb-1-a"}', encoding="utf-8")
        unreadable.chmod(0o000)

        try:
            with pytest.raises(FeedbackReadError, match="fb-1-a.json"):
                read_local_headers(store)
        finally:
            unreadable.chmod(0o600)

    def test_local_globs_match_the_store_naming_contract(self):
        """The reader's globs and the writer's filenames are one contract."""
        from osprey.interfaces.web_terminal.feedback_store import CONTEXT_GLOB, HEADER_GLOB

        assert feedback_cmd._HEADER_GLOB == HEADER_GLOB
        assert feedback_cmd._CONTEXT_GLOB == CONTEXT_GLOB


class TestPairRecords:
    """Joining each header to its session context on the way to an export."""

    def test_pair_records_joins_headers_drops_an_orphan_and_flags_the_pruned(self):
        """The fixture carries one of everything an export has to handle."""
        headers = [
            {"id": "fb-1", "excerpt": "one"},
            {"id": "fb-2", "excerpt": "two"},
            {"id": "fb-3", "excerpt": "three", "context_pruned": True},
        ]
        contexts = iter(
            [
                {"id": "fb-2", "context": "second"},
                {"id": "fb-404", "context": "no header carries this id"},
                {"id": "fb-1", "context": "first"},
            ]
        )
        written: list[dict[str, Any]] = []

        summary = pair_records(headers, contexts, written.append)

        assert summary.paired == 2
        assert summary.orphans == 1
        assert summary.pruned == 1
        assert summary.missing == 0
        assert written == [
            {"id": "fb-2", "excerpt": "two", "context": "second"},
            {"id": "fb-1", "excerpt": "one", "context": "first"},
            {"id": "fb-3", "excerpt": "three", "context_pruned": True},
        ]

    def test_pair_records_writes_each_record_before_the_iterator_is_exhausted(self):
        """Records reach the sink as they complete, so ``--output`` streams."""
        written: list[dict[str, Any]] = []
        depth: list[int] = []

        def _contexts() -> Iterator[dict[str, Any]]:
            yield {"id": "fb-1", "context": "first"}
            depth.append(len(written))
            yield {"id": "fb-2", "context": "second"}

        pair_records([{"id": "fb-1"}, {"id": "fb-2"}], _contexts(), written.append)

        assert depth == [1], "the first record must be written before the second is read"
        assert len(written) == 2

    def test_pair_records_flags_a_header_whose_context_never_arrived(self):
        """An unexplained missing context is marked, not silently dropped."""
        written: list[dict[str, Any]] = []

        summary = pair_records([{"id": "fb-1", "excerpt": "one"}], iter([]), written.append)

        assert summary.missing == 1
        assert written == [{"id": "fb-1", "excerpt": "one", "context_missing": True}]

    def test_pair_records_flags_a_header_that_carries_no_id(self):
        """A header with no id can never be paired; it is still exported."""
        written: list[dict[str, Any]] = []

        summary = pair_records([{"excerpt": "one"}], iter([]), written.append)

        assert summary.missing == 1
        assert written == [{"excerpt": "one", "context_missing": True}]

    def test_pair_records_keeps_the_header_authoritative_on_a_collision(self):
        """The header is the record's metadata; a context cannot rewrite it."""
        written: list[dict[str, Any]] = []

        pair_records(
            [{"id": "fb-1", "excerpt": "from the header"}],
            iter([{"id": "fb-1", "excerpt": "from the context", "scrollback": "s"}]),
            written.append,
        )

        assert written == [{"id": "fb-1", "excerpt": "from the header", "scrollback": "s"}]


class TestFeedbackListCommand:
    """`osprey feedback list` -- what the table says, and when it refuses."""

    def test_list_command_shows_a_row_per_submission_newest_first(
        self, tmp_path: Path, cli_runner: CliRunner, fake_runtime: _FakeRuntime
    ):
        """Every roster user's store contributes its rows to one table."""
        repo = _make_repo(tmp_path, _config(["alice", "bob"]))
        fake_runtime.listed = [_agent_data("alice"), _agent_data("bob")]
        fake_runtime.stores[_agent_data("alice")] = (
            0,
            _header_bytes(
                {
                    "id": "fb-1700000000000-aaaa1111",
                    "channel": "github",
                    "excerpt": "the plot axes are swapped",
                }
            ),
            b"",
        )
        fake_runtime.stores[_agent_data("bob")] = (
            0,
            _header_bytes(
                {
                    "id": "fb-1800000000000-bbbb2222",
                    "channel": "local",
                    "excerpt": "cannot find the scan panel",
                }
            ),
            b"",
        )

        result = cli_runner.invoke(feedback, ["list", "--repo", str(repo)])

        assert result.exit_code == 0, result.output
        flat = _flat(result.output)
        assert "aaaa1111" in flat
        assert "bbbb2222" in flat
        assert "alice" in flat
        assert "bob" in flat
        assert "github" in flat
        assert "the plot axes are swapped" in flat
        assert flat.index("bbbb2222") < flat.index("aaaa1111"), "newest submission comes first"

    def test_list_command_reports_a_deployment_nobody_has_written_from(
        self, tmp_path: Path, cli_runner: CliRunner, fake_runtime: _FakeRuntime
    ):
        """An empty store is a plain sentence, not an empty table."""
        repo = _make_repo(tmp_path, _config(["alice"]))
        fake_runtime.listed = [_agent_data("alice")]
        fake_runtime.stores[_agent_data("alice")] = (3, b"", b"")

        result = cli_runner.invoke(feedback, ["list", "--repo", str(repo)])

        assert result.exit_code == 0, result.output
        assert "No feedback has been sent" in _flat(result.output)

    def test_list_command_lists_the_rest_when_one_workspace_fails(
        self, tmp_path: Path, cli_runner: CliRunner, fake_runtime: _FakeRuntime
    ):
        """One unreadable workspace is reported; the others are still listed."""
        repo = _make_repo(tmp_path, _config(["alice", "bob"]))
        fake_runtime.listed = [_agent_data("bob")]
        fake_runtime.stores[_agent_data("bob")] = (
            0,
            _header_bytes({"id": "fb-1800000000000-bbbb2222", "channel": "local"}),
            b"",
        )

        result = cli_runner.invoke(feedback, ["list", "--repo", str(repo)])

        assert result.exit_code == 0, result.output
        flat = _flat(result.output)
        assert "alice" in flat, "the unreadable workspace is named"
        assert "bbbb2222" in flat, "the readable one is still listed"

    def test_list_command_stops_when_nothing_at_all_could_be_read(
        self, tmp_path: Path, cli_runner: CliRunner, fake_runtime: _FakeRuntime
    ):
        """A deployment that read nothing is not a deployment with no feedback."""
        repo = _make_repo(tmp_path, _config(["alice"]))
        fake_runtime.listed = []

        result = cli_runner.invoke(feedback, ["list", "--repo", str(repo)])

        assert result.exit_code != 0
        flat = _flat(result.output)
        assert "Could not read any feedback" in flat
        assert "No feedback has been sent" not in flat

    def test_list_command_reads_a_single_user_deployment_off_disk(
        self, tmp_path: Path, cli_runner: CliRunner
    ):
        """With no roster there is one store, and no container is started."""
        repo = _make_repo(tmp_path, _single_user_config())
        _write_store(
            repo,
            headers={
                "fb-1700000000000-aaaa1111.json": json.dumps(
                    {
                        "id": "fb-1700000000000-aaaa1111",
                        "channel": "email",
                        "excerpt": "the export button does nothing",
                        "identity": "operator",
                    }
                )
            },
            contexts={},
        )

        result = cli_runner.invoke(feedback, ["list", "--repo", str(repo)])

        assert result.exit_code == 0, result.output
        flat = _flat(result.output)
        assert "aaaa1111" in flat
        assert "operator" in flat
        assert "email" in flat

    def test_list_command_renders_the_columns_from_the_record(
        self, tmp_path: Path, cli_runner: CliRunner
    ):
        """Every column: the stamp, the identity, the channel, the excerpt."""
        repo = _make_repo(tmp_path, _single_user_config())
        _write_store(
            repo,
            headers={
                "fb-1700000000000-aaaa1111.json": json.dumps(
                    {
                        "id": "fb-1700000000000-aaaa1111",
                        "identity": {"name": "dana"},
                        "channel": "github",
                        "excerpt": "x" * 80,
                    }
                )
            },
            contexts={},
        )

        result = cli_runner.invoke(feedback, ["list", "--repo", str(repo)])

        assert result.exit_code == 0, result.output
        flat = _flat(result.output)
        assert "2023-11-14 22:13" in flat, "the id's own milliseconds, rendered in UTC"
        assert "dana" in flat, "an identity mapping is read for a name"
        assert "github" in flat
        assert "x" * 53 + "..." in flat, "a long excerpt is cut at 56 characters"

    def test_list_command_prefers_the_headers_own_timestamp(
        self, tmp_path: Path, cli_runner: CliRunner
    ):
        """``created_at`` is what the web terminal writes; it wins over the id."""
        repo = _make_repo(tmp_path, _single_user_config())
        _write_store(
            repo,
            headers={
                "fb-1700000000000-aaaa1111.json": json.dumps(
                    {
                        "id": "fb-1700000000000-aaaa1111",
                        "created_at": "2024-03-05T09:41:07.123456+00:00",
                    }
                )
            },
            contexts={},
        )

        result = cli_runner.invoke(feedback, ["list", "--repo", str(repo)])

        assert result.exit_code == 0, result.output
        flat = _flat(result.output)
        assert "2024-03-05 09:41" in flat
        assert "2023-11-14" not in flat, "the id is the fallback, not the source"

    def test_list_command_survives_a_record_with_an_unreadable_id(
        self, tmp_path: Path, cli_runner: CliRunner
    ):
        """One malformed id must not take the whole listing down with it.

        ``"²".isdigit()`` is true and ``int("²")`` raises, so a store holding
        such an id used to abort the command before it printed a single row.
        """
        repo = _make_repo(tmp_path, _single_user_config())
        _write_store(
            repo,
            headers={
                "fb-bad.json": json.dumps({"id": "fb-²-x", "channel": "local"}),
                "fb-none.json": json.dumps({"channel": "email"}),
                "fb-1700000000000-aaaa1111.json": json.dumps({"id": "fb-1700000000000-aaaa1111"}),
            },
            contexts={},
        )

        result = cli_runner.invoke(feedback, ["list", "--repo", str(repo)])

        assert result.exit_code == 0, result.output
        flat = _flat(result.output)
        assert "aaaa1111" in flat, "the well-formed record is still listed"
        assert "local" in flat, "so is the one with the unreadable id"
        assert "email" in flat, "and the one with no id at all"
        unreadable_id = next(line for line in result.output.splitlines() if "local" in line)
        no_id = next(line for line in result.output.splitlines() if "email" in line)
        assert " - " in _flat(unreadable_id), "its stamp falls back to a dash"
        assert _flat(no_id).count(" - ") >= 2, "both its id and its stamp fall back"

    def test_list_command_refuses_a_home_relative_agent_data_dir(
        self, tmp_path: Path, cli_runner: CliRunner
    ):
        """The known limit is reported, not read as an empty deployment."""
        repo = _make_repo(tmp_path, {"agent_data": {"base_dir": "~/agent_data"}})

        result = cli_runner.invoke(feedback, ["list", "--repo", str(repo)])

        assert result.exit_code != 0
        assert "agent_data.base_dir" in _flat(result.output)


class TestFeedbackExportCommand:
    """`osprey feedback export` -- the document, and where it goes."""

    def test_export_command_writes_a_json_array_to_the_output_file(
        self, tmp_path: Path, cli_runner: CliRunner
    ):
        """--output holds the export; each record carries its own session."""
        repo = _make_repo(tmp_path, _single_user_config())
        _write_store(
            repo,
            headers={
                "fb-1700000000000-aaaa1111.json": json.dumps(
                    {"id": "fb-1700000000000-aaaa1111", "excerpt": "one"}
                )
            },
            contexts={
                "ctx-1700000000000-aaaa1111.json": json.dumps(
                    {"id": "fb-1700000000000-aaaa1111", "context": "the session"}
                )
            },
        )
        destination = tmp_path / "out.json"

        result = cli_runner.invoke(
            feedback, ["export", "--repo", str(repo), "--output", str(destination)]
        )

        assert result.exit_code == 0, result.output
        assert json.loads(destination.read_text(encoding="utf-8")) == [
            {"id": "fb-1700000000000-aaaa1111", "excerpt": "one", "context": "the session"}
        ]
        assert "Exported 1 feedback record " in _flat(result.output), "counted, not pluralized"

    def test_export_command_writes_the_document_to_clean_stdout(
        self, tmp_path: Path, cli_runner: CliRunner
    ):
        """With no --output, stdout is the export and carries nothing else."""
        repo = _make_repo(tmp_path, _single_user_config())
        _write_store(
            repo,
            headers={
                "fb-1700000000000-aaaa1111.json": json.dumps(
                    {"id": "fb-1700000000000-aaaa1111", "excerpt": "one"}
                )
            },
            contexts={
                "ctx-1700000000000-aaaa1111.json": json.dumps(
                    {"id": "fb-1700000000000-aaaa1111", "context": "the session"}
                )
            },
        )

        result = cli_runner.invoke(feedback, ["export", "--repo", str(repo)])

        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout) == [
            {"id": "fb-1700000000000-aaaa1111", "excerpt": "one", "context": "the session"}
        ]
        assert "Exported" in _flat(result.stderr), "the summary belongs on standard error"

    def test_export_command_writes_an_empty_array_for_an_empty_store(
        self, tmp_path: Path, cli_runner: CliRunner
    ):
        """Nothing to export is still a whole JSON document."""
        repo = _make_repo(tmp_path, _single_user_config())

        result = cli_runner.invoke(feedback, ["export", "--repo", str(repo)])

        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout) == []

    def test_export_command_marks_a_submission_with_no_session(
        self, tmp_path: Path, cli_runner: CliRunner
    ):
        """A header whose context never arrived is exported, and flagged."""
        repo = _make_repo(tmp_path, _single_user_config())
        _write_store(
            repo,
            headers={
                "fb-1700000000000-aaaa1111.json": json.dumps(
                    {"id": "fb-1700000000000-aaaa1111", "excerpt": "one"}
                )
            },
            contexts={},
        )

        result = cli_runner.invoke(feedback, ["export", "--repo", str(repo)])

        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout) == [
            {"id": "fb-1700000000000-aaaa1111", "excerpt": "one", "context_missing": True}
        ]

    def test_export_command_exports_every_roster_users_store(
        self, tmp_path: Path, cli_runner: CliRunner, fake_runtime: _FakeRuntime, fake_popen
    ):
        """A multi-user export reads each verified volume, and only those."""
        repo = _make_repo(tmp_path, _config(["alice", "bob"]))
        fake_runtime.listed = [_agent_data("alice"), _agent_data("bob")]
        fake_runtime.stores[_agent_data("alice")] = (
            0,
            _header_bytes({"id": "fb-1700000000000-aaaa1111"}),
            b"",
        )
        fake_runtime.stores[_agent_data("bob")] = (
            0,
            _header_bytes({"id": "fb-1800000000000-bbbb2222"}),
            b"",
        )
        fake_popen.scripted[_agent_data("alice")] = (
            0,
            json.dumps({"id": "fb-1700000000000-aaaa1111", "context": "alice's"}).encode(),
            b"",
        )
        fake_popen.scripted[_agent_data("bob")] = (3, b"", b"")

        result = cli_runner.invoke(feedback, ["export", "--repo", str(repo)])

        assert result.exit_code == 0, result.output
        exported = json.loads(result.stdout)
        assert {record["id"] for record in exported} == {
            "fb-1700000000000-aaaa1111",
            "fb-1800000000000-bbbb2222",
        }
        assert exported[0]["context"] == "alice's"
        assert exported[1]["context_missing"] is True
        mounted = {_mounted_volume(argv) for argv in fake_popen.calls}
        assert mounted == {_agent_data("alice"), _agent_data("bob")}

    def test_export_command_closes_the_document_when_a_store_stops_reading(
        self, tmp_path: Path, cli_runner: CliRunner
    ):
        """A corrupt store fails loudly, and the file holds what was read.

        The failure copy promises the export holds what was read, so the
        headers that parsed have to be in it -- a document that stopped at
        ``[]`` would make that sentence untrue.
        """
        repo = _make_repo(tmp_path, _single_user_config())
        store = _write_store(
            repo,
            headers={
                "fb-1700000000000-aaaa1111.json": json.dumps(
                    {"id": "fb-1700000000000-aaaa1111", "excerpt": "one"}
                )
            },
            contexts={},
        )
        (store / "ctx-1700000000000-aaaa1111.json").write_text('{"id": ', encoding="utf-8")
        destination = tmp_path / "out.json"

        result = cli_runner.invoke(
            feedback, ["export", "--repo", str(repo), "--output", str(destination)]
        )

        assert result.exit_code != 0
        assert "stopped part-way" in _flat(result.output)
        exported = json.loads(destination.read_text(encoding="utf-8"))
        assert [record["id"] for record in exported] == ["fb-1700000000000-aaaa1111"]
        assert exported[0]["context_missing"] is True

    def test_export_command_names_the_workspaces_it_could_not_read(
        self, tmp_path: Path, cli_runner: CliRunner, fake_runtime: _FakeRuntime, fake_popen
    ):
        """A partial export says so, and says whose feedback is missing from it."""
        repo = _make_repo(tmp_path, _config(["alice", "bob"]))
        fake_runtime.listed = [_agent_data("bob")]
        fake_runtime.stores[_agent_data("bob")] = (
            0,
            _header_bytes({"id": "fb-1800000000000-bbbb2222"}),
            b"",
        )
        fake_popen.scripted[_agent_data("bob")] = (3, b"", b"")

        result = cli_runner.invoke(feedback, ["export", "--repo", str(repo)])

        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout)[0]["id"] == "fb-1800000000000-bbbb2222"
        summary = _flat(result.stderr)
        assert "could not be read" in summary
        assert "alice" in summary

    def test_export_command_does_not_reopen_a_workspace_that_already_failed(
        self, tmp_path: Path, cli_runner: CliRunner, fake_runtime: _FakeRuntime, fake_popen
    ):
        """No second container for a volume whose headers already failed."""
        repo = _make_repo(tmp_path, _config(["alice", "bob"]))
        fake_runtime.listed = [_agent_data("alice"), _agent_data("bob")]
        fake_runtime.stores[_agent_data("alice")] = (1, b"", b"cat: cannot open")
        fake_runtime.stores[_agent_data("bob")] = (
            0,
            _header_bytes({"id": "fb-1800000000000-bbbb2222"}),
            b"",
        )
        fake_popen.scripted[_agent_data("bob")] = (3, b"", b"")

        result = cli_runner.invoke(feedback, ["export", "--repo", str(repo)])

        assert result.exit_code == 0, result.output
        mounted = {_mounted_volume(argv) for argv in fake_popen.calls}
        assert mounted == {_agent_data("bob")}, "the failed volume is not opened again"
        summary = _flat(result.stderr)
        assert "not in this export: alice" in summary

    def test_export_command_still_reads_contexts_for_a_partly_read_workspace(
        self, tmp_path: Path, cli_runner: CliRunner, fake_runtime: _FakeRuntime, fake_popen
    ):
        """A read that failed after emitting headers keeps its context read.

        Failing is not the same as producing nothing: those headers are real
        submissions, and their sessions come from a second read that has every
        chance of succeeding. Skipping it would export them as though their
        sessions were gone.
        """
        repo = _make_repo(tmp_path, _config(["alice"]))
        fake_runtime.listed = [_agent_data("alice")]
        fake_runtime.stores[_agent_data("alice")] = (
            1,
            _header_bytes({"id": "fb-1700000000000-aaaa1111"}),
            b"find: /data/feedback/fb-later.json: Input/output error",
        )
        fake_popen.scripted[_agent_data("alice")] = (
            0,
            json.dumps({"id": "fb-1700000000000-aaaa1111", "context": "hers"}).encode(),
            b"",
        )

        result = cli_runner.invoke(feedback, ["export", "--repo", str(repo)])

        assert result.exit_code == 0, result.output
        exported = json.loads(result.stdout)
        assert exported[0]["context"] == "hers", "its context read was not skipped"
        assert "context_missing" not in exported[0]
        summary = _flat(result.stderr)
        assert "only partly readable: alice" in summary
        assert "not in this export" not in summary, "its records ARE in the export"

    def test_export_command_reports_corruption_even_when_the_sink_fails(
        self, tmp_path: Path, cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ):
        """A sink that dies on the way out must not bury the real failure."""
        repo = _make_repo(tmp_path, _single_user_config())
        store = _write_store(
            repo,
            headers={
                "fb-1700000000000-aaaa1111.json": json.dumps({"id": "fb-1700000000000-aaaa1111"})
            },
            contexts={},
        )
        (store / "ctx-1700000000000-aaaa1111.json").write_text('{"id": ', encoding="utf-8")

        def _closed_pipe(*args: Any, **kwargs: Any) -> None:
            raise BrokenPipeError(32, "Broken pipe")

        monkeypatch.setattr(feedback_cmd.click, "echo", _closed_pipe)

        result = cli_runner.invoke(feedback, ["export", "--repo", str(repo)])

        assert result.exit_code != 0, "the corrupt store is still the reported failure"
        assert "stopped part-way" in _flat(result.output)

    @pytest.mark.skipif(
        not (Path(sys.executable).parent / "osprey").exists() or shutil.which("bash") is None,
        reason="needs the installed osprey console script and bash",
    )
    def test_export_command_survives_a_real_closed_pipe(self, tmp_path: Path):
        """`osprey feedback export | head` exits cleanly from a real shell.

        No in-process double reaches this. Catching the broken pipe is only
        half of it: the interpreter flushes whatever is still in stdout's
        buffer on the way out, long after the command has returned, and that
        flush is what turns a clean exit into 120. So the store here is large
        enough to leave a buffer behind, and the pipeline is a real one.
        """
        repo = _make_repo(tmp_path, _single_user_config())
        _write_store(
            repo,
            headers={
                f"fb-17000000{index:05d}-aaaa{index:04d}.json": json.dumps(
                    {"id": f"fb-17000000{index:05d}-aaaa{index:04d}", "excerpt": "x" * 2048}
                )
                for index in range(120)
            },
            contexts={},
        )
        osprey = shlex.quote(str(Path(sys.executable).parent / "osprey"))
        pipeline = (
            f"{osprey} feedback export --repo {shlex.quote(str(repo))} "
            "| head -c 60; exit ${PIPESTATUS[0]}"
        )

        finished = subprocess.run(
            ["bash", "-c", pipeline], capture_output=True, text=True, timeout=300, check=False
        )

        assert finished.returncode == 0, finished.stderr
        assert "BrokenPipeError" not in finished.stderr
        assert "Exception ignored" not in finished.stderr

    def test_export_command_ends_quietly_when_the_reader_closes_the_pipe(
        self, tmp_path: Path, cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ):
        """Piping into `head` closes stdout early; that is not a failure."""
        repo = _make_repo(tmp_path, _single_user_config())
        _write_store(
            repo,
            headers={
                "fb-1700000000000-aaaa1111.json": json.dumps({"id": "fb-1700000000000-aaaa1111"})
            },
            contexts={},
        )

        def _closed_pipe(*args: Any, **kwargs: Any) -> None:
            raise BrokenPipeError(32, "Broken pipe")

        monkeypatch.setattr(feedback_cmd.click, "echo", _closed_pipe)

        result = cli_runner.invoke(feedback, ["export", "--repo", str(repo)])

        assert result.exit_code == 0, result.output
        assert result.exception is None

    def test_export_command_refuses_an_output_that_is_a_directory(
        self, tmp_path: Path, cli_runner: CliRunner, fake_runtime: _FakeRuntime
    ):
        """A destination that cannot hold a file fails before anything is read."""
        repo = _make_repo(tmp_path, _config(["alice"]))
        fake_runtime.listed = [_agent_data("alice")]

        result = cli_runner.invoke(
            feedback, ["export", "--repo", str(repo), "--output", str(tmp_path)]
        )
        empty = cli_runner.invoke(feedback, ["export", "--repo", str(repo), "--output", ""])

        assert result.exit_code == 2, result.output
        assert empty.exit_code == 2, "an empty --output is a mistake, not a switch to stdout"
        assert fake_runtime.calls == [], "no workspace is read for an export with nowhere to go"

    def test_export_command_refuses_a_repo_that_was_never_built(
        self, tmp_path: Path, cli_runner: CliRunner
    ):
        """Without a rendered config there is no deployment to read."""
        repo = tmp_path / "unbuilt"
        repo.mkdir()
        (repo / "profile.yml").write_text("preset: control-assistant\n", encoding="utf-8")

        result = cli_runner.invoke(feedback, ["export", "--repo", str(repo)])

        assert result.exit_code != 0
        assert "No build found" in _flat(result.output)


class TestFeedbackRegistration:
    """The verb has to be reachable from the CLI, not just importable."""

    def test_feedback_is_registered_in_both_main_literals(self):
        """The dispatch map and the help list are two separate literals.

        `tests/cli/test_main.py` catches a name in the list that does not
        resolve; nothing catches the reverse, so the map entry is pinned here.
        """
        import click as click_module

        from osprey.cli.main import cli

        context = click_module.Context(cli)

        assert "feedback" in cli.list_commands(context)
        assert cli.get_command(context, "feedback") is feedback

    def test_feedback_group_offers_both_verbs(self, cli_runner: CliRunner):
        """--help names the two verbs a maintainer reaches for."""
        result = cli_runner.invoke(feedback, ["--help"])

        assert result.exit_code == 0
        flat = _flat(result.output)
        assert "list" in flat
        assert "export" in flat
