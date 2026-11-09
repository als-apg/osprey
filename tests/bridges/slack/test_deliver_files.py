"""``SlackOps.deliver_files``: the run's artifacts, uploaded where the answer went.

Pinned here: each artifact is fetched from the worker (``core.artifacts.fetch_artifact``)
and uploaded into the answer's placement — the thread, or the top level of a 1:1 DM;
names come from ``unique_stems`` / ``upload_name``, the extension from the served bytes;
a failed fetch or upload costs only its own file and is named in ONE skipped-files
note; nothing raises; and the return value is ``{}`` always, because nothing Slack
hosts is fetchable by an unauthenticated GET.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from osprey.bridges.core import MAX_DELIVERED_DOC_BYTES, CoreConfig, FetchedArtifact
from osprey.bridges.core.errors import UndeliverableError
from osprey.bridges.slack import ops as ops_module
from osprey.bridges.slack.client import SlackApiError
from osprey.bridges.slack.ops import skipped_files_note
from tests.bridges.slack._posting import (
    CHAN,
    DM,
    PARENT,
    TS,
    RecordingClient,
    make_entry,
    make_ops,
)

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


class UploadClient(RecordingClient):
    """A recording client that also records ``upload``, failing on named files."""

    def __init__(
        self, fail_names: dict[str, BaseException] | None = None, *, post_fails: bool = False
    ) -> None:
        super().__init__(SlackApiError("down", method="chat.postMessage") if post_fails else None)
        self.fail_names = fail_names or {}
        self.uploads: list[tuple[str, str | None, str, bytes, str]] = []

    def upload(
        self, channel: str, thread_ts: str | None, name: str, data: bytes, mime: str
    ) -> dict[str, Any]:
        self.uploads.append((channel, thread_ts, name, data, mime))
        if name in self.fail_names:
            raise self.fail_names[name]
        return {"ok": True}


class Fetcher:
    """Stands in for ``fetch_artifact``: serves per artifact id, records each call."""

    def __init__(self, served: dict[str, FetchedArtifact | None | BaseException]) -> None:
        self.served = served
        self.calls: list[tuple[str, str, int]] = []
        self.cfgs: list[CoreConfig] = []
        self.https: list[Any] = []

    def __call__(
        self,
        http: httpx.Client,
        cfg: CoreConfig,
        run_id: str,
        artifact_id: str,
        *,
        max_bytes: int = MAX_DELIVERED_DOC_BYTES,
    ) -> FetchedArtifact | None:
        self.calls.append((run_id, artifact_id, max_bytes))
        self.cfgs.append(cfg)
        self.https.append(http)
        outcome = self.served.get(artifact_id)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


@pytest.fixture
def fetcher(monkeypatch: pytest.MonkeyPatch) -> Fetcher:
    stub = Fetcher({})
    monkeypatch.setattr(ops_module, "fetch_artifact", stub)
    return stub


def result(*artifacts: Any, run_id: Any = "R1") -> dict[str, Any]:
    out: dict[str, Any] = {
        "status": "completed",
        "text_output": "done",
        "artifacts": list(artifacts),
    }
    if run_id is not None:
        out["run_id"] = run_id
    return out


def art(artifact_id: str, filename: str = "", mime: str = "") -> dict[str, Any]:
    descriptor: dict[str, Any] = {"artifact_id": artifact_id}
    if filename:
        descriptor["filename"] = filename
    if mime:
        descriptor["delivered_mime"] = mime
    return descriptor


def test_each_artifact_is_uploaded_into_the_answers_thread(fetcher: Fetcher) -> None:
    fetcher.served = {
        "a1": FetchedArtifact(PNG, "image/png"),
        "a2": FetchedArtifact(b"x,y\n1,2\n", "text/csv"),
    }
    client = UploadClient()
    ops, _ = make_ops(client)
    out = ops.deliver_files(
        make_entry(), result(art("a1", "orbit.png", "image/png"), art("a2", "data.csv", "text/csv"))
    )
    assert out == {}
    assert client.uploads == [
        (CHAN, TS, "orbit.png", PNG, "image/png"),
        (CHAN, TS, "data.csv", b"x,y\n1,2\n", "text/csv"),
    ]
    assert [(run, aid) for run, aid, _ in fetcher.calls] == [("R1", "a1"), ("R1", "a2")]
    assert client.calls == []


def test_the_fetch_reads_the_core_config(fetcher: Fetcher) -> None:
    fetcher.served = {"a1": FetchedArtifact(PNG, "image/png")}
    ops, _ = make_ops(UploadClient())
    ops.deliver_files(make_entry(), result(art("a1")))
    assert fetcher.cfgs == [ops._cfg.core]
    assert isinstance(fetcher.https[0], httpx.Client)


def test_a_top_level_dm_uploads_at_the_top_level(fetcher: Fetcher) -> None:
    fetcher.served = {"a1": FetchedArtifact(PNG, "image/png")}
    client = UploadClient()
    ops, _ = make_ops(client)
    ops.deliver_files(make_entry("im"), result(art("a1", "plot.png")))
    assert [(channel, thread) for channel, thread, *_ in client.uploads] == [(DM, None)]


def test_a_threaded_dm_uploads_into_its_thread(fetcher: Fetcher) -> None:
    fetcher.served = {"a1": FetchedArtifact(PNG, "image/png")}
    client = UploadClient()
    ops, _ = make_ops(client)
    ops.deliver_files(make_entry("im", thread_ts=PARENT), result(art("a1", "plot.png")))
    assert [(channel, thread) for channel, thread, *_ in client.uploads] == [(DM, PARENT)]


def test_the_served_bytes_decide_the_extension(fetcher: Fetcher) -> None:
    # Predicted a PNG, served the original CSV (a failed conversion).
    fetcher.served = {
        "a1": FetchedArtifact(b"a,b\n", "text/csv"),
        "a2": FetchedArtifact(PNG, None),
    }
    client = UploadClient()
    ops, _ = make_ops(client)
    ops.deliver_files(
        make_entry(), result(art("a1", "chart.png", "image/png"), art("a2", "shot", "text/csv"))
    )
    assert [(name, mime) for *_, name, _, mime in client.uploads] == [
        ("chart.csv", "text/csv"),
        ("shot.png", "image/png"),
    ]


def test_an_untyped_non_png_ships_as_octet_stream(fetcher: Fetcher) -> None:
    fetcher.served = {"a1": FetchedArtifact(b"\x00\x01", None)}
    client = UploadClient()
    ops, _ = make_ops(client)
    ops.deliver_files(make_entry(), result(art("a1", "blob")))
    assert [(name, mime) for *_, name, _, mime in client.uploads] == [
        ("blob.bin", "application/octet-stream")
    ]


def test_two_artifacts_with_one_name_get_distinct_names(fetcher: Fetcher) -> None:
    fetcher.served = {
        "a1": FetchedArtifact(PNG, "image/png"),
        "a2": FetchedArtifact(PNG, "image/png"),
    }
    client = UploadClient()
    ops, _ = make_ops(client)
    ops.deliver_files(make_entry(), result(art("a1", "plot.png"), art("a2", "plot.png")))
    names = [name for *_, name, _, _ in client.uploads]
    assert len(names) == 2 and len(set(names)) == 2
    assert all(name.endswith(".png") for name in names)


def test_a_failed_fetch_is_named_in_one_note(fetcher: Fetcher) -> None:
    fetcher.served = {
        "a1": None,
        "a2": FetchedArtifact(PNG, "image/png"),
        "a3": RuntimeError("fetcher broke its promise"),
    }
    client = UploadClient()
    ops, _ = make_ops(client)
    out = ops.deliver_files(
        make_entry(),
        result(
            art("a1", "summary", "application/pdf"),
            art("a2", "ok.png"),
            art("a3", "plot", "image/png"),
        ),
    )
    assert out == {}
    assert [name for *_, name, _, _ in client.uploads] == ["ok.png"]
    assert client.calls == [(CHAN, skipped_files_note(["summary.pdf", "plot.png"]), TS)]


def test_a_failed_upload_costs_only_itself_and_is_named(fetcher: Fetcher) -> None:
    fetcher.served = {
        "a1": FetchedArtifact(PNG, "image/png"),
        "a2": FetchedArtifact(b"x", "text/csv"),
        "a3": FetchedArtifact(b"y", "text/plain"),
    }
    client = UploadClient(
        {
            "one.png": SlackApiError("HTTP 500", method="files.upload"),
            "three.txt": UndeliverableError("channel_not_found"),
        }
    )
    ops, _ = make_ops(client)
    out = ops.deliver_files(
        make_entry(), result(art("a1", "one.png"), art("a2", "two.csv"), art("a3", "three.txt"))
    )
    assert out == {}
    assert [name for *_, name, _, _ in client.uploads] == ["one.png", "two.csv", "three.txt"]
    assert client.texts == [skipped_files_note(["one.png", "three.txt"])]


def test_a_failing_note_post_does_not_raise(
    fetcher: Fetcher, caplog: pytest.LogCaptureFixture
) -> None:
    fetcher.served = {"a1": None}
    client = UploadClient(post_fails=True)
    ops, _ = make_ops(client)
    with caplog.at_level("WARNING"):
        assert ops.deliver_files(make_entry(), result(art("a1", "x.png", "image/png"))) == {}
    assert len(client.calls) == 1
    assert caplog.records


def test_a_fully_delivered_run_posts_no_note_and_returns_empty(fetcher: Fetcher) -> None:
    fetcher.served = {"a1": FetchedArtifact(PNG, "image/png")}
    client = UploadClient()
    ops, _ = make_ops(client)
    assert ops.deliver_files(make_entry(), result(art("a1", "p.png"))) == {}
    assert client.calls == []


def test_no_artifacts_makes_no_call(fetcher: Fetcher) -> None:
    client = UploadClient()
    ops, _ = make_ops(client)
    assert ops.deliver_files(make_entry(), result()) == {}
    assert ops.deliver_files(make_entry(), {"status": "completed", "run_id": "R1"}) == {}
    assert ops.deliver_files(make_entry(), result("", 7, {"no": "id"})) == {}
    assert fetcher.calls == [] and client.uploads == [] and client.calls == []


def test_no_run_id_fetches_nothing(fetcher: Fetcher) -> None:
    client = UploadClient()
    ops, _ = make_ops(client)
    assert ops.deliver_files(make_entry(), result(art("a1"), run_id=None)) == {}
    assert ops.deliver_files(make_entry(), result(art("a1"), run_id="")) == {}
    assert fetcher.calls == [] and client.uploads == [] and client.calls == []


def test_the_run_id_falls_back_to_the_entry(fetcher: Fetcher) -> None:
    fetcher.served = {"a1": FetchedArtifact(PNG, "image/png")}
    ops, _ = make_ops(UploadClient())
    entry = make_entry()
    entry["run_id"] = "R-entry"
    ops.deliver_files(entry, result(art("a1"), run_id=None))
    assert [run for run, _, _ in fetcher.calls] == ["R-entry"]


def test_bare_id_strings_are_delivered(fetcher: Fetcher) -> None:
    fetcher.served = {"a1": FetchedArtifact(PNG, "image/png")}
    client = UploadClient()
    ops, _ = make_ops(client)
    ops.deliver_files(make_entry(), result("a1"))
    assert len(client.uploads) == 1
    assert client.uploads[0][2].endswith(".png")


def test_a_malformed_entry_delivers_nothing_and_does_not_raise(fetcher: Fetcher) -> None:
    fetcher.served = {"a1": FetchedArtifact(PNG, "image/png")}
    client = UploadClient()
    ops, _ = make_ops(client)
    assert ops.deliver_files({"sl_channel_type": "channel"}, result(art("a1"))) == {}
    assert client.uploads == [] and client.calls == []


def test_a_note_names_escape_mrkdwn(fetcher: Fetcher) -> None:
    fetcher.served = {"a1": None}
    client = UploadClient()
    ops, _ = make_ops(client)
    ops.deliver_files(make_entry(), result(art("a1", "<!here>", "image/png")))
    assert len(client.texts) == 1
    assert "<!here>" not in client.texts[0]
