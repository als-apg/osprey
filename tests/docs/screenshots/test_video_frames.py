"""Unit tests for the screencast frame sink.

CI-safe: the CDP ``Page.screencastFrame`` payloads are built by hand, so no
browser and no Playwright are involved.
"""

from __future__ import annotations

import base64
import contextlib
from pathlib import Path

from docs.screenshots import video_frames
from docs.screenshots.video_frames import FrameSink


def _payload(body: bytes, timestamp: float | None) -> dict:
    metadata: dict = {"offsetTop": 0, "pageScaleFactor": 1}
    if timestamp is not None:
        metadata["timestamp"] = timestamp
    return {
        "data": base64.b64encode(body).decode("ascii"),
        "metadata": metadata,
        "sessionId": 1,
    }


def _noop() -> None:
    return None


def test_first_frame_is_written_immediately(tmp_path: Path) -> None:
    sink = FrameSink(tmp_path)
    sink.on_frame(_payload(b"first", 100.0), _noop)
    assert len(sink.frames) == 1
    path, t = sink.frames[0]
    assert t == 100.0
    assert Path(path).read_bytes() == b"first"
    assert Path(path).parent == tmp_path


def test_burst_then_silence_writes_last_frame_after_flush(tmp_path: Path) -> None:
    sink = FrameSink(tmp_path)
    # A burst of frames 10 ms apart: only the first clears the 40 ms spacing.
    for i, body in enumerate([b"f0", b"f1", b"f2", b"f3"]):
        sink.on_frame(_payload(body, 100.0 + i * 0.010), _noop)
    assert [Path(p).read_bytes() for p, _ in sink.frames] == [b"f0"]

    # Silence follows; the final state of the burst must not be lost.
    sink.flush()
    assert [Path(p).read_bytes() for p, _ in sink.frames] == [b"f0", b"f3"]
    assert sink.frames[-1][1] == 100.030

    # A second flush with nothing held writes nothing.
    sink.flush()
    assert len(sink.frames) == 2


def test_frame_after_spacing_is_written_and_drops_older_pending(tmp_path: Path) -> None:
    sink = FrameSink(tmp_path)
    sink.on_frame(_payload(b"a", 10.000), _noop)
    sink.on_frame(_payload(b"b", 10.020), _noop)  # held
    sink.on_frame(_payload(b"c", 10.045), _noop)  # >= 40 ms: written, supersedes b
    assert [Path(p).read_bytes() for p, _ in sink.frames] == [b"a", b"c"]
    sink.flush()
    assert [Path(p).read_bytes() for p, _ in sink.frames] == [b"a", b"c"]


def test_exactly_forty_ms_is_written(tmp_path: Path) -> None:
    sink = FrameSink(tmp_path)
    sink.on_frame(_payload(b"a", 1.0), _noop)
    sink.on_frame(_payload(b"b", 1.0 + video_frames.MIN_INTERVAL_S), _noop)
    assert len(sink.frames) == 2


def test_frame_paths_are_distinct_and_ordered(tmp_path: Path) -> None:
    sink = FrameSink(tmp_path)
    for i in range(3):
        sink.on_frame(_payload(bytes([i]), 5.0 + i * 0.1), _noop)
    paths = [str(p) for p, _ in sink.frames]
    assert len(set(paths)) == 3
    assert paths == sorted(paths)
    assert all(p.endswith(".jpg") for p in paths)


def test_ack_comes_before_any_write(tmp_path: Path, monkeypatch) -> None:
    events: list[str] = []
    real_write_bytes = Path.write_bytes

    def recording_write_bytes(self: Path, data: bytes) -> int:
        events.append("write")
        return real_write_bytes(self, data)

    monkeypatch.setattr(Path, "write_bytes", recording_write_bytes)
    sink = FrameSink(tmp_path)

    sink.on_frame(_payload(b"x", 50.0), lambda: events.append("ack"))
    assert events == ["ack", "write"]

    # A held frame is still acked, and no write happens.
    sink.on_frame(_payload(b"y", 50.001), lambda: events.append("ack"))
    assert events == ["ack", "write", "ack"]


def test_ack_is_called_even_when_decoding_fails(tmp_path: Path) -> None:
    acked: list[bool] = []
    sink = FrameSink(tmp_path)
    with contextlib.suppress(Exception):
        sink.on_frame(
            {"data": "!!not base64!!", "metadata": {"timestamp": 1.0}},
            lambda: acked.append(True),
        )
    assert acked == [True]


def test_missing_timestamp_falls_back_to_wall_clock(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(video_frames.time, "time", lambda: 1234.5)
    sink = FrameSink(tmp_path)
    sink.on_frame(_payload(b"no-ts", None), _noop)
    assert sink.frames[0][1] == 1234.5


def test_missing_metadata_falls_back_to_wall_clock(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(video_frames.time, "time", lambda: 99.0)
    sink = FrameSink(tmp_path)
    sink.on_frame({"data": base64.b64encode(b"m").decode("ascii"), "sessionId": 3}, _noop)
    assert sink.frames == [(sink.frames[0][0], 99.0)]


def test_out_dir_is_created(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "frames"
    sink = FrameSink(out)
    sink.on_frame(_payload(b"z", 1.0), _noop)
    assert Path(sink.frames[0][0]).parent == out


def test_module_does_not_sleep() -> None:
    # Sync Playwright only delivers events inside Playwright calls; a sleep here
    # would starve the screencast.
    source = Path(video_frames.__file__).read_text()
    assert "time.sleep" not in source
