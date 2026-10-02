"""Tests for the panel sidecars' start-outcome records."""

from __future__ import annotations

import json

from osprey.interfaces.web_terminal.sidecar_status import (
    PANEL_STATUS_DIRNAME,
    SidecarStatus,
    clear_status,
    failure_reason,
    read_status,
    status_message,
    status_path,
    write_status,
)


def test_a_written_status_reads_back(tmp_path):
    status = SidecarStatus.failed("boom")

    write_status(tmp_path, "jupyter", status)

    assert status_path(tmp_path, "jupyter") == tmp_path / PANEL_STATUS_DIRNAME / "jupyter.json"
    assert read_status(tmp_path, "jupyter") == status

    clear_status(tmp_path, "jupyter")
    assert read_status(tmp_path, "jupyter") is None
    clear_status(tmp_path, "jupyter")  # a missing record is not an error


def test_an_absent_or_damaged_record_reads_as_none(tmp_path):
    assert read_status(tmp_path, "jupyter") is None

    path = status_path(tmp_path, "jupyter")
    path.parent.mkdir(parents=True)
    path.write_text("{not json")
    assert read_status(tmp_path, "jupyter") is None

    path.write_text(json.dumps({"state": "exploded", "reason": None, "recorded_at": "x"}))
    assert read_status(tmp_path, "jupyter") is None

    path.write_text(json.dumps({"state": "failed", "reason": 7, "recorded_at": "x"}))
    assert read_status(tmp_path, "jupyter") is None


def test_a_failed_status_without_a_reason_reads_as_none(tmp_path):
    path = status_path(tmp_path, "jupyter")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"state": "failed", "reason": None, "recorded_at": "x"}))

    assert read_status(tmp_path, "jupyter") is None
    assert SidecarStatus.from_json({"state": "failed", "recorded_at": "x"}) is None


def test_the_reason_is_one_line_with_the_last_stderr_line():
    reason = failure_reason(
        "Notebook sidecar exited with status 1 before it was ready\nTraceback ...",
        "Traceback (most recent call last):\n  ...\nModuleNotFoundError: No module named 'x'\n",
        None,
    )

    assert reason == (
        "Notebook sidecar exited with status 1 before it was ready: "
        "ModuleNotFoundError: No module named 'x'"
    )
    assert "\n" not in reason
    # A tail line the message already carries is not repeated.
    assert failure_reason("no interpreter: boom", "boom", None) == "no interpreter: boom"


def test_the_reason_never_carries_the_launch_token():
    reason = failure_reason(
        "did not answer at http://127.0.0.1:1/?token=s3cret",
        "GET /api/status?token=s3cret 403",
        "s3cret",
    )

    assert "s3cret" not in reason
    assert "<token>" in reason


def test_a_long_reason_is_cut():
    reason = failure_reason("x" * 1000, "", None)

    assert len(reason) == 240
    assert reason.endswith("…")


def test_the_failure_message_is_the_ruled_sentence():
    assert (
        status_message("jupyter", SidecarStatus.failed("boom")) == "JUPYTER failed to start: boom"
    )
    assert status_message("jupyter", SidecarStatus.starting()) == "JUPYTER is starting"


def test_a_running_status_has_no_message():
    assert status_message("jupyter", SidecarStatus.running()) is None
    assert status_message("jupyter", None) is None


def test_a_write_under_an_unwritable_root_does_not_raise(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where the root should be")

    write_status(blocker, "jupyter", SidecarStatus.starting())

    assert read_status(blocker, "jupyter") is None
