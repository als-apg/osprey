"""Model-level tests for dispatch input files.

Covers the ``InputFile`` / ``DispatchRequest`` pydantic models: the additive
``input_files`` field, the ``ingest`` default, and the filename-sanitisation
field validator. Pure model construction — no network, no HTTP client.
"""

from __future__ import annotations

import pytest

from osprey.mcp_server.dispatch_worker.dispatch_api import DispatchRequest, InputFile
from osprey.mcp_server.dispatch_worker.input_files_policy import sanitize_filename


def test_input_files_model_ingest_defaults_true():
    f = InputFile(filename="a.png", mime="image/png", content_b64="AAAA")
    assert f.ingest is True


def test_input_files_model_ingest_explicit_false():
    f = InputFile(filename="a.png", mime="image/png", content_b64="AAAA", ingest=False)
    assert f.ingest is False


def test_input_files_model_defaults_to_none_on_request():
    req = DispatchRequest(prompt="hi", allowed_tools=["Read"])
    assert req.input_files is None


def test_input_files_model_carries_batch():
    req = DispatchRequest(
        prompt="hi",
        allowed_tools=["Read"],
        input_files=[
            {"filename": "a.png", "mime": "image/png", "content_b64": "AAAA"},
            {"filename": "b.csv", "mime": "text/csv", "content_b64": "AAAA", "ingest": False},
        ],
    )
    assert req.input_files is not None
    assert len(req.input_files) == 2
    assert req.input_files[0].ingest is True
    assert req.input_files[1].ingest is False


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("../../etc/passwd", "passwd"),  # basename only, traversal stripped
        ("plot.png", "plot.png"),
        ("a/b/c.txt", "c.txt"),
        ("weird name!.csv", "weird_name_.csv"),
        ("..", "file"),
        (".", "file"),
        ("", "file"),
        (".hidden", "hidden"),
        ("with\x00nul.png", "withnul.png"),
        ("back\\slash\\x.json", "x.json"),
    ],
)
def test_input_files_model_sanitize_filename_helper(raw, expected):
    assert sanitize_filename(raw) == expected


def test_input_files_model_sanitizes_traversal_on_construction():
    f = InputFile(filename="../../secret.png", mime="image/png", content_b64="AAAA")
    assert f.filename == "secret.png"
    assert "/" not in f.filename and ".." not in f.filename


def test_input_files_model_sanitizes_via_request():
    req = DispatchRequest(
        prompt="hi",
        allowed_tools=["Read"],
        input_files=[
            {"filename": "a/b/../evil.json", "mime": "application/json", "content_b64": "AAAA"}
        ],
    )
    assert req.input_files[0].filename == "evil.json"
