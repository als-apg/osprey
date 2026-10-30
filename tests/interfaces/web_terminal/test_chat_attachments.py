"""Tests for the chat attachment routes (`routes/chat_attachments.py`).

The composer uploads each file the operator attaches, and the stored path is
what the agent is given to read. These tests hold the store to what it promises:
a file is refused unless its name and its content are both on the list; a name
can not reach outside its chat's directory; the caps bind while the body is
still arriving; nothing refused or abandoned is left behind; and the gate in
front of the route refuses an unauthenticated or cross-origin write.

The size tests drive a raw ASGI request rather than the test client, because they
count how many body chunks the app pulled before it refused.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.common_middleware import OPERATOR_SECRET_HEADER, WebAuthMiddleware
from osprey.interfaces.web_auth import WebCredentials
from osprey.interfaces.web_terminal.app import create_app
from osprey.interfaces.web_terminal.routes import chat_attachments
from osprey.interfaces.web_terminal.routes.chat_attachments import (
    BATCH_HEADER,
    NAME_HEADER,
    router,
)

from .conftest import bare_route_app

# Every gated app here is handed its credential explicitly, or none at all. The
# suite-wide seam would add the operator secret to each request and admit the
# very uploads the gate tests expect refused.
pytestmark = pytest.mark.no_auth_seam

CHAT = "aaaaaaaa-1111-2222-3333-444444444444"
OTHER_CHAT = "bbbbbbbb-1111-2222-3333-444444444444"
BATCH = "batch-0001-aaaa"
COOKIE_NAME = "osprey_terminal_session_8080"
OPERATOR_SECRET = "operator-secret-value"
PANEL_TOKEN = "panel-token-value"

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + b"\x00" * 64
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64
GIF = b"GIF89a" + b"\x01\x00" * 8
WEBP = b"RIFF\x24\x00\x00\x00WEBPVP8 " + b"\x00" * 32
PDF = b"%PDF-1.7\n1 0 obj\n<< >>\nendobj\n%%EOF\n"
TEXT = b"timestamp,value\n2026-10-22T12:00,3.1\n"
HTML = b"<!DOCTYPE html>\n<html><body><script>alert(1)</script></body></html>"

_STORED = re.compile(r"[0-9a-f]{16}-(?P<name>.+)")


@pytest.fixture
def agent_root(tmp_path, monkeypatch) -> Path:
    """Pin the agent-data root into *tmp_path*, so nothing lands in the repository."""
    root = tmp_path / "agent_data"
    root.mkdir()
    monkeypatch.setattr(chat_attachments, "resolve_agent_data_root", lambda app=None: str(root))
    return root


def _headers(name: str | None, batch: str | None = BATCH) -> dict[str, str]:
    headers: dict[str, str] = {}
    if name is not None:
        headers[NAME_HEADER] = quote(name, safe="")
    if batch is not None:
        headers[BATCH_HEADER] = batch
    return headers


def _upload(client, name, body, *, chat=CHAT, batch=BATCH):
    return client.post(
        f"/api/chat/{chat}/attachments",
        content=body,
        headers=_headers(name, batch),
    )


def _stored_files(root: Path, chat: str = CHAT) -> list[str]:
    chat_dir = root / "uploads" / chat
    if not chat_dir.is_dir():
        return []
    return sorted(entry.name for entry in chat_dir.iterdir())


def _refusal(response) -> tuple[int, str]:
    detail = response.json()["detail"]
    return response.status_code, detail["error"]


@pytest.fixture
def client(agent_root):  # noqa: ARG001 - the fixture pins the root the route reads
    """The attachment router on a bare app: no gate, so the route alone is under test."""
    with TestClient(bare_route_app(router)) as test_client:
        yield test_client


def test_the_composer_states_the_limits_the_server_enforces():
    """The composer refuses a file before uploading it and offers the picker a type
    list. Both copy the server's limits, and a drift would let the operator pick a
    file the server then refuses, or withhold one it would take."""
    source = (
        Path(__file__).parents[3] / "src/osprey/interfaces/web_terminal/static/js/chat-attach.js"
    ).read_text(encoding="utf-8")
    files = re.search(r"export const MAX_FILES_PER_MESSAGE = (\d+);", source)
    size = re.search(r"export const MAX_FILE_BYTES = (\d+) \* 1024 \* 1024;", source)
    accept = re.search(r"const ACCEPT = '([^']+)';", source)
    assert files and size and accept, "the composer no longer states its limits as before"
    assert int(files[1]) == chat_attachments.MAX_FILES_PER_MESSAGE
    assert int(size[1]) * 1024 * 1024 == chat_attachments.MAX_ATTACHMENT_BYTES
    assert {ext.lstrip(".") for ext in accept[1].split(",")} == set(chat_attachments._ALLOWED)


# ---- Storing a file ----


def test_a_png_is_stored_under_the_root_and_its_path_is_returned(client, agent_root):
    """The returned path is what the agent reads, from whatever directory it runs in."""
    response = _upload(client, "alarm panel.png", PNG)

    assert response.status_code == 201
    body = response.json()
    assert set(body) == {"name", "path", "size", "type"}
    assert body["name"] == "alarm_panel.png"
    assert body["size"] == len(PNG)
    assert body["type"] == "image/png"

    stored = Path(body["path"])
    assert stored.is_absolute()
    assert stored.read_bytes() == PNG
    assert stored.is_relative_to(agent_root / "uploads" / CHAT)
    assert _STORED.fullmatch(stored.name)["name"] == "alarm_panel.png"


@pytest.mark.parametrize(
    ("name", "content", "expected_type"),
    [
        ("shot.jpeg", JPEG, "image/jpeg"),
        ("shot.gif", GIF, "image/gif"),
        ("shot.webp", WEBP, "image/webp"),
        ("manual.pdf", PDF, "application/pdf"),
        ("notes.txt", TEXT, "text/plain"),
        ("run.log", TEXT, "text/plain"),
        ("readings.csv", TEXT, "text/csv"),
        ("state.json", b'{"ok": true}\n', "application/json"),
        ("config.yaml", b"a: 1\n", "application/yaml"),
        ("config.yml", b"b: 2\n", "application/yaml"),
        ("readme.md", "# Title\n\nSummary, \u00e9t\u00e9.\n".encode(), "text/markdown"),
        ("probe.py", b"print('ok')\n", "text/x-python"),
        pytest.param("scan.jpg", PNG, "image/png", id="a-jpg-holding-png-bytes"),
        pytest.param("bom.txt", b"\xef\xbb\xbfhello\n", "text/plain", id="text-after-a-bom"),
        pytest.param(
            "quote.log",
            b"A log line that quotes <html> in the middle is still a log.\n",
            "text/plain",
            id="text-mentioning-html-later-on",
        ),
    ],
)
def test_each_allowed_kind_is_accepted_with_the_type_its_content_proves(
    client, name, content, expected_type
):
    response = _upload(client, name, content)
    assert response.status_code == 201, response.text
    assert response.json()["type"] == expected_type


# ---- The type allowlist and the content sniff ----


@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("tool.exe", PNG),
        ("archive.tar.gz", TEXT),
        ("noextension", TEXT),
        ("page.html", HTML),
        ("drawing.svg", b'<svg xmlns="http://www.w3.org/2000/svg"></svg>'),
        ("shot.png", TEXT),
        ("notes.txt", PNG),
        ("notes.txt", HTML),
        ("notes.log", b"<!doctype html><title>x</title>"),
        ("notes.md", b"  \n<HTML lang=en></HTML>"),
        ("notes.txt", b"plain\x00binary"),
        ("notes.txt", b"\xff\xfe\xfd not utf-8"),
        ("manual.pdf", TEXT),
    ],
)
def test_content_must_match_what_the_name_claims(client, agent_root, name, content):
    response = _upload(client, name, content)
    assert _refusal(response) == (415, "unsupported_type")
    assert _stored_files(agent_root) == []


# ---- Names ----


@pytest.mark.parametrize(
    ("name", "stored_name"),
    [
        ("../../etc/passwd.txt", "passwd.txt"),
        ("..\\..\\evil.txt", "evil.txt"),
        ("/srv/secret/shot.txt", "shot.txt"),
        ("weird name (final).txt", "weird_name__final_.txt"),
        ("na\u00efve.txt", "na_ve.txt"),
        ("semi;rm -rf.txt", "semi_rm_-rf.txt"),
        ("a\x00b.txt", "a_b.txt"),
    ],
)
def test_odd_names_are_neutralised_and_stay_in_the_chat_directory(
    client, agent_root, name, stored_name
):
    response = _upload(client, name, TEXT)
    assert response.status_code == 201, response.text
    body = response.json()
    stored = Path(body["path"])

    assert body["name"] == stored_name
    assert stored.parent == agent_root / "uploads" / CHAT
    assert _STORED.fullmatch(stored.name)["name"] == stored_name


def test_a_long_stem_is_cut_and_the_extension_kept(client):
    body = _upload(client, "x" * 300 + ".txt", TEXT).json()
    stem = body["name"].removesuffix(".txt")
    assert len(stem) == 100
    assert body["name"].endswith(".txt")


@pytest.mark.parametrize("name", [None, "", "..", ".env.txt", ".png", "../"])
def test_a_missing_empty_or_dotfile_name_is_refused(client, agent_root, name):
    response = _upload(client, name, TEXT)
    assert _refusal(response) == (400, "invalid_name")
    assert _stored_files(agent_root) == []


# ---- Bodies ----


def test_an_empty_body_is_refused(client, agent_root):
    response = _upload(client, "empty.txt", b"")
    assert _refusal(response) == (400, "empty_file")
    assert _stored_files(agent_root) == []


def test_a_chat_id_outside_the_session_key_grammar_is_refused(client, agent_root):
    response = _upload(client, "shot.png", PNG, chat="not-a-uuid")
    assert _refusal(response) == (400, "invalid_chat_id")
    upper = CHAT.upper()
    assert _refusal(_upload(client, "shot.png", PNG, chat=upper)) == (400, "invalid_chat_id")
    assert not (agent_root / "uploads").exists()


@pytest.mark.parametrize("batch", [None, "", "x", "../../etc", "has space in it"])
def test_an_upload_must_name_a_well_formed_message(client, agent_root, batch):
    response = client.post(
        f"/api/chat/{CHAT}/attachments",
        content=PNG,
        headers=_headers("shot.png", batch),
    )
    assert _refusal(response) == (400, "invalid_batch")
    assert _stored_files(agent_root) == []


# ---- The caps ----


async def _drive(app, path, *, chunks, headers=(), method="POST", disconnect_after=None):
    """POST *chunks* to *app* as a raw ASGI request; report the status, the reads, the body."""
    pending = list(chunks)
    reads = 0
    sent: list[dict] = []

    async def receive():
        nonlocal reads
        reads += 1
        if disconnect_after is not None and reads > disconnect_after:
            return {"type": "http.disconnect"}
        if pending:
            return {"type": "http.request", "body": pending.pop(0), "more_body": bool(pending)}
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "root_path": "",
        "query_string": b"",
        "headers": [(k.encode("latin-1"), v.encode("latin-1")) for k, v in headers],
        "client": ("127.0.0.1", 50000),
        "server": ("testserver", 80),
    }
    await app(scope, receive, send)
    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return status, reads, body


async def test_the_file_cap_is_enforced_while_the_body_is_read(agent_root, monkeypatch):
    monkeypatch.setattr(chat_attachments, "MAX_ATTACHMENT_BYTES", 4096)
    chunks = [b"a" * 1024] * 100
    headers = list(_headers("big.txt").items())

    status, reads, body = await _drive(
        bare_route_app(router), f"/api/chat/{CHAT}/attachments", chunks=chunks, headers=headers
    )

    assert status == 413
    assert json.loads(body)["detail"]["error"] == "file_too_large"
    # Four chunks fit under the cap; the fifth is the one that is refused.
    assert reads == 5
    assert _stored_files(agent_root) == []


async def test_a_declared_length_over_the_cap_is_refused_before_any_byte_is_read(
    agent_root, monkeypatch
):
    monkeypatch.setattr(chat_attachments, "MAX_ATTACHMENT_BYTES", 4096)
    headers = [*_headers("big.txt").items(), ("content-length", "9999")]

    status, reads, body = await _drive(
        bare_route_app(router), f"/api/chat/{CHAT}/attachments", chunks=[b"a" * 10], headers=headers
    )

    assert status == 413
    assert json.loads(body)["detail"]["error"] == "file_too_large"
    assert reads == 0
    assert _stored_files(agent_root) == []


def test_the_chat_cap_counts_what_is_already_stored(client, agent_root, monkeypatch):
    monkeypatch.setattr(chat_attachments, "MAX_CHAT_ATTACHMENT_BYTES", 3000)
    assert _upload(client, "a.txt", b"a" * 2000).status_code == 201
    refused = _upload(client, "b.txt", b"b" * 2000)
    assert _refusal(refused) == (413, "chat_full")

    first = _upload(client, "c.txt", b"c" * 900)
    assert first.status_code == 201
    assert _refusal(_upload(client, "d.txt", b"d" * 200)) == (413, "chat_full")

    on_disk = sum(
        (agent_root / "uploads" / CHAT / n).stat().st_size for n in _stored_files(agent_root)
    )
    assert on_disk <= 3000


def test_removing_a_file_gives_its_bytes_back_to_the_chat(client, monkeypatch):
    monkeypatch.setattr(chat_attachments, "MAX_CHAT_ATTACHMENT_BYTES", 3000)
    kept = _upload(client, "a.txt", b"a" * 2000).json()
    assert _refusal(_upload(client, "b.txt", b"b" * 2000)) == (413, "chat_full")

    removed = client.delete(
        f"/api/chat/{CHAT}/attachments/{Path(kept['path']).name}",
        headers={BATCH_HEADER: BATCH},
    )
    assert removed.status_code == 204
    assert _upload(client, "b.txt", b"b" * 2000).status_code == 201


def test_the_chat_cap_measures_files_already_on_disk_on_first_use(agent_root, monkeypatch):
    """A restarted server does not forget what an earlier one stored."""
    monkeypatch.setattr(chat_attachments, "MAX_CHAT_ATTACHMENT_BYTES", 3000)
    chat_dir = agent_root / "uploads" / CHAT
    chat_dir.mkdir(parents=True)
    (chat_dir / "0123456789abcdef-old.txt").write_bytes(b"o" * 2500)

    with TestClient(bare_route_app(router)) as client:
        assert _refusal(_upload(client, "new.txt", b"n" * 1000)) == (413, "chat_full")


def test_every_chat_together_is_capped_whatever_ids_are_minted(agent_root, monkeypatch):
    """A fresh chat id does not buy a fresh allowance: the tree as a whole is bounded."""
    monkeypatch.setattr(chat_attachments, "MAX_UPLOADS_BYTES", 3000)
    old_dir = agent_root / "uploads" / "cccccccc-1111-2222-3333-444444444444"
    old_dir.mkdir(parents=True)
    (old_dir / "0123456789abcdef-old.txt").write_bytes(b"o" * 1500)

    with TestClient(bare_route_app(router)) as client:
        kept = _upload(client, "a.txt", b"a" * 1000).json()
        assert _refusal(_upload(client, "b.txt", b"b" * 1000, chat=OTHER_CHAT)) == (
            413,
            "uploads_full",
        )
        removed = client.delete(f"/api/chat/{CHAT}/attachments/{Path(kept['path']).name}")
        assert removed.status_code == 204
        assert _upload(client, "b.txt", b"b" * 1000, chat=OTHER_CHAT).status_code == 201


def test_a_message_carries_at_most_ten_files(client, agent_root):
    for index in range(chat_attachments.MAX_FILES_PER_MESSAGE):
        assert _upload(client, f"f{index}.txt", TEXT).status_code == 201

    refused = _upload(client, "eleventh.txt", TEXT)
    assert _refusal(refused) == (422, "too_many_files")
    assert len(_stored_files(agent_root)) == chat_attachments.MAX_FILES_PER_MESSAGE

    assert _upload(client, "next.txt", TEXT, batch="batch-0002-bbbb").status_code == 201


def test_removing_a_file_frees_a_place_in_its_message(client):
    names = []
    for index in range(chat_attachments.MAX_FILES_PER_MESSAGE):
        names.append(Path(_upload(client, f"f{index}.txt", TEXT).json()["path"]).name)

    assert _refusal(_upload(client, "over.txt", TEXT)) == (422, "too_many_files")
    client.delete(f"/api/chat/{CHAT}/attachments/{names[0]}", headers={BATCH_HEADER: BATCH})
    assert _upload(client, "again.txt", TEXT).status_code == 201


async def test_a_client_that_disconnects_mid_upload_leaves_nothing_behind(agent_root):
    status, reads, _body = await _drive(
        bare_route_app(router),
        f"/api/chat/{CHAT}/attachments",
        chunks=[b"x" * 512] * 10,
        headers=list(_headers("half.txt").items()),
        disconnect_after=2,
    )

    assert status == 204
    assert reads >= 2
    assert _stored_files(agent_root) == []


def test_a_root_that_cannot_hold_files_answers_503(tmp_path, monkeypatch):
    blocked = tmp_path / "not_a_directory"
    blocked.write_text("in the way")
    monkeypatch.setattr(chat_attachments, "resolve_agent_data_root", lambda app=None: str(blocked))
    with TestClient(bare_route_app(router)) as client:
        assert _refusal(_upload(client, "shot.png", PNG)) == (503, "store_write_failed")


# ---- Removing a file ----


def test_a_file_removed_before_sending_is_gone(client, agent_root):
    stored = Path(_upload(client, "gone.txt", TEXT).json()["path"]).name
    assert client.delete(f"/api/chat/{CHAT}/attachments/{stored}").status_code == 204
    assert _stored_files(agent_root) == []
    # Removing it again is not an error: the answer is the same either way.
    assert client.delete(f"/api/chat/{CHAT}/attachments/{stored}").status_code == 204


def test_a_name_that_opens_with_a_dash_can_be_removed_again(client, agent_root):
    stored = Path(_upload(client, "-notes.txt", TEXT).json()["path"]).name
    assert stored.endswith("--notes.txt")
    assert client.delete(f"/api/chat/{CHAT}/attachments/{stored}").status_code == 204
    assert _stored_files(agent_root) == []


def test_a_stored_name_is_only_reachable_through_its_own_chat(client, agent_root):
    stored = Path(_upload(client, "theirs.txt", TEXT, chat=OTHER_CHAT).json()["path"]).name

    assert client.delete(f"/api/chat/{CHAT}/attachments/{stored}").status_code == 204
    assert stored in _stored_files(agent_root, OTHER_CHAT)


@pytest.mark.parametrize(
    "stored", ["not-stored.txt", "0123456789abcdef-.hidden", "0123456789abcdef-", "x" * 200]
)
def test_a_removal_names_only_a_stored_file(client, stored):
    response = client.delete(f"/api/chat/{CHAT}/attachments/{stored}")
    assert _refusal(response) == (400, "invalid_name")


def test_nothing_serves_an_upload_back(client):
    stored = Path(_upload(client, "private.txt", TEXT).json()["path"]).name

    assert client.get(f"/api/chat/{CHAT}/attachments").status_code == 405
    assert client.get(f"/api/chat/{CHAT}/attachments/{stored}").status_code in (404, 405)


# ---- Retention: sent files are kept, never-sent leftovers go after a day ----

DAY = chat_attachments.UNSENT_RETENTION_S


def _age(path: Path, seconds: float) -> None:
    """Make *path* look *seconds* old."""
    stamp = time.time() - seconds
    os.utime(path, (stamp, stamp), follow_symlinks=False)


def _composer_prompt(*stored: dict) -> str:
    """The prompt the composer sends: typed text, then the block naming each stored file."""
    lines = "".join(f"\n- {s['path']} ({s['type']}, {s['size']} B)" for s in stored)
    return f"What are these?\n\nAttached files (read them with the Read tool):{lines}"


def _sweep(app, agent_root: Path) -> None:
    chat_attachments.sweep_uploads(agent_root / "uploads", chat_attachments._ledger(app))


def test_a_file_sent_with_a_message_survives_past_a_day(client, agent_root):
    stored = _upload(client, "sent.txt", TEXT).json()
    chat_attachments.mark_sent(client.app, CHAT, _composer_prompt(stored))
    _age(Path(stored["path"]), DAY + 3600)

    _sweep(client.app, agent_root)

    assert Path(stored["path"]).is_file()
    assert Path(stored["path"]).name in _stored_files(agent_root)


def test_a_file_never_sent_is_deleted_after_a_day_and_a_younger_one_stays(client, agent_root):
    old = _upload(client, "old.txt", TEXT).json()
    young = _upload(client, "young.txt", TEXT).json()
    _age(Path(old["path"]), DAY + 60)
    _age(Path(young["path"]), DAY - 3600)

    _sweep(client.app, agent_root)

    assert not Path(old["path"]).exists()
    assert Path(young["path"]).is_file()


def test_a_stale_part_file_is_swept_and_the_manifest_is_kept(client, agent_root):
    stored = _upload(client, "kept.txt", TEXT).json()
    chat_attachments.mark_sent(client.app, CHAT, _composer_prompt(stored))
    chat_dir = agent_root / "uploads" / CHAT
    part = chat_dir / ".0123456789abcdef.part"
    part.write_bytes(b"half an upload")
    _age(part, DAY + 60)
    _age(chat_dir / chat_attachments.SENT_MANIFEST, DAY + 60)

    _sweep(client.app, agent_root)

    assert not part.exists()
    assert (chat_dir / chat_attachments.SENT_MANIFEST).is_file()
    assert Path(stored["path"]).is_file()


def test_a_prompt_marks_only_its_own_chats_files(client, agent_root):
    """A message of one chat cannot keep another chat's leftover alive, nor any other path."""
    theirs = _upload(client, "theirs.txt", TEXT, chat=OTHER_CHAT).json()
    elsewhere = agent_root / "elsewhere.txt"
    elsewhere.write_bytes(TEXT)
    prompt = _composer_prompt(theirs, {"path": elsewhere, "type": "text/plain", "size": 1})
    chat_attachments.mark_sent(client.app, CHAT, prompt)
    _age(Path(theirs["path"]), DAY + 60)

    _sweep(client.app, agent_root)

    assert not (agent_root / "uploads" / CHAT / chat_attachments.SENT_MANIFEST).exists()
    assert not Path(theirs["path"]).exists()


def test_the_sweep_does_not_follow_a_symlink(client, agent_root, tmp_path):
    """Neither a linked file in a chat directory nor a linked chat directory is entered."""
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "0123456789abcdef-precious.txt"
    target.write_bytes(TEXT)
    _age(target, DAY + 60)
    chat_dir = agent_root / "uploads" / CHAT
    chat_dir.mkdir(parents=True)
    linked_file = chat_dir / "fedcba9876543210-link.txt"
    linked_file.symlink_to(target)
    linked_dir = agent_root / "uploads" / OTHER_CHAT
    linked_dir.symlink_to(outside, target_is_directory=True)
    _age(linked_file, DAY + 60)

    _sweep(client.app, agent_root)

    assert target.is_file()
    assert linked_file.is_symlink()
    assert linked_dir.is_symlink()


def test_the_caps_measure_the_tree_after_the_sweep(client, agent_root, monkeypatch):
    """A leftover the sweep deletes gives its bytes back to the chat and to the tree."""
    monkeypatch.setattr(chat_attachments, "MAX_CHAT_ATTACHMENT_BYTES", 3000)
    monkeypatch.setattr(chat_attachments, "MAX_UPLOADS_BYTES", 3000)
    old = _upload(client, "old.txt", b"o" * 2000).json()
    assert _refusal(_upload(client, "b.txt", b"b" * 2000)) == (413, "chat_full")
    _age(Path(old["path"]), DAY + 60)

    _sweep(client.app, agent_root)

    ledger = chat_attachments._ledger(client.app)
    assert ledger.chat_bytes[CHAT] == 0
    assert ledger.total_bytes == 0
    assert _upload(client, "b.txt", b"b" * 2000).status_code == 201


def test_an_upload_sweeps_first_when_the_last_sweep_is_old(agent_root, monkeypatch):
    """The caps a new upload is held to are those of the swept tree."""
    monkeypatch.setattr(chat_attachments, "MAX_CHAT_ATTACHMENT_BYTES", 3000)
    chat_dir = agent_root / "uploads" / CHAT
    chat_dir.mkdir(parents=True)
    leftover = chat_dir / "0123456789abcdef-leftover.txt"
    leftover.write_bytes(b"o" * 2500)
    _age(leftover, DAY + 60)

    with TestClient(bare_route_app(router)) as client:
        assert _upload(client, "new.txt", b"n" * 1000).status_code == 201
        assert not leftover.exists()
        assert chat_attachments._ledger(client.app).swept_at is not None


def test_the_full_app_sweeps_at_startup(agent_root, isolated_app):
    """A restart deletes the leftovers nobody sent before any upload, and keeps the sent.

    Booted through the shared whole-app builder, which launches no panel server: a
    real launch would probe the host's companion ports from inside a unit test.
    """
    chat_dir = agent_root / "uploads" / CHAT
    chat_dir.mkdir(parents=True)
    leftover = chat_dir / "0123456789abcdef-leftover.txt"
    leftover.write_bytes(TEXT)
    _age(leftover, DAY + 60)
    kept = chat_dir / "fedcba9876543210-kept.txt"
    kept.write_bytes(TEXT)
    _age(kept, DAY + 60)
    (chat_dir / chat_attachments.SENT_MANIFEST).write_text(json.dumps({"sent": [kept.name]}))

    with isolated_app():
        pass

    assert not leftover.exists()
    assert kept.is_file()


# ---- The full app, with its real middleware stack ----


@pytest.fixture
def full_app(agent_root):  # noqa: ARG001 - the fixture pins the root the route reads
    """The web-terminal app as the server builds it, with a known operator secret.

    Built without the test client's lifespan, as the session-endpoint tests build it:
    the stack under test is the middleware, not the terminal's background machinery.
    """
    application = create_app(shell_command=["echo"])
    application.state.web_credentials = WebCredentials(
        operator_secret=OPERATOR_SECRET, panel_token=PANEL_TOKEN
    )
    return application


def test_the_full_app_stores_an_upload_for_the_operator(full_app, agent_root):
    response = TestClient(full_app).post(
        f"/api/chat/{CHAT}/attachments",
        content=PNG,
        headers={**_headers("shot.png"), OPERATOR_SECRET_HEADER: OPERATOR_SECRET},
    )

    assert response.status_code == 201, response.text
    assert Path(response.json()["path"]).read_bytes() == PNG
    assert Path(response.json()["path"]).is_relative_to(agent_root / "uploads" / CHAT)


def test_the_full_app_refuses_a_stream_over_the_cap_and_keeps_nothing(
    full_app, agent_root, monkeypatch
):
    monkeypatch.setattr(chat_attachments, "MAX_ATTACHMENT_BYTES", 4096)
    chunks = (b"a" * 1024 for _ in range(100))

    response = TestClient(full_app).post(
        f"/api/chat/{CHAT}/attachments",
        content=chunks,
        headers={**_headers("big.txt"), OPERATOR_SECRET_HEADER: OPERATOR_SECRET},
    )

    assert _refusal(response) == (413, "file_too_large")
    assert _stored_files(agent_root) == []


# ---- The gate in front of the route ----


@pytest.fixture
def gated(agent_root):  # noqa: ARG001 - the fixture pins the root the route reads
    """The router behind the real gate, with a known operator secret and panel token."""
    app = bare_route_app(router)
    app.state.web_credentials = WebCredentials(
        operator_secret=OPERATOR_SECRET, panel_token=PANEL_TOKEN
    )
    app.add_middleware(WebAuthMiddleware, cookie_name=COOKIE_NAME)
    with TestClient(app, client=("127.0.0.1", 54321)) as test_client:
        test_client.cookies.clear()
        yield SimpleNamespace(client=test_client, credentials=app.state.web_credentials)


@pytest.mark.parametrize(
    "credential",
    [
        pytest.param({}, id="none"),
        pytest.param({"authorization": f"Bearer {PANEL_TOKEN}"}, id="panel-token"),
    ],
)
def test_an_upload_without_the_operator_credential_is_refused(gated, agent_root, credential):
    """The upload is operator-tier: a panel companion is not the operator."""
    response = gated.client.post(
        f"/api/chat/{CHAT}/attachments",
        content=PNG,
        headers={**_headers("s.png"), **credential},
    )
    assert response.status_code == 401
    assert _stored_files(agent_root) == []


def test_the_operator_secret_uploads(gated):
    response = gated.client.post(
        f"/api/chat/{CHAT}/attachments",
        content=PNG,
        headers={**_headers("s.png"), OPERATOR_SECRET_HEADER: OPERATOR_SECRET},
    )
    assert response.status_code == 201


@pytest.mark.parametrize(
    "origin",
    [
        pytest.param({"origin": "https://evil.example"}, id="another-origin"),
        pytest.param({}, id="no-origin"),
    ],
)
def test_a_cookie_upload_not_from_the_app_origin_is_refused(gated, agent_root, origin):
    gated.client.cookies.set(COOKIE_NAME, gated.credentials.create_session())
    response = gated.client.post(
        f"/api/chat/{CHAT}/attachments",
        content=PNG,
        headers={**_headers("s.png"), **origin},
    )
    assert response.status_code == 403
    assert _stored_files(agent_root) == []


def test_a_cookie_upload_from_the_app_origin_is_stored(gated):
    gated.client.cookies.set(COOKIE_NAME, gated.credentials.create_session())
    response = gated.client.post(
        f"/api/chat/{CHAT}/attachments",
        content=PNG,
        headers={**_headers("s.png"), "origin": "http://testserver"},
    )
    assert response.status_code == 201


def test_a_cross_origin_removal_is_refused(gated):
    gated.client.cookies.set(COOKIE_NAME, gated.credentials.create_session())
    stored = Path(
        gated.client.post(
            f"/api/chat/{CHAT}/attachments",
            content=PNG,
            headers={**_headers("s.png"), "origin": "http://testserver"},
        ).json()["path"]
    ).name
    refused = gated.client.delete(
        f"/api/chat/{CHAT}/attachments/{stored}", headers={"origin": "https://evil.example"}
    )
    assert refused.status_code == 403
