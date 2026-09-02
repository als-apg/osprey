"""Tests for ``GET/PUT/DELETE /api/bar-items`` — the per-user bar layout.

The store (``bar_items_store.py``) is pure persistence and raises; this module
pins the HTTP contract built on top of it, which is the half a browser sees:

* **GET** answers the document this deployment would render *right now* — the
  operator's saved arrangement when they have one, the deployment default when
  they do not. It reads the lifespan's cache, never the disk.
* **PUT** carries the revision the editor believed it was holding. A revision
  that no longer matches disk is a **409** carrying the current document, so a
  second tab can adopt it, re-apply its edit and retry rather than clobbering a
  save it never saw. A document this build cannot store is a **422** naming the
  class of problem in the store's own vocabulary. A store that cannot be
  written — a read-only mount, a missing volume — is a **503**, because nothing
  was saved and the operator must be told so rather than shown a success. A
  body with no usable ``rev`` is its own reason, ``bad-rev``, kept apart from
  the store's ``malformed``; an oversized body is a **413** refused before it
  is parsed.
* **DELETE** removes the document. It does not write the deployment default
  back: "saved nothing" and "saved something equal to the default" are
  different states, and only the first follows a later edit of ``web.bar_items``.

Every accepted write refreshes ``app.state.bar_items_effective`` under the same
lock that guarded the write, which is what makes the *next* server-rendered
first paint show the arrangement that was just saved. The SSR test below is the
one that would notice a route that wrote the file and forgot the cache.

The suite drives the **full application** rather than a bare router, because
every one of those facts lives on ``app.state`` and is put there by the
lifespan: the store directory, the vocabulary, the lock and the cache.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
import yaml
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal import bar_items_store
from osprey.interfaces.web_terminal.app import (
    BAR_LAYOUT_VERSION,
    DEFAULT_BAR_LAYOUT,
    MAX_BAR_ITEMS_PER_HOST,
    create_app,
)
from osprey.interfaces.web_terminal.bar_items_store import LAYOUT_FILENAME
from osprey.interfaces.web_terminal.routes import bar_items as bar_items_routes
from osprey.interfaces.web_terminal.routes.bar_items import MAX_REQUEST_BYTES

# ── fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def agent_data_root(tmp_path):
    """A throwaway agent-data root for the lifespan to site both stores under.

    Patched at the resolver, not through ``OSPREY_AGENT_DATA_ROOT``:
    :func:`~osprey_connectors.workspace.resolve_shared_data_root` reads
    ``agent_data.base_dir`` anchored on the project root and does not consult
    that variable, so a test relying on it would write this suite's layouts
    into the repository's own ``var/agent_data`` — shared by every test in the
    session, which is exactly the pollution a per-test store must not have.
    """
    root = tmp_path / "agent_data"
    root.mkdir()
    return root


@pytest.fixture
def workspace_dir(tmp_path):
    ws = tmp_path / "_watch"
    ws.mkdir()
    return ws


@pytest.fixture
def config_path(tmp_path):
    """A minimal render — this suite is about the routes, not ``web.bar_items``."""
    path = tmp_path / "config.yml"
    path.write_text(
        yaml.safe_dump({"project_name": "bar-items-routes"}, sort_keys=False),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def client(agent_data_root, workspace_dir, config_path):
    """The whole app, lifespan run, watching a throwaway workspace."""
    with (
        patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(workspace_dir)},
        ),
        patch(
            "osprey.utils.workspace.resolve_shared_data_root",
            return_value=agent_data_root,
        ),
    ):
        app = create_app(shell_command="echo")
        with TestClient(app) as test_client:
            test_client.app.state.config_path = config_path
            yield test_client


@pytest.fixture
def store_dir(client) -> Path:
    """Where the lifespan decided this operator's document lives."""
    return Path(client.app.state.bar_items_dir)


# ── helpers ────────────────────────────────────────────────────────────────


def document(
    rev: int,
    *,
    header: list | None = None,
    status: list | None = None,
    header_visible: bool = True,
    status_visible: bool = True,
) -> dict:
    """A document this build stores, at *rev*.

    ``logo`` is a single-node type and ``clock`` takes options, so the default
    pair exercises both halves of the vocabulary without any test spelling them
    out.
    """
    return {
        "version": BAR_LAYOUT_VERSION,
        "rev": rev,
        "header": [{"type": "logo"}] if header is None else header,
        "status": [{"type": "clock"}] if status is None else status,
        "header_visible": header_visible,
        "status_visible": status_visible,
    }


def types(layout: dict, host: str) -> list[str]:
    """The item types placed in *host*, in order."""
    return [item["type"] for item in layout[host]]


def error_of(response: httpx.Response) -> str:
    """The machine-readable reason a refusal names."""
    return response.json()["detail"]["error"]


# ── GET ────────────────────────────────────────────────────────────────────


class TestGet:
    def test_answers_the_deployment_default_when_nothing_is_saved(self, client, store_dir):
        response = client.get("/api/bar-items")

        assert response.status_code == 200
        body = response.json()
        assert set(body) == {
            "version",
            "rev",
            "header",
            "status",
            "header_visible",
            "status_visible",
        }
        assert body["version"] == BAR_LAYOUT_VERSION
        assert body["rev"] == 0
        assert body["header_visible"] is True
        assert body["status_visible"] is True
        assert body["header"] and body["status"]
        assert not (store_dir / LAYOUT_FILENAME).exists(), "a read must never create the store"

    def test_answers_the_saved_document_after_a_put(self, client):
        client.put("/api/bar-items", json=document(0, status=[{"type": "clock"}]))

        body = client.get("/api/bar-items").json()

        assert body["rev"] == 1
        assert types(body, "header") == ["logo"]
        assert types(body, "status") == ["clock"]


# ── PUT ────────────────────────────────────────────────────────────────────


class TestPut:
    def test_round_trip_assigns_the_next_revision(self, client, store_dir):
        response = client.put("/api/bar-items", json=document(0))

        assert response.status_code == 200
        body = response.json()
        assert body["rev"] == 1
        assert types(body, "header") == ["logo"]
        assert (store_dir / LAYOUT_FILENAME).exists()

        second = client.put("/api/bar-items", json=document(1, header=[{"type": "identity"}]))
        assert second.status_code == 200
        assert second.json()["rev"] == 2

    def test_completes_declared_options_from_their_defaults(self, client):
        body = client.put("/api/bar-items", json=document(0)).json()

        assert body["status"] == [
            {"type": "clock", "options": {"zone": "none", "format": "24h", "seconds": False}}
        ]

    def test_stores_the_visibility_flag(self, client):
        body = client.put("/api/bar-items", json=document(0, status_visible=False)).json()

        assert body["status_visible"] is False
        assert client.get("/api/bar-items").json()["status_visible"] is False

    def test_a_hidden_header_round_trips_like_the_status_bar(self, client):
        body = client.put("/api/bar-items", json=document(0, header_visible=False)).json()

        assert body["header_visible"] is False
        assert body["status_visible"] is True
        assert client.get("/api/bar-items").json()["header_visible"] is False

    def test_a_stale_revision_is_a_409_carrying_the_current_document(self, client):
        client.put("/api/bar-items", json=document(0, header=[{"type": "logo"}]))

        response = client.put("/api/bar-items", json=document(0, header=[{"type": "identity"}]))

        assert response.status_code == 409
        detail = response.json()["detail"]
        assert detail["error"] == "rev_conflict"
        assert detail["layout"]["rev"] == 1
        assert types(detail["layout"], "header") == ["logo"], "the 409 must carry what IS stored"

    def test_the_refused_save_left_the_stored_document_alone(self, client):
        client.put("/api/bar-items", json=document(0, header=[{"type": "logo"}]))
        client.put("/api/bar-items", json=document(0, header=[{"type": "identity"}]))

        assert types(client.get("/api/bar-items").json(), "header") == ["logo"]

    @pytest.mark.parametrize(
        ("body", "reason"),
        [
            pytest.param(document(0, header=[{"type": "nonesuch"}]), "unknown-type", id="unknown"),
            pytest.param(
                document(0, header=[{"type": "clock"}] * (MAX_BAR_ITEMS_PER_HOST + 1)),
                "overflow",
                id="overflow",
            ),
            pytest.param(document(0, header=["logo"]), "malformed", id="entry-not-an-object"),
            pytest.param(document(0, header="logo"), "malformed", id="host-not-a-list"),
            pytest.param(
                document(0, status_visible="yes"), "malformed", id="visibility-not-a-boolean"
            ),
            pytest.param(
                document(0, header_visible=0), "malformed", id="header-visibility-not-a-boolean"
            ),
            pytest.param(
                document(0, status=[{"type": "clock", "options": {"zone": "mars"}}]),
                "bad-option",
                id="enum-out-of-spec",
            ),
            pytest.param(
                document(0, status=[{"type": "space", "options": {"width": 4000}}]),
                "bad-option",
                id="number-out-of-range",
            ),
            pytest.param(
                document(0, status=[{"type": "docs"}, {"type": "docs"}]),
                "duplicate",
                id="single-node-type-twice",
            ),
        ],
    )
    def test_a_document_this_build_cannot_store_is_a_422(self, client, body, reason, store_dir):
        response = client.put("/api/bar-items", json=body)

        assert response.status_code == 422
        assert error_of(response) == reason
        assert response.json()["detail"]["message"]
        assert not (store_dir / LAYOUT_FILENAME).exists(), "a refusal must write nothing"

    def test_a_version_this_build_cannot_read_is_a_422(self, client):
        body = document(0)
        body["version"] = BAR_LAYOUT_VERSION + 1

        response = client.put("/api/bar-items", json=body)

        assert response.status_code == 422
        assert error_of(response) == "version"

    @pytest.mark.parametrize(
        "rev",
        [
            pytest.param(None, id="absent"),
            pytest.param("1", id="a-string"),
            pytest.param(True, id="a-boolean"),
            pytest.param(-1, id="negative"),
        ],
    )
    def test_a_body_without_a_usable_revision_is_a_422_bad_rev(self, client, rev):
        """``bad-rev``, not ``malformed`` — a missing protocol field is a client
        bug, and 2.6 branches on the two differently."""
        body = document(0)
        if rev is None:
            del body["rev"]
        else:
            body["rev"] = rev

        response = client.put("/api/bar-items", json=body)

        assert response.status_code == 422
        assert error_of(response) == "bad-rev"

    def test_a_body_that_is_not_an_object_is_malformed_not_bad_rev(self, client):
        """The document is the broken thing here, so the store's word for it wins."""
        response = client.put("/api/bar-items", json=[{"type": "logo"}])

        assert response.status_code == 422
        assert error_of(response) == "malformed"

    def test_a_body_that_is_not_json_is_a_422(self, client):
        response = client.put(
            "/api/bar-items",
            content=b"{ not json",
            headers={"content-type": "application/json"},
        )

        assert response.status_code == 422
        assert error_of(response) == "malformed"

    def test_an_unwritable_store_is_a_503(self, client):
        with patch(
            "osprey.interfaces.web_terminal.routes.bar_items.save_layout",
            side_effect=OSError("read-only file system"),
        ):
            response = client.put("/api/bar-items", json=document(0))

        assert response.status_code == 503
        assert error_of(response) == "store_write_failed"
        assert client.get("/api/bar-items").json()["rev"] == 0, "the cache must not have moved"

    def test_a_deployment_with_no_store_is_a_503(self, client):
        client.app.state.bar_items_dir = None

        response = client.put("/api/bar-items", json=document(0))

        assert response.status_code == 503
        assert error_of(response) == "store_unavailable"

    def test_a_deployment_with_no_lock_is_a_503(self, client, store_dir):
        """Refused, never served with a privately built lock.

        A lock made per request would look like a safety net and would not be
        one: two concurrent requests would each build their own and each
        overwrite the other's, serializing nothing.
        """
        client.app.state.bar_items_lock = None

        response = client.put("/api/bar-items", json=document(0))

        assert response.status_code == 503
        assert error_of(response) == "store_unavailable"
        assert not (store_dir / LAYOUT_FILENAME).exists()

    def test_a_body_over_the_ceiling_is_a_413(self, client, store_dir):
        oversized = document(0, header=[{"type": "logo", "label": "x" * MAX_REQUEST_BYTES}])

        response = client.put("/api/bar-items", json=oversized)

        assert response.status_code == 413
        assert error_of(response) == "too_large"
        assert not (store_dir / LAYOUT_FILENAME).exists()

    def test_the_ceiling_is_checked_before_the_body_is_parsed(self, client):
        """The refusal comes from the declared size, not from the parse.

        The point of the ceiling is that an oversized save costs nothing to
        refuse — so the handler must never reach the document at all.
        """
        oversized = document(0, header=[{"type": "logo", "label": "x" * MAX_REQUEST_BYTES}])

        with patch.object(
            bar_items_routes, "_requested_rev", side_effect=AssertionError("the body was parsed")
        ):
            response = client.put("/api/bar-items", json=oversized)

        assert response.status_code == 413


# ── DELETE ─────────────────────────────────────────────────────────────────


class TestDelete:
    def test_resets_to_the_deployment_default(self, client, store_dir):
        default = client.get("/api/bar-items").json()
        client.put("/api/bar-items", json=document(0, header=[{"type": "identity"}]))

        response = client.delete("/api/bar-items")

        assert response.status_code == 200
        assert response.json() == default
        assert client.get("/api/bar-items").json() == default
        assert not (store_dir / LAYOUT_FILENAME).exists()

    def test_clears_the_cache_rather_than_writing_the_default_back(self, client):
        client.put("/api/bar-items", json=document(0))

        client.delete("/api/bar-items")

        assert client.app.state.bar_items_effective is None

    def test_with_nothing_saved_it_still_answers_the_default(self, client):
        default = client.get("/api/bar-items").json()

        response = client.delete("/api/bar-items")

        assert response.status_code == 200
        assert response.json() == default

    def test_a_document_that_cannot_be_removed_is_a_503(self, client):
        client.put("/api/bar-items", json=document(0, header=[{"type": "identity"}]))

        with patch(
            "osprey.interfaces.web_terminal.routes.bar_items.reset_layout",
            side_effect=OSError("read-only file system"),
        ):
            response = client.delete("/api/bar-items")

        assert response.status_code == 503
        assert error_of(response) == "store_write_failed"
        assert types(client.get("/api/bar-items").json(), "header") == ["identity"]

    def test_a_deployment_with_no_store_is_a_503(self, client):
        client.app.state.bar_items_dir = None

        response = client.delete("/api/bar-items")

        assert response.status_code == 503
        assert error_of(response) == "store_unavailable"


# ── the cache and the first paint ──────────────────────────────────────────


#: Directory mode bits only mean something here on POSIX, and not to root.
_MODE_BITS_BIND = os.name == "posix" and getattr(os, "geteuid", lambda: 0)() != 0


@pytest.mark.skipif(
    not _MODE_BITS_BIND, reason="directory mode bits do not refuse writes here (Windows, or root)"
)
class TestARealReadOnlyStore:
    """The 503 rung against a genuinely unwritable directory.

    The patched-``OSError`` tests above pin the handler's *mapping*. They
    cannot see the chain the contract actually rests on: ``mkstemp`` in a
    read-only directory, ``write_json_atomic`` letting the error out, the route
    answering 503. A ``_json_store`` change that swallowed that error would
    leave those tests green while every save on a read-only mount answered 200
    having written nothing — the silent failure this rung exists to prevent.
    """

    @pytest.fixture
    def readonly_store(self, client, store_dir):
        """The resolved store directory, present and not writable."""
        store_dir.mkdir(parents=True, exist_ok=True)
        store_dir.chmod(0o500)
        yield store_dir
        store_dir.chmod(0o700)

    def test_a_save_into_it_is_a_503(self, client, readonly_store):
        response = client.put("/api/bar-items", json=document(0))

        assert response.status_code == 503
        assert error_of(response) == "store_write_failed"
        assert not (readonly_store / LAYOUT_FILENAME).exists()

    def test_removing_a_stored_document_from_it_is_a_503(self, client, store_dir):
        client.put("/api/bar-items", json=document(0, header=[{"type": "identity"}]))
        store_dir.chmod(0o500)
        try:
            response = client.delete("/api/bar-items")
        finally:
            store_dir.chmod(0o700)

        assert response.status_code == 503
        assert error_of(response) == "store_write_failed"
        assert types(client.get("/api/bar-items").json(), "header") == ["identity"]

    def test_removing_a_document_that_was_never_there_still_succeeds(self, client, readonly_store):
        """Nothing to unlink is not a write, so the mode bits never come into it."""
        response = client.delete("/api/bar-items")

        assert response.status_code == 200


class TestServerRenderedFirstPaint:
    """The reason an accepted write refreshes the cache at all."""

    def test_the_next_render_paints_the_saved_arrangement(self, client):
        assert 'data-bar-item="separator"' not in client.get("/").text

        client.put(
            "/api/bar-items",
            json=document(0, status=[{"type": "separator"}, {"type": "clock"}]),
        )

        assert 'data-bar-item="separator"' in client.get("/").text

    def test_a_reset_paints_the_deployment_default_again(self, client):
        client.put("/api/bar-items", json=document(0, status=[{"type": "separator"}]))

        client.delete("/api/bar-items")

        assert 'data-bar-item="separator"' not in client.get("/").text


# ── concurrency ────────────────────────────────────────────────────────────


class TestTwoTabsSavingAtOnce:
    """What ``app.state.bar_items_lock`` is for.

    The window the lock closes is between the store's own read of the stored
    revision and its write: both saves claim ``rev`` 0, and without
    serialization both can read 0 and both can write, so the second silently
    overwrites an arrangement its editor never saw. Held, the loser is told.
    """

    async def test_the_second_of_two_concurrent_puts_gets_a_409(self, client):
        transport = httpx.ASGITransport(app=client.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as web:
            first, second = await asyncio.gather(
                web.put("/api/bar-items", json=document(0, header=[{"type": "logo"}])),
                web.put("/api/bar-items", json=document(0, header=[{"type": "identity"}])),
            )

        statuses = sorted(response.status_code for response in (first, second))
        assert statuses == [200, 409], "exactly one save may win"

        loser = first if first.status_code == 409 else second
        assert loser.json()["detail"]["layout"]["rev"] == 1
        assert client.get("/api/bar-items").json()["rev"] == 1, "one save, one revision"


# ── a stored document this build cannot read ───────────────────────────────


@contextmanager
def _app_over(stored: bytes | None, *, agent_data_root: Path, workspace_dir: Path, config_path):
    """The whole app, booted with *stored* already sitting in the store.

    The same wiring as the ``client`` fixture, with the document written before
    the lifespan runs — which is the only way to reach the boot-time load with
    something on disk.
    """
    if stored is not None:
        store = agent_data_root / "bar_items"
        store.mkdir(parents=True, exist_ok=True)
        (store / LAYOUT_FILENAME).write_bytes(stored)
    with (
        patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(workspace_dir)},
        ),
        patch(
            "osprey.utils.workspace.resolve_shared_data_root",
            return_value=agent_data_root,
        ),
    ):
        app = create_app(shell_command="echo")
        with TestClient(app) as test_client:
            test_client.app.state.config_path = config_path
            yield test_client


#: Two ways a stored document can be unreadable, each carrying a marker item
#: (``separator``, which the deployment default never places) so a render that
#: adopted it would be visible.
_UNREADABLE_DOCUMENTS = {
    "invalid-json": b'{"version": 1, "rev": 2, "status": [{"type": "separator"',
    "unreadable-version": json.dumps(
        {
            "version": BAR_LAYOUT_VERSION + 1,
            "rev": 7,
            "header": [{"type": "logo"}],
            "status": [{"type": "separator"}],
            "status_visible": True,
        }
    ).encode("utf-8"),
}


class TestAStoredDocumentThisBuildCannotRead:
    """The route half of the corrupt-prefs contract.

    ``test_file_watcher.py`` pins that neither of these stops the boot. What a
    browser sees is pinned here: the API answers the deployment default at 200
    rather than a 500 or a half-read document, the first paint is that same
    default, and the operator can save over the damaged file — a reset is not
    the only way out of one.
    """

    @pytest.fixture(params=sorted(_UNREADABLE_DOCUMENTS), ids=sorted(_UNREADABLE_DOCUMENTS))
    def unreadable_client(self, request, agent_data_root, workspace_dir, config_path):
        with _app_over(
            _UNREADABLE_DOCUMENTS[request.param],
            agent_data_root=agent_data_root,
            workspace_dir=workspace_dir,
            config_path=config_path,
        ) as test_client:
            yield test_client

    def test_the_route_answers_the_deployment_default(self, unreadable_client):
        response = unreadable_client.get("/api/bar-items")

        assert response.status_code == 200
        layout = response.json()
        assert layout["rev"] == 0, "nothing readable is stored, so nothing has been saved"
        assert types(layout, "status") == types(DEFAULT_BAR_LAYOUT, "status")
        assert types(layout, "header") == types(DEFAULT_BAR_LAYOUT, "header")

    def test_the_first_paint_is_the_deployment_default(self, unreadable_client):
        page = unreadable_client.get("/").text

        assert 'data-bar-item="separator"' not in page, "the damaged document must not render"
        assert 'data-bar-item="clock"' in page

    def test_the_cache_is_empty_rather_than_holding_the_damaged_document(self, unreadable_client):
        assert unreadable_client.app.state.bar_items_effective is None

    def test_the_operator_can_save_over_it_at_revision_zero(self, unreadable_client):
        """``rev`` 0 is what a client holding the default believes, and an
        unreadable file stores no revision to conflict with."""
        response = unreadable_client.put(
            "/api/bar-items", json=document(0, header=[{"type": "identity"}])
        )

        assert response.status_code == 200
        assert response.json()["rev"] == 1
        assert types(unreadable_client.get("/api/bar-items").json(), "header") == ["identity"]


# ── the watcher does not announce a save ───────────────────────────────────


def _broadcast_paths(queue, *, until: str, timeout: float = 15.0) -> list[str]:
    """Every path broadcast up to and shortly after *until* arrives.

    The control write is made **after** the save, so its frame arriving is
    proof the observer has caught up past the save — the ordering that makes an
    assertion about the save's *absence* a real one rather than a race. A short
    drain afterwards catches a coalesced frame delivered a beat late.
    """
    paths: list[str] = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            paths.append(queue.get_nowait()["path"])
        except asyncio.QueueEmpty:
            time.sleep(0.05)
            continue
        if paths[-1] == until:
            break
    time.sleep(0.5)
    while True:
        try:
            paths.append(queue.get_nowait()["path"])
        except asyncio.QueueEmpty:
            return paths


@pytest.mark.real_workspace_watcher
class TestALayoutSaveIsNotAFileChange:
    """A save must be as silent as filing feedback.

    The store sits under the agent-data root, which on a stock deployment is
    inside the watched workspace. Unconcealed, every rearrangement would push
    an SSE ``created``/``modified`` frame at every connected browser — the file
    panel would blink each time anyone moved a clock. ``test_file_watcher.py``
    pins that the lifespan hands the store's relative path to the watcher; this
    is the same claim end to end, with a **real** observer and a real PUT.
    """

    @pytest.fixture
    def watched_workspace(self, tmp_path):
        workspace = tmp_path / "watched"
        (workspace / "agent_data").mkdir(parents=True)
        return workspace

    @pytest.fixture
    def watched_client(self, watched_workspace, config_path):
        """The app watching a tree that *contains* its own store."""
        with _app_over(
            None,
            agent_data_root=watched_workspace / "agent_data",
            workspace_dir=watched_workspace,
            config_path=config_path,
        ) as test_client:
            yield test_client

    def test_the_store_is_inside_the_watched_tree(self, watched_client):
        """The premise of the test below: with the store outside, concealment
        is not what would be keeping the frames away."""
        assert watched_client.app.state.bar_items_rel is not None

    def test_a_save_broadcasts_nothing_while_a_plain_write_still_does(
        self, watched_client, watched_workspace
    ):
        """The steady state: the store directory exists, so the save touches
        nothing but the document and its atomic-write temp file."""
        queue = watched_client.app.state.broadcaster.subscribe()
        # Create the store, then let its frames drain: creating a directory
        # modifies its parent, and that parent frame belongs to the first-save
        # test below rather than to this one.
        watched_client.put("/api/bar-items", json=document(0))
        (watched_workspace / "settle-note.txt").write_text("first")
        _broadcast_paths(queue, until="settle-note.txt")

        saved = watched_client.put(
            "/api/bar-items", json=document(1, status=[{"type": "separator"}])
        )
        assert saved.status_code == 200
        (watched_workspace / "control-note.txt").write_text("ordinary workspace content")

        paths = _broadcast_paths(queue, until="control-note.txt")

        assert "control-note.txt" in paths, "the observer is live and the tree is watched"
        # Everything *below* the agent-data root, not the root's own directory
        # frame: that one is legitimate (see the sibling test) and the FSEvents
        # backend can replay it from before the observer started.
        assert [path for path in paths if path.startswith("agent_data/")] == [], (
            "a layout save must not reach the file panel"
        )

    def test_without_concealment_the_same_save_does_reach_the_file_panel(
        self, watched_workspace, config_path
    ):
        """The assertion above, mutation-checked in place rather than by hand.

        With ``resolve_store_rel`` answering ``None``, the lifespan hands the
        watcher an empty ``concealed`` collection and the identical save
        broadcasts the document it wrote. This is what stops the narrowed
        ``agent_data/`` prefix from being an assertion that can no longer fail.
        """
        with patch(
            "osprey.interfaces.web_terminal.app.resolve_store_rel",
            return_value=None,
        ):
            with _app_over(
                None,
                agent_data_root=watched_workspace / "agent_data",
                workspace_dir=watched_workspace,
                config_path=config_path,
            ) as unconcealed:
                assert unconcealed.app.state.bar_items_rel is None
                queue = unconcealed.app.state.broadcaster.subscribe()

                unconcealed.put("/api/bar-items", json=document(0))
                (watched_workspace / "control-note.txt").write_text("ordinary content")

                paths = _broadcast_paths(queue, until="control-note.txt")

        assert [path for path in paths if path.startswith("agent_data/")] != []

    def test_the_first_ever_save_never_names_the_store_or_the_document(
        self, watched_client, watched_workspace
    ):
        """The first save also *creates* the store directory, and creating a
        directory modifies its parent — so the agent-data root itself can fire
        one frame, exactly as the first feedback record makes it fire one. What
        must never appear is the store or the document inside it: those are
        concealed, and they are the paths that would repeat on every edit."""
        queue = watched_client.app.state.broadcaster.subscribe()

        watched_client.put("/api/bar-items", json=document(0, status=[{"type": "separator"}]))
        (watched_workspace / "control-note.txt").write_text("ordinary workspace content")

        paths = _broadcast_paths(queue, until="control-note.txt")

        assert "control-note.txt" in paths
        assert [path for path in paths if path.startswith("agent_data/bar_items")] == []
        assert [path for path in paths if LAYOUT_FILENAME in path] == []


# ── the refusal vocabulary ─────────────────────────────────────────────────


_STORE_SOURCE = inspect.getsource(bar_items_store)

#: Every reason the *store* raises, read off its own source rather than copied.
#: A new reason added there fails the union test below until this route has
#: a case for it — which is the whole point of deriving it.
STORE_REASONS = frozenset(re.findall(r'BarLayoutInvalid\(\s*"([a-z-]+)"', _STORE_SOURCE))

#: How many times the store raises at all. The scrape above only reads a
#: double-quoted literal of lowercase letters and hyphens, so a reason raised
#: through a variable or spelled some other way would be invisible to it and
#: every assertion here would still pass. Comparing the two counts turns that
#: silent under-count into a red test.
STORE_RAISE_SITES = len(re.findall(r"raise BarLayoutInvalid\(", _STORE_SOURCE))

#: The 422 reason this route adds on top. A body that carries no usable ``rev``
#: is a client bug, not a document the store judged.
ROUTE_ONLY_REASONS = frozenset({"bad-rev"})

#: Every machine-readable ``detail.error`` token the *route* spells for itself,
#: derived the same way. The 422 pass-through (``_refuse(422, exc.reason, …)``)
#: carries no literal and is therefore absent — those are STORE_REASONS.
ROUTE_TOKENS = frozenset(
    re.findall(
        r'_refuse\(\s*(?:\d+,\s*)?"([a-z_-]+)"',
        inspect.getsource(bar_items_routes),
    )
)

#: The tokens outside the 422 family: one per non-422 rung of the ladder.
NON_422_TOKENS = frozenset({"rev_conflict", "too_large", "store_unavailable", "store_write_failed"})


class TestTheRefusalVocabulary:
    """Every ``detail.error`` token the route can emit, derived not repeated.

    Two families, and they are spelled differently. The **422** family is the
    store's own six plus this module's ``bad-rev``, all kebab-case, matching
    ``bar-layout.js``'s client-side drop reasons word for word. The other rungs
    of the ladder — 409, 413 and the two 503s — carry snake_case tokens
    (``rev_conflict``, ``too_large``, ``store_unavailable``,
    ``store_write_failed``), the spelling the rest of the app already uses for
    the same failures. The module docstring names both families and states that
    rule; this class is what holds it to them, because a reason the route can
    emit and no test knows about is exactly the hole it exists to close, and the
    casing split is the trap for anyone matching on the word.

    The client does **not** switch on any of them: ``bar-sync.js`` classifies a
    refusal by *status* (``reasonFor()``: 409 → ``conflict``, 422 → ``invalid``,
    503 → ``unavailable``, anything else → ``network``), so the whole 422 family
    collapses into one ``BarSyncError`` class there and a 413 is classified
    ``network``. That is what makes the server-side union the thing worth
    pinning: a new token needs no client change, and nothing else would notice
    it appearing.
    """

    #: One body per store reason, plus the route's own.
    REFUSALS = {
        "malformed": document(0, header="logo"),
        "version": {**document(0), "version": BAR_LAYOUT_VERSION + 1},
        "unknown-type": document(0, header=[{"type": "nonesuch"}]),
        "overflow": document(0, header=[{"type": "clock"}] * (MAX_BAR_ITEMS_PER_HOST + 1)),
        "duplicate": document(0, status=[{"type": "docs"}, {"type": "docs"}]),
        "bad-option": document(0, status=[{"type": "clock", "options": {"zone": "mars"}}]),
        "bad-rev": {key: value for key, value in document(0).items() if key != "rev"},
    }

    def test_the_store_reasons_are_the_six_the_route_documents(self):
        assert STORE_REASONS == {
            "malformed",
            "version",
            "unknown-type",
            "overflow",
            "duplicate",
            "bad-option",
        }

    def test_the_scrape_read_every_place_the_store_raises(self):
        """Otherwise the derivation above could under-count in silence, and a
        new reason spelled some other way would leave every assertion in
        this class green while the route emitted a word nobody pinned."""
        matched = re.findall(r'raise BarLayoutInvalid\(\s*"[a-z-]+"', _STORE_SOURCE)

        assert len(matched) == STORE_RAISE_SITES

    def test_every_reason_has_a_body_that_provokes_it(self):
        assert set(self.REFUSALS) == STORE_REASONS | ROUTE_ONLY_REASONS

    def test_each_body_provokes_its_own_reason_and_nothing_else(self, client):
        """Per body, not just per set: the store's checks are ordered on
        purpose (a type has to be known before its hosts can be asked), and a
        reordering that swapped two reasons would survive a set comparison."""
        emitted = set()
        for reason, body in self.REFUSALS.items():
            response = client.put("/api/bar-items", json=body)
            assert response.status_code == 422, f"{reason} must be a 422"
            assert error_of(response) == reason
            emitted.add(error_of(response))

        assert emitted == STORE_REASONS | ROUTE_ONLY_REASONS

    def test_the_documented_list_names_every_422_reason(self):
        """The module docstring is what a client author reads. It names the
        whole 422 family; the four tokens below are described there by status
        and by consequence, but not by word — see the class docstring."""
        docstring = inspect.getdoc(bar_items_routes) or ""

        for reason in STORE_REASONS | ROUTE_ONLY_REASONS:
            assert f"``{reason}``" in docstring, f"{reason} is emitted but not documented"

    def test_the_rest_of_the_ladder_spells_exactly_four_more_tokens(self):
        """Everything the route names for itself, minus the 422 family. A fifth
        appearing here is a rung nothing below exercises."""
        assert ROUTE_TOKENS - (STORE_REASONS | ROUTE_ONLY_REASONS) == NON_422_TOKENS

    def test_each_of_those_four_is_reachable_over_http(self, client, store_dir):
        """Derived tokens are worth nothing unless a request produces them.

        Four requests, one per rung, in the order that lets each one set up the
        next: a save to have a revision to conflict with, then the ceiling, then
        a write that fails, then a deployment with the store taken away.
        """
        assert client.put("/api/bar-items", json=document(0)).status_code == 200

        oversized = document(0, header=[{"type": "logo", "label": "x" * MAX_REQUEST_BYTES}])
        with patch(
            "osprey.interfaces.web_terminal.routes.bar_items.save_layout",
            side_effect=OSError("read-only file system"),
        ):
            write_failed = client.put("/api/bar-items", json=document(1))

        emitted = [
            (409, error_of(client.put("/api/bar-items", json=document(0)))),
            (413, error_of(client.put("/api/bar-items", json=oversized))),
            (write_failed.status_code, error_of(write_failed)),
        ]
        client.app.state.bar_items_dir = None
        no_store = client.put("/api/bar-items", json=document(1))
        emitted.append((no_store.status_code, error_of(no_store)))

        assert emitted == [
            (409, "rev_conflict"),
            (413, "too_large"),
            (503, "store_write_failed"),
            (503, "store_unavailable"),
        ]
        assert {token for _, token in emitted} == NON_422_TOKENS
