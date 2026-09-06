"""Tests for the durability of the deployment's write-posture narrowings.

The narrowings live in one field of one file — ``posture`` in
``control_target/control_context.json`` — and they span restarts on purpose: a
container recreation must not silently revert a narrowed deployment to writes.
Three properties make that true, and they are all pinned here:

* **The shared parser decides the shape.** Every reader of the record decodes
  ``posture`` through :func:`osprey_connectors.control_context.parse_posture`,
  which is the posture store's own entry grammar
  (:func:`osprey_connectors.posture_store.parse_posture_value`): the legacy bare
  ``"sandbox"`` narrows *every* target, a bare ``"writes"`` disappears (absence
  is how a narrowing spells the writes posture), and anything unrecognised is
  dropped. The web server does not get its own filter.

* **A narrowing written by one process is what the next one reads.** The point
  of persisting at all. The web terminal keeps no posture in memory to fall
  back on: a fresh app answers from the file its predecessor left, and a
  narrowing cleared before the restart stays cleared.

* **There is one location, and the retired one is inert.** The per-session
  posture store the record replaced wrote ``session-postures.json`` into the
  same directory. Nothing reads it any more, and nothing is written on its
  account — a reader that still consulted it would honour narrowings the
  operator can no longer see or clear.

Route-level tests mirror ``test_posture_routes.py``: each builds its own app
through ``create_app`` under a patched ``_load_web_config``, entered as a
``TestClient`` context manager so the lifespan runs.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal import control_context_owner
from osprey.interfaces.web_terminal.app import create_app
from osprey_connectors import control_context, posture_store
from osprey_connectors.types import CONTROL_TARGETS
from tests._control_context_fixtures import write_control_context

SANDBOX = posture_store.POSTURE_SANDBOX

#: A session id the POST route accepts. It names who made the gesture and
#: decides nothing about what the gesture does — which is the whole reason
#: this file no longer has a key grammar to test.
SESSION_A = "aaaaaaaa-1111-2222-3333-444444444444"

#: What the legacy bare ``"sandbox"`` means once parsed.
EVERY_TARGET = dict.fromkeys(CONTROL_TARGETS, SANDBOX)

#: The retired per-session store's file name, restated: ``posture_store`` no
#: longer publishes it, and one test here is about that file staying dead.
RETIRED_STORE_NAME = "session-postures.json"

STANDIN_PORT = 5074


# -- the deployment these tests render ---------------------------------------


def _gateways(port):
    row = {"address": "localhost", "port": port, "use_name_server": True}
    return {"read_only": dict(row), "write_access": dict(row)}


def write_config(tmp_path):
    """A three-target render, so ``standin`` is a target that can be narrowed."""
    path = tmp_path / "config.yml"
    path.write_text(
        yaml.safe_dump(
            {
                "control_system": {
                    "type": "live_standin",
                    "writes_enabled": True,
                    "connector": {
                        "epics": {"gateways": _gateways(5064)},
                        "live_standin": {"gateways": _gateways(STANDIN_PORT)},
                        "virtual_accelerator": {
                            "simulation_file": "data/sim.json",
                            "gateways": _gateways(5064),
                        },
                    },
                },
                "services": {
                    "live_standin": {"port": STANDIN_PORT},
                    "virtual_accelerator": {"port": 5064},
                },
                "deployed_services": ["virtual_accelerator", "live_standin"],
            }
        ),
        encoding="utf-8",
    )
    return path


# -- fixtures ----------------------------------------------------------------


@pytest.fixture
def agent_data_root(tmp_path, monkeypatch):
    """One throwaway agent-data root, stamped, with both caches dropped.

    ``OSPREY_AGENT_DATA_ROOT`` is the stamp the record reader and
    ``posture_store`` both prefer — the one this feature puts in every session
    child's environment. Patching ``resolve_shared_data_root`` instead would
    redirect only one of them: ``posture_store`` binds the resolver at import,
    so the other half would read the repository's own ``var/agent_data``.
    """
    root = tmp_path / "agent_data"
    (root / posture_store.STATE_DIR_NAME).mkdir(parents=True)
    monkeypatch.setenv(posture_store.AGENT_DATA_ROOT_ENV_VAR, str(root))
    monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)
    posture_store.invalidate_cache()
    yield root
    posture_store.invalidate_cache()


@pytest.fixture
def workspace_dir(tmp_path):
    ws = tmp_path / "_watch"
    ws.mkdir()
    return ws


@pytest.fixture
def make_client(agent_data_root, workspace_dir, tmp_path):
    """Build an app + TestClient, repeatably, over the same stamped root.

    Repeatably is the point: a "restart" here is a second ``_make()``, which
    is a genuinely fresh app whose only inheritance from the first is the
    directory. The owner task's loop is stubbed out — every tick these tests
    need has happened by the time the client is handed over, and a loop
    ticking underneath the assertions would be a race rather than coverage.
    """

    @contextmanager
    def _make():
        with (
            patch.object(control_context_owner.ControlContextOwnerTask, "start", lambda self: None),
            patch(
                "osprey.interfaces.web_terminal.app._load_web_config",
                return_value={"watch_dir": str(workspace_dir)},
            ),
        ):
            app = create_app(shell_command="echo")
            with TestClient(app) as test_client:
                test_client.app.state.config_path = write_config(tmp_path)
                yield test_client

    return _make


@pytest.fixture
def started(agent_data_root):
    """A record on disk before any app starts, as a predecessor would leave it.

    The lifespan claim merges into an existing record rather than minting one
    from ``load_osprey_config``, which this process has no workspace for — so
    every client fixture below needs this to have run first.
    """
    write_control_context(agent_data_root, target="live", generation=1)
    return agent_data_root


# -- harness -----------------------------------------------------------------


def recorded_posture() -> dict[str, str]:
    """The deployment's narrowings, re-read from disk."""
    posture_store.invalidate_cache()
    return dict(posture_store.recorded_posture())


def written_posture(root: Path):
    """The ``posture`` field exactly as it sits in the file."""
    path = control_context.record_path_under(root)
    return json.loads(path.read_text(encoding="utf-8"))["posture"]


def post_posture(client, *, target="standin", posture="sandbox"):
    return client.post(
        "/api/terminal/posture",
        json={"session_id": SESSION_A, "target": target, "posture": posture},
    )


def row_for(client, target):
    rows = client.get("/api/terminal/posture").json()["targets"]
    return next(row for row in rows if row["target"] == target)


# ── The shared parser decides every shape ────────────────────────────────────


class TestTheSharedParserDecidesTheShape:
    """One grammar, applied on the way out of the record and nowhere else."""

    def test_legacy_bare_sandbox_narrows_every_target(self, agent_data_root):
        """The upgrade case: a deployment sandboxed before targets existed.

        It meant "this deployment writes nothing", so it has to keep meaning
        that — every target, not none.
        """
        write_control_context(agent_data_root, posture="sandbox")

        assert recorded_posture() == EVERY_TARGET

    def test_bare_writes_is_dropped(self, agent_data_root):
        """Absence is the only spelling of the writes posture.

        Nothing in this field may widen anything, so a stored ``"writes"`` is
        not an assertion to honour — it is an entry with nothing in it.
        """
        write_control_context(agent_data_root, posture="writes")

        assert recorded_posture() == {}

    def test_per_target_entries_keep_only_the_narrowings(self, agent_data_root):
        write_control_context(
            agent_data_root,
            posture={"live": "sandbox", "va": "writes", "standin": "bogus"},
        )

        assert recorded_posture() == {"live": SANDBOX}

    def test_an_entry_that_narrows_nothing_is_empty(self, agent_data_root):
        write_control_context(agent_data_root, posture={"live": "writes"})

        assert recorded_posture() == {}

    @pytest.mark.parametrize("value", [["live"], 7, "sandboxed", True], ids=str)
    def test_a_field_of_the_wrong_shape_narrows_nothing(self, agent_data_root, value):
        """A hand-edited record must not wedge every write path.

        A narrowing can only refuse, so failing to read one leaves whatever
        the deployment ceiling already decided — which is why the parser
        drops what it cannot read instead of raising.
        """
        write_control_context(agent_data_root, posture=value)

        assert recorded_posture() == {}

    def test_a_corrupt_record_narrows_nothing(self, agent_data_root):
        """Not repairable from the browser, so it must not stop the server."""
        control_context.record_path_under(agent_data_root).write_text("{not json", encoding="utf-8")

        assert recorded_posture() == {}

    def test_no_record_at_all_narrows_nothing(self, agent_data_root):
        assert control_context.record_path_under(agent_data_root).exists() is False
        assert recorded_posture() == {}


# ── A narrowing survives a restart ───────────────────────────────────────────


class TestANarrowingSurvivesARestart:
    """The property the file exists for, driven through the real route."""

    def test_a_narrowing_set_on_one_app_is_read_by_the_next(self, started, make_client):
        with make_client() as client:
            assert post_posture(client).status_code == 200
            assert row_for(client, "standin")["posture"] == "sandbox"

        with make_client() as restarted:
            assert row_for(restarted, "standin")["posture"] == "sandbox"

    def test_the_file_is_what_carries_it_not_process_memory(self, started, make_client):
        """The narrowing is on disk the moment the route answers.

        A posture held in memory that never reached the file is a badge
        saying sandboxed over a deployment whose next write is still
        permitted — worse than a refusal the operator can retry.
        """
        with make_client() as client:
            post_posture(client)

            assert written_posture(started) == {"standin": SANDBOX}

    def test_clearing_a_narrowing_survives_too(self, started, make_client):
        """Absence is how the record spells writes, across a restart as well.

        A restored ``{}`` entry would be a second spelling of the same thing,
        and a restart that resurrected a cleared narrowing would be the silent
        revert this file guards against, pointing the other way.
        """
        with make_client() as client:
            post_posture(client)
            assert post_posture(client, posture="writes").status_code == 200
            assert written_posture(started) == {}

        with make_client() as restarted:
            assert row_for(restarted, "standin")["posture"] == "writes"

    def test_a_narrowing_left_by_a_predecessor_governs_the_first_read(
        self, agent_data_root, make_client
    ):
        """The real shape of a container recreation.

        Nothing in this process set the posture: the record was on disk before
        the app existed, and the very first read answers from it.
        """
        write_control_context(
            agent_data_root, target="live", generation=1, posture={"standin": SANDBOX}
        )

        with make_client() as client:
            assert row_for(client, "standin")["posture"] == "sandbox"
            assert row_for(client, "standin")["effective"] is False

    def test_the_narrowing_is_deployment_wide_not_per_session(self, started, make_client):
        """Two session ids, one answer — there is no key to file it under.

        The narrowing the first gesture recorded governs the deployment, so a
        second session's read cannot be a session that was never narrowed.
        """
        with make_client() as client:
            post_posture(client)

            other = client.get(
                "/api/terminal/posture",
                params={"session_id": "bbbbbbbb-1111-2222-3333-444444444444"},
            ).json()

        standin = next(row for row in other["targets"] if row["target"] == "standin")
        assert standin["posture"] == "sandbox"

    def test_the_other_targets_are_untouched(self, started, make_client):
        """One target narrowed is one target narrowed."""
        with make_client() as client:
            post_posture(client, target="standin")

        assert recorded_posture() == {"standin": SANDBOX}


# ── One location, and the retired one ────────────────────────────────────────


class TestTheRecordHasOneLocation:
    def test_the_record_is_the_one_file_that_is_written(self, started, make_client):
        """``<root>/control_target/control_context.json`` and nothing beside it.

        A narrowing the writer puts in one file and a reader looks for in
        another is a narrowing that silently never applies.
        """
        with make_client() as client:
            post_posture(client)

        record = control_context.record_path_under(started)
        assert record == started / posture_store.STATE_DIR_NAME / "control_context.json"
        assert record.exists()
        assert not (record.parent / RETIRED_STORE_NAME).exists()

    def test_a_file_at_the_retired_location_is_neither_read_nor_migrated(
        self, agent_data_root, make_client
    ):
        """The per-session store is retired: it narrows nothing.

        It sat in this very directory, so a reader that still globbed for it
        would honour narrowings no surface can show or clear.
        """
        write_control_context(agent_data_root, target="live", generation=1)
        retired = agent_data_root / posture_store.STATE_DIR_NAME / RETIRED_STORE_NAME
        retired.write_text(json.dumps({SESSION_A: {"standin": "sandbox"}}), encoding="utf-8")

        assert recorded_posture() == {}

        with make_client() as client:
            assert row_for(client, "standin")["posture"] == "writes"

        assert retired.read_text(encoding="utf-8"), "the retired file was rewritten"


# ── No location at all ───────────────────────────────────────────────────────


class TestNoRecordLocation:
    def test_an_unresolvable_root_narrows_nothing_and_does_not_raise(self, monkeypatch):
        """One transient config failure must not become a permissive default.

        ``recorded_posture`` sits on every write path, so an unreadable root
        answers "nothing is narrowed" — which grants nothing, because the
        deployment ceiling still decides.
        """
        monkeypatch.delenv(posture_store.AGENT_DATA_ROOT_ENV_VAR, raising=False)
        monkeypatch.setattr(
            posture_store, "resolve_shared_data_root", _raise_no_root, raising=False
        )
        posture_store.invalidate_cache()

        assert posture_store.recorded_posture() == {}

    def test_the_next_read_answers_once_the_root_is_back(self, tmp_path, monkeypatch):
        """Nothing is cached against a root that did not resolve."""
        monkeypatch.delenv(posture_store.AGENT_DATA_ROOT_ENV_VAR, raising=False)
        monkeypatch.setattr(
            posture_store, "resolve_shared_data_root", _raise_no_root, raising=False
        )
        posture_store.invalidate_cache()
        assert posture_store.recorded_posture() == {}

        root = tmp_path / "recovered"
        (root / posture_store.STATE_DIR_NAME).mkdir(parents=True)
        write_control_context(root, posture={"standin": SANDBOX})
        monkeypatch.setenv(posture_store.AGENT_DATA_ROOT_ENV_VAR, str(root))

        assert recorded_posture() == {"standin": SANDBOX}


def _raise_no_root() -> Path:
    raise RuntimeError("no project root")
