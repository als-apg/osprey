"""Every surface that lists a session's artifacts lists the same ones.

One artifact store and one hub session ledger hold every kind of session the
ownership rule has to tell apart:

* two live web sessions, ``A`` (with a script run: head, preview, notebook and
  output) and ``B`` (with an extension-type artifact);
* a key ``C`` that was ``/clear``ed, with an artifact before the clear and one
  saved in the first turn after it;
* a seeded demo session ``S``, stored the way demo-history's contract stores a
  seed (stamped transcript, one ledger row per part), resumed on its last part;
* a keyless process, whose save belongs to no session;
* an ``osprey chat`` conversation that saved, ``/clear``ed (the turn-state
  hook's writer moved its pointer) and ran a script in its first new turn;
* a watchdog judge artifact and a dispatch-run artifact, both carrying key A.

The literal per-transcript ids declared below are the oracle. For every
transcript the gallery sidebar (``session-list {session}``, ``runs="grouped"``)
must equal it first; then, for each key's current transcript, the agent's
``artifact_list``, the session page route, the gallery's session-view body
(contract C6, replayed as ``state.js`` posts it) and the feedback bundle's
owned section each equal it. Saves take their session from the environment
(the default tag), as a real agent process's do.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.agent_record.client import ClientAnswer
from osprey.interfaces.artifacts.app import create_app
from osprey.interfaces.web_terminal.feedback_composer import collect_material
from osprey.interfaces.web_terminal.routes.session import router as session_router
from osprey.stores import artifact_store as artifact_store_module
from osprey.stores import session_ledger
from osprey.stores.artifact_inventory import listed_rows
from osprey.stores.artifact_scope import read_ledger, scope_for
from osprey.stores.artifact_store import RUN_RECORD_KEY, WATCHDOG_ORIGIN, ArtifactStore
from osprey.templates.claude_code.claude.hooks.osprey_turn_state import write_cli_pointer
from osprey_connectors import control_context, posture_store
from tests.agent_record.fakes import FakeRecordClient
from tests.mcp_server.conftest import get_tool_fn

# ---- the cast ---------------------------------------------------------------

KEY_A = "11111111-1111-4111-8111-111111111111"
KEY_B = "22222222-2222-4222-8222-222222222222"
KEY_C = "33333333-3333-4333-8333-333333333333"
T_C2 = "44444444-4444-4444-8444-444444444444"  # C's transcript after the /clear
KEY_S = "55555555-5555-4555-8555-555555555555"
T_S1 = "66666666-6666-4666-8666-666666666666"  # the seed's first part
T_S2 = "77777777-7777-4777-8777-777777777777"  # its second part, resumed
SCENARIO = "injector-tuning"

CLI_LAUNCH = "eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee"
CLI_TERMINAL = "chat-ffffffff-ffff-4fff-8fff-ffffffffffff"
T_CLI1 = "88888888-8888-4888-8888-888888888888"
T_CLI2 = "99999999-9999-4999-8999-999999999999"  # the CLI's transcript after its /clear

EXTENSION_KIND = "lattice"
EXTENSION_LABEL = "Ring lattice"

#: Every variable the scope resolver or a save's default tag reads.
_SCOPE_ENV = (
    "OSPREY_SESSION_ID",
    "OSPREY_DISPATCH_RUN_ID",
    "OSPREY_DISPATCH_RUN",
    "OSPREY_TERMINAL_SESSION",
    "OSPREY_CLI_LAUNCH",
    "OSPREY_TELEMETRY_SESSION_ID",
    "CLAUDE_CODE_SESSION_ID",
    "OSPREY_ARTIFACT_ORIGIN",
)

WEB_ENV = {key: {"OSPREY_SESSION_ID": key} for key in (KEY_A, KEY_B, KEY_C, KEY_S)}
CLI_ENV = {"OSPREY_TERMINAL_SESSION": CLI_TERMINAL, "OSPREY_CLI_LAUNCH": CLI_LAUNCH}
KEYLESS_ENV: dict[str, str] = {}

#: Each web key's current transcript.
CURRENT = {KEY_A: KEY_A, KEY_B: KEY_B, KEY_C: T_C2, KEY_S: T_S2}
#: Transcripts a key has moved off; checked over ``scope_for`` only.
PAST = (KEY_C, T_S1)

EXEC_A = "toolu_web_run"
EXEC_CLI = "toolu_cli_run"
CLEAR_C_AT = 30


def _at(minute: int) -> str:
    return f"2026-11-04T10:{minute:02d}:00+00:00"


# ---- the world --------------------------------------------------------------


@dataclass
class World:
    root: Path
    state_dir: Path
    store: ArtifactStore
    hub: TestClient
    page: TestClient
    ids: dict[str, str] = field(default_factory=dict)
    """Every saved artifact's id by its role in the fixture."""

    def oracle(self) -> dict[str, set[str]]:
        """Each transcript's literal expectation: the ids its sidebar row lists."""
        i = self.ids
        return {
            KEY_A: {i["a_plot"], i["a_run_head"], i["a_run_output"]},
            KEY_B: {i["b_plot"], i["b_extension"]},
            KEY_C: {i["c_before_clear"]},
            T_C2: {i["c_first_turn"]},
            T_S1: {i["seed_1"]},
            T_S2: {i["seed_2"]},
        }

    def cli_post_clear(self) -> set[str]:
        """The CLI conversation's listed artifacts after its /clear (a run: head and output)."""
        return {self.ids["cli_run_head"], self.ids["cli_run_output"]}

    def cli_all(self) -> set[str]:
        return {self.ids["cli_before_clear"], *self.cli_post_clear()}


def _use_env(monkeypatch, env: dict[str, str]) -> None:
    """Make this process the one *env* describes, and nothing else."""
    for name in _SCOPE_ENV:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)


def _file(store, title, *, minute, artifact_type="image", **kwargs):
    metadata = {"first_ts": _at(minute), **kwargs.pop("metadata", {})}
    return store.save_file(
        b"png",
        f"{title}.png",
        artifact_type,
        title,
        mime_type=kwargs.pop("mime_type", "image/png"),
        tool_source=kwargs.pop("tool_source", "create_static_plot"),
        metadata=metadata,
        **kwargs,
    )


def _run(store, exec_id, *, minute) -> dict[str, str]:
    """A script run: preview, notebook, result head and a figure output."""
    preview = _file(
        store,
        "Review: fit",
        minute=minute,
        artifact_type="notebook",
        mime_type="application/x-ipynb+json",
        tool_source="osprey_approval",
        metadata={"run_part": "preview", RUN_RECORD_KEY: True},
        exec_id=exec_id,
    )
    notebook = _file(
        store,
        "Notebook: fit",
        minute=minute,
        artifact_type="notebook",
        mime_type="application/x-ipynb+json",
        tool_source="execute",
        metadata={"run_part": "code", RUN_RECORD_KEY: True},
        exec_id=exec_id,
    )
    output = _file(
        store,
        "Figure",
        minute=minute,
        tool_source="execute",
        metadata={"run_part": "output", "figure": True},
        exec_id=exec_id,
    )
    head = store.save_data(
        tool="execute",
        data={"stdout": "ok"},
        title="fit",
        metadata={"run_part": "result", RUN_RECORD_KEY: True, "first_ts": _at(minute)},
        exec_id=exec_id,
    )
    return {
        "preview": preview.id,
        "notebook": notebook.id,
        "output": output.id,
        "head": head.id,
    }


@pytest.fixture
def world(tmp_path, monkeypatch) -> World:
    from osprey.stores import type_registry

    monkeypatch.setattr(
        type_registry,
        "_EXTENSION_TYPES",
        {EXTENSION_KIND: type_registry.ExtensionType(EXTENSION_KIND, EXTENSION_LABEL, "tables")},
    )
    _use_env(monkeypatch, KEYLESS_ENV)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_DIR_ENV_VAR, str(state_dir))
    rows = session_ledger.append_many(
        state_dir,
        [
            (KEY_A, KEY_A, session_ledger.EPOCH),
            (KEY_B, KEY_B, session_ledger.EPOCH),
            (KEY_C, KEY_C, session_ledger.EPOCH),
            (KEY_C, T_C2, _at(CLEAR_C_AT)),
            # demo-history's seed: one row per part, at that part's first turn.
            (KEY_S, T_S1, _at(5)),
            (KEY_S, T_S2, _at(20)),
        ],
    )
    assert len(rows) == 6
    control_context.invalidate_cache()

    root = tmp_path / "data"
    store = ArtifactStore(workspace_root=root, auto_launch=False)
    ids: dict[str, str] = {}

    # Live web session A: a plot, a script run, and two artifacts carrying its
    # key that it never owns (a watchdog judge's and a dispatch run's).
    _use_env(monkeypatch, WEB_ENV[KEY_A])
    ids["a_plot"] = _file(store, "Orbit A", minute=1).id
    run = _run(store, EXEC_A, minute=2)
    ids.update({f"a_run_{part}": value for part, value in run.items()})
    ids["watchdog"] = _file(store, "Watchdog verdict", minute=3, origin=WATCHDOG_ORIGIN).id
    ids["dispatch"] = _file(store, "Dispatched plot", minute=4, run_id="run-0001").id

    # Live web session B: a plot and an extension-type artifact.
    _use_env(monkeypatch, WEB_ENV[KEY_B])
    ids["b_plot"] = _file(store, "Orbit B", minute=6).id
    ids["b_extension"] = store.save_file(
        b"{}",
        "model.json",
        "json",
        "Model",
        mime_type="application/json",
        tool_source="test",
        metadata={"first_ts": _at(7)},
        kind=EXTENSION_KIND,
    ).id

    # Key C: one artifact before its /clear, one in the first turn after.
    _use_env(monkeypatch, WEB_ENV[KEY_C])
    ids["c_before_clear"] = _file(store, "Before clear", minute=10).id
    ids["c_first_turn"] = _file(store, "First turn", minute=CLEAR_C_AT + 1).id

    # The seeded session, stored as demo-history's contract stores a seed.
    _use_env(monkeypatch, KEYLESS_ENV)
    for name, transcript, minute in (("seed_1", T_S1, 6), ("seed_2", T_S2, 21)):
        ids[name] = _file(
            store,
            f"Seeded {name}",
            minute=minute,
            session_id=KEY_S,
            origin="",
            run_id="",
            metadata={"transcript_id": transcript, "scenario": SCENARIO},
        ).id

    # A keyless process: its save belongs to no session.
    ids["keyless"] = _file(store, "Keyless", minute=12).id

    # An osprey chat conversation: a save, a /clear (the turn-state hook's
    # writer moves the pointer), then a script run in its first new turn.
    _use_env(monkeypatch, CLI_ENV)
    assert write_cli_pointer(root, CLI_LAUNCH, T_CLI1)
    ids["cli_before_clear"] = _file(store, "CLI before clear", minute=13).id
    assert write_cli_pointer(root, CLI_LAUNCH, T_CLI2)
    cli_run = _run(store, EXEC_CLI, minute=14)
    ids["cli_run_head"] = cli_run["head"]
    ids["cli_run_output"] = cli_run["output"]
    ids["cli_run_notebook"] = cli_run["notebook"]

    _use_env(monkeypatch, KEYLESS_ENV)
    # The agent tools read the module's store singleton.
    monkeypatch.setattr(artifact_store_module, "_artifact_store", store)

    page_app = FastAPI()
    page_app.include_router(session_router)
    page_app.state.workspace_dir = root
    yield World(
        root=root,
        state_dir=state_dir,
        store=store,
        hub=TestClient(create_app(workspace_root=root)),
        page=TestClient(page_app),
        ids=ids,
    )
    control_context.invalidate_cache()


# ---- the legs ---------------------------------------------------------------


def _post(world: World, path: str, body: dict) -> dict:
    response = world.hub.post(path, json=body)
    assert response.status_code == 200, response.text
    return response.json()


def _sidebar(world: World, transcript: str) -> dict:
    return _post(
        world,
        "/api/artifacts/session-list",
        {"session": transcript, "runs": "grouped", "limit": 200},
    )


def _gallery(world: World, key: str) -> dict:
    """The body ``state.js`` posts in the session view (contract C6), replayed."""
    return _post(
        world,
        "/api/artifacts/session-list",
        {"key": key, "search": "", "cursor": None, "runs": "grouped"},
    )


def _session_page(world: World, key: str | None) -> dict:
    params = {"session_id": key} if key else {}
    response = world.page.get("/api/session-summary", params=params)
    assert response.status_code == 200, response.text
    return response.json()


def _artifact_list(monkeypatch, env: dict[str, str], **kwargs) -> dict:
    """The agent's ``artifact_list``, run as the process *env* describes."""
    from osprey.mcp_server.workspace.tools.artifact_query import artifact_list

    _use_env(monkeypatch, env)
    try:
        return json.loads(asyncio.run(get_tool_fn(artifact_list)(**kwargs)))
    finally:
        _use_env(monkeypatch, KEYLESS_ENV)


def _feedback_owned(world: World, key: str, transcript: str) -> set[str]:
    """The feedback bundle's owned section, built as the route builds its scope."""
    node = {
        "key": "k-prompt",
        "kind": "prompt",
        "ts": "2026-11-04T10:00:00+00:00",
        "body": {"text": "hi"},
        "decision": None,
        "result": None,
        "subagent": None,
    }
    client = FakeRecordClient({"timeline": [ClientAnswer(data=[node], complete=True)]})
    scope = scope_for(transcript, *read_ledger())
    material = collect_material(client, world.store, key, "", {}, scope=scope)
    return {e.id for e in material.owned}


def _ids(body: dict) -> set[str]:
    return {a["id"] for a in body["artifacts"]}


def _listed(answer: dict) -> set[str]:
    return {row["artifact_id"] for row in answer["entries"]}


# ---- parity -----------------------------------------------------------------


@pytest.mark.parametrize("key", [KEY_A, KEY_B, KEY_C, KEY_S])
def test_every_surface_lists_the_current_transcripts_oracle(world, monkeypatch, key):
    transcript = CURRENT[key]
    expected = world.oracle()[transcript]

    # The sidebar first: it is the row every other surface must equal.
    sidebar = _sidebar(world, transcript)
    assert sidebar["ledger"] == "ok"
    assert _ids(sidebar) == expected

    gallery = _gallery(world, key)
    assert gallery["transcript"] == transcript
    assert _ids(gallery) == expected

    page = _session_page(world, key)
    assert page["session"] == transcript
    assert {e["id"] for e in page["entries"]} == expected

    listing = _artifact_list(monkeypatch, WEB_ENV[key], last_n=200)
    assert listing["session"] == transcript
    assert "reason" not in listing
    assert listing["ledger"] == "ok"
    assert _listed(listing) == expected
    assert listing["total"] == len(expected) and listing["omitted"] == 0

    assert _feedback_owned(world, key, transcript) == expected

    counts = _post(world, "/api/artifacts/session-summary", {"sessions": [transcript]})
    assert counts["counts"][transcript]["total"] == len(expected)


def test_every_transcript_sidebar_row_equals_the_oracle(world):
    for transcript, expected in world.oracle().items():
        assert _ids(_sidebar(world, transcript)) == expected, transcript


@pytest.mark.parametrize("transcript", PAST)
def test_a_transcript_moved_off_keeps_its_rows_over_scope_for(world, transcript):
    key = KEY_C if transcript == KEY_C else KEY_S
    expected = world.oracle()[transcript]
    assert _ids(_sidebar(world, transcript)) == expected
    assert _feedback_owned(world, key, transcript) == expected
    # Its key's current views have moved on.
    assert not expected & _ids(_gallery(world, key))


def test_the_run_lists_once_at_its_head_with_its_parts(world):
    ids = world.ids
    (head,) = [a for a in _sidebar(world, KEY_A)["artifacts"] if a["id"] == ids["a_run_head"]]
    assert head["form"] == "runs"
    assert {p["id"] for p in head["run_parts"]} == {
        ids["a_run_preview"],
        ids["a_run_notebook"],
        ids["a_run_output"],
    }


def test_watchdog_and_dispatch_artifacts_carrying_a_key_are_owned_by_nobody(world, monkeypatch):
    stray = {world.ids["watchdog"], world.ids["dispatch"]}
    assert {world.store.get_entry(i).session_id for i in stray} == {KEY_A}
    for transcript in world.oracle():
        assert not stray & _ids(_sidebar(world, transcript))
    every = _artifact_list(monkeypatch, KEYLESS_ENV, scope="all", last_n=200)
    assert not stray & _listed(every)


# ---- the extension type -------------------------------------------------------


def test_the_extension_artifact_is_its_kind_everywhere(world, monkeypatch):
    ext = world.ids["b_extension"]

    (row,) = [a for a in _sidebar(world, KEY_B)["artifacts"] if a["id"] == ext]
    assert row["form"] == EXTENSION_KIND
    assert row["form_label"] == EXTENSION_LABEL

    (row,) = [a for a in _gallery(world, KEY_B)["artifacts"] if a["id"] == ext]
    assert row["form"] == EXTENSION_KIND

    (row,) = [e for e in _session_page(world, KEY_B)["entries"] if e["id"] == ext]
    assert row["form"] == EXTENSION_KIND
    assert row["form_label"] == EXTENSION_LABEL

    listing = _artifact_list(monkeypatch, WEB_ENV[KEY_B], last_n=200)
    (row,) = [r for r in listing["entries"] if r["artifact_id"] == ext]
    assert row["form"] == EXTENSION_KIND
    assert row["form_label"] == EXTENSION_LABEL
    totals = {t["key"]: t["count"] for t in listing["form_totals"]}
    assert totals[EXTENSION_KIND] == 1
    narrowed = _artifact_list(monkeypatch, WEB_ENV[KEY_B], form=EXTENSION_KIND)
    assert _listed(narrowed) == {ext}

    forms = {
        e.id: form for e, form, _label in listed_rows(world.store, [world.store.get_entry(ext)])
    }
    assert forms == {ext: EXTENSION_KIND}

    counts = _post(world, "/api/artifacts/session-summary", {"sessions": [KEY_B]})
    assert counts["counts"][KEY_B]["by_form"][EXTENSION_KIND] == 1
    assert "tables" not in counts["counts"][KEY_B]["by_form"]


# ---- the CLI conversation -----------------------------------------------------


def test_the_cli_listing_is_its_post_clear_artifacts(world, monkeypatch):
    # The default tag followed the pointer the hook's writer moved.
    tag = world.store.get_entry
    assert tag(world.ids["cli_before_clear"]).session_id == "cli:" + T_CLI1
    assert tag(world.ids["cli_run_head"]).session_id == "cli:" + T_CLI2

    listing = _artifact_list(monkeypatch, CLI_ENV, last_n=200)
    assert listing["session"] == T_CLI2
    assert "reason" not in listing
    assert _listed(listing) == world.cli_post_clear()


def test_every_cli_artifact_has_no_owner_in_the_web_listings(world, monkeypatch):
    every = _artifact_list(monkeypatch, WEB_ENV[KEY_A], scope="all", last_n=200)
    owners = {row["artifact_id"]: row["owner"] for row in every["entries"]}
    assert world.cli_all() <= owners.keys()
    assert {owners[i] for i in world.cli_all()} == {None}
    assert world.ids["cli_run_notebook"] not in owners  # a part lists under its head

    counts = _post(world, "/api/artifacts/session-summary", {"sessions": list(CURRENT.values())})
    assert counts["owned_total"] == sum(len(ids) for ids in world.oracle().values())
    for transcript in world.oracle():
        assert not world.cli_all() & _ids(_sidebar(world, transcript))


# ---- the keyless process --------------------------------------------------------


def test_the_keyless_listing_is_empty_with_a_reason(world, monkeypatch):
    listing = _artifact_list(monkeypatch, KEYLESS_ENV, last_n=200)
    assert listing["session"] is None
    assert listing["entries"] == []
    assert listing["reason"]

    page = _session_page(world, None)
    assert page["entries"] == [] and page["session"] is None and page["reason"]

    every = _artifact_list(monkeypatch, WEB_ENV[KEY_A], scope="all", last_n=200)
    owners = {row["artifact_id"]: row["owner"] for row in every["entries"]}
    assert owners[world.ids["keyless"]] is None


# ---- carried r3-CF-1: the seed tripwire --------------------------------------------


def test_a_resumed_seeded_session_lists_its_seed_and_no_other_session_does(world, monkeypatch):
    """Fails the day demo-history's interim hiding hides a seed from its own session."""
    seeds = {T_S1: world.ids["seed_1"], T_S2: world.ids["seed_2"]}
    for transcript, seed_id in seeds.items():
        seed = world.store.get_entry(seed_id)
        assert seed.session_id == KEY_S
        assert seed.origin == ""
        assert seed.run_id == ""
        assert seed.metadata["transcript_id"] == transcript
        assert seed.metadata["scenario"] == SCENARIO

    # The resumed key lists its current part's seed.
    resumed = _artifact_list(monkeypatch, WEB_ENV[KEY_S], last_n=200)
    assert resumed["session"] == T_S2
    assert seeds[T_S2] in _listed(resumed)

    # No other key, the CLI conversation or a keyless process does.
    for env in [WEB_ENV[k] for k in (KEY_A, KEY_B, KEY_C)] + [CLI_ENV, KEYLESS_ENV]:
        assert not set(seeds.values()) & _listed(_artifact_list(monkeypatch, env, last_n=200))

    # scope="all" names each seed's transcript, whoever asks.
    every = _artifact_list(monkeypatch, WEB_ENV[KEY_A], scope="all", last_n=200)
    owners = {row["artifact_id"]: row["owner"] for row in every["entries"]}
    assert owners[seeds[T_S2]] == T_S2
    assert owners[seeds[T_S1]] == T_S1
