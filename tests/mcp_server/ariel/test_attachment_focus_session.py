"""``attachment_to_artifact`` focuses the kept picture in the agent's own session.

The gallery's focus endpoint is captured at ``_post_json_with_response``; the
session comes from the process environment through ``current_scope()``, so a
web session's focus names its transcript and a dispatch run's names none.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from osprey.services.ariel_search.attachments import attachment_id_for
from osprey.stores import artifact_store as artifact_store_module
from osprey.stores.artifact_store import ArtifactStore
from tests.mcp_server.ariel.conftest import KEYSET_ENTRY_ID, get_tool_fn, keyset_entries
from tests.mcp_server.conftest import _SCOPE_ENV

_PNG = keyset_entries()[0]["attachments"][0]
PNG_ID = attachment_id_for(KEYSET_ENTRY_ID, _PNG)


@pytest.fixture
def store(tmp_path, monkeypatch) -> ArtifactStore:
    store = ArtifactStore(workspace_root=tmp_path / "agent_data", auto_launch=False)
    monkeypatch.setattr(artifact_store_module, "_artifact_store", store)
    return store


@pytest.fixture
def gallery():
    """The gallery's focus endpoint, answering 200."""
    with patch("osprey.mcp_server.http._post_json_with_response", return_value=(200, {})) as focus:
        yield focus


def _focus_body(gallery) -> dict:
    """The one body posted to ``/api/focus``."""
    calls = [c for c in gallery.call_args_list if str(c.args[0]).endswith("/api/focus")]
    assert len(calls) == 1
    return calls[0].args[1]


async def _keep() -> dict:
    from osprey.mcp_server.ariel.tools.attachment import attachment_to_artifact

    return json.loads(await get_tool_fn(attachment_to_artifact)(attachment_id=PNG_ID))


@pytest.mark.usefixtures("keyset_harness", "store")
async def test_web_session_focus_carries_the_transcript(gallery, web_session):
    result = await _keep()

    assert result["focused"] is True
    assert _focus_body(gallery) == {
        "artifact_id": result["artifact_id"],
        "transcript_id": web_session.transcript,
    }


async def test_web_session_focus_follows_a_clear(keyset_harness, store, gallery, web_session):  # noqa: ARG001
    web_session.clear("22222222-2222-4222-8222-222222222222")

    result = await _keep()

    assert _focus_body(gallery)["transcript_id"] == "22222222-2222-4222-8222-222222222222"
    assert result["focused"] is True


@pytest.mark.parametrize(
    "dispatch_env",
    [{"OSPREY_DISPATCH_RUN_ID": "run-1"}, {"OSPREY_DISPATCH_RUN": "1"}],
)
async def test_dispatch_run_focus_carries_no_transcript(
    keyset_harness,  # noqa: ARG001
    store,  # noqa: ARG001
    gallery,
    web_session,  # noqa: ARG001
    monkeypatch,
    dispatch_env,
):
    for name, value in dispatch_env.items():
        monkeypatch.setenv(name, value)

    result = await _keep()

    assert _focus_body(gallery) == {"artifact_id": result["artifact_id"]}


async def test_no_session_focus_is_unscoped(keyset_harness, store, gallery, monkeypatch):  # noqa: ARG001
    for name in _SCOPE_ENV:
        monkeypatch.delenv(name, raising=False)

    result = await _keep()

    assert _focus_body(gallery) == {"artifact_id": result["artifact_id"]}
