"""Category retired: the consumers that keyed on it read the new fields instead.

Each consumer that used to read ``entry.category`` — ``archiver_downsample``, the activity bookkeeping filter, the session summary,
the channel-finder review link, ARIEL picture retention — now reads what the
writer states (``tool_source``, ``metadata.channel``, ``metadata.run_part``,
the form). Internal writers no longer pass
``category=``, which an AST scan over ``src/osprey`` pins.
"""

from __future__ import annotations

import ast
import asyncio
import json
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest

from osprey.stores.artifact_store import RUN_RECORD_KEY, ArtifactStore
from osprey_connectors import control_context

SRC = Path(__file__).resolve().parents[2] / "src" / "osprey"
CHANNEL = "SR:C01:BPM1:X"


def _reading(store: ArtifactStore, channel: str = CHANNEL, **kw):
    return store.save_data(
        tool="channel_read",
        data={"query": {}, "series": {channel: {"timestamps": [0], "values": [1.0]}}},
        title=f"{channel} reading",
        metadata={"data_type": "timeseries", "channel": channel},
        **kw,
    )


def _legacy(store: ArtifactStore, entry, **fields):
    """Give *entry* the stored fields a pre-retirement save would have left."""
    store.update_entry_metadata(entry.id, **fields)
    return store.get_entry(entry.id)


# ---- archiver_downsample -----------------------------------------------------


def _downsample():
    from osprey.mcp_server.workspace.tools import archiver_downsample as mod

    return getattr(mod.archiver_downsample, "fn", mod.archiver_downsample)


@pytest.fixture
def singleton(store, monkeypatch):
    """Route the tools' ``get_artifact_store()`` to the test store."""
    monkeypatch.setattr("osprey.stores.artifact_store.get_artifact_store", lambda: store)
    return store


class TestDownsample:
    def test_accepts_an_archiver_entry_saved_without_category(self, singleton):
        entry = singleton.save_data(
            tool="archiver_read",
            data={
                "query": {},
                "series": {CHANNEL: {"timestamps": [1.0, 2.0, 3.0], "values": [1.0, 2.0, 3.0]}},
            },
            title="archiver",
            metadata={"data_type": "timeseries"},
        )
        assert entry.category == ""

        result = json.loads(asyncio.run(_downsample()(entry.id, max_points=10)))

        assert [d["channel"] for d in result["datasets"]] == [CHANNEL]

    def test_rejects_an_array_channel_read(self, singleton):
        from fastmcp.exceptions import ToolError

        entry = _reading(singleton)
        assert entry.metadata["data_type"] == "timeseries"

        with pytest.raises(ToolError, match="archiver_read"):
            asyncio.run(_downsample()(entry.id))

    def test_a_legacy_archiver_data_category_alone_does_not_admit(self, singleton):
        from fastmcp.exceptions import ToolError

        entry = _legacy(singleton, _reading(singleton), category="archiver_data")

        with pytest.raises(ToolError):
            asyncio.run(_downsample()(entry.id))


# ---- activity bookkeeping ----------------------------------------------------


def _entry(**fields):
    from osprey.stores.artifact_store import ArtifactEntry

    base = {
        "id": "a1",
        "artifact_type": "json",
        "title": "t",
        "description": "",
        "filename": "a1.json",
        "mime_type": "application/json",
        "size_bytes": 1,
        "timestamp": "2026-11-04T00:00:00+00:00",
    }
    base.update(fields)
    return ArtifactEntry(**base)


class TestActivityVerdict:
    @pytest.mark.parametrize("part", ["preview", "code", "result"])
    def test_run_parts_other_than_output_are_bookkeeping(self, part):
        from osprey.mcp_server.artifact_activity import _is_bookkeeping

        assert _is_bookkeeping(_entry(tool_source="execute", metadata={"run_part": part}))

    def test_a_run_output_emits(self):
        from osprey.mcp_server.artifact_activity import _is_bookkeeping

        entry = _entry(
            tool_source="execute",
            artifact_type="notebook",
            title="Notebook: saved on purpose",
            metadata={"run_part": "output", RUN_RECORD_KEY: True},
        )
        assert not _is_bookkeeping(entry)

    def test_legacy_result_record_is_bookkeeping(self):
        from osprey.mcp_server.artifact_activity import _is_bookkeeping

        assert _is_bookkeeping(_entry(tool_source="execute", category="code_output"))
        assert _is_bookkeeping(_entry(tool_source="execute", metadata={RUN_RECORD_KEY: True}))

    def test_a_deliberate_save_emits_whatever_its_category(self):
        from osprey.mcp_server.artifact_activity import _is_bookkeeping

        assert not _is_bookkeeping(_entry(tool_source="create_static_plot"))
        assert not _is_bookkeeping(_entry(tool_source="artifact_register", category="code_output"))


# ---- summary by form ---------------------------------------------------------


@pytest.fixture
def beam_profile(monkeypatch):
    from osprey.stores.type_registry import ExtensionType

    registry = {"beam_profile": ExtensionType("beam_profile", "Beam Profile", "images")}
    monkeypatch.setattr(
        "osprey.mcp_server.workspace.tools.artifact_query.extension_registry", lambda: registry
    )
    # The web terminal's session page reads the registry through the store.
    monkeypatch.setattr("osprey.stores.type_registry.get_extension_types", lambda: registry)
    return registry


def _seed_forms(store):
    archiver = store.save_data(
        tool="archiver_read", data={}, title="arch", metadata={"data_type": "timeseries"}
    )
    plot = store.save_file(
        file_content=b"\x89PNG",
        filename="p.png",
        artifact_type="image",
        title="plot",
        mime_type="image/png",
        tool_source="create_static_plot",
    )
    plot = _legacy(store, plot, category="visualization")
    ext = store.save_file(
        file_content=b"\x89PNG",
        filename="b.png",
        artifact_type="image",
        title="beam",
        mime_type="image/png",
        tool_source="artifact_register",
        kind="beam_profile",
    )
    return archiver, plot, ext


class TestSummaryByForm:
    @pytest.mark.usefixtures("beam_profile")
    def test_mcp_tool_counts_by_form_and_extension_kind(self, singleton):
        from osprey.mcp_server.workspace.tools import artifact_query as mod

        archiver, plot, ext = _seed_forms(singleton)
        fn = getattr(mod.artifact_list, "fn", mod.artifact_list)
        result = json.loads(asyncio.run(fn(scope="all")))

        by_id = {e["artifact_id"]: e for e in result["entries"]}
        assert (by_id[archiver.id]["form"], by_id[archiver.id]["form_label"]) == (
            "trends",
            "Trends",
        )
        assert by_id[plot.id]["form"] == "plots"
        assert (by_id[ext.id]["form"], by_id[ext.id]["form_label"]) == (
            "beam_profile",
            "Beam Profile",
        )
        assert "category" not in by_id[plot.id]
        assert {f["key"]: f["count"] for f in result["form_totals"]} == {
            "trends": 1,
            "plots": 1,
            "beam_profile": 1,
        }

    @pytest.mark.usefixtures("beam_profile")
    def test_hub_route_counts_by_form(self, store, agent_data_root, tmp_path, monkeypatch):
        from osprey.interfaces.web_terminal.routes.session import session_artifacts
        from osprey.stores import session_ledger
        from osprey_connectors import posture_store

        key = "11111111-1111-4111-8111-111111111111"
        state = tmp_path / "state"
        state.mkdir()
        monkeypatch.setenv(posture_store.CONTROL_CONTEXT_DIR_ENV_VAR, str(state))
        control_context.invalidate_cache()
        assert session_ledger.append(state, key, key, "2000-01-01T00:00:00+00:00")
        for entry in _seed_forms(store):
            store.update_entry_metadata(entry.id, session_id=key)
        request = SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(workspace_dir=agent_data_root)),
            query_params={"session_id": key},
        )
        result = asyncio.run(session_artifacts(request))
        control_context.invalidate_cache()

        totals = result["totals"]
        assert "categories" not in totals
        assert {f["label"]: f["count"] for f in totals["forms"]} == {
            "Trends": 1,
            "Plots": 1,
            "Beam Profile": 1,
        }
        assert {e["form"] for e in result["entries"]} == {"trends", "plots", "beam_profile"}


# ---- channel-finder link -----------------------------------------------------


class TestChannelFinderLink:
    def test_links_a_channel_finder_artifact_saved_without_category(self, tmp_path, monkeypatch):
        from osprey.interfaces.channel_finder import pending_review_api as mod

        (tmp_path / "artifacts.json").write_text(
            json.dumps(
                [
                    {
                        "id": "cf1",
                        "title": "Channels",
                        "filename": "cf1.md",
                        "source_agent": "channel-finder",
                        "metadata": {"entry_ids": ["A:1", "A:2"]},
                    },
                    {
                        "id": "other",
                        "title": "Not the finder",
                        "filename": "o.md",
                        "source_agent": "data-visualizer",
                        "category": "channel_finder",
                        "metadata": {"entry_ids": ["A:1", "A:2"]},
                    },
                ]
            )
        )
        monkeypatch.setattr(mod, "_artifacts_dir", lambda: tmp_path)

        item = {"tool_response": json.dumps({"channels": ["A:1"]})}
        found = mod._resolve_artifact(item, str(tmp_path))

        assert found == {"id": "cf1", "title": "Channels", "filename": "cf1.md"}


# ---- internal saves warn nothing ---------------------------------------------


class TestInternalSavesDoNotWarn:
    def test_viz_collect_saves_without_category(self, singleton, tmp_path):
        from osprey.mcp_server.workspace.tools._viz_common import collect_and_register_artifacts

        fig = tmp_path / "fig.png"
        fig.write_bytes(b"\x89PNG")
        exec_result = SimpleNamespace(
            artifacts=[
                {
                    "path": fig,
                    "artifact_type": "image",
                    "title": "fig",
                    "description": "",
                    "mime_type": "image/png",
                    "category": "visualization",
                }
            ]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            ids = collect_and_register_artifacts(exec_result, tool_source="create_static_plot")

        assert len(ids) == 1
        entry = singleton.get_entry(ids[0])
        assert entry.category == ""
        assert entry.source_agent == "data-visualizer"

    def test_channel_read_series_save(self, store):
        import numpy as np

        from osprey.mcp_server.control_system.tools.channel_read import _save_series_reading

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            handle = _save_series_reading(store, CHANNEL, np.arange(4.0), {}, np)

        entry = store.get_entry(handle["artifact_id"])
        assert entry.category == ""
        assert entry.tool_source == "channel_read"
        assert entry.metadata["channel"] == CHANNEL

    def test_ariel_picture_retention(self, singleton):
        from osprey.services.ariel_search.attachments.artifacts import (
            save_viewable_picture_as_artifact,
        )
        from osprey.stores.artifact_scope import SessionScope

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            response = save_viewable_picture_as_artifact(
                {"rendition_bytes": b"\x89PNG-pic", "rendition_mime": "image/png"},
                {"entry_id": "e1"},
                {"attachment_id": "att1", "filename": "pic.png"},
                scope=SessionScope("none", None, None),
            )

        assert singleton.get_entry(response["artifact_id"]).category == ""


# ---- AST scan ----------------------------------------------------------------


def _save_call_name(call: ast.Call) -> str:
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _is_save(name: str) -> bool:
    return name == "save_artifact" or name.startswith("save_")


def test_no_internal_save_passes_category():
    offenders: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not _is_save(_save_call_name(node)):
                continue
            for kw in node.keywords:
                where = f"{path.relative_to(SRC)}:{node.lineno}"
                if kw.arg == "category":
                    offenders.append(where)
                # save_or_touch_by_sha256 forwards a kwargs dict to a save.
                if kw.arg == "save_kwargs" and isinstance(kw.value, ast.Dict):
                    keys = [k.value for k in kw.value.keys if isinstance(k, ast.Constant)]
                    if "category" in keys:
                        offenders.append(where)
    assert offenders == []


def test_category_symbols_are_gone_but_old_badges_still_render():
    import osprey.stores as stores
    from osprey.stores import artifact_store, type_registry

    for name in ("valid_category_keys", "register_category"):
        assert not hasattr(type_registry, name)
        assert not hasattr(stores, name)
        assert name not in stores.__all__
    assert not hasattr(artifact_store, "UNREGISTERED_CATEGORY_WARNING")
    # An entry saved before the retirement still carries its category; the
    # gallery badge labels it from the registry payload.
    assert type_registry.registry_to_api_dict()["categories"]["archiver_data"]["label"]
