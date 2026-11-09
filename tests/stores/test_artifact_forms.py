"""Tests for the intrinsic artifact form mapping (``osprey.stores.artifact_forms``)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from osprey.stores import artifact_store
from osprey.stores.artifact_forms import (
    BASE_FORMS,
    RUN_RECORD_KEY,
    form_label,
    form_of,
    is_interactive,
    is_run_record,
)
from osprey.stores.artifact_store import ArtifactEntry
from osprey.stores.type_registry import ARTIFACT_TYPES


def _entry(
    artifact_type: str = "json",
    *,
    mime_type: str = "application/json",
    tool_source: str = "",
    metadata: dict | None = None,
    category: str = "",
    title: str = "t",
) -> ArtifactEntry:
    return ArtifactEntry(
        id="a1",
        artifact_type=artifact_type,
        title=title,
        description="",
        filename="f",
        mime_type=mime_type,
        size_bytes=0,
        timestamp="2026-11-04T00:00:00+00:00",
        tool_source=tool_source,
        metadata=metadata or {},
        category=category,
    )


EXPECTED_BY_TYPE = {
    "plot_html": "plots",
    "plot_png": "plots",
    "table_html": "tables",
    "html": "documents",
    "markdown": "documents",
    "json": "data",
    "image": "images",
    "text": "documents",
    "file": "data",
    "binary": "data",
    "notebook": "documents",
    "dashboard_html": "dashboards",
}


class TestBaseForms:
    def test_order_and_members(self):
        assert BASE_FORMS == (
            "runs",
            "plots",
            "trends",
            "tables",
            "documents",
            "dashboards",
            "images",
            "data",
        )

    @pytest.mark.parametrize("key", BASE_FORMS)
    def test_every_form_has_a_label(self, key):
        assert form_label(key) == key.title()

    def test_unknown_label_is_derived(self):
        assert form_label("beam_logs") == "Beam Logs"

    def test_registered_kind_label_comes_from_registry(self):
        registry = {"beam_logs": SimpleNamespace(base_form="documents", label="Beam logbook")}
        assert form_label("beam_logs", registry) == "Beam logbook"

    def test_registered_kind_label_from_mapping(self):
        registry = {"beam_logs": {"base_form": "documents", "label": "Beam logbook"}}
        assert form_label("beam_logs", registry) == "Beam logbook"

    def test_get_only_registry_label(self):
        class GetOnly:
            def get(self, key):
                if key == "beam_logs":
                    return SimpleNamespace(base_form="documents", label="Beam logbook")
                return None

        assert form_label("beam_logs", GetOnly()) == "Beam logbook"

    @pytest.mark.parametrize("base_form", ["runs", "widgets"])
    def test_ineligible_kind_label_is_derived(self, base_form):
        registry = {"beam_logs": SimpleNamespace(base_form=base_form, label="Beam logbook")}
        assert form_label("beam_logs", registry) == "Beam Logs"

    def test_base_form_label_ignores_registry(self):
        registry = {"plots": SimpleNamespace(base_form="plots", label="Pictures")}
        assert form_label("plots", registry) == "Plots"

    def test_registry_defaults_to_none_for_label(self):
        assert form_label("beam_logs", None) == "Beam Logs"


class TestEveryArtifactType:
    def test_expectations_cover_registry(self):
        assert set(EXPECTED_BY_TYPE) == set(ARTIFACT_TYPES)

    @pytest.mark.parametrize("artifact_type", sorted(ARTIFACT_TYPES))
    def test_maps_to_a_listable_form(self, artifact_type):
        form = form_of(_entry(artifact_type, mime_type=""), None)
        assert form in BASE_FORMS
        assert form != "runs"
        assert form == EXPECTED_BY_TYPE[artifact_type]


class TestFormOf:
    @pytest.mark.parametrize(
        "entry, expected",
        [
            pytest.param(
                _entry(
                    "image", mime_type="image/png", tool_source="execute", metadata={"figure": True}
                ),
                "plots",
                id="executor-figure",
            ),
            pytest.param(
                _entry(
                    "image",
                    mime_type="image/png",
                    tool_source="execute_file",
                    metadata={"figure": True},
                ),
                "plots",
                id="executor-file-figure",
            ),
            pytest.param(
                _entry("image", mime_type="image/png", tool_source="create_static_plot"),
                "plots",
                id="viz-static-figure",
            ),
            pytest.param(
                _entry("plot_html", mime_type="text/html", tool_source="create_interactive_plot"),
                "plots",
                id="viz-plotly",
            ),
            pytest.param(
                _entry("image", mime_type="image/png", tool_source="execute"),
                "images",
                id="in-script-image",
            ),
            pytest.param(
                _entry(
                    "image",
                    mime_type="image/png",
                    tool_source="execute",
                    metadata={"figure": "yes"},
                ),
                "images",
                id="figure-flag-truthy-non-boolean",
            ),
            pytest.param(
                _entry("json", tool_source="screenshot_capture", category="screenshot"),
                "images",
                id="screenshot",
            ),
            pytest.param(
                _entry("json", tool_source="phoebus_snapshot"),
                "images",
                id="phoebus-snapshot",
            ),
            pytest.param(
                _entry(
                    "json",
                    tool_source="archiver_read",
                    category="archiver_data",
                    metadata={"data_type": "timeseries"},
                ),
                "trends",
                id="archiver-reading",
            ),
            pytest.param(
                _entry("json", tool_source="channel_read", category="channel_values"),
                "trends",
                id="channel-series",
            ),
            pytest.param(
                _entry("file", mime_type="application/octet-stream", tool_source="channel_read"),
                "trends",
                id="channel-npy",
            ),
            pytest.param(
                _entry("image", mime_type="image/png", tool_source="channel_read"),
                "trends",
                id="channel-reading-image",
            ),
            pytest.param(
                _entry("json", tool_source="execute", metadata={"data_type": "timeseries"}),
                "trends",
                id="timeseries-data",
            ),
            pytest.param(
                _entry("html", mime_type="text/html", tool_source="artifact_register"),
                "documents",
                id="html",
            ),
            pytest.param(
                _entry("file", mime_type="application/pdf", tool_source="create_document"),
                "documents",
                id="pdf",
            ),
            pytest.param(
                _entry(
                    "notebook",
                    mime_type="application/x-ipynb+json",
                    tool_source="osprey_approval",
                    metadata={RUN_RECORD_KEY: True},
                ),
                "documents",
                id="approval-preview",
            ),
            pytest.param(
                _entry(
                    "notebook",
                    mime_type="application/x-ipynb+json",
                    tool_source="execute",
                    title="Notebook: x.py",
                    metadata={RUN_RECORD_KEY: True},
                ),
                "documents",
                id="executor-notebook",
            ),
            pytest.param(
                _entry(
                    "json",
                    tool_source="execute",
                    category="code_output",
                    metadata={RUN_RECORD_KEY: True},
                ),
                "data",
                id="result-record",
            ),
            pytest.param(
                _entry("hologram", mime_type="application/x-unknown"),
                "data",
                id="unknown-type",
            ),
            pytest.param(_entry("", mime_type=""), "data", id="empty-type"),
            pytest.param(
                _entry("hologram", mime_type="image/jpeg"),
                "images",
                id="unknown-type-image-mime",
            ),
            pytest.param(
                _entry("hologram", mime_type="text/csv"),
                "data",
                id="unknown-type-csv-mime",
            ),
            pytest.param(
                _entry("hologram", mime_type="text/x-rst"),
                "documents",
                id="unknown-type-text-mime",
            ),
        ],
    )
    def test_mapping(self, entry, expected):
        assert form_of(entry, None) == expected

    def test_result_record_never_runs(self):
        entry = _entry(
            "json", tool_source="execute", category="code_output", metadata={RUN_RECORD_KEY: True}
        )
        assert is_run_record(entry)
        assert form_of(entry, None) != "runs"

    def test_registry_defaults_to_none(self):
        assert form_of(_entry("table_html")) == "tables"


class TestKind:
    def _kind_entry(self, kind: str, artifact_type: str = "json"):
        return SimpleNamespace(
            kind=kind,
            artifact_type=artifact_type,
            mime_type="application/json",
            tool_source="execute",
            metadata={},
        )

    def test_registered_kind_wins(self):
        registry = {"beam_profile": SimpleNamespace(base_form="plots")}
        assert form_of(self._kind_entry("beam_profile"), registry) == "plots"

    def test_registered_kind_as_mapping(self):
        registry = {"logbook": {"base_form": "documents"}}
        assert form_of(self._kind_entry("logbook"), registry) == "documents"

    def test_unknown_kind_falls_back(self):
        registry = {"beam_profile": SimpleNamespace(base_form="plots")}
        assert form_of(self._kind_entry("other", "table_html"), registry) == "tables"

    def test_kind_without_registry_falls_back(self):
        assert form_of(self._kind_entry("beam_profile"), None) == "data"

    def test_kind_declaring_runs_is_ignored(self):
        registry = {"sneaky": SimpleNamespace(base_form="runs")}
        assert form_of(self._kind_entry("sneaky"), registry) == "data"

    def test_kind_declaring_unknown_form_is_ignored(self):
        registry = {"odd": SimpleNamespace(base_form="widgets")}
        assert form_of(self._kind_entry("odd"), registry) == "data"

    def test_broken_registry_never_raises(self):
        class Broken:
            def get(self, _key):
                raise RuntimeError("boom")

        assert form_of(self._kind_entry("x"), Broken()) == "data"

    def test_entry_without_kind_attribute(self):
        entry = _entry("plot_png", mime_type="image/png")
        assert not hasattr(entry, "kind") or entry.kind == ""
        assert form_of(entry, {"": SimpleNamespace(base_form="tables")}) == "plots"


class TestInteractive:
    def test_plotly_is_interactive(self):
        assert is_interactive(_entry("plot_html", mime_type="text/html"))

    @pytest.mark.parametrize("artifact_type", ["plot_png", "image", "dashboard_html", "html"])
    def test_others_are_not(self, artifact_type):
        assert not is_interactive(_entry(artifact_type))


class TestReExport:
    def test_artifact_store_re_exports_the_same_objects(self):
        assert artifact_store.RUN_RECORD_KEY is RUN_RECORD_KEY
        assert artifact_store.is_run_record is is_run_record
