"""archiver_read names a labelled channel's state by its label, beside its index.

Archives store a ``bool`` or ``enum`` channel as its option index. The read tool
loads the labels once from the channel roster and returns, per sample, the
index and the label, whichever archive served the series, so an agent says
"User operation" rather than "0". A labelled channel is always requested raw
(the last real sample per bin): a mean of option indices names no state.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

import osprey.channel_roster as channel_roster
from osprey.mcp_server.control_system.server_context import initialize_server_context
from osprey_connectors.archiver._timerange import long_frame
from tests._facility_file import write_facility_file
from tests.facility.served_tree import served_tree
from tests.mcp_server.conftest import extract_response_dict, get_tool_fn

MODE = "SR:OPS:MODE:RB"
DCCT = "SR:DCCT"
MODES = ["User operation", "Injection", "Machine studies", "Off"]

_STAMPS = pd.to_datetime(
    [
        "2024-01-15T10:00:00",
        "2024-01-15T10:01:00",
        "2024-01-15T10:02:00",
        "2024-01-15T10:03:00",
        "2024-01-15T10:04:00",
    ],
    utc=True,
)


@pytest.fixture(autouse=True)
def cold_cache() -> Iterator[None]:
    channel_roster._roster_cache.clear()
    yield
    channel_roster._roster_cache.clear()


def _tool():
    from osprey.mcp_server.control_system.tools.archiver_read import archiver_read

    return get_tool_fn(archiver_read)


def _frame(channels_data: dict[str, list]) -> pd.DataFrame:
    return long_frame(
        {
            ch: pd.Series(values, index=_STAMPS[: len(values)])
            for ch, values in channels_data.items()
        }
    )


def _data_file(data: dict) -> dict:
    """The saved data file, resolved against the artifact store's own repo root."""
    from osprey.stores.artifact_store import get_artifact_store

    return json.loads((get_artifact_store().repo_root / data["data_file"]).read_text())


def _labelled_tree() -> dict:
    return {
        "records/channels.yaml": [
            {"id": MODE, "value_type": "enum", "options": MODES},
            {"id": "SR:VALVE:OPEN", "value_type": "bool"},
            {"id": DCCT},
        ]
    }


# ---------------------------------------------------------------------------
# An archive serving indices (the MongoDB shape, or any other backend)
# ---------------------------------------------------------------------------


@pytest.fixture
def stubbed(tmp_path, monkeypatch):
    """``(tool, archive, project)``: a project with a facility file and a stub archive.

    The stub stands for any backend that serves the archived option index
    (the MongoDB store does): the labels come from the roster, not from it.
    """
    monkeypatch.chdir(tmp_path)
    # The config names the in-process archiver so the session's target opens
    # an archive; the archive itself is the stub below.
    (tmp_path / "config.yml").write_text("archiver:\n  type: mock_archiver\n")
    write_facility_file(tmp_path, _labelled_tree())
    initialize_server_context()

    archive = _stub_archive()
    with patch(
        "osprey_connectors.factory.ConnectorFactory.create_service",
        new_callable=AsyncMock,
        return_value=archive,
    ):
        yield _tool(), archive, tmp_path


def _stub_archive() -> AsyncMock:
    """A stand-in archive whose coverage answers every channel archived, boundless."""
    from osprey_connectors.archiver import ArchiveCoverage

    archive = AsyncMock()
    archive.coverage.side_effect = lambda chans, **_: {
        ch: ArchiveCoverage(address=ch, is_archived=True) for ch in chans
    }
    return archive


class TestIndexAndLabel:
    async def test_each_sample_carries_its_index_and_its_label(self, stubbed):
        fn, archive, _ = stubbed
        archive.history.return_value = _frame({MODE: [0, 1, 3]})

        data = extract_response_dict(
            await fn(channels=[MODE], start_time="2024-01-15T10:00:00", bin_size=60)
        )

        assert data["status"] == "success"
        series = _data_file(data)["series"][MODE]
        assert series["values"] == [0, 1, 3]
        assert series["labels"] == ["User operation", "Injection", "Off"]
        assert series["options"] == MODES
        json.dumps(data, allow_nan=False)

    async def test_the_summary_names_the_last_state_by_its_label(self, stubbed):
        fn, archive, _ = stubbed
        archive.history.return_value = _frame({MODE: [2, 0]})

        data = extract_response_dict(
            await fn(channels=[MODE], start_time="2024-01-15T10:00:00", bin_size=60)
        )

        entry = data["summary"]["per_channel"][MODE]
        assert entry["points"] == 2
        assert entry["options"] == MODES
        assert entry["last"]["index"] == 0
        assert entry["last"]["label"] == "User operation"
        assert entry["last"]["timestamp"].startswith("2024-01-15T10:01:00")
        # A mean of option indices names no state.
        assert "mean" not in entry
        assert "min" not in entry
        assert "max" not in entry

    async def test_float_indices_from_a_numeric_column_are_read_as_indices(self, stubbed):
        fn, archive, _ = stubbed
        archive.history.return_value = _frame({MODE: [0.0, 2.0], DCCT: [500.1, 500.2]})

        data = extract_response_dict(
            await fn(channels=[MODE, DCCT], start_time="2024-01-15T10:00:00", bin_size=60)
        )

        series = _data_file(data)["series"][MODE]
        assert series["values"] == [0, 2]
        assert series["labels"] == ["User operation", "Machine studies"]

    async def test_a_bool_without_options_takes_the_default_labels(self, stubbed):
        fn, archive, _ = stubbed
        archive.history.return_value = _frame({"SR:VALVE:OPEN": [False, True]})

        data = extract_response_dict(
            await fn(channels=["SR:VALVE:OPEN"], start_time="2024-01-15T10:00:00", bin_size=60)
        )

        series = _data_file(data)["series"]["SR:VALVE:OPEN"]
        assert series["values"] == [0, 1]
        assert series["labels"] == ["FALSE", "TRUE"]

    async def test_a_backend_serving_the_label_gets_its_index(self, stubbed):
        fn, archive, _ = stubbed
        archive.history.return_value = _frame({MODE: ["Injection", "User operation"]})

        data = extract_response_dict(
            await fn(channels=[MODE], start_time="2024-01-15T10:00:00", bin_size=60)
        )

        series = _data_file(data)["series"][MODE]
        assert series["values"] == [1, 0]
        assert series["labels"] == ["Injection", "User operation"]

    async def test_a_null_sample_has_no_index_and_no_label(self, stubbed):
        fn, archive, _ = stubbed
        archive.history.return_value = _frame({MODE: [0.0, float("nan"), 1.0]})

        data = extract_response_dict(
            await fn(channels=[MODE], start_time="2024-01-15T10:00:00", bin_size=60)
        )

        series = _data_file(data)["series"][MODE]
        assert series["values"] == [0, None, 1]
        assert series["labels"] == ["User operation", None, "Injection"]
        assert data["summary"]["per_channel"][MODE]["points"] == 3
        json.dumps(data, allow_nan=False)

    async def test_an_index_outside_the_options_keeps_its_value_and_has_no_label(self, stubbed):
        fn, archive, _ = stubbed
        archive.history.return_value = _frame({MODE: [7]})

        data = extract_response_dict(
            await fn(channels=[MODE], start_time="2024-01-15T10:00:00", bin_size=60)
        )

        series = _data_file(data)["series"][MODE]
        assert series["values"] == [7]
        assert series["labels"] == [None]

    async def test_an_unlabelled_channel_is_untouched(self, stubbed):
        fn, archive, _ = stubbed
        archive.history.return_value = _frame({DCCT: [500.1, 500.3]})

        data = extract_response_dict(
            await fn(channels=[DCCT], start_time="2024-01-15T10:00:00", bin_size=60)
        )

        series = _data_file(data)["series"][DCCT]
        assert series == {
            "timestamps": series["timestamps"],
            "values": [500.1, 500.3],
        }
        entry = data["summary"]["per_channel"][DCCT]
        assert entry["mean"] == pytest.approx(500.2)
        assert "options" not in entry


class TestLabelledChannelsAreReadRaw:
    async def test_raw_reads_everything_in_one_request(self, stubbed):
        fn, archive, _ = stubbed
        archive.history.return_value = _frame({MODE: [0], DCCT: [500.0]})

        await fn(channels=[MODE, DCCT], start_time="2024-01-15T10:00:00", bin_size=60)

        assert archive.history.await_count == 1
        assert archive.history.call_args.kwargs["processing"] == "raw"

    async def test_an_aggregate_read_requests_the_labelled_channels_raw_at_the_same_bin(
        self, stubbed
    ):
        fn, archive, _ = stubbed

        def serve(channels, *_window, **_request):
            data = {}
            if MODE in channels:
                data[MODE] = [0, 1]
            if DCCT in channels:
                data[DCCT] = [500.2]
            return _frame(data)

        archive.history.side_effect = serve

        data = extract_response_dict(
            await fn(
                channels=[MODE, DCCT],
                start_time="2024-01-15T10:00:00",
                end_time="2024-01-15T10:05:00",
                processing="mean",
                bin_size=60,
            )
        )

        calls = {
            tuple(c.args[0]): {k: c.kwargs[k] for k in ("precision_ms", "processing")}
            for c in archive.history.await_args_list
        }
        assert calls == {
            (DCCT,): {"precision_ms": 60_000, "processing": "mean"},
            (MODE,): {"precision_ms": 60_000, "processing": "raw"},
        }
        assert data["status"] == "success"
        assert data["summary"]["per_channel"][MODE]["processing"] == "raw"
        assert data["summary"]["per_channel"][MODE]["last"]["label"] == "Injection"
        assert _data_file(data)["series"][MODE]["labels"] == [
            "User operation",
            "Injection",
        ]

    async def test_an_aggregate_read_of_only_labelled_channels_never_aggregates(self, stubbed):
        fn, archive, _ = stubbed
        archive.history.return_value = _frame({MODE: [3]})

        await fn(
            channels=[MODE],
            start_time="2024-01-15T10:00:00",
            processing="max",
            bin_size=60,
        )

        assert [c.kwargs["processing"] for c in archive.history.await_args_list] == ["raw"]


class TestTheRosterIsLoadedOnce:
    async def test_one_read_loads_the_labels_once(self, stubbed):
        fn, archive, _ = stubbed
        archive.history.return_value = _frame({MODE: [0], "SR:VALVE:OPEN": [1]})

        from osprey.mcp_server.control_system.tools import archiver_read as module

        with patch.object(
            module, "registered_channels", wraps=module.registered_channels
        ) as roster:
            await fn(
                channels=[MODE, "SR:VALVE:OPEN", DCCT],
                start_time="2024-01-15T10:00:00",
                bin_size=60,
            )

        assert roster.call_count == 1

    async def test_no_facility_file_leaves_every_channel_as_served(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "config.yml").write_text("archiver:\n  type: mock_archiver\n")
        initialize_server_context()
        archive = _stub_archive()
        archive.history.return_value = _frame({MODE: [0, 1]})
        with patch(
            "osprey_connectors.factory.ConnectorFactory.create_service",
            new_callable=AsyncMock,
            return_value=archive,
        ):
            data = extract_response_dict(
                await _tool()(channels=[MODE], start_time="2024-01-15T10:00:00", bin_size=60)
            )

        assert data["status"] == "success"
        series = _data_file(data)["series"][MODE]
        assert series["values"] == [0, 1]
        assert "labels" not in series


# ---------------------------------------------------------------------------
# The real in-process archiver
# ---------------------------------------------------------------------------


class TestTheInProcessArchiver:
    @pytest.fixture
    def project(self, tmp_path, monkeypatch):
        """A rendered project serving the operation-mode enum and a DCCT."""
        project = served_tree(
            tmp_path,
            readings=[MODE, DCCT],
            channels={
                MODE: {
                    "value_type": "enum",
                    "options": MODES,
                    "simulation": {"nominal": "User operation"},
                },
                DCCT: {"simulation": {"nominal": 500.0, "noise": {"absolute": 1.0}}},
            },
        ).parent.parent
        monkeypatch.chdir(project)
        (project / "config.yml").write_text("archiver:\n  type: mock_archiver\n")
        initialize_server_context()
        return project

    @pytest.mark.usefixtures("project")
    async def test_the_operation_mode_reads_as_user_operation(self):
        data = extract_response_dict(
            await _tool()(
                channels=[MODE],
                start_time="2024-01-15T10:00:00",
                end_time="2024-01-15T10:05:00",
                bin_size=60,
            )
        )

        assert data["status"] == "success"
        entry = data["summary"]["per_channel"][MODE]
        assert entry["last"] == {
            "timestamp": entry["last"]["timestamp"],
            "index": 0,
            "label": "User operation",
        }
        series = _data_file(data)["series"][MODE]
        assert series["labels"] and set(series["labels"]) == {"User operation"}
        assert set(series["values"]) == {0}

    @pytest.mark.usefixtures("project")
    async def test_a_mean_read_keeps_the_enum_raw_and_averages_the_rest(self):
        data = extract_response_dict(
            await _tool()(
                channels=[MODE, DCCT],
                start_time="2024-01-15T10:00:00",
                end_time="2024-01-15T10:05:00",
                processing="mean",
                bin_size=60,
            )
        )

        assert data["status"] == "success"
        mode = data["summary"]["per_channel"][MODE]
        assert mode["processing"] == "raw"
        assert mode["last"]["label"] == "User operation"
        assert "mean" not in mode
        assert data["summary"]["per_channel"][DCCT]["mean"] == pytest.approx(500.0, abs=2.0)
        series = _data_file(data)["series"][MODE]
        # Raw at a 60 s bin: one real sample per bin, each an option index.
        assert all(isinstance(v, int) for v in series["values"])
        assert set(series["labels"]) == {"User operation"}
