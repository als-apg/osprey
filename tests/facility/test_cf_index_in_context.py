"""The in_context channel-finder index: the build's tagged channels as the flat database.

``data/channel_finder/in_context.json`` carries
``"schema": "osprey.facility.channel_finder/1"`` and one row per channel tagged
``in_context``, sorted by address: ``channel`` is the first ``names`` entry, else
the address; ``address``; ``description``. A render carries it when its
``channel_finder.pipeline_mode`` is ``in_context``; a facility with no tagged
channel then stops with ``view-unsupported``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from osprey.facility.errors import FacilityBuildError
from osprey.facility.views.channel_finder import (
    CHANNEL_FINDER_SCHEMA,
    IN_CONTEXT_FILE,
    in_context_document,
    in_context_selected,
    write_in_context,
)
from tests.facility.test_cf_view_parity import load_golden

if TYPE_CHECKING:
    from tests.facility.conftest import BuiltProject


def _inputs(doc: dict[str, Any], rendered_config: dict[str, Any] | None = None) -> Any:
    from osprey.facility.views import ViewInputs

    return ViewInputs(
        doc=doc,
        rendered_config=rendered_config or {},
        facility_dir=Path("."),
        served=[],
    )


def _channel(id_: str, **fields: Any) -> dict[str, Any]:
    return {"id": id_, "role": "readback", **fields}


# --- the document ------------------------------------------------------------------


def test_a_tagged_channel_is_one_row_named_by_its_first_name() -> None:
    document = in_context_document(
        {
            "channels": [
                _channel(
                    "B:RB", names=["Beta", "B2"], description="beta", tags=["in_context", "x"]
                ),
                _channel("A:RB", description="alpha", tags=["in_context"]),
                _channel("C:RB", names=["Gamma"], tags=["other"]),
                _channel("D:RB", names=[]),
            ]
        }
    )

    assert document == {
        "schema": CHANNEL_FINDER_SCHEMA,
        "channels": [
            {"channel": "A:RB", "address": "A:RB", "description": "alpha"},
            {"channel": "Beta", "address": "B:RB", "description": "beta"},
        ],
    }


def test_a_channel_without_a_description_carries_none() -> None:
    document = in_context_document({"channels": [_channel("A:RB", tags=["in_context"])]})
    (row,) = document["channels"]
    assert row == {"channel": "A:RB", "address": "A:RB", "description": None}


# --- the predicate -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("rendered_config", "selected"),
    [
        ({"channel_finder": {"pipeline_mode": "in_context", "pipelines": {}}}, True),
        ({"channel_finder": {"pipeline_mode": "hierarchical"}}, False),
        ({"channel_finder": {"pipeline_mode": "graph", "pipelines": None}}, False),
        ({"channel_finder": {"pipeline_mode": None}}, False),
        ({"channel_finder": None}, False),
        ({}, False),
    ],
)
def test_the_view_is_written_when_in_context_is_selected(
    rendered_config: dict[str, Any], selected: bool
) -> None:
    assert in_context_selected(_inputs({"channels": []}, rendered_config)) is selected


def test_the_view_is_registered_under_channel_finder() -> None:
    from osprey.facility.views import VIEWS

    (view,) = [view for view in VIEWS if view.name == "in_context"]
    assert (view.path, view.reason, view.selected_by) == (
        "channel_finder",
        "channel_finder.pipeline_mode",
        "channel_finder.pipeline_mode",
    )


def test_a_render_that_selects_another_index_names_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from osprey.facility import views
    from osprey.facility.render import FACILITY_FILE, render_facility_outputs

    (view,) = [view for view in views.VIEWS if view.name == "in_context"]
    monkeypatch.setattr(views, "VIEWS", (view,))

    written = render_facility_outputs(
        tmp_path, {"channels": []}, {"channel_finder": {"pipeline_mode": "graph"}}, tmp_path
    )

    assert written == [tmp_path / FACILITY_FILE]
    assert capsys.readouterr() == ("", "")


# --- the writer --------------------------------------------------------------------


def test_the_writer_writes_the_index_with_its_header(tmp_path: Path) -> None:
    root = tmp_path / "data" / "channel_finder"
    written = write_in_context(
        root, _inputs({"channels": [_channel("A:RB", names=["A"], tags=["in_context"])]})
    )

    assert written == [root / IN_CONTEXT_FILE]
    raw = written[0].read_bytes()
    assert raw.endswith(b"}\n")
    assert json.loads(raw)["schema"] == CHANNEL_FINDER_SCHEMA


def test_zero_tagged_channels_stop_with_view_unsupported(tmp_path: Path) -> None:
    with pytest.raises(FacilityBuildError) as caught:
        write_in_context(tmp_path, _inputs({"channels": [_channel("A:RB", tags=["other"])]}))

    assert caught.value.kind == "view-unsupported"
    assert caught.value.format_message().startswith(
        "facility: view-unsupported: path channel_finder.pipeline_mode — "
    )
    assert not (tmp_path / IN_CONTEXT_FILE).exists()


# --- the demo ----------------------------------------------------------------------


@pytest.mark.slow
def test_the_demo_index_holds_the_569_golden_rows(
    built_control_assistant: BuiltProject, tmp_path: Path
) -> None:
    golden = load_golden("in_context_size.json")
    (target,) = write_in_context(tmp_path, _inputs(built_control_assistant.facility))

    rows = json.loads(target.read_bytes())["channels"]

    assert len(rows) == golden["size"] == 569
    assert {row["address"]: row["channel"] for row in rows} == {
        row["address"]: row["channel"] for row in golden["rows"]
    }
    assert all(row["description"] for row in rows)


@pytest.mark.slow
def test_an_in_context_build_writes_the_index(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from osprey.cli.init_cmd import init
    from tests._builds import run_build

    repo = tmp_path / "demo"
    result = CliRunner().invoke(
        init,
        [
            str(repo),
            "--preset",
            "control-assistant",
            "--no-git",
            "--set",
            "channel_finder_mode=in_context",
        ],
    )
    assert result.exit_code == 0, result.output

    built = run_build(repo)

    assert built.exit_code == 0, built.output
    index = repo / "build" / "data" / "channel_finder" / IN_CONTEXT_FILE
    assert json.loads(index.read_bytes())["schema"] == CHANNEL_FINDER_SCHEMA
