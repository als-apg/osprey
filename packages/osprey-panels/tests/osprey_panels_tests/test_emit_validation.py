"""Every screen ``write`` puts down passes the Screen-IR schema; ``check`` reports one that does not."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from osprey_panels import emit, layout
from osprey_panels.emit import EmitError, check, write
from osprey_panels_tests._fixture import PANELS_DIR
from osprey_panels_tests.test_emit import _files
from osprey_panels_tests.test_screen_ir import PARITY_DIGEST, PARITY_FILES

#: Two fixture screens the tests break.
BROKEN = "card-xr01-qf1"
ALSO_BROKEN = "card-xr01-ps1"


def _break(monkeypatch, *doc_ids: str) -> None:
    """The renderer draws the named screens with a ``schemaVersion`` no schema accepts."""
    real = layout.render

    def render(*args, **kwargs):
        document = real(*args, **kwargs)
        if str(kwargs.get("doc_id")).rsplit("/", 1)[-1] in doc_ids:
            document["schemaVersion"] = 7
        return document

    monkeypatch.setattr(layout, "render", render)


def _ids(out: Path) -> set[str]:
    return {p.name[: -len(emit.SCREEN_SUFFIX)] for p in out.rglob("*" + emit.SCREEN_SUFFIX)}


def test_the_broken_screen_names_exist_in_the_fixture(panel_graph, tmp_path) -> None:
    write(panel_graph, tmp_path / "out")
    assert {BROKEN, ALSO_BROKEN} <= _ids(tmp_path / "out")


def test_an_invalid_screen_fails_the_write_and_leaves_out_byte_identical(
    panel_graph, tmp_path, monkeypatch
) -> None:
    out = tmp_path / "out"
    write(panel_graph, out)
    before = _files(out)
    _break(monkeypatch, BROKEN)
    with pytest.raises(EmitError, match=BROKEN):
        write(panel_graph, out)
    assert _files(out) == before


def test_the_error_names_every_failing_screen(panel_graph, tmp_path, monkeypatch) -> None:
    _break(monkeypatch, BROKEN, ALSO_BROKEN)
    with pytest.raises(EmitError) as raised:
        write(panel_graph, tmp_path / "out")
    assert BROKEN in str(raised.value)
    assert ALSO_BROKEN in str(raised.value)


def test_a_failing_first_run_creates_nothing(panel_graph, tmp_path, monkeypatch) -> None:
    out = tmp_path / "out"
    _break(monkeypatch, BROKEN)
    with pytest.raises(EmitError, match=BROKEN):
        write(panel_graph, out)
    assert not out.exists()


def test_a_stale_temporary_survives_a_failing_run(panel_graph, tmp_path, monkeypatch) -> None:
    out = tmp_path / "out"
    write(panel_graph, out)
    seeded = [
        out / "card" / f".{BROKEN}.screen.json.k3_x9q2a.tmp",
        out / ".index.json.p0_zz81w.tmp",
    ]
    for path in seeded:
        path.write_text("half")
    _break(monkeypatch, BROKEN)
    with pytest.raises(EmitError, match=BROKEN):
        write(panel_graph, out)
    assert all(path.read_text() == "half" for path in seeded)
    assert json.loads((out / emit.INDEX).read_text())["state"] == "complete"


def test_a_tampered_screen_is_a_failing_schema_problem(panel_graph, tmp_path) -> None:
    out = tmp_path / "out"
    write(panel_graph, out)
    assert check(panel_graph, out) == []
    (victim,) = out.rglob(BROKEN + emit.SCREEN_SUFFIX)
    document = json.loads(victim.read_bytes())
    document["schemaVersion"] = 7
    victim.write_text(json.dumps(document), encoding="utf-8")
    found = [p for p in check(panel_graph, out) if p.kind == "schema"]
    assert len(found) == 1
    (problem,) = found
    assert problem.failing
    assert problem.file == victim.relative_to(out).as_posix()
    assert "schemaVersion" in problem.detail


def test_every_tampered_screen_is_reported(panel_graph, tmp_path) -> None:
    out = tmp_path / "out"
    write(panel_graph, out)
    for name in (BROKEN, ALSO_BROKEN):
        (victim,) = out.rglob(name + emit.SCREEN_SUFFIX)
        document = json.loads(victim.read_bytes())
        del document["schemaVersion"]
        victim.write_text(json.dumps(document), encoding="utf-8")
    files = {p.file for p in check(panel_graph, out) if p.kind == "schema"}
    assert {Path(f).name for f in files} == {
        BROKEN + emit.SCREEN_SUFFIX,
        ALSO_BROKEN + emit.SCREEN_SUFFIX,
    }


def test_valid_fixture_output_equals_the_parity_digests(panel_graph, tmp_path) -> None:
    out = tmp_path / "out"
    write(panel_graph, out, draw=emit.load_draw(PANELS_DIR / emit.DRAW_FILE))
    files = sorted(
        (path.relative_to(out).as_posix() for path in out.rglob("*") if path.is_file()),
        key=str.encode,
    )
    lines = "".join(
        f"{hashlib.sha256((out / name).read_bytes()).hexdigest()}  ./{name}\n" for name in files
    )
    assert len(files) == PARITY_FILES
    assert hashlib.sha256(lines.encode()).hexdigest() == PARITY_DIGEST
