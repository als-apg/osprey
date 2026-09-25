"""The ARIEL database an end-to-end project is built against, and the one its skip guard probes.

``tests/e2e/sdk_helpers.py`` names one ARIEL database per run: the URI in
``OSPREY_ARIEL_DB_URI`` (the model-matrix runner provisions one per (model, seed)
cell), otherwise the host Postgres on the standard port. It pins that database
as ``ariel.database.uri`` at ``osprey init``, checks the render names it, and
probes the same one in ``ariel_db_skip_reason``. All three are exercised here,
in the unit lane, because the lane runs ``pytest tests/ --ignore=tests/e2e`` and
a test under ``tests/e2e/`` would never run where a regression has to fail.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NoReturn

import psycopg
import pytest
import yaml

from osprey.port_layout import resolve_port_base
from osprey.services.ariel_search.config import ARIELConfig, resolve_ariel_dsn
from tests.e2e import sdk_helpers
from tests.e2e.provider import FORCE_PROVIDER_ENV

CELL_URI = "postgresql://ariel:ariel@localhost:5432/ariel_model_under_test_seed1"
DEFAULT_URI = sdk_helpers._DEFAULT_ARIEL_DB_URI


class _InitReached(Exception):
    """Raised by the stand-in for ``osprey init`` so the build never runs."""


def _export(monkeypatch: pytest.MonkeyPatch, uri: str | None) -> None:
    """Export ``uri`` as ``OSPREY_ARIEL_DB_URI``, or leave the variable unset for ``None``."""
    if uri is None:
        monkeypatch.delenv("OSPREY_ARIEL_DB_URI", raising=False)
    else:
        monkeypatch.setenv("OSPREY_ARIEL_DB_URI", uri)


def _probe(monkeypatch: pytest.MonkeyPatch) -> tuple[str, str]:
    """Run the skip guard against a refusing server; return the URI it dialled and its reason."""
    dialled: list[str] = []

    def _refuse(conninfo: str, **_options: object) -> NoReturn:
        dialled.append(conninfo)
        raise psycopg.OperationalError("connection refused")

    monkeypatch.setattr(psycopg, "connect", _refuse)
    reason = sdk_helpers.ariel_db_skip_reason()
    assert reason is not None
    return dialled[0], reason


def _write_render(render: Path, config: dict) -> Path:
    """Write ``config`` as the render's ``config.yml`` and return the render."""
    render.mkdir(parents=True, exist_ok=True)
    (render / "config.yml").write_text(yaml.safe_dump(config), encoding="utf-8")
    return render


# A render of an ARIEL preset states ARIEL settings and a Postgres service, and
# no ``ariel.database`` at all: the DSN is derived from ``services.postgresql``.
_ARIEL_RENDER = {
    "ariel": {"default_search_mode": "hybrid"},
    "services": {"postgresql": {"port_host": 25800, "database_name": "ariel"}},
}


# ---------------------------------------------------------------------------
# The pin stated at init
# ---------------------------------------------------------------------------


def test_a_run_that_exported_no_database_pins_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Outside the matrix a project names the host database the skip guard probes."""
    _export(monkeypatch, None)

    assert sdk_helpers._ariel_db_pins("control_assistant") == {"ariel.database.uri": DEFAULT_URI}


def test_an_ariel_preset_is_pinned_to_the_exported_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OSPREY_ARIEL_DB_URI", CELL_URI)

    assert sdk_helpers._ariel_db_pins("control_assistant") == {"ariel.database.uri": CELL_URI}


def test_a_preset_that_extends_an_ariel_preset_is_pinned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ARIEL settings a variant inherits are the parent's, read through ``extends:``."""
    monkeypatch.setenv("OSPREY_ARIEL_DB_URI", CELL_URI)

    assert sdk_helpers._ariel_db_pins("control_assistant_readonly") == {
        "ariel.database.uri": CELL_URI
    }


def test_a_preset_without_ariel_is_not_pinned(monkeypatch: pytest.MonkeyPatch) -> None:
    """An ``ariel:`` block would make ``apply_scenarios`` seed a logbook the project lacks."""
    monkeypatch.setenv("OSPREY_ARIEL_DB_URI", CELL_URI)

    assert sdk_helpers._ariel_db_pins("hello_world") == {}


@pytest.mark.parametrize(
    ("template", "exported", "pinned"),
    [
        ("control_assistant", CELL_URI, CELL_URI),
        ("control_assistant", None, DEFAULT_URI),
        ("hello_world", CELL_URI, None),
    ],
    ids=["exported", "default", "no-ariel"],
)
def test_init_project_states_the_pin_at_init(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    template: str,
    exported: str | None,
    pinned: str | None,
) -> None:
    _export(monkeypatch, exported)
    calls: list[list[str]] = []

    def _record(verb: str, args: list[str], *, timeout: int) -> None:  # noqa: ARG001 - stands in for _run_osprey, whose callers name timeout
        calls.append([verb, *args])
        raise _InitReached

    monkeypatch.setattr(sdk_helpers, "_run_osprey", _record)
    with pytest.raises(_InitReached):
        sdk_helpers.init_project(tmp_path, "proj", template=template, provider="anthropic")

    verb, *argv = calls[0]
    assert verb == "init"
    uri_sets = [a for a in argv if a.startswith("config.ariel.database.uri=")]
    assert uri_sets == ([f"config.ariel.database.uri={json.dumps(pinned)}"] if pinned else [])


# ---------------------------------------------------------------------------
# The skip guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("exported", [CELL_URI, None, ""], ids=["exported", "unset", "empty"])
def test_the_guard_probes_the_database_a_project_is_pinned_to(
    monkeypatch: pytest.MonkeyPatch, exported: str | None
) -> None:
    """An empty export is no export: the guard and the pin both fall back to the default."""
    _export(monkeypatch, exported)

    dialled, _reason = _probe(monkeypatch)

    assert dialled == sdk_helpers._ariel_db_pins("control_assistant")["ariel.database.uri"]
    assert dialled == (exported or DEFAULT_URI)


def test_the_skip_reason_names_the_database_without_its_password(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _export(monkeypatch, CELL_URI)

    _dialled, reason = _probe(monkeypatch)

    assert "at localhost:5432/ariel_model_under_test_seed1 not reachable" in reason
    assert "ariel:ariel@" not in reason


# ---------------------------------------------------------------------------
# The render check
# ---------------------------------------------------------------------------


def test_a_render_that_names_no_database_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The shape every ARIEL render has when the pin did not reach it."""
    monkeypatch.setenv("OSPREY_ARIEL_DB_URI", CELL_URI)
    render = _write_render(tmp_path / "build", _ARIEL_RENDER)

    with pytest.raises(AssertionError, match="the ARIEL database ariel_db_skip_reason probes"):
        sdk_helpers._assert_render_names_ariel_db(render)


def test_a_render_that_names_another_database_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OSPREY_ARIEL_DB_URI", CELL_URI)
    other = {**_ARIEL_RENDER, "ariel": {"database": {"uri": sdk_helpers._DEFAULT_ARIEL_DB_URI}}}
    render = _write_render(tmp_path / "build", other)

    with pytest.raises(AssertionError, match="ariel_model_under_test"):
        sdk_helpers._assert_render_names_ariel_db(render)


def test_a_render_that_names_the_database_passes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OSPREY_ARIEL_DB_URI", CELL_URI)
    pinned = {**_ARIEL_RENDER, "ariel": {"database": {"uri": CELL_URI}}}

    sdk_helpers._assert_render_names_ariel_db(_write_render(tmp_path / "build", pinned))


def test_a_render_without_ariel_passes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OSPREY_ARIEL_DB_URI", CELL_URI)

    sdk_helpers._assert_render_names_ariel_db(_write_render(tmp_path / "build", {"model": "m"}))


def test_without_an_export_a_render_must_name_the_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _export(monkeypatch, None)
    render = _write_render(tmp_path / "build", _ARIEL_RENDER)

    with pytest.raises(AssertionError, match="localhost:5432/ariel'"):
        sdk_helpers._assert_render_names_ariel_db(render)


# ---------------------------------------------------------------------------
# A real render
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("exported", [CELL_URI, None], ids=["exported", "default"])
def test_the_agent_loads_the_database_the_guard_probes_from_a_real_render(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, exported: str | None
) -> None:
    """``osprey init`` + ``osprey build`` of the control-assistant preset, read back
    the way the ARIEL MCP server reads it, against the URI the skip guard dials.

    The last assertion is what makes the others mean something: the same render
    with its ``database`` block removed derives a DSN from the deployment's own
    Postgres service, which is a different database.
    """
    _export(monkeypatch, exported)
    monkeypatch.delenv(FORCE_PROVIDER_ENV, raising=False)

    repo = sdk_helpers.init_project(tmp_path, "cell", provider="anthropic")

    config = yaml.safe_load((sdk_helpers.render_dir(repo) / "config.yml").read_text("utf-8"))
    postgresql = (config.get("services") or {}).get("postgresql") or {}
    base = resolve_port_base(config)
    loaded = ARIELConfig.from_dict(config["ariel"], postgresql, base=base)
    dialled, _reason = _probe(monkeypatch)
    assert loaded.database.uri == dialled == (exported or DEFAULT_URI)

    unpinned = {key: value for key, value in config["ariel"].items() if key != "database"}
    assert resolve_ariel_dsn(unpinned, postgresql, env={}, base=base) != dialled
