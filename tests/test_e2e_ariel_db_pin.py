"""The per-cell ARIEL database an end-to-end project is built against.

The model-matrix runner gives each (model, seed) cell a Postgres database of its
own and exports its URI as ``OSPREY_ARIEL_DB_URI``. ``tests/e2e/sdk_helpers.py``
turns that into an ``ariel.database.uri`` pin at ``osprey init`` and checks the
render names it. Both halves are exercised here, in the unit lane, because the
lane runs ``pytest tests/ --ignore=tests/e2e`` and a test under ``tests/e2e/``
would never run where a regression has to fail.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from osprey.port_layout import resolve_port_base
from osprey.services.ariel_search.config import ARIELConfig, resolve_ariel_dsn
from tests.e2e import sdk_helpers
from tests.e2e.provider import FORCE_PROVIDER_ENV

CELL_URI = "postgresql://ariel:ariel@localhost:5432/ariel_model_under_test_seed1"


class _InitReached(Exception):
    """Raised by the stand-in for ``osprey init`` so the build never runs."""


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


def test_a_run_that_exported_no_database_pins_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OSPREY_ARIEL_DB_URI", raising=False)

    assert sdk_helpers._ariel_db_pins("control_assistant") == {}


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
    ("template", "pinned"),
    [("control_assistant", True), ("hello_world", False)],
)
def test_init_project_states_the_pin_at_init(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, template: str, pinned: bool
) -> None:
    monkeypatch.setenv("OSPREY_ARIEL_DB_URI", CELL_URI)
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
    assert uri_sets == ([f"config.ariel.database.uri={json.dumps(CELL_URI)}"] if pinned else [])


# ---------------------------------------------------------------------------
# The render check
# ---------------------------------------------------------------------------


def test_a_render_that_names_no_database_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The shape every ARIEL render has when the pin did not reach it."""
    monkeypatch.setenv("OSPREY_ARIEL_DB_URI", CELL_URI)
    render = _write_render(tmp_path / "build", _ARIEL_RENDER)

    with pytest.raises(AssertionError, match="OSPREY_ARIEL_DB_URI this run exported"):
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


def test_the_check_is_inert_when_no_database_was_exported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("OSPREY_ARIEL_DB_URI", raising=False)

    sdk_helpers._assert_render_names_ariel_db(_write_render(tmp_path / "build", _ARIEL_RENDER))


# ---------------------------------------------------------------------------
# A real render
# ---------------------------------------------------------------------------


def test_the_agent_loads_the_exported_database_from_a_real_render(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``osprey init`` + ``osprey build`` of the control-assistant preset, read back
    the way the ARIEL MCP server reads it.

    The second assertion is what makes the first one mean something: the same
    render with its ``database`` block removed derives a DSN from the deployment's
    own Postgres service, which is a different database.
    """
    monkeypatch.setenv("OSPREY_ARIEL_DB_URI", CELL_URI)
    monkeypatch.delenv(FORCE_PROVIDER_ENV, raising=False)

    repo = sdk_helpers.init_project(tmp_path, "cell", provider="anthropic")

    config = yaml.safe_load((sdk_helpers.render_dir(repo) / "config.yml").read_text("utf-8"))
    postgresql = (config.get("services") or {}).get("postgresql") or {}
    base = resolve_port_base(config)
    loaded = ARIELConfig.from_dict(config["ariel"], postgresql, base=base)
    assert loaded.database.uri == CELL_URI

    unpinned = {key: value for key, value in config["ariel"].items() if key != "database"}
    assert resolve_ariel_dsn(unpinned, postgresql, env={}, base=base) != CELL_URI
