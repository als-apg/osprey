"""``--set`` pairs are a document too: the retired write keys are rewritten there.

A command line's pairs pass through the same normalizer as a file (as a root
named ``--set``) before they edit the resolved profile, so a retired-key pair
states a ``control_system.writes`` value that overrides the resolved one, and a
command line spelling both forms is refused. ``osprey init --set`` emits through
the same edit, so it never writes a retired leaf or a mixed document, and
``osprey set`` writes the tier key into a materialized profile. The command
itself refuses a retired key before any of that runs.

The oracle is the render (each config key applied in order, as the build
does) read through ``tier_writes``, never the normalizer.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli import build_profile_document as doc
from osprey.cli.build_profile_document import _read_profile_document
from osprey.cli.build_profile_emit import emit_standalone_profile_yaml
from osprey.cli.build_profile_resolve import (
    cli_edit_layer,
    resolve_build_profile,
    write_back_cli_overrides,
)
from osprey.cli.set_cmd import set as set_command
from osprey.utils.config_writer import config_update_fields
from osprey_connectors.errors import BuildProfileError
from osprey_connectors.types import tier_allows, tier_writes

_LOGGER_NAME = doc.__name__


@pytest.fixture(autouse=True)
def _fresh_state():
    doc._WARNED_RETIRED_KEYS.clear()
    doc._RETIRED_WRITE_ORIGINS.clear()
    yield
    doc._WARNED_RETIRED_KEYS.clear()
    doc._RETIRED_WRITE_ORIGINS.clear()


def _cli_warnings(caplog) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == _LOGGER_NAME
        and r.levelno == logging.WARNING
        and r.getMessage().startswith("--set: ")
    ]


def _rendered_section(config: dict[str, Any], tmp_path: Path) -> dict[str, Any]:
    target = tmp_path / "rendered-config.yml"
    target.write_text("{}\n", encoding="utf-8")
    config_update_fields(target, config)
    rendered = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    section = rendered.get("control_system")
    assert isinstance(section, dict)
    return section


def _no_retired_leaf(section: dict[str, Any]) -> bool:
    connector = section.get("connector")
    blocks = connector.values() if isinstance(connector, dict) else ()
    return "writes_enabled" not in section and not any(
        isinstance(block, dict) and "writes_enabled" in block for block in blocks
    )


def test_a_retired_key_pair_is_rewritten_in_the_edit_layer(caplog):
    caplog.set_level(logging.WARNING, logger=_LOGGER_NAME)
    assert cli_edit_layer(("config.control_system.writes_enabled=false",)) == {
        "config": {"control_system.writes": "never"}
    }
    assert cli_edit_layer(("config.control_system.connector.va.writes_enabled=true",)) == {
        "config": {"control_system.writes": "allowed"}
    }
    lines = _cli_warnings(caplog)
    assert len(lines) == 2
    assert all("control_system.writes" in line for line in lines)
    assert all("osprey init --revalidate" in line for line in lines)


@pytest.mark.parametrize(
    ("preset", "pair", "expected"),
    [
        ("control-assistant", "config.control_system.writes_enabled=false", "never"),
        ("hello-world", "config.control_system.writes_enabled=true", "allowed"),
        (
            "hello-world",
            "config.control_system.connector.virtual_accelerator.writes_enabled=true",
            "allowed",
        ),
    ],
)
def test_a_retired_key_pair_overrides_the_resolved_tier(preset, pair, expected, caplog, tmp_path):
    """The preset build path (`resolve_build_profile` with ``--set`` pairs).

    ``osprey build`` takes no ``--set``; this is the path every ``--set`` build
    edit goes through (preset builds, ``osprey init``, the scaffold verbs).
    """
    caplog.set_level(logging.WARNING, logger=_LOGGER_NAME)
    profile, _ = resolve_build_profile(None, preset, set_pairs=(pair,))
    section = _rendered_section(profile.config, tmp_path)
    assert section["writes"] == expected
    assert tier_writes(section) == expected
    assert _no_retired_leaf(section)
    assert len(_cli_warnings(caplog)) == 1


@pytest.mark.parametrize(
    "pairs",
    [
        ("config.control_system.writes_enabled=true", "config.control_system.writes=allowed"),
        (
            "config.control_system.connector.epics.writes_enabled=false",
            "config.control_system.writes=never",
        ),
        ("config.control_system={writes: never, writes_enabled: true}",),
    ],
)
def test_a_command_line_spelling_both_forms_is_refused(pairs):
    with pytest.raises(BuildProfileError) as excinfo:
        cli_edit_layer(pairs)
    message = str(excinfo.value)
    assert "--set" in message
    assert "`control_system.writes`" in message
    with pytest.raises(BuildProfileError):
        resolve_build_profile(None, "hello-world", set_pairs=pairs)


def test_an_invalid_tier_value_on_the_command_line_is_refused():
    with pytest.raises(BuildProfileError):
        resolve_build_profile(None, "hello-world", set_pairs=("config.control_system.writes=yes",))


def _emitted_lines(text: str) -> list[str]:
    return [
        line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")
    ]


def test_init_set_emits_the_tier_key_and_no_retired_leaf(tmp_path):
    text = emit_standalone_profile_yaml(
        "hello-world",
        ("config.control_system.writes_enabled=true",),
        "Demo",
        project_name="demo",
    )
    lines = _emitted_lines(text)
    assert not [line for line in lines if "writes_enabled" in line.split("#")[0]]
    emitted = yaml.safe_load(text)
    section = _rendered_section(emitted["config"], tmp_path)
    assert section["writes"] == "allowed"
    # The emitted document reads back clean: no retired leaf, nothing mixed.
    path = tmp_path / "profile.yml"
    path.write_text(text, encoding="utf-8")
    assert _rendered_section(_read_profile_document(path)["config"], tmp_path)["writes"] == (
        "allowed"
    )


def test_osprey_set_writes_the_tier_key_into_a_materialized_profile(tmp_path):
    path = tmp_path / "profile.yml"
    path.write_text(
        emit_standalone_profile_yaml("control-assistant", (), "Demo", project_name="demo"),
        encoding="utf-8",
    )
    assert tier_allows(_rendered_section(_read_profile_document(path)["config"], tmp_path))
    write_back_cli_overrides(path, ("config.control_system.writes_enabled=false",))
    text = path.read_text(encoding="utf-8")
    assert not [line for line in _emitted_lines(text) if "writes_enabled" in line.split("#")[0]]
    reread = _read_profile_document(path)
    assert _rendered_section(reread["config"], tmp_path)["writes"] == "never"


@pytest.mark.parametrize(
    "pair",
    [
        "config.control_system.writes_enabled=false",
        "config.control_system.connector.epics.writes_enabled=false",
    ],
    ids=["global", "per-type"],
)
def test_the_set_command_refuses_a_retired_key_before_writing(pair, tmp_path):
    """The verb stops short of the write-back above: a retired key never reaches it."""
    path = tmp_path / "profile.yml"
    path.write_text(
        emit_standalone_profile_yaml("control-assistant", (), "Demo", project_name="demo"),
        encoding="utf-8",
    )
    before = path.read_bytes()

    result = CliRunner().invoke(set_command, ["--repo", str(tmp_path), pair])

    assert result.exit_code != 0
    assert "retired" in result.output and "never | asked | allowed" in result.output
    assert path.read_bytes() == before
