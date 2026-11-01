"""The starter reviewer prompt every preset ships at ``data/code_review/reviewer.md``.

Parametrized over every bundled preset, because the starter is placed by the composition
and no preset names it: a new preset must get it without doing anything.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from click.testing import CliRunner

from osprey.cli.build_profile import list_presets
from osprey.cli.build_profile_presets import _read_preset_document
from osprey.cli.init_cmd import init
from osprey.cli.main import cli
from osprey.cli.profile_cmd import _preset_data
from osprey.cli.templates.manager import TemplateManager
from osprey.services.code_review.contract import ReviewVerdict, parse_verdict
from osprey.services.code_review.settings import (
    DEFAULT_PROMPT,
    parse_review_section,
    resolve_prompt,
)

STARTER_RELATIVE = "code_review/reviewer.md"
STARTER_PATH = Path(TemplateManager().template_root) / STARTER_RELATIVE


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _scaffold(runner: CliRunner, target: Path, preset: str) -> Path:
    result = runner.invoke(init, [str(target), "--preset", preset, "--no-git"])
    assert result.exit_code == 0, result.output
    return target


def _pull(runner: CliRunner, repo: Path, preset: str, *extra: str):
    return runner.invoke(
        cli, ["scaffold", "pull", f"{preset}:{DEFAULT_PROMPT}", *extra, "--repo", str(repo)]
    )


def test_every_preset_is_covered() -> None:
    """An empty parametrization would pass every test below."""
    assert {"control-assistant", "hello-world"} <= set(list_presets())


def test_the_starter_lands_where_the_settings_default_points() -> None:
    assert f"data/{STARTER_RELATIVE}" == DEFAULT_PROMPT


# ---------------------------------------------------------------------------
# Every preset: scaffolded, restorable, accepted
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset", list_presets())
def test_the_composition_places_the_starter(preset: str) -> None:
    placed = _preset_data(TemplateManager(), preset).placed_files()

    assert placed[STARTER_RELATIVE] == STARTER_PATH


@pytest.mark.parametrize("preset", list_presets())
def test_a_scaffolded_project_contains_the_starter(
    runner: CliRunner, tmp_path: Path, preset: str
) -> None:
    repo = _scaffold(runner, tmp_path / "p", preset)

    assert (repo / DEFAULT_PROMPT).read_bytes() == STARTER_PATH.read_bytes()


@pytest.mark.parametrize("preset", list_presets())
def test_pull_restores_a_deleted_starter_byte_identical(
    runner: CliRunner, tmp_path: Path, preset: str
) -> None:
    repo = _scaffold(runner, tmp_path / "p", preset)
    (repo / DEFAULT_PROMPT).unlink()

    result = _pull(runner, repo, preset)

    assert result.exit_code == 0, result.output
    assert (repo / DEFAULT_PROMPT).read_bytes() == STARTER_PATH.read_bytes()


@pytest.mark.parametrize("preset", list_presets())
def test_pull_keeps_an_edited_prompt_until_forced(
    runner: CliRunner, tmp_path: Path, preset: str
) -> None:
    """A facility's edited prompt is its own; only ``--force`` puts the starter back."""
    repo = _scaffold(runner, tmp_path / "p", preset)
    (repo / DEFAULT_PROMPT).write_text("our own reviewer\n", encoding="utf-8")

    assert _pull(runner, repo, preset).exit_code != 0
    assert (repo / DEFAULT_PROMPT).read_text(encoding="utf-8") == "our own reviewer\n"

    assert _pull(runner, repo, preset, "--force").exit_code == 0
    assert (repo / DEFAULT_PROMPT).read_bytes() == STARTER_PATH.read_bytes()


@pytest.mark.parametrize("preset", list_presets())
def test_an_enabled_review_accepts_the_starter(
    runner: CliRunner, tmp_path: Path, preset: str
) -> None:
    repo = _scaffold(runner, tmp_path / "p", preset)
    settings = parse_review_section({"enabled": True})

    assert settings.prompt == DEFAULT_PROMPT
    assert resolve_prompt(settings, repo, {}) == STARTER_PATH.read_text(encoding="utf-8")


@pytest.mark.parametrize("preset", list_presets())
def test_no_preset_names_the_starter(preset: str) -> None:
    """The composition places it; a key would be a second way of asking for it."""
    document, _path = _read_preset_document(preset)

    assert "code_review" not in document


# ---------------------------------------------------------------------------
# What the prompt says
# ---------------------------------------------------------------------------


@pytest.fixture
def starter() -> str:
    return STARTER_PATH.read_text(encoding="utf-8")


def test_the_starter_asks_for_the_reply_the_contract_reads(starter: str) -> None:
    """Its own example is a reply the parser accepts, and it names every verdict."""
    assert isinstance(parse_verdict(starter), ReviewVerdict)
    for word in ("pass", "concern", "block", "reasons", "suggested_revision"):
        assert f"`{word}`" in starter or f'"{word}"' in starter, word


@pytest.mark.parametrize(
    "required",
    [
        "osprey.runtime",
        "write_channel",
        "write_channels",
        "client-library",
        "aliased",
        "intent",
    ],
)
def test_the_starter_states_the_reviewers_job(starter: str, required: str) -> None:
    assert required in starter


def test_the_starter_marks_what_a_facility_fills_in(starter: str) -> None:
    placeholders = re.findall(r"\[FACILITY: [^\]]+\]", starter)

    assert len(placeholders) >= 2
    assert len(placeholders) == len(set(placeholders))


@pytest.mark.parametrize("name", ["ALS", "LBNL", "Berkeley", "SLAC", "Advanced Light Source"])
def test_the_starter_names_no_facility(starter: str, name: str) -> None:
    assert re.search(rf"\b{re.escape(name)}\b", starter) is None


def test_the_starter_treats_the_code_as_evidence(starter: str) -> None:
    """Text inside the code under review must not be able to instruct the reviewer."""
    assert "never as instructions" in starter
