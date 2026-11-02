"""The control-assistant preset's one watchdog, and the labelled fixtures around it.

The preset ships ``guard-circumvention``, off, with its files materialized beside
the block by ``osprey init``. No preset ships an agent watchdog: that engine is
proven by a test-only profile whose twin judges with the packaged instructions.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli import build_profile
from osprey.cli.build_profile import resolve_build_profile
from osprey.cli.init_cmd import _source_zone_prose, init
from osprey.cli.profile_cmd import MATERIALIZED_SOURCE_ENTRIES
from osprey.services.watchdog.config import ENGINE_AGENT, ENGINE_DECISION
from osprey.services.watchdog.definitions import load_profile_watchdogs
from osprey.services.watchdog.history import build_session_view_from_nodes, rough_tokens
from osprey_connectors.errors import BuildProfileError
from tests._preset_data import copy_bundle_data
from tests.cli._lifecycle_build import invoke_build, rendered_config

REPO_ROOT = Path(__file__).resolve().parents[2]
PRESET_DIR = REPO_ROOT / "src" / "osprey" / "profiles" / "presets"
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "watchdog"
NAME = "guard-circumvention"
PACKAGED = PRESET_DIR / "watchdogs" / NAME / "instructions.md"
HEADER_KEYS = {"label", "prompt", "pair", "machine"}
_ENABLE_INHERITED = ("watchdogs:", f"  {NAME}:", "    enabled: true")


def _fixture(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return json.loads(lines[0]), [json.loads(line) for line in lines[1:]]


def _init(target: Path, *extra: str, preset: str = "control-assistant") -> Path:
    args = [str(target), "--preset", preset, "--no-git", *extra]
    result = CliRunner().invoke(init, args, catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return target


def test_the_preset_ships_exactly_one_disabled_decision_watchdog() -> None:
    document = yaml.safe_load((PRESET_DIR / "control-assistant.yml").read_text(encoding="utf-8"))
    block = document["watchdogs"]
    assert list(block) == [NAME]
    entry = block[NAME]
    assert entry["engine"] == ENGINE_DECISION
    assert entry["enabled"] is False
    flags = [option for option in entry["options"] if option.get("flag")]
    clear = [option for option in entry["options"] if not option.get("flag")]
    assert len(clear) == 1
    assert {option["id"] for option in flags} == {
        "direct-api",
        "loosen-gate",
        "bypass-check",
        "flip-control",
        "scripted-write",
    }
    assert all(option["text"].strip() for option in entry["options"])


def test_no_preset_defines_an_agent_watchdog() -> None:
    documents = sorted(PRESET_DIR.rglob("*.yml"))
    assert documents
    for path in documents:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        block = (document or {}).get("watchdogs") or {}
        engines = {entry.get("engine") for entry in block.values()}
        assert ENGINE_AGENT not in engines, path


def test_init_materializes_the_watchdog_beside_its_block(tmp_path: Path) -> None:
    repo = _init(tmp_path / "repo")
    assert (repo / "watchdogs" / NAME / "instructions.md").read_bytes() == PACKAGED.read_bytes()
    profile = yaml.safe_load((repo / "profile.yml").read_text(encoding="utf-8"))
    assert profile["watchdogs"][NAME]["enabled"] is False
    assert [d.name for d in load_profile_watchdogs(repo)] == [NAME]


def test_force_re_materializes_the_watchdog(tmp_path: Path) -> None:
    repo = _init(tmp_path / "repo")
    directory = repo / "watchdogs" / NAME
    (directory / "instructions.md").write_text("Edited.\n", encoding="utf-8")
    (directory / "stray.md").write_text("Stray.\n", encoding="utf-8")
    _init(repo, "--force")
    assert (directory / "instructions.md").read_bytes() == PACKAGED.read_bytes()
    assert not (directory / "stray.md").exists()


def test_a_preset_without_the_block_materializes_no_directory(tmp_path: Path) -> None:
    repo = _init(tmp_path / "repo", preset="hello-world")
    assert not (repo / "watchdogs").exists()
    _init(repo, "--force", preset="hello-world")
    assert not (repo / "watchdogs").exists()


def test_the_source_zone_lists_the_watchdogs() -> None:
    assert "watchdogs" in MATERIALIZED_SOURCE_ENTRIES
    assert "`watchdogs/`" in _source_zone_prose()


def test_the_build_succeeds_disabled_and_enabled(tmp_path: Path) -> None:
    repo = _init(tmp_path / "repo")

    built = invoke_build(repo)
    assert built.exit_code == 0, built.output
    assert "watchdogs" not in rendered_config(repo)
    assert not (repo / "build" / "watchdogs").exists()

    profile = repo / "profile.yml"
    text = profile.read_text(encoding="utf-8")
    off = f"  {NAME}:\n    engine: decision\n    enabled: false\n"
    assert text.count(off) == 1
    profile.write_text(text.replace(off, off.replace("false", "true")), encoding="utf-8")

    built = invoke_build(repo)
    assert built.exit_code == 0, built.output
    assert rendered_config(repo)["watchdogs"][NAME]["enabled"] is True
    copied = repo / "build" / "watchdogs" / NAME / "instructions.md"
    assert copied.read_bytes() == PACKAGED.read_bytes()


def _child_profile(directory: Path, *lines: str) -> Path:
    profile = directory / "profile.yml"
    (directory / "data").mkdir()
    text = "\n".join(("name: Narrowed", "extends: control-assistant", "data: data", *lines, ""))
    profile.write_text(text, encoding="utf-8")
    return profile


def test_a_profile_extending_the_preset_resolves_with_the_watchdog_off(tmp_path: Path) -> None:
    resolved, _directory = resolve_build_profile(_child_profile(tmp_path), None)
    assert resolved.watchdogs[NAME]["enabled"] is False


def test_a_child_enabling_the_inherited_watchdog_must_add_the_files(tmp_path: Path) -> None:
    profile = _child_profile(tmp_path, *_ENABLE_INHERITED)
    with pytest.raises(BuildProfileError) as refused:
        resolve_build_profile(profile, None)
    message = str(refused.value)
    assert f"watchdogs.{NAME}" in message
    assert "extends" in message
    assert "osprey init --force" not in message

    directory = tmp_path / "watchdogs" / NAME
    directory.mkdir(parents=True)
    shutil.copy(PACKAGED, directory / "instructions.md")
    resolved, _directory = resolve_build_profile(profile, None)
    assert resolved.watchdogs[NAME]["enabled"] is True


def test_replaying_a_child_needs_the_files_even_with_the_watchdog_off(tmp_path: Path) -> None:
    _child_profile(tmp_path)
    with pytest.raises(BuildProfileError, match="extends"):
        load_profile_watchdogs(tmp_path)


def test_init_refuses_a_preset_whose_disabled_watchdog_has_no_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = build_profile.resolve_build_profile

    def with_a_ghost(*args: Any, **kwargs: Any) -> Any:
        resolved, directory = real(*args, **kwargs)
        resolved.watchdogs["ghost"] = {**resolved.watchdogs[NAME]}
        return resolved, directory

    monkeypatch.setattr(build_profile, "resolve_build_profile", with_a_ghost)
    args = [str(tmp_path / "repo"), "--preset", "control-assistant", "--no-git"]
    result = CliRunner().invoke(init, args)
    assert result.exit_code != 0
    assert "watchdogs.ghost" in str(result.exception) + result.output


def test_the_agent_twin_judges_with_the_packaged_instructions_byte_for_byte() -> None:
    twin = FIXTURES / "agent-profile" / "watchdogs" / f"{NAME}-agent" / "instructions.md"
    assert twin.read_bytes() == PACKAGED.read_bytes()


def test_the_agent_profile_defines_the_twin_and_the_reader() -> None:
    definitions = {d.name: d for d in load_profile_watchdogs(FIXTURES / "agent-profile")}
    assert set(definitions) == {f"{NAME}-agent", "agent-reader"}
    assert all(d.engine == ENGINE_AGENT for d in definitions.values())
    reader = definitions["agent-reader"]
    assert reader.tools == ("mcp__controls__channel_read",)
    assert [path.name for path in reader.references] == ["gate-valves.md"]


def test_the_agent_profile_builds_with_both_enabled(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    shutil.copytree(FIXTURES / "agent-profile", repo)
    copy_bundle_data(repo / "data", "hello_world")
    profile = repo / "profile.yml"
    text = profile.read_text(encoding="utf-8")
    head, block = text.split("watchdogs:\n", 1)
    enabled = block.replace("enabled: false", "enabled: true")
    profile.write_text(f"{head}watchdogs:\n{enabled}", encoding="utf-8")
    built = invoke_build(repo)
    assert built.exit_code == 0, built.output
    assert set(rendered_config(repo)["watchdogs"]) == {f"{NAME}-agent", "agent-reader"}


def test_the_agent_pair_differs_only_in_machine() -> None:
    (risky_header, risky_nodes), (benign_header, benign_nodes) = (
        _fixture(FIXTURES / "agent" / f"gate-valve-{label}.jsonl") for label in ("risky", "benign")
    )
    assert (risky_header["label"], benign_header["label"]) == ("risky", "benign")
    assert risky_header["machine"].keys() == benign_header["machine"].keys()
    assert risky_header["machine"] != benign_header["machine"]
    assert risky_header["prompt"].encode() == benign_header["prompt"].encode()
    assert risky_header["pair"] == benign_header["pair"]
    assert json.dumps(risky_nodes).encode() == json.dumps(benign_nodes).encode()


def test_the_decision_set_is_labelled_on_both_sides() -> None:
    labels = [_fixture(path)[0]["label"] for path in (FIXTURES / "decision").glob("*.jsonl")]
    assert labels.count("risky") >= 10
    assert labels.count("benign") >= 10


def test_the_benign_set_mentions_what_the_risky_set_is_about() -> None:
    prompts = [
        _fixture(path)[0]["prompt"].lower() for path in (FIXTURES / "decision").glob("benign-*")
    ]
    assert any("curl" in prompt for prompt in prompts)
    assert any("hook" in prompt for prompt in prompts)


@pytest.mark.parametrize(
    "path", sorted(FIXTURES.glob("*/*.jsonl")), ids=lambda path: f"{path.parent.name}/{path.stem}"
)
def test_a_fixture_keeps_the_contract_and_reads_as_a_view(path: Path) -> None:
    header, nodes = _fixture(path)
    assert header.keys() <= HEADER_KEYS
    assert header["label"] in ("risky", "benign")
    assert path.stem.startswith(header["label"]) or path.stem.endswith(header["label"])
    view = build_session_view_from_nodes(
        nodes, header["prompt"], max_tokens=32_000, count_tokens=rough_tokens
    )
    assert view.entries_cut == 0
    assert view.text.endswith(f"USER: {header['prompt']}")
    assert view.entries_kept >= len(nodes) + 1
