"""``osprey init --revalidate`` fills a profile from its preset and refreshes providers.yml.

A profile written while ``templates/apps/<name>/config.yml.j2`` still existed
spells only what its facility changed. The template is gone, so such a profile
is refused outright, and the one migration verb is the way back: it writes every
``config:`` key the preset documents into the profile itself, under the preset's
own comments, drops the retired ``app_template:`` key, refreshes the provider
entries OSPREY ships and re-stamps provenance. It never invents the deployment's
own selections — the roster, the shared apps, the default tier, role tiers, the
landing groups — and names each one it leaves out.

The shape under test throughout is a REAL one — a repo ``osprey init`` made,
edited back into the old shape — rather than a hand-written stub, because what
the verb has to reproduce is the emitted file, comments and all, and only an
emitted file specifies that.
"""

from __future__ import annotations

import io
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.build_profile_deploy import (
    IMAGE_SOURCE_CONFIG_KEY,
    config_image_source_spelling,
)
from osprey.cli.build_profile_document import RETIRED_CATALOG_KEY, RETIRED_DEFAULT_TIER_KEY
from osprey.cli.build_profile_emit import emit_standalone_profile_yaml, materialized_profile
from osprey.cli.build_profile_merge import compute_preset_hash
from osprey.cli.build_profile_presets import PRESET_DATA_BUNDLE_KEY
from osprey.cli.init_cmd import init
from osprey.cli.main import cli
from osprey.cli.validate_cmd import validate
from osprey.profiles.providers import PROVIDERS_FILENAME, packaged_catalog_path
from tests.cli._lifecycle_build import invoke_build, rendered_config

#: The four ``approval`` leaves plus their two neighbours, as a "whole branch
#: and then some" removal: the branch the proposal names, and two single keys
#: whose comment blocks sit in the middle of other sections.
DROPPED_BRANCHES = ("approval.", "archiver.", "hooks.debug")

#: The frozen hand-built reference deployment. Two things about it are what the
#: tests below need and no ``osprey init`` repo has: it states ``image_source``
#: once, in its ``deploy:`` block, and its roster runs two tiers and shares none
#: of the apps the control-assistant preset ships.
EXEMPLAR = Path(__file__).resolve().parents[1] / "deployment" / "goldens" / "exemplar-profile"

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The selection keys a preset only ever gives an example of.
SELECTIONS = (
    "modules.web_terminals.shared",
    "modules.web_terminals.users",
    "modules.web_terminals.default_tier",
    "modules.web_terminals.landing.groups",
)

SITE_GATEWAY = (
    "  # Our own gateway.\n"
    "  site-gateway:\n"
    "    base_url: https://gw.example.org/v1\n"
    "    api_key: ${SITE_GATEWAY_KEY}\n"
    "    default_model: site-model\n"
    "    models: [site-model]\n"
)


@pytest.fixture(autouse=True)
def _no_harvested_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep a developer's exported API keys out of the repos these tests make."""
    from osprey.cli.templates.scaffolding import provider_api_key_entries

    for entry in provider_api_key_entries():
        monkeypatch.delenv(entry["var"], raising=False)


def _init(runner: CliRunner, target: Path, preset: str) -> None:
    """Materialize a deployment repo — the only way one comes into existence."""
    result = runner.invoke(init, [str(target), "--preset", preset, "--no-" + "git"])
    assert result.exit_code == 0, result.output


def _revalidate(runner: CliRunner, repo: Path, *extra: str):
    return runner.invoke(init, [str(repo), "--revalidate", *extra])


def _repo_files(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in root.rglob("*")
        if p.is_file() and "build" not in p.relative_to(root).parts[:1]
    }


def _config_lines(text: str) -> list[str]:
    """Every line of the document's ``config:`` block, the header excluded."""
    lines: list[str] = []
    inside = False
    for line in text.splitlines():
        if line.startswith("config:"):
            inside = True
            continue
        if inside and line.strip() and not line.startswith((" ", "\t")):
            break
        if inside:
            lines.append(line)
    return lines


def _comment_block_above(text: str, key: str) -> list[str]:
    """The comment lines directly above ``key`` in a profile's config block.

    Text-level on purpose: what the filled file has to reproduce is what an
    operator READS above the key, so the assertion reads it the same way rather
    than through the round-trip machinery that wrote it.
    """
    lines = _config_lines(text)
    marker = f"  {key}:"
    for index, line in enumerate(lines):
        if not line.startswith(marker):
            continue
        block: list[str] = []
        for previous in reversed(lines[:index]):
            if previous.strip().startswith("#"):
                block.insert(0, previous.strip())
            elif not previous.strip():
                continue
            else:
                break
        return block
    return []


def _config_keys(text: str) -> list[str]:
    """Dotted keys the config block spells, in file order."""
    return [
        line.strip().split(":", 1)[0]
        for line in _config_lines(text)
        if line.startswith("  ") and not line.strip().startswith("#") and ":" in line
    ]


def _to_old_shape(
    profile: Path,
    *,
    drop: Sequence[str] = DROPPED_BRANCHES,
    template: str | None = "hello_world",
    keep_provenance: bool = True,
) -> list[str]:
    """Edit an emitted profile back into the shape the fill migrates.

    Each dropped key goes with the comment block above it, which is what makes
    the fixture honest: an old profile did not carry the template's prose any
    more than it carried the template's keys.

    Returns:
        The dotted keys removed, so a test can assert on exactly them.
    """
    removed: list[str] = []
    kept: list[str] = []
    pending: list[str] = []
    for line in profile.read_text(encoding="utf-8").splitlines(keepends=True):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            pending.append(line)
            continue
        key = stripped.split(":", 1)[0]
        if line.startswith("  ") and any(key.startswith(prefix) for prefix in drop):
            removed.append(key)
            pending = []
            continue
        kept.extend(pending)
        pending = []
        kept.append(line)
    text = "".join([*kept, *pending])

    if not keep_provenance:
        text = _without_top_level_block(text, "provenance")
    if template is not None:
        text = text.replace("\nconfig:\n", f"\n{PRESET_DATA_BUNDLE_KEY}: {template}\nconfig:\n", 1)
    profile.write_text(text, encoding="utf-8")
    return removed


def _without_top_level_block(text: str, key: str) -> str:
    """Drop the top-level ``key:`` block — the shape a profile predating it has."""
    out: list[str] = []
    inside = False
    for line in text.splitlines(keepends=True):
        if line.startswith(f"{key}:"):
            inside = True
            continue
        if inside and line.strip() and not line.startswith((" ", "\t")):
            inside = False
        if not inside:
            out.append(line)
    return "".join(out)


def _profile_data(profile: Path) -> dict:
    return yaml.safe_load(profile.read_text(encoding="utf-8"))


def _dotted_keys_below(value: object, prefix: str = "") -> list[str]:
    """Every key with a dot in its NAME anywhere below *value*, dotted-path form."""
    if not isinstance(value, dict):
        return []
    found: list[str] = []
    for key, nested in value.items():
        path = f"{prefix}{key}"
        if "." in str(key):
            found.append(path)
        found.extend(_dotted_keys_below(nested, f"{path}."))
    return found


def _block_above(text: str, line_start: str) -> list[str]:
    """The comment lines directly above the first line starting with ``line_start``.

    The whole-document counterpart of :func:`_comment_block_above`, for a
    top-level key rather than a ``config:`` leaf. A blank line ENDS the block
    here rather than being skipped: at the margin a blank separates one section
    from the next, so what is above the blank introduces something else.
    """
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if not line.startswith(line_start):
            continue
        block: list[str] = []
        for previous in reversed(lines[:index]):
            if not previous.strip().startswith("#"):
                break
            block.insert(0, previous.strip())
        return block
    return []


def _exemplar(tmp_path: Path) -> Path:
    """A writable copy of the frozen exemplar deployment.

    Copied rather than used in place: the rewrite WRITES, and the golden is a
    tracked reference several other suites compare against byte for byte.
    """
    repo = tmp_path / "demo-facility"
    shutil.copytree(EXEMPLAR, repo)
    return repo


def _flatten(mapping: dict, prefix: str = "") -> list[tuple[str, object]]:
    out: list[tuple[str, object]] = []
    for key, value in mapping.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict) and value:
            out.extend(_flatten(value, f"{dotted}."))
        else:
            out.append((dotted, value))
    return out


def _preset_config(preset: str) -> dict[str, object]:
    """The preset's rendered ``config:`` leaves, dotted — values included."""
    materialized = materialized_profile(preset, project_name="facility", profile_name="Facility")
    return dict(_flatten(materialized.get("config") or {}))


def _edit_round_trip(profile: Path, edit) -> None:
    """Apply *edit* to the profile's round-trip document and write it back."""
    from osprey.utils.config_writer import _yaml

    document = _yaml.load(profile.read_text(encoding="utf-8"))
    edit(document)
    buffer = io.StringIO()
    _yaml.dump(document, buffer)
    profile.write_text(buffer.getvalue(), encoding="utf-8")


# ---------------------------------------------------------------------------
# Acceptance: the whole journey
# ---------------------------------------------------------------------------


def test_old_shape_refused_then_revalidate_then_build_green(
    runner: CliRunner, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The migration, end to end: refused, revalidated, built, and stable."""
    repo = tmp_path / "facility"
    _init(runner, repo, "hello-world")
    profile = repo / "profile.yml"
    removed = _to_old_shape(profile)
    assert removed, "fixture removed nothing"

    refused = invoke_build(repo)
    assert refused.exit_code != 0
    # The refusal reaches the operator through the build's logger.
    assert "osprey init --revalidate" in caplog.text

    revalidated = _revalidate(runner, repo)
    assert revalidated.exit_code == 0, revalidated.output

    text = profile.read_text(encoding="utf-8")
    assert PRESET_DATA_BUNDLE_KEY not in _profile_data(profile)
    assert [key for key in removed if key not in _config_keys(text)] == []

    built = invoke_build(repo)
    assert built.exit_code == 0, built.output
    assert (repo / "build" / "config.yml").is_file()

    again = _revalidate(runner, repo)
    assert again.exit_code == 0, again.output
    assert profile.read_text(encoding="utf-8") == text


# ---------------------------------------------------------------------------
# What lands in the file
# ---------------------------------------------------------------------------


def test_every_filled_key_carries_the_presets_comment(runner: CliRunner, tmp_path: Path) -> None:
    """The profile documents what it now spells, as the preset's emitted file does."""
    repo = tmp_path / "facility"
    _init(runner, repo, "hello-world")
    profile = repo / "profile.yml"
    removed = _to_old_shape(profile)

    assert _revalidate(runner, repo).exit_code == 0

    filled = profile.read_text(encoding="utf-8")
    reference = emit_standalone_profile_yaml("hello-world", (), "Facility", project_name="facility")
    differing = {
        key: (_comment_block_above(filled, key), _comment_block_above(reference, key))
        for key in removed
        if _comment_block_above(filled, key) != _comment_block_above(reference, key)
    }
    assert differing == {}
    # Not vacuous: most of the removed keys do carry prose.
    assert sum(bool(_comment_block_above(filled, key)) for key in removed) >= 5


def test_a_whole_missing_branch_comes_back_as_its_leaves(runner: CliRunner, tmp_path: Path) -> None:
    """A profile with no ``approval`` key at all gets the branch, leaf by leaf."""
    repo = tmp_path / "facility"
    _init(runner, repo, "hello-world")
    profile = repo / "profile.yml"
    removed = _to_old_shape(profile, drop=("approval.",))
    assert len(removed) >= 6, removed
    assert [
        key
        for key in _config_keys(profile.read_text(encoding="utf-8"))
        if key.startswith("approval")
    ] == []

    result = _revalidate(runner, repo)
    assert result.exit_code == 0, result.output

    text = profile.read_text(encoding="utf-8")
    assert [key for key in removed if key not in _config_keys(text)] == []
    reference = emit_standalone_profile_yaml("hello-world", (), "Facility", project_name="facility")
    assert [
        key
        for key in removed
        if _comment_block_above(text, key) != _comment_block_above(reference, key)
    ] == []
    assert f"Filled {len(removed)} key(s) from preset 'hello-world'" in result.output
    assert invoke_build(repo).exit_code == 0


def test_filled_values_are_the_presets(runner: CliRunner, tmp_path: Path) -> None:
    repo = tmp_path / "facility"
    _init(runner, repo, "hello-world")
    profile = repo / "profile.yml"
    before = _profile_data(profile)["config"]
    removed = _to_old_shape(profile)

    assert _revalidate(runner, repo).exit_code == 0

    after = _profile_data(profile)["config"]
    assert {key: after[key] for key in removed} == {key: before[key] for key in removed}


def test_a_value_the_operator_changed_is_not_overwritten(runner: CliRunner, tmp_path: Path) -> None:
    """The fill only ADDS: the profile is the source of truth once it exists."""
    repo = tmp_path / "facility"
    _init(runner, repo, "hello-world")
    profile = repo / "profile.yml"
    _to_old_shape(profile)
    profile.write_text(
        profile.read_text(encoding="utf-8").replace(
            "  control_system.read_inline_max_elements: 2000",
            "  control_system.read_inline_max_elements: 7",
        ),
        encoding="utf-8",
    )

    assert _revalidate(runner, repo).exit_code == 0

    assert _profile_data(profile)["config"]["control_system.read_inline_max_elements"] == 7


def test_a_nested_spelling_is_left_alone(runner: CliRunner, tmp_path: Path) -> None:
    """A profile nesting a key already spells it; lacking leaves go INSIDE the mapping."""
    repo = tmp_path / "facility"
    _init(runner, repo, "hello-world")
    profile = repo / "profile.yml"
    _to_old_shape(profile, drop=("approval.",))
    profile.write_text(
        profile.read_text(encoding="utf-8").replace(
            "\nconfig:\n",
            "\nconfig:\n  approval:\n    enabled: true\n    default_policy: always\n",
            1,
        ),
        encoding="utf-8",
    )

    assert _revalidate(runner, repo).exit_code == 0

    text = profile.read_text(encoding="utf-8")
    keys = _config_keys(text)
    assert "approval.enabled" not in keys
    assert "approval.default_policy" not in keys
    # Not beside the mapping...
    assert "approval.tools.channel_write" not in keys

    # ...but inside it, addressing the same rendered leaf — asserted on the RENDER.
    assert invoke_build(repo).exit_code == 0
    fresh = tmp_path / "reference"
    _init(runner, fresh, "hello-world")
    assert invoke_build(fresh).exit_code == 0
    approval = rendered_config(repo)["approval"]
    assert approval["tools"] == rendered_config(fresh)["approval"]["tools"]
    assert _dotted_keys_below(approval) == []


def test_a_nested_parent_gets_real_mappings_not_a_dotted_key(
    runner: CliRunner, tmp_path: Path
) -> None:
    """A leaf below a mapping the profile nests lands under REAL intermediates.

    ``default_policy: selective`` makes a collapse visible: a per-tool policy
    written as a literal dotted key would be ignored and fall back to it.
    """
    repo = tmp_path / "facility"
    _init(runner, repo, "hello-world")
    profile = repo / "profile.yml"
    _to_old_shape(profile, drop=("approval.",))
    profile.write_text(
        profile.read_text(encoding="utf-8").replace(
            "\nconfig:\n",
            "\nconfig:\n  approval:\n    enabled: true\n    default_policy: selective\n",
            1,
        ),
        encoding="utf-8",
    )

    assert _revalidate(runner, repo).exit_code == 0
    assert invoke_build(repo).exit_code == 0

    approval = rendered_config(repo)["approval"]
    assert approval["default_policy"] == "selective"
    assert approval["tools"]["channel_write"] == "always"
    assert _dotted_keys_below(approval) == []


def _drop_nested_leaf(profile: Path) -> str:
    """Remove one scalar leaf from under ``modules.web_terminals``, returning its dotted key."""
    lines = profile.read_text(encoding="utf-8").splitlines(keepends=True)
    stack: dict[int, str] = {}
    inside = False
    for index, line in enumerate(lines):
        if line.startswith("config:"):
            inside = True
            continue
        if inside and line.strip() and not line.startswith((" ", "\t")):
            break
        stripped = line.strip()
        if not inside or not stripped or stripped.startswith(("#", "-")) or ":" not in stripped:
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, _, value = stripped.partition(":")
        stack = {column: name for column, name in stack.items() if column < indent}
        stack[indent] = key
        path = [stack[column] for column in sorted(stack)]
        if value.strip() and path[0] == "modules.web_terminals" and len(path) >= 3:
            dotted = ".".join(path)
            if dotted.startswith("modules.web_terminals.landing.groups"):
                continue
            del lines[index]
            profile.write_text("".join(lines), encoding="utf-8")
            return dotted
    raise AssertionError("no nested leaf found under modules.web_terminals")


def test_a_nested_preset_branch_fills_its_leaves(runner: CliRunner, tmp_path: Path) -> None:
    """The control-assistant presets spell ``modules.web_terminals:`` nested."""
    repo = tmp_path / "facility"
    _init(runner, repo, "control-assistant")
    profile = repo / "profile.yml"
    text = profile.read_text(encoding="utf-8")
    assert "  modules.web_terminals:" in text, "preset no longer nests this branch"
    profile.write_text(
        text.replace("\nconfig:\n", f"\n{PRESET_DATA_BUNDLE_KEY}: control_assistant\nconfig:\n", 1),
        encoding="utf-8",
    )
    dropped = _drop_nested_leaf(profile)

    assert _revalidate(runner, repo).exit_code == 0

    flattened = dict(_flatten(_profile_data(profile)["config"]))
    assert dropped in flattened


# ---------------------------------------------------------------------------
# Which preset, and the provenance stamp
# ---------------------------------------------------------------------------


def test_a_profile_without_provenance_fills_from_the_retired_key(
    runner: CliRunner, tmp_path: Path
) -> None:
    """The shape with nothing else to go on — and the block is written back."""
    repo = tmp_path / "facility"
    _init(runner, repo, "hello-world")
    profile = repo / "profile.yml"
    removed = _to_old_shape(profile, keep_provenance=False)
    assert "provenance" not in _profile_data(profile)

    result = _revalidate(runner, repo)
    assert result.exit_code == 0, result.output

    data = _profile_data(profile)
    assert data["provenance"]["preset"] == "hello-world"
    assert data["provenance"]["preset_hash"] == compute_preset_hash("hello-world")
    assert [key for key in removed if key not in data["config"]] == []
    assert invoke_build(repo).exit_code == 0


def test_provenance_outranks_the_retired_key(runner: CliRunner, tmp_path: Path) -> None:
    """``provenance.preset`` names the exact preset; the retired key named a bundle."""
    bundle = _preset_config("control-assistant")
    recorded = _preset_config("hello-world")
    serving = "control_system.connector.virtual_accelerator.serving"
    epics_timeout = "control_system.connector.epics.timeout_s"
    # Measured, not assumed: the two presets disagree on these keys.
    assert bundle["control_system.writes"] == "allowed"
    assert recorded["control_system.writes"] == "never"
    assert serving not in bundle
    assert recorded[serving] == "in_process"
    assert epics_timeout in bundle
    assert epics_timeout not in recorded

    repo = tmp_path / "facility"
    _init(runner, repo, "hello-world")
    profile = repo / "profile.yml"
    removed = _to_old_shape(profile, drop=("control_system.",), template="control_assistant")

    result = _revalidate(runner, repo)
    assert result.exit_code == 0, result.output

    data = _profile_data(profile)
    assert data["provenance"]["preset"] == "hello-world"
    assert data["provenance"]["preset_hash"] == compute_preset_hash("hello-world")
    assert PRESET_DATA_BUNDLE_KEY not in data
    # The fill never writes the tier key: a root that does not state it reads
    # as `never`, so neither preset's value is taken, and the output says so.
    assert [key for key in removed if key not in data["config"]] == ["control_system.writes"]
    assert "control_system.writes" in result.output
    assert data["config"][serving] == "in_process"
    assert epics_timeout not in data["config"]


def test_preset_overrides_every_other_source(runner: CliRunner, tmp_path: Path) -> None:
    repo = tmp_path / "facility"
    _init(runner, repo, "hello-world")
    profile = repo / "profile.yml"
    _to_old_shape(profile)

    result = _revalidate(runner, repo, "--preset", "control-assistant")
    assert result.exit_code == 0, result.output

    data = _profile_data(profile)
    assert data["provenance"]["preset"] == "control-assistant"
    assert PRESET_DATA_BUNDLE_KEY not in data


def test_an_unshipped_preset_is_refused_with_nothing_written(
    runner: CliRunner, tmp_path: Path
) -> None:
    """``--preset`` is the operator saying it, so it never quietly loses."""
    repo = tmp_path / "facility"
    _init(runner, repo, "hello-world")
    _to_old_shape(repo / "profile.yml")
    before = _repo_files(repo)

    result = _revalidate(runner, repo, "--preset", "neutrino-standalone")

    assert result.exit_code == 2
    assert "neutrino-standalone" in result.output
    assert "hello-world" in result.output
    assert _repo_files(repo) == before


def test_a_tier_named_as_the_preset_gets_the_layer_refusal(
    runner: CliRunner, tmp_path: Path
) -> None:
    """``readonly`` is a real bundled file — a tier — so "unshipped" would mislead."""
    repo = tmp_path / "facility"
    _init(runner, repo, "hello-world")
    _to_old_shape(repo / "profile.yml")
    before = _repo_files(repo)

    result = _revalidate(runner, repo, "--preset", "readonly")

    assert result.exit_code == 2
    assert "is a tier, not a deployment" in result.output
    assert "does not ship" not in result.output
    assert _repo_files(repo) == before


def test_underscore_spelling_of_the_preset_is_accepted(runner: CliRunner, tmp_path: Path) -> None:
    """``app_template: hello_world`` is the underscore spelling ``--preset`` takes."""
    repo = tmp_path / "facility"
    _init(runner, repo, "hello-world")
    profile = repo / "profile.yml"
    _to_old_shape(profile, keep_provenance=False, template="hello_world")

    assert _revalidate(runner, repo).exit_code == 0
    assert _profile_data(profile)["provenance"]["preset"] == "hello-world"


def test_a_profile_naming_no_preset_skips_the_fill_and_says_how_to_name_one(
    runner: CliRunner, tmp_path: Path
) -> None:
    repo = tmp_path / "facility"
    _init(runner, repo, "hello-world")
    profile = repo / "profile.yml"
    removed = _to_old_shape(profile, template=None, keep_provenance=False)

    result = _revalidate(runner, repo)

    assert result.exit_code == 0, result.output
    assert "osprey init --revalidate --preset NAME" in result.output
    assert "no key was filled in" in result.output
    keys = _config_keys(profile.read_text(encoding="utf-8"))
    assert [key for key in removed if key in keys] == []
    assert "provenance" not in _profile_data(profile)


def test_an_unknown_recorded_preset_skips_the_fill_but_still_rewrites_the_roster(
    runner: CliRunner, tmp_path: Path
) -> None:
    """A profile from a release that shipped a preset this one does not."""
    repo = tmp_path / "facility"
    _init(runner, repo, "control-assistant")
    profile = repo / "profile.yml"
    text = profile.read_text(encoding="utf-8")
    assert "    default_tier: readonly" in text
    profile.write_text(
        text.replace("  preset: control-assistant", "  preset: neutrino-standalone").replace(
            "    default_tier: readonly", f"    {RETIRED_DEFAULT_TIER_KEY}: readonly"
        ),
        encoding="utf-8",
    )
    _to_old_shape(profile, drop=("approval.",), template=None)

    result = _revalidate(runner, repo)

    assert result.exit_code == 0, result.output
    assert "osprey init --revalidate --preset NAME" in result.output
    after = profile.read_text(encoding="utf-8")
    assert "    default_tier: readonly" in after
    assert RETIRED_DEFAULT_TIER_KEY not in after
    assert [key for key in _config_keys(after) if key.startswith("approval")] == []
    assert _profile_data(profile)["provenance"]["preset"] == "neutrino-standalone"


def test_nothing_lacking_leaves_the_repo_untouched(runner: CliRunner, tmp_path: Path) -> None:
    """A repo straight out of ``init`` is already current, byte for byte."""
    repo = tmp_path / "facility"
    _init(runner, repo, "hello-world")
    before = _repo_files(repo)

    result = _revalidate(runner, repo)

    assert result.exit_code == 0, result.output
    assert _repo_files(repo) == before
    assert "already current; nothing to rewrite." in result.output


# ---------------------------------------------------------------------------
# providers.yml — refreshed on every run
# ---------------------------------------------------------------------------


def test_providers_refreshes_packaged_entries_and_keeps_yours_on_every_run(
    runner: CliRunner, tmp_path: Path
) -> None:
    repo = tmp_path / "facility"
    _init(runner, repo, "hello-world")
    catalog = repo / PROVIDERS_FILENAME
    catalog.write_text("providers:\n" + SITE_GATEWAY, encoding="utf-8")

    for _ in range(2):
        result = _revalidate(runner, repo)
        assert result.exit_code == 0, result.output

        text = catalog.read_text(encoding="utf-8")
        assert "site-gateway" in text
        assert "# Our own gateway." in text
        packaged = yaml.safe_load(packaged_catalog_path().read_text(encoding="utf-8"))
        entries = yaml.safe_load(text)["providers"]
        assert set(packaged["providers"]) <= set(entries)
        assert "Kept your own provider entries: site-gateway" in result.output


def test_providers_refresh_is_in_the_diff_and_restamps_the_hash(
    runner: CliRunner, tmp_path: Path
) -> None:
    """The stamp describes the file the repo ends up holding, and the diff shows it."""
    from osprey.profiles.providers import compute_providers_hash

    repo = tmp_path / "facility"
    _init(runner, repo, "hello-world")
    profile = repo / "profile.yml"
    catalog = repo / PROVIDERS_FILENAME
    catalog.write_text("providers:\n" + SITE_GATEWAY, encoding="utf-8")

    result = _revalidate(runner, repo)
    assert result.exit_code == 0, result.output

    assert f"+++ b/{PROVIDERS_FILENAME}" in result.output
    assert _profile_data(profile)["provenance"]["providers_hash"] == compute_providers_hash(catalog)


def test_an_unparsable_catalog_refuses_with_the_repo_untouched(
    runner: CliRunner, tmp_path: Path
) -> None:
    repo = tmp_path / "facility"
    _init(runner, repo, "hello-world")
    _to_old_shape(repo / "profile.yml")
    (repo / PROVIDERS_FILENAME).write_text("providers: [unclosed\n", encoding="utf-8")
    before = _repo_files(repo)

    result = _revalidate(runner, repo)

    assert result.exit_code == 2
    assert PROVIDERS_FILENAME in result.output or "provider catalog" in result.output
    assert _repo_files(repo) == before


# ---------------------------------------------------------------------------
# Shapes hello-world cannot show: a key after a trailing list, a deploy block,
# and a roster the facility chose
# ---------------------------------------------------------------------------


def test_control_assistant_keeps_every_section_header_where_it_belongs(
    runner: CliRunner, tmp_path: Path
) -> None:
    """Two comment blocks stored on a key other than the one they introduce.

    Every control-assistant preset ends ``config:`` with the block that
    introduces ``dispatch:``, and nests ``modules.web_terminals:`` — whose last
    key ``shared`` is a LIST — in front of a key whose own header ruamel
    therefore stores on that list's last item. Appending to the block must move
    the first and copy the second.
    """
    repo = tmp_path / "facility"
    _init(runner, repo, "control-assistant")
    profile = repo / "profile.yml"
    before = profile.read_text(encoding="utf-8")
    heading = _block_above(before, "dispatch:")
    assert heading, "preset no longer heads dispatch: with a comment block"
    removed = _to_old_shape(
        profile, drop=("approval.", "execution.execution_method"), template="control_assistant"
    )
    assert "execution.execution_method" in removed

    assert _revalidate(runner, repo).exit_code == 0

    text = profile.read_text(encoding="utf-8")
    # A comment at the margin INSIDE config: is one that belongs to the block
    # after it — the whole shape of the relocation failure.
    assert [line for line in _config_lines(text) if line.startswith("#")] == []
    assert _block_above(text, "dispatch:") == heading
    reference = emit_standalone_profile_yaml(
        "control-assistant", (), "Facility", project_name="facility"
    )
    assert [
        key
        for key in removed
        if _comment_block_above(text, key) != _comment_block_above(reference, key)
    ] == []
    # Not vacuous: the key after the trailing list has a header of its own.
    assert _comment_block_above(reference, "execution.execution_method")
    assert invoke_build(repo).exit_code == 0


def test_a_deploy_block_keeps_image_source_out_of_the_config_block(
    runner: CliRunner, tmp_path: Path
) -> None:
    """One fact, one home: ``deploy.image_source`` claims the config leaf."""
    repo = tmp_path / "facility"
    _init(runner, repo, "control-assistant")
    profile = repo / "profile.yml"
    text = profile.read_text(encoding="utf-8")
    assert "\n    image_source:" in text, "preset no longer nests image_source"
    text = "".join(
        line for line in text.splitlines(keepends=True) if not line.startswith("    image_source:")
    )
    text += "deploy:\n  ci: gitlab\n  image_source: local\n  host:\n    name: demo-deploy\n"
    text += "    user: osprey\n    project_path: /opt/demo\n"
    profile.write_text(text, encoding="utf-8")

    result = _revalidate(runner, repo)
    assert result.exit_code == 0, result.output

    config = _profile_data(profile)["config"]
    assert config_image_source_spelling(config) is None
    assert IMAGE_SOURCE_CONFIG_KEY in result.output  # the skip is reported
    validated = runner.invoke(validate, ["--repo", str(repo), "--drift", "warn"])
    assert validated.exit_code == 0, validated.output


def test_the_exemplar_gains_none_of_the_deployments_own_selections(
    runner: CliRunner, tmp_path: Path
) -> None:
    """Who is on the roster and which apps are shared is the deployment's choice.

    The preset shares three apps and this repo shares none, so filling in the
    lacking leaf would invent three app renders the facility never asked for,
    each naming an ``apps/<name>.yml`` the repo does not hold.
    """
    repo = _exemplar(tmp_path)
    profile = repo / "profile.yml"
    before_config = _profile_data(profile)["config"]
    before = before_config["modules.web_terminals"]
    assert "shared" not in before
    assert sorted(p.stem for p in (repo / "tiers").iterdir()) == ["readonly", "readwrite"]
    assert not (repo / "apps").exists()
    # Not vacuous: the preset documents shared apps this repo does not declare.
    shipped = _preset_config("control-assistant")["modules.web_terminals.shared"]
    assert set(shipped) == {"logbook", "knowledge", "channel-finder"}

    result = _revalidate(runner, repo, "--preset", "control-assistant")
    assert result.exit_code == 0, result.output

    config = _profile_data(profile)["config"]
    after = config["modules.web_terminals"]
    for key in ("shared", "users", "default_tier"):
        assert after.get(key) == before.get(key), key
    assert after.get("landing", {}).get("groups") == before.get("landing", {}).get("groups")
    strays = [
        key
        for key in dict(_flatten(config))
        if key.startswith(
            (
                "modules.web_terminals.renders",
                f"modules.web_terminals.{RETIRED_CATALOG_KEY}",
            )
        )
    ]
    assert strays == []
    assert config_image_source_spelling(config) is None
    flat_output = " ".join(result.output.split())
    spelled = dict(_flatten(before_config))
    lacking_selections = [key for key in SELECTIONS if key not in spelled]
    assert "modules.web_terminals.shared" in lacking_selections
    for key in lacking_selections:
        assert (
            f"Left for you: {key}: `{key}` is this deployment's own selection; "
            "a preset never fills it in."
        ) in flat_output
    assert not (repo / "apps").exists()


def test_the_exemplar_validates_after_revalidate(runner: CliRunner, tmp_path: Path) -> None:
    """The golden holds no data tree, so it never builds on a host; it validates."""
    repo = _exemplar(tmp_path)

    assert _revalidate(runner, repo, "--preset", "control-assistant").exit_code == 0

    validated = runner.invoke(validate, ["--repo", str(repo), "--drift", "warn"])
    assert validated.exit_code == 0, validated.output


def test_revalidating_the_exemplar_twice_changes_nothing(runner: CliRunner, tmp_path: Path) -> None:
    """Byte-level idempotence on a hand-built repo, not only on an emitted one."""
    repo = _exemplar(tmp_path)

    assert _revalidate(runner, repo, "--preset", "control-assistant").exit_code == 0
    once = _repo_files(repo)
    again = _revalidate(runner, repo, "--preset", "control-assistant")
    assert again.exit_code == 0, again.output

    assert _repo_files(repo) == once
    assert "nothing to rewrite" in again.output


def test_a_re_pointed_profile_warns_about_the_drift_it_turns_on(
    runner: CliRunner, tmp_path: Path
) -> None:
    """Stamping provenance turns the preset-drift lint on, so the verb says so."""
    repo = _exemplar(tmp_path)

    result = _revalidate(runner, repo, "--preset", "control-assistant")
    assert result.exit_code == 0, result.output

    assert "DEVIATION" in result.output
    assert "--drift=warn" in result.output
    if "Left for you to do by hand" in result.output:
        assert result.output.index("--drift=warn") < result.output.index(
            "Left for you to do by hand"
        )
    refused = runner.invoke(validate, ["--repo", str(repo)])
    assert refused.exit_code != 0
    warned = runner.invoke(validate, ["--repo", str(repo), "--drift", "warn"])
    assert warned.exit_code == 0, warned.output


def test_an_old_control_assistant_profile_without_shared_does_not_gain_it(
    runner: CliRunner, tmp_path: Path
) -> None:
    repo = tmp_path / "facility"
    _init(runner, repo, "control-assistant")
    profile = repo / "profile.yml"
    _edit_round_trip(profile, lambda doc: doc["config"]["modules.web_terminals"].pop("shared"))
    assert "shared" not in _profile_data(profile)["config"]["modules.web_terminals"]

    result = _revalidate(runner, repo)
    assert result.exit_code == 0, result.output

    assert "shared" not in _profile_data(profile)["config"]["modules.web_terminals"]
    assert "Left for you: modules.web_terminals.shared" in result.output
    built = invoke_build(repo)
    assert built.exit_code == 0, built.output


def test_a_profile_without_dispatch_gains_no_dispatch_block(
    runner: CliRunner, tmp_path: Path
) -> None:
    """The fill reads ``config:`` only, so a top-level block is never filled."""
    repo = tmp_path / "facility"
    _init(runner, repo, "control-assistant")
    profile = repo / "profile.yml"
    assert "dispatch" in _profile_data(profile)
    profile.write_text(
        _without_top_level_block(profile.read_text(encoding="utf-8"), "dispatch"),
        encoding="utf-8",
    )
    assert "dispatch" not in _profile_data(profile)

    result = _revalidate(runner, repo)
    assert result.exit_code == 0, result.output

    assert "dispatch" not in _profile_data(profile)


def test_a_dispatch_block_without_a_tier_gains_none(runner: CliRunner, tmp_path: Path) -> None:
    repo = tmp_path / "facility"
    _init(runner, repo, "control-assistant")
    profile = repo / "profile.yml"
    _edit_round_trip(profile, lambda doc: doc["dispatch"].pop("tier"))
    assert "tier" not in _profile_data(profile)["dispatch"]

    result = _revalidate(runner, repo)
    assert result.exit_code == 0, result.output

    assert "tier" not in _profile_data(profile)["dispatch"]


def test_authorization_roles_are_never_filled(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A preset that spelled role tiers would still not hand them to a profile."""
    from osprey.cli import build_profile_drift, init_revalidate

    roles = "modules.web_terminals.authorization.roles"
    lacking = [roles, f"{roles}.operator"]
    monkeypatch.setattr(build_profile_drift, "lacking_config_keys", lambda _doc, _preset: lacking)
    real_reference = init_revalidate._reference_document

    def reference_with_roles(preset: str, **kwargs):
        from ruamel.yaml.comments import CommentedMap

        document = real_reference(preset, **kwargs)
        document["config"][roles] = CommentedMap(operator="readwrite")
        return document

    monkeypatch.setattr(init_revalidate, "_reference_document", reference_with_roles)
    repo = tmp_path / "facility"
    _init(runner, repo, "control-assistant")
    profile = repo / "profile.yml"

    result = _revalidate(runner, repo)
    assert result.exit_code == 0, result.output

    flattened = dict(_flatten(_profile_data(profile)["config"]))
    assert [key for key in flattened if "authorization.roles" in key] == []
    for key in lacking:
        assert f"Left for you: {key}:" in result.output
    assert "Filled" not in result.output


def test_the_header_prose_names_the_preset_provenance_now_names(
    runner: CliRunner, tmp_path: Path
) -> None:
    """``--preset`` re-points the stamp, so the header above it stops disagreeing."""
    repo = tmp_path / "facility"
    _init(runner, repo, "hello-world")
    profile = repo / "profile.yml"
    _to_old_shape(profile)
    assert "Made from the bundled `hello-world` preset" in profile.read_text(encoding="utf-8")

    assert _revalidate(runner, repo, "--preset", "control-assistant").exit_code == 0

    text = profile.read_text(encoding="utf-8")
    assert "Made from the bundled `control-assistant` preset" in text
    assert "`hello-world` preset" not in text
    assert f"preset content hash: {compute_preset_hash('control-assistant')}" in text


def _empty_config(profile: Path) -> None:
    """Strip every line of the profile's ``config:`` block, header kept."""
    out: list[str] = []
    inside = False
    for line in profile.read_text(encoding="utf-8").splitlines(keepends=True):
        if line.startswith("config:"):
            inside = True
            out.append("config:\n")
            continue
        if inside:
            if line.strip() and not line.startswith((" ", "\t")):
                inside = False
            else:
                continue
        out.append(line)
    profile.write_text("".join(out), encoding="utf-8")


def test_an_empty_config_block_is_filled_at_the_blocks_own_indent(
    runner: CliRunner, tmp_path: Path
) -> None:
    """A profile whose ``config:`` says nothing still reads like an emitted one."""
    repo = tmp_path / "facility"
    _init(runner, repo, "hello-world")
    profile = repo / "profile.yml"
    _to_old_shape(profile)
    _empty_config(profile)
    assert _config_keys(profile.read_text(encoding="utf-8")) == []

    assert _revalidate(runner, repo).exit_code == 0

    text = profile.read_text(encoding="utf-8")
    assert _config_keys(text)
    assert [line for line in _config_lines(text) if line.startswith("#")] == []
    assert invoke_build(repo).exit_code == 0


# ---------------------------------------------------------------------------
# One migration verb
# ---------------------------------------------------------------------------


def test_the_retired_expand_verb_is_gone(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["profile", "expand"])

    assert result.exit_code == 2


def test_set_names_the_retired_key_for_what_it_is(runner: CliRunner, tmp_path: Path) -> None:
    """``osprey set app_template=...`` gets the retired-key message, naming revalidate."""
    from osprey.cli.set_cmd import set as set_cmd

    repo = tmp_path / "facility"
    _init(runner, repo, "hello-world")

    result = runner.invoke(set_cmd, ["--repo", str(repo), f"{PRESET_DATA_BUNDLE_KEY}=hello_world"])

    assert result.exit_code == 0, result.output
    assert "osprey init --revalidate" in result.output


def test_every_remedy_names_the_one_migration_verb() -> None:
    """No tracked text tells an operator to run the retired verb."""
    retired = "profile " + "expand"
    listed = subprocess.run(
        ["git", "ls-files", "--", "src", "docs", "plugins", "tests"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    ).stdout.splitlines()
    this_file = Path(__file__).resolve().relative_to(REPO_ROOT).as_posix()
    offenders = []
    for rel in listed:
        if rel in (this_file, "CHANGELOG.md") or rel.startswith(
            "tests/fixtures/previous_release_repo/"
        ):
            continue
        path = REPO_ROOT / rel
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if retired in text:
            offenders.append(rel)
    assert offenders == []
