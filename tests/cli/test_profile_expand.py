"""``osprey profile expand`` — the one-time migration of a pre-template profile.

A profile written while ``templates/apps/<name>/config.yml.j2`` still existed
spells only what its facility changed. The template is gone, so such a profile
is refused outright, and this verb is the way back: it writes every ``config:``
key the preset documents into the profile itself, under the preset's own
comments, drops the retired ``app_template:`` key, and re-stamps provenance.

The shape under test throughout is a REAL one — a repo ``osprey init`` made,
edited back into the old shape — rather than a hand-written stub, because what
the verb has to reproduce is the emitted file, comments and all, and only an
emitted file specifies that.
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.build_cmd import build
from osprey.cli.build_profile_deploy import (
    IMAGE_SOURCE_CONFIG_KEY,
    config_image_source_spelling,
)
from osprey.cli.build_profile_emit import emit_standalone_profile_yaml, materialized_profile
from osprey.cli.build_profile_merge import compute_preset_hash
from osprey.cli.init_cmd import init
from osprey.cli.main import cli
from osprey.cli.profile_expand import RETIRED_TEMPLATE_KEY, expand
from osprey.cli.validate_cmd import validate
from osprey.profiles.providers import PROVIDERS_FILENAME, packaged_catalog_path

#: The four ``approval`` leaves plus their two neighbours, as a "whole branch
#: and then some" removal: the branch the proposal names, and two single keys
#: whose comment blocks sit in the middle of other sections.
DROPPED_BRANCHES = ("approval.", "archiver.", "hooks.debug")

#: The frozen hand-built reference deployment. Two things about it are what the
#: tests below need and no ``osprey init`` repo has: it states ``image_source``
#: once, in its ``deploy:`` block, and its persona catalog names two of the five
#: personas the control-assistant preset ships.
EXEMPLAR = Path(__file__).resolve().parents[1] / "deployment" / "goldens" / "exemplar-profile"


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


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


def _build(runner: CliRunner, repo: Path):
    return runner.invoke(build, ["--repo", str(repo), "--skip-deps", "--skip-lifecycle"])


def _expand(runner: CliRunner, repo: Path, *extra: str):
    return runner.invoke(expand, ["--repo", str(repo), *extra])


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

    Text-level on purpose: what the expanded file has to reproduce is what an
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
    """Edit an emitted profile back into the shape this verb migrates.

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
        text = _without_provenance(text)
    if template is not None:
        text = text.replace("\nconfig:\n", f"\n{RETIRED_TEMPLATE_KEY}: {template}\nconfig:\n", 1)
    profile.write_text(text, encoding="utf-8")
    return removed


def _without_provenance(text: str) -> str:
    """Drop the ``provenance:`` block — the shape a profile predating it has."""
    out: list[str] = []
    inside = False
    for line in text.splitlines(keepends=True):
        if line.startswith("provenance:"):
            inside = True
            continue
        if inside and line.strip() and not line.startswith((" ", "\t")):
            inside = False
        if not inside:
            out.append(line)
    return "".join(out)


def _profile_data(profile: Path) -> dict:
    return yaml.safe_load(profile.read_text(encoding="utf-8"))


def _rendered_config(repo: Path) -> dict:
    """The build's own ``build/config.yml`` — what the RUNTIME reads.

    The profile is the input; this is the output, and the two are not the same
    document. Dotted keys are expanded on the way here, but only at the top
    level of ``config:``: a dotted key nested inside a mapping arrives as a
    literal key with dots in its name, which no runtime reader ever looks up.
    So a claim about what the deployment will actually do has to be made here.
    """
    return yaml.safe_load((repo / "build" / "config.yml").read_text(encoding="utf-8"))


def _dotted_keys_below(value: object, prefix: str = "") -> list[str]:
    """Every key with a dot in its NAME anywhere below *value*, dotted-path form.

    The shape a rendered config must never contain below its top level: a key
    the applier did not expand, sitting where the reader expects a mapping.
    """
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

    Copied rather than used in place: expand WRITES, and the golden is a
    tracked reference several other suites compare against byte for byte.
    """
    repo = tmp_path / "demo-facility"
    shutil.copytree(EXEMPLAR, repo)
    return repo


def _preset_config(preset: str) -> dict[str, object]:
    """The preset's rendered ``config:`` leaves, dotted — values included."""
    materialized = materialized_profile(preset, repo_name="facility", profile_name="Facility")
    return dict(_flatten(materialized.get("config") or {}))


# ---------------------------------------------------------------------------
# Acceptance: the whole journey
# ---------------------------------------------------------------------------


def test_old_shape_refused_then_expand_then_build_green(
    runner: CliRunner, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The migration, end to end: refused, expanded, built, and stable.

    One test rather than four because the sequence IS the requirement — a verb
    that expands a profile the build still refuses, or produces one that will
    not build, has done nothing for the operator standing in front of it.
    """
    repo = tmp_path / "facility"
    _init(runner, repo, "hello-world")
    profile = repo / "profile.yml"
    removed = _to_old_shape(profile)
    assert removed, "fixture removed nothing"

    refused = _build(runner, repo)
    assert refused.exit_code != 0
    # The refusal reaches the operator through the build's logger rather than
    # through the command's own stdout, so that is where it is read from.
    assert "osprey profile expand" in caplog.text

    expanded = _expand(runner, repo)
    assert expanded.exit_code == 0, expanded.output

    text = profile.read_text(encoding="utf-8")
    assert RETIRED_TEMPLATE_KEY not in _profile_data(profile)
    assert [key for key in removed if key not in _config_keys(text)] == []

    built = _build(runner, repo)
    assert built.exit_code == 0, built.output
    assert (repo / "build" / "config.yml").is_file()

    again = _expand(runner, repo)
    assert again.exit_code == 0, again.output
    assert profile.read_text(encoding="utf-8") == text


# ---------------------------------------------------------------------------
# What lands in the file
# ---------------------------------------------------------------------------


def test_every_expanded_key_carries_the_presets_comment(runner: CliRunner, tmp_path: Path) -> None:
    """The point of the verb: the profile documents what it now spells.

    Compared against the preset's own emitted profile rather than against a
    transcription, so a preset that rewords a key's prose moves both sides at
    once and this stays a statement about equality, not about wording.
    """
    repo = tmp_path / "facility"
    _init(runner, repo, "hello-world")
    profile = repo / "profile.yml"
    removed = _to_old_shape(profile)

    assert _expand(runner, repo).exit_code == 0

    expanded = profile.read_text(encoding="utf-8")
    reference = emit_standalone_profile_yaml("hello-world", (), "Facility")
    differing = {
        key: (_comment_block_above(expanded, key), _comment_block_above(reference, key))
        for key in removed
        if _comment_block_above(expanded, key) != _comment_block_above(reference, key)
    }
    assert differing == {}
    # Not vacuous: most of the removed keys do carry prose.
    assert sum(bool(_comment_block_above(expanded, key)) for key in removed) >= 5


def test_a_whole_missing_branch_comes_back_as_its_leaves(runner: CliRunner, tmp_path: Path) -> None:
    """A profile with no ``approval`` key at all gets the branch, leaf by leaf.

    The shape the proposal names, on its own: not a key here and a key there,
    but a whole section the old profile left entirely to the app template. What
    lands is what the preset DOCUMENTS — one dotted key per leaf, each with its
    own prose — rather than one opaque mapping the operator would have to go
    and look up.
    """
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

    assert _expand(runner, repo).exit_code == 0

    text = profile.read_text(encoding="utf-8")
    assert [key for key in removed if key not in _config_keys(text)] == []
    reference = emit_standalone_profile_yaml("hello-world", (), "Facility")
    assert [
        key
        for key in removed
        if _comment_block_above(text, key) != _comment_block_above(reference, key)
    ] == []
    assert _build(runner, repo).exit_code == 0


def test_expanded_values_are_the_presets(runner: CliRunner, tmp_path: Path) -> None:
    repo = tmp_path / "facility"
    _init(runner, repo, "hello-world")
    profile = repo / "profile.yml"
    before = _profile_data(profile)["config"]
    removed = _to_old_shape(profile)

    assert _expand(runner, repo).exit_code == 0

    after = _profile_data(profile)["config"]
    assert {key: after[key] for key in removed} == {key: before[key] for key in removed}


def test_a_value_the_operator_changed_is_not_overwritten(runner: CliRunner, tmp_path: Path) -> None:
    """Expansion only ADDS: the profile is the source of truth once it exists."""
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

    assert _expand(runner, repo).exit_code == 0

    assert _profile_data(profile)["config"]["control_system.read_inline_max_elements"] == 7


def test_a_nested_spelling_is_left_alone(runner: CliRunner, tmp_path: Path) -> None:
    """A profile nesting a key already spells it — mixed spellings are legal.

    Without this the verb would write a dotted duplicate of a key the profile
    holds in the nested form the build documents, and the operator would find
    the same setting in two places.

    The leaves that ARE lacking go inside the mapping the profile nests rather
    than beside it: a dotted ``approval.tools.channel_write`` next to an
    ``approval:`` mapping addresses one block at two depths, which is the shape
    ``osprey validate`` refuses outright for ``control_system:`` and which
    leaves every other block's reader looking in two places.
    """
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

    assert _expand(runner, repo).exit_code == 0

    text = profile.read_text(encoding="utf-8")
    keys = _config_keys(text)
    assert "approval.enabled" not in keys
    assert "approval.default_policy" not in keys
    # Not beside the mapping...
    assert "approval.tools.channel_write" not in keys

    # ...but inside it, addressing the same rendered leaf — asserted on the
    # RENDER, because the profile data cannot tell a nested path from a dotted
    # key sitting inside a mapping, and only one of those two the runtime reads.
    assert _build(runner, repo).exit_code == 0
    fresh = tmp_path / "reference"
    _init(runner, fresh, "hello-world")
    assert _build(runner, fresh).exit_code == 0
    approval = _rendered_config(repo)["approval"]
    assert approval["tools"] == _rendered_config(fresh)["approval"]["tools"]
    assert _dotted_keys_below(approval) == []


def test_a_nested_parent_gets_real_mappings_not_a_dotted_key(
    runner: CliRunner, tmp_path: Path
) -> None:
    """A leaf below a mapping the profile nests lands under REAL intermediates.

    The dangerous shape: ``approval:`` nested with no ``tools`` under it, and
    the per-tool policies written back as ``tools.channel_write: always``
    INSIDE it. Nothing expands a dotted key there — the applier only does that
    at the top level of ``config:`` — so the render carries a literal key with a
    dot in its name, the approval hook reads ``approval['tools']`` and finds
    nothing, and every per-tool policy silently falls back to
    ``default_policy``.

    ``default_policy: skip`` is what makes the collapse visible: a channel write
    the preset asks to prompt for would proceed unprompted, on a profile that
    was correct before the expansion touched it.
    """
    repo = tmp_path / "facility"
    _init(runner, repo, "hello-world")
    profile = repo / "profile.yml"
    _to_old_shape(profile, drop=("approval.",))
    profile.write_text(
        profile.read_text(encoding="utf-8").replace(
            "\nconfig:\n",
            "\nconfig:\n  approval:\n    enabled: true\n    default_policy: skip\n",
            1,
        ),
        encoding="utf-8",
    )

    assert _expand(runner, repo).exit_code == 0
    assert _build(runner, repo).exit_code == 0

    approval = _rendered_config(repo)["approval"]
    # The operator's own value is untouched, so a policy that fell back to it
    # would read `skip` rather than the preset's `always`.
    assert approval["default_policy"] == "skip"
    assert approval["tools"]["channel_write"] == "always"
    assert _dotted_keys_below(approval) == []


def test_a_nested_preset_branch_expands_to_its_leaves(runner: CliRunner, tmp_path: Path) -> None:
    """The control-assistant presets spell ``modules.web_terminals:`` nested.

    Its leaves are addressed by dotted keys like every other, so a profile
    missing one gets that key back — proof the walk that collects values and
    comments recurses rather than reading only the block's top level.
    """
    repo = tmp_path / "facility"
    _init(runner, repo, "control-assistant")
    profile = repo / "profile.yml"
    text = profile.read_text(encoding="utf-8")
    assert "  modules.web_terminals:" in text, "preset no longer nests this branch"
    profile.write_text(
        text.replace("\nconfig:\n", f"\n{RETIRED_TEMPLATE_KEY}: control_assistant\nconfig:\n", 1),
        encoding="utf-8",
    )
    dropped = _drop_nested_leaf(profile)

    assert _expand(runner, repo).exit_code == 0

    config = _profile_data(profile)["config"]
    flattened = dict(_flatten(config))
    assert dropped in flattened


def _drop_nested_leaf(profile: Path) -> str:
    """Remove one leaf from under ``modules.web_terminals``, returning its dotted key.

    Walked by indentation rather than parsed, because the line has to be cut
    out of the FILE: a round trip would rewrite the block being tested.
    """
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
            del lines[index]
            profile.write_text("".join(lines), encoding="utf-8")
            return ".".join(path)
    raise AssertionError("no nested leaf found under modules.web_terminals")


def _flatten(mapping: dict, prefix: str = "") -> list[tuple[str, object]]:
    out: list[tuple[str, object]] = []
    for key, value in mapping.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict) and value:
            out.extend(_flatten(value, f"{dotted}."))
        else:
            out.append((dotted, value))
    return out


# ---------------------------------------------------------------------------
# Which preset, and the provenance stamp
# ---------------------------------------------------------------------------


def test_a_profile_without_provenance_expands_from_the_retired_key(
    runner: CliRunner, tmp_path: Path
) -> None:
    """The shape with nothing else to go on — and the block is written back."""
    repo = tmp_path / "facility"
    _init(runner, repo, "hello-world")
    profile = repo / "profile.yml"
    removed = _to_old_shape(profile, keep_provenance=False)
    assert "provenance" not in _profile_data(profile)

    result = _expand(runner, repo)
    assert result.exit_code == 0, result.output

    data = _profile_data(profile)
    assert data["provenance"]["preset"] == "hello-world"
    assert data["provenance"]["preset_hash"] == compute_preset_hash("hello-world")
    assert [key for key in removed if key not in data["config"]] == []
    assert _build(runner, repo).exit_code == 0


def test_provenance_outranks_the_retired_key(runner: CliRunner, tmp_path: Path) -> None:
    """``provenance.preset`` names the exact preset; the retired key named a bundle.

    A repo materialized from a persona preset carries ``app_template:
    control_assistant`` and ``provenance.preset:
    control-assistant-readonly``. Expanding from the bundle would write the
    base preset's keys and then re-stamp the profile as if it had come from
    there, so the more precise source wins.

    Asserted on the VALUES as well as on the stamp, because the two presets
    differ on exactly the keys the persona exists for: the base enables writes,
    the readonly persona disables them and adds two connector-level flags the
    base has no key for at all. An expansion from the wrong preset would put
    ``true`` back and skip those two.
    """
    # Measured, not assumed: the two presets have to disagree on these keys for
    # anything below to distinguish them. A preset edit that erased the
    # difference would otherwise leave this test green and vacuous.
    base = _preset_config("control-assistant")
    persona = _preset_config("control-assistant-readonly")
    assert base["control_system.writes_enabled"] is True
    assert persona["control_system.writes_enabled"] is False
    assert "control_system.connector.epics.writes_enabled" not in base
    assert persona["control_system.connector.epics.writes_enabled"] is False

    repo = tmp_path / "facility"
    _init(runner, repo, "control-assistant-readonly")
    profile = repo / "profile.yml"
    removed = _to_old_shape(profile, drop=("control_system.",), template="control_assistant")

    assert _expand(runner, repo).exit_code == 0

    data = _profile_data(profile)
    assert data["provenance"]["preset"] == "control-assistant-readonly"
    assert data["provenance"]["preset_hash"] == compute_preset_hash("control-assistant-readonly")
    # Popped even though provenance, not the retired key, decided the preset.
    assert RETIRED_TEMPLATE_KEY not in data
    assert [key for key in removed if key not in data["config"]] == []
    assert data["config"]["control_system.writes_enabled"] is False
    assert data["config"]["control_system.connector.epics.writes_enabled"] is False


def test_from_overrides_every_other_source(runner: CliRunner, tmp_path: Path) -> None:
    repo = tmp_path / "facility"
    _init(runner, repo, "hello-world")
    profile = repo / "profile.yml"
    _to_old_shape(profile)

    assert _expand(runner, repo, "--from", "channel-finder-standalone").exit_code == 0

    data = _profile_data(profile)
    assert data["provenance"]["preset"] == "channel-finder-standalone"
    assert RETIRED_TEMPLATE_KEY not in data


def test_an_unshipped_from_is_refused_rather_than_falling_through(
    runner: CliRunner, tmp_path: Path
) -> None:
    """``--from`` is the operator saying it, so it never quietly loses.

    The two sources inside the file DO fall through to each other. A preset the
    operator named and this OSPREY does not ship is a different situation: the
    honest answer is the list of presets it does ship, not an expansion from
    whatever the file happened to record.
    """
    repo = tmp_path / "facility"
    _init(runner, repo, "hello-world")
    profile = repo / "profile.yml"
    _to_old_shape(profile)
    before = profile.read_bytes()

    result = _expand(runner, repo, "--from", "neutrino-standalone")

    assert result.exit_code != 0
    assert "neutrino-standalone" in result.output
    assert "hello-world" in result.output
    assert profile.read_bytes() == before


def test_underscore_spelling_of_the_preset_is_accepted(runner: CliRunner, tmp_path: Path) -> None:
    """``app_template: hello_world`` is the underscore spelling ``--preset`` takes."""
    repo = tmp_path / "facility"
    _init(runner, repo, "hello-world")
    profile = repo / "profile.yml"
    _to_old_shape(profile, keep_provenance=False, template="hello_world")

    assert _expand(runner, repo).exit_code == 0
    assert _profile_data(profile)["provenance"]["preset"] == "hello-world"


def test_a_profile_naming_no_preset_is_refused_asking_for_from(
    runner: CliRunner, tmp_path: Path
) -> None:
    repo = tmp_path / "facility"
    _init(runner, repo, "hello-world")
    profile = repo / "profile.yml"
    _to_old_shape(profile, template=None, keep_provenance=False)

    result = _expand(runner, repo)

    assert result.exit_code != 0
    assert "--from PRESET" in result.output


def test_an_unknown_preset_is_refused_asking_for_from(runner: CliRunner, tmp_path: Path) -> None:
    """A profile from a release that shipped a preset this one does not."""
    repo = tmp_path / "facility"
    _init(runner, repo, "hello-world")
    profile = repo / "profile.yml"
    _to_old_shape(profile, template=None)
    profile.write_text(
        profile.read_text(encoding="utf-8").replace(
            "  preset: hello-world", "  preset: neutrino-standalone"
        ),
        encoding="utf-8",
    )

    result = _expand(runner, repo)

    assert result.exit_code != 0
    assert "neutrino-standalone" in result.output
    assert "--from PRESET" in result.output


def test_nothing_to_expand_leaves_the_file_untouched(runner: CliRunner, tmp_path: Path) -> None:
    """A profile straight out of ``init`` is already expanded, byte for byte."""
    repo = tmp_path / "facility"
    _init(runner, repo, "hello-world")
    profile = repo / "profile.yml"
    before = profile.read_bytes()

    result = _expand(runner, repo)

    assert result.exit_code == 0, result.output
    assert profile.read_bytes() == before
    assert "Nothing to expand" in result.output


# ---------------------------------------------------------------------------
# --providers
# ---------------------------------------------------------------------------


def test_providers_refreshes_packaged_entries_and_keeps_yours(
    runner: CliRunner, tmp_path: Path
) -> None:
    repo = tmp_path / "facility"
    _init(runner, repo, "hello-world")
    catalog = repo / PROVIDERS_FILENAME
    catalog.write_text(
        "providers:\n"
        "  # Our own gateway.\n"
        "  site-gateway:\n"
        "    base_url: https://gw.example.org/v1\n"
        "    api_key: ${SITE_GATEWAY_KEY}\n",
        encoding="utf-8",
    )

    result = _expand(runner, repo, "--providers")
    assert result.exit_code == 0, result.output

    text = catalog.read_text(encoding="utf-8")
    assert "site-gateway" in text
    assert "# Our own gateway." in text
    packaged = yaml.safe_load(packaged_catalog_path().read_text(encoding="utf-8"))["providers"]
    entries = yaml.safe_load(text)["providers"]
    assert set(packaged) <= set(entries)


def test_providers_restamps_the_catalog_hash(runner: CliRunner, tmp_path: Path) -> None:
    """The stamp describes the file the repo ends up holding."""
    from osprey.profiles.providers import compute_providers_hash

    repo = tmp_path / "facility"
    _init(runner, repo, "hello-world")
    profile = repo / "profile.yml"
    catalog = repo / PROVIDERS_FILENAME
    catalog.write_text(
        catalog.read_text(encoding="utf-8")
        + "  site-gateway:\n    base_url: https://gw.example.org/v1\n",
        encoding="utf-8",
    )

    assert _expand(runner, repo, "--providers").exit_code == 0

    assert _profile_data(profile)["provenance"]["providers_hash"] == compute_providers_hash(catalog)


def test_expand_without_providers_stamps_the_catalog_already_there(
    runner: CliRunner, tmp_path: Path
) -> None:
    """No flag, no rewrite — but the stamp still describes what a build reads."""
    from osprey.profiles.providers import compute_providers_hash

    repo = tmp_path / "facility"
    _init(runner, repo, "hello-world")
    profile = repo / "profile.yml"
    catalog = repo / PROVIDERS_FILENAME
    catalog.write_text(
        catalog.read_text(encoding="utf-8")
        + "  site-gateway:\n    base_url: https://gw.example.org/v1\n",
        encoding="utf-8",
    )
    before = catalog.read_bytes()
    _to_old_shape(profile)

    assert _expand(runner, repo).exit_code == 0

    assert catalog.read_bytes() == before
    assert _profile_data(profile)["provenance"]["providers_hash"] == compute_providers_hash(catalog)


# ---------------------------------------------------------------------------
# Shapes hello-world cannot show: a config block with a successor, a deploy
# block, and a persona catalog the facility trimmed
# ---------------------------------------------------------------------------


def test_control_assistant_keeps_every_section_header_where_it_belongs(
    runner: CliRunner, tmp_path: Path
) -> None:
    """Two comment blocks stored on a key other than the one they introduce.

    ``hello-world`` ends its file with keys that carry their own prose, so
    nothing is parked on the ``config:`` block's last leaf. Every
    control-assistant preset ends ``config:`` with the block that introduces
    ``dispatch:``, and nests ``modules.web_terminals:`` in front of a key whose
    own header ruamel therefore stores on that branch's deepest leaf. Appending
    to the block must move the first and copy the second.
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

    assert _expand(runner, repo).exit_code == 0

    text = profile.read_text(encoding="utf-8")
    # A comment at the margin INSIDE config: is one that belongs to the block
    # after it — the whole shape of the relocation failure.
    assert [line for line in _config_lines(text) if line.startswith("#")] == []
    assert _block_above(text, "dispatch:") == heading
    reference = emit_standalone_profile_yaml("control-assistant", (), "Facility")
    assert [
        key
        for key in removed
        if _comment_block_above(text, key) != _comment_block_above(reference, key)
    ] == []
    # Not vacuous: the key after the nested branch is the one whose header was
    # dropped, and it does have one.
    assert _comment_block_above(reference, "execution.execution_method")
    assert _build(runner, repo).exit_code == 0


def test_a_deploy_block_keeps_image_source_out_of_the_config_block(
    runner: CliRunner, tmp_path: Path
) -> None:
    """One fact, one home: ``deploy.image_source`` claims the config leaf.

    The build writes the deploy block's value into
    ``modules.web_terminals.image_source`` for the facility, so a profile that
    also spells that leaf is refused. Expanding it in would break the build of
    every repo that has deploy coordinates.
    """
    repo = tmp_path / "facility"
    _init(runner, repo, "control-assistant")
    profile = repo / "profile.yml"
    text = profile.read_text(encoding="utf-8")
    assert "\n    image_source:" in text, "preset no longer nests image_source"
    # Take the leaf out, so an expansion has it to put back, and state the fact
    # in its real home instead.
    text = "".join(
        line for line in text.splitlines(keepends=True) if not line.startswith("    image_source:")
    )
    text += "deploy:\n  ci: gitlab\n  image_source: local\n  host:\n    name: demo-deploy\n"
    text += "    user: osprey\n    project_path: /opt/demo\n"
    profile.write_text(text, encoding="utf-8")

    result = _expand(runner, repo)
    assert result.exit_code == 0, result.output

    config = _profile_data(profile)["config"]
    assert config_image_source_spelling(config) is None
    assert IMAGE_SOURCE_CONFIG_KEY in result.output  # the skip is reported
    validated = runner.invoke(validate, ["--repo", str(repo), "--drift", "warn"])
    assert validated.exit_code == 0, validated.output


def test_the_exemplar_deployment_expands_without_inventing_personas(
    runner: CliRunner, tmp_path: Path
) -> None:
    """The persona catalog is data, not schema — so expand stops at it.

    The manifest marks the catalog a data map: persona NAMES are the
    deployment's own. A preset ships five and this repo declares two, so
    filling in the lacking leaves would invent three personas the facility
    never asked for, with ``project_path`` values derived from whichever
    directory the expansion ran in.
    """
    repo = _exemplar(tmp_path)
    profile = repo / "profile.yml"
    before = _profile_data(profile)["config"]["modules.web_terminals"]["personas"]
    assert set(before) == {"readonly", "readwrite"}
    # Not vacuous: the preset documents personas this repo does not declare.
    shipped = {
        key.split(".")[3]
        for key in _preset_config("control-assistant")
        if key.startswith("modules.web_terminals.personas.")
    }
    assert shipped - set(before) == {"admin", "logbook", "knowledge"}

    result = _expand(runner, repo, "--from", "control-assistant")
    assert result.exit_code == 0, result.output

    config = _profile_data(profile)["config"]
    assert config["modules.web_terminals"]["personas"] == before
    strays = [
        key
        for key in dict(_flatten(config))
        if key.startswith("modules.web_terminals.personas.")
        and key.split(".")[3] not in set(before)
    ]
    assert strays == []
    assert config_image_source_spelling(config) is None


def test_expanding_the_exemplar_twice_changes_nothing(runner: CliRunner, tmp_path: Path) -> None:
    """Byte-level idempotence on a hand-built repo, not only on an emitted one."""
    repo = _exemplar(tmp_path)
    profile = repo / "profile.yml"

    assert _expand(runner, repo, "--from", "control-assistant").exit_code == 0
    once = profile.read_bytes()
    assert _expand(runner, repo, "--from", "control-assistant").exit_code == 0

    assert profile.read_bytes() == once


def test_a_re_pointed_profile_warns_about_the_drift_it_turns_on(
    runner: CliRunner, tmp_path: Path
) -> None:
    """Stamping provenance turns the preset-drift lint on, so expand says so.

    Before the stamp there is nothing to compare the profile with and
    ``osprey validate`` says nothing about drift. After it, every structural
    difference from the preset is a refusal — which is right, and which an
    operator has to hear from the verb that caused it rather than from the next
    build.
    """
    repo = _exemplar(tmp_path)

    result = _expand(runner, repo, "--from", "control-assistant")
    assert result.exit_code == 0, result.output

    assert "DEVIATION" in result.output
    assert "--drift=warn" in result.output
    refused = runner.invoke(validate, ["--repo", str(repo)])
    assert refused.exit_code != 0
    warned = runner.invoke(validate, ["--repo", str(repo), "--drift", "warn"])
    assert warned.exit_code == 0, warned.output


def test_the_header_prose_names_the_preset_provenance_now_names(
    runner: CliRunner, tmp_path: Path
) -> None:
    """``--from`` re-points the stamp, so the header above it stops disagreeing."""
    repo = tmp_path / "facility"
    _init(runner, repo, "hello-world")
    profile = repo / "profile.yml"
    _to_old_shape(profile)
    assert "Made from the bundled `hello-world` preset" in profile.read_text(encoding="utf-8")

    assert _expand(runner, repo, "--from", "channel-finder-standalone").exit_code == 0

    text = profile.read_text(encoding="utf-8")
    assert "Made from the bundled `channel-finder-standalone` preset" in text
    assert "`hello-world` preset" not in text
    assert f"preset content hash: {compute_preset_hash('channel-finder-standalone')}" in text


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

    assert _expand(runner, repo).exit_code == 0

    text = profile.read_text(encoding="utf-8")
    assert _config_keys(text)
    assert [line for line in _config_lines(text) if line.startswith("#")] == []
    assert _build(runner, repo).exit_code == 0


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


# ---------------------------------------------------------------------------
# The verb's place in the CLI
# ---------------------------------------------------------------------------


def test_the_verb_is_registered_on_the_profile_group(runner: CliRunner) -> None:
    """The refusal names ``osprey profile expand``; the group has to answer to it."""
    result = runner.invoke(cli, ["profile", "expand", "--help"])

    assert result.exit_code == 0
    assert "--providers" in result.output
    assert "--from" in result.output


def test_set_names_the_retired_key_for_what_it_is(runner: CliRunner, tmp_path: Path) -> None:
    """``osprey set app_template=...`` gets the retired-key message, not the generic one.

    The generic warning says the key is not in the schema and to prefix it with
    ``config.``, which is wrong for this one: the answer is this verb.
    """
    from osprey.cli.set_cmd import set as set_cmd

    repo = tmp_path / "facility"
    _init(runner, repo, "hello-world")

    result = runner.invoke(set_cmd, ["--repo", str(repo), f"{RETIRED_TEMPLATE_KEY}=hello_world"])

    assert result.exit_code == 0, result.output
    assert "osprey profile expand" in result.output
