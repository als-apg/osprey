"""Annotated profile keys — the two keys an emitted profile has to teach.

``web_panels`` and ``env`` are the keys whose shape a facility cannot guess
from the empty default the emitter writes, so they carry a worked example
instead of the one-line synthesis rationale every other explicit key gets.
These tests pin what that buys: the example is valid YAML, it is entirely
commented (so it never changes what the profile resolves to), it lands
immediately above its key in every bundled preset, and it displaces the
rationale line rather than sitting next to it.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.build_cmd import build
from osprey.cli.build_profile import BuildProfile, list_presets, resolve_build_profile
from osprey.cli.build_profile_emit import (
    _ANNOTATIONS,
    _EXPLICIT_KEYS,
    _FIELD_TO_YAML,
    _SYNTHESIS_RATIONALE,
    emit_standalone_profile_yaml,
)
from osprey.cli.init_cmd import init

# The lines the edited-text gates below apply: the block-style snippet
# inside each annotation, which is the part a facility copies and edits. Keyed
# by the annotated field so the guard below can prove the set has not drifted
# from _ANNOTATIONS, and each fragment is proved to be a real run of lines out
# of its annotation rather than a hand-copied paraphrase.
_ANNOTATION_FRAGMENTS: dict[str, tuple[str, ...]] = {
    "web_panels": (
        "#   web_panels:",
        "#     - elog",
    ),
    "env": (
        "#   env:",
        "#     required: [EPICS_CA_ADDR_LIST]",
        "#     pinned: [ARIEL_DB_PASSWORD]",
        "#     defaults:",
        "#       EPICS_CA_ADDR_LIST: 127.0.0.1",
        "#     file: env/facility.env",
    ),
}


def _emit(preset: str) -> str:
    return emit_standalone_profile_yaml(preset, (), (), "Emitted")


def _uncomment(line: str) -> str:
    """Drop the leading ``#`` and the single space that follows it, if any."""
    body = line[1:]
    return body[1:] if body.startswith(" ") else body


def _snippet(lines: tuple[str, ...]) -> str:
    """The block-style YAML example inside an annotation, dedented.

    The example is the annotation's indented run: prose sits at column 0 once
    the comment marker is off, the example is indented under it.
    """
    body = [_uncomment(line) for line in lines]
    indented = [index for index, line in enumerate(body) if line.startswith(" ")]
    assert indented, "annotation carries no block-style example"
    assert indented == list(range(indented[0], indented[-1] + 1)), (
        "the block-style example must be one contiguous run of lines"
    )
    return textwrap.dedent("\n".join(body[indented[0] : indented[-1] + 1]) + "\n")


def _yaml_key(field: str) -> str:
    return _FIELD_TO_YAML.get(field, field)


# ---------------------------------------------------------------------------
# (a) the table itself
# ---------------------------------------------------------------------------


def test_annotated_fields_are_explicit_members() -> None:
    """Only a key that is always written can be annotated above itself."""
    assert set(_ANNOTATIONS) <= _EXPLICIT_KEYS


@pytest.mark.parametrize("field", sorted(_ANNOTATIONS))
def test_every_annotation_line_is_a_comment(field: str) -> None:
    """No line may open a bare YAML key — an annotation must not change the doc."""
    for line in _ANNOTATIONS[field]:
        assert line.startswith("#"), f"{field}: {line!r} is not commented"


@pytest.mark.parametrize("field", sorted(_ANNOTATIONS))
def test_annotation_example_is_valid_yaml(field: str) -> None:
    """What we tell a facility to uncomment has to parse once uncommented."""
    loaded = yaml.safe_load(_snippet(_ANNOTATIONS[field]))
    assert isinstance(loaded, dict) and loaded, f"{field}: example is not a mapping"


def test_web_panels_names_the_dotted_url_key() -> None:
    """A custom panel is only real once ``web.panels.<id>.url`` is set."""
    text = "\n".join(_ANNOTATIONS["web_panels"])
    for leaf in ("url", "label", "path", "health_endpoint"):
        assert f"web.panels.elog.{leaf}" in text


def test_env_names_every_env_config_member() -> None:
    """All four EnvConfig members, so none of them is discovered by accident."""
    example = yaml.safe_load(_snippet(_ANNOTATIONS["env"]))["env"]
    assert set(example) == {"required", "pinned", "defaults", "file"}


# ---------------------------------------------------------------------------
# (b) the annotation reaches every emitted profile, above its key
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset", list_presets())
@pytest.mark.parametrize("field", sorted(_ANNOTATIONS))
def test_annotation_sits_immediately_above_its_key(preset: str, field: str) -> None:
    annotation = list(_ANNOTATIONS[field])
    key = _yaml_key(field)
    lines = _emit(preset).splitlines()

    assert sum(1 for line in lines if line == annotation[0]) == 1, (
        f"{preset}: the {field} annotation must appear exactly once"
    )
    key_lines = [index for index, line in enumerate(lines) if line.startswith(f"{key}:")]
    assert len(key_lines) == 1, f"{preset}: expected one top-level `{key}:`"
    index = key_lines[0]
    assert lines[index - len(annotation) : index] == annotation, (
        f"{preset}: the {field} annotation is not the block directly above `{key}:`"
    )


def test_annotation_lands_below_a_preset_authored_header() -> None:
    """control-assistant heads its ``env:`` block; our block goes under it."""
    lines = _emit("control-assistant").splitlines()
    index = next(i for i, line in enumerate(lines) if line.startswith("env:"))
    above = lines[: index - len(_ANNOTATIONS["env"])]
    assert above[-1].startswith("#")
    assert "Passwords for the webhook service above" in "\n".join(above[-6:])


# ---------------------------------------------------------------------------
# (c) an annotated key gets no rationale line
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", sorted(_ANNOTATIONS))
def test_annotated_key_displaces_its_rationale(field: str) -> None:
    """hello-world synthesizes ``env``; the rationale line must not survive."""
    text = _emit("hello-world")
    rationale = _SYNTHESIS_RATIONALE.get(field)
    assert rationale is not None, f"{field} has no rationale to displace"
    assert f"# {rationale}" not in text


# ---------------------------------------------------------------------------
# (d) the edited-text fragments the gates below apply
# ---------------------------------------------------------------------------


def test_fragments_cover_exactly_the_annotated_fields() -> None:
    assert set(_ANNOTATION_FRAGMENTS) == set(_ANNOTATIONS)


@pytest.mark.parametrize("field", sorted(_ANNOTATION_FRAGMENTS))
def test_fragment_is_a_contiguous_run_of_its_annotation(field: str) -> None:
    annotation = list(_ANNOTATIONS[field])
    fragment = list(_ANNOTATION_FRAGMENTS[field])
    start = annotation.index(fragment[0])
    assert annotation[start : start + len(fragment)] == fragment


# ---------------------------------------------------------------------------
# (e) the edited text resolves, and keeps what the key already resolved to
# ---------------------------------------------------------------------------

# The dotted half of the web_panels annotation. The list entry alone is not a
# panel — the annotation's prose says so, and the resolver enforces it — so the
# edit a facility actually makes is the snippet plus these three keys.
_ELOG_URL = "http://127.0.0.1:8020"
_ELOG_CONFIG_LINES = (
    f"  web.panels.elog.url: {_ELOG_URL}",
    "  web.panels.elog.label: ELOG",
    "  web.panels.elog.path: /",
)

# What each edit is supposed to have added, expressed the same way
# ``_resolved_entries`` expresses what the profile resolved to. The empty case
# applies the annotation's own example verbatim, so what it adds is that
# example's contents; the populated case adds the one new entry a facility
# would append beside what the preset already set.
_EMPTY_CASE_ADDITIONS: dict[str, set[tuple[str, ...]]] = {
    "web_panels": {("panel", "elog")},
    "env": {
        ("required", "EPICS_CA_ADDR_LIST"),
        ("pinned", "ARIEL_DB_PASSWORD"),
        ("defaults", "EPICS_CA_ADDR_LIST", "127.0.0.1"),
    },
}
_POPULATED_CASE_ADDITIONS: dict[str, set[tuple[str, ...]]] = {
    "web_panels": {("panel", "elog")},
    "env": {("required", "MY_TOKEN")},
}


def _sole_key_line(lines: list[str], key: str) -> int:
    """Index of the one top-level ``<key>:`` line, proved to be the only one.

    A YAML mapping with the key twice parses fine and silently keeps the last
    one, so "it still loads" says nothing about an edit. Counting the key is
    what makes the resolved-value comparison below mean anything.
    """
    matches = [
        index
        for index, line in enumerate(lines)
        if line[:1] not in ("", " ", "#") and line.split("#", 1)[0].rstrip().startswith(f"{key}:")
    ]
    assert len(matches) == 1, f"expected exactly one top-level `{key}:`, found {len(matches)}"
    return matches[0]


def _resolve(profile_dir: Path, lines: list[str]) -> BuildProfile:
    """Write *lines* as the profile of a bare directory and resolve them."""
    path = profile_dir / "profile.yml"
    path.write_text("\n".join(lines) + "\n")
    return resolve_build_profile(path, None)[0]


def _resolved_entries(field: str, profile: BuildProfile) -> set[tuple[str, ...]]:
    """The annotated key's resolved content, as comparable entries."""
    if field == "web_panels":
        return {("panel", panel) for panel in profile.web_panels}
    env = profile.env
    return (
        {("required", name) for name in env.required}
        | {("pinned", name) for name in env.pinned}
        | {("defaults", name, str(value)) for name, value in env.defaults.items()}
    )


def _edit_empty_key(field: str, lines: list[str], profile_dir: Path) -> list[str]:
    """Uncomment the annotation's example straight over the empty key it documents."""
    edited = list(lines)
    index = _sole_key_line(edited, _yaml_key(field))
    snippet = _snippet(_ANNOTATIONS[field]).splitlines()
    edited = edited[:index] + snippet + edited[index + 1 :]

    if field == "web_panels":
        config = _sole_key_line(edited, "config")
        edited = edited[: config + 1] + list(_ELOG_CONFIG_LINES) + edited[config + 1 :]
    else:
        # The example's `file:` is a profile-relative path, and the resolver
        # refuses one that is not there. Making it exist keeps the whole
        # example under test rather than trimming the line out of it.
        (profile_dir / "env").mkdir(exist_ok=True)
        (profile_dir / "env" / "facility.env").write_text("EPICS_CA_ADDR_LIST=127.0.0.1\n")
    return edited


def _edit_populated_key(field: str, lines: list[str]) -> list[str]:
    """Follow the annotation's populated-case instruction: add under what is there."""
    edited = list(lines)
    if field == "web_panels":
        index = _sole_key_line(edited, "web_panels")
        edited = edited[: index + 1] + ["  - elog"] + edited[index + 1 :]
        config = _sole_key_line(edited, "config")
        return edited[: config + 1] + list(_ELOG_CONFIG_LINES) + edited[config + 1 :]

    index = _sole_key_line(edited, "env")
    required = next(
        offset
        for offset in range(index + 1, len(edited))
        if edited[offset].split("#", 1)[0].rstrip() == "  required:"
    )
    return edited[: required + 1] + ["    - MY_TOKEN"] + edited[required + 1 :]


@pytest.mark.parametrize("preset", ["hello-world", "control-assistant"])
@pytest.mark.parametrize("field", sorted(_ANNOTATIONS))
def test_edited_annotation_keeps_everything_the_key_already_resolved(
    field: str, preset: str, tmp_path: Path
) -> None:
    """The example is only worth shipping if editing it in is additive.

    hello-world leaves both keys empty and control-assistant populates both, so
    the two presets are the two edits a facility actually makes: uncomment the
    example over an empty key, or add one entry beside existing ones. Either
    way the key must still resolve, must still be the only one of its name, and
    must still carry everything it carried before the edit.
    """
    key = _yaml_key(field)
    lines = _emit(preset).splitlines()
    _sole_key_line(lines, key)

    before = _resolved_entries(field, _resolve(tmp_path, lines))
    assert bool(before) == (preset == "control-assistant"), (
        f"{preset}: expected `{key}:` to be "
        f"{'populated' if preset == 'control-assistant' else 'empty'} before the edit"
    )

    if before:
        edited = _edit_populated_key(field, lines)
        expected_new = _POPULATED_CASE_ADDITIONS[field]
    else:
        edited = _edit_empty_key(field, lines, tmp_path)
        expected_new = _EMPTY_CASE_ADDITIONS[field]

    _sole_key_line(edited, key)
    after = _resolved_entries(field, _resolve(tmp_path, edited))

    assert before <= after, f"{preset}: the edit dropped {sorted(before - after)}"
    assert expected_new <= after, f"{preset}: the edit did not add {sorted(expected_new - after)}"


# ---------------------------------------------------------------------------
# (f) the fragments, applied as an override, survive init and build
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _init_with_fragment(runner: CliRunner, tmp_path: Path, fragment: dict[str, object]):
    """Materialize control-assistant with *fragment* layered on top of it."""
    override = tmp_path / "frag.yml"
    override.write_text(yaml.safe_dump(fragment))
    target = tmp_path / "facility"
    result = runner.invoke(
        init,
        [str(target), "--preset", "control-assistant", "--no-git", "-O", str(override)],
    )
    return target, result


def _build_from(runner: CliRunner, repo: Path):
    return runner.invoke(build, ["--repo", str(repo), "--skip-deps", "--skip-lifecycle"])


def test_fragment_web_panel_reaches_the_built_config(runner: CliRunner, tmp_path: Path) -> None:
    """The panel the annotation teaches has to arrive in build/config.yml.

    Resolving is not the claim the example makes — it says the facility gets a
    tab pointed at its own address, and only the rendered config can show that.
    """
    repo, result = _init_with_fragment(
        runner,
        tmp_path,
        {
            "web_panels": ["elog"],
            "config": {
                "web.panels.elog.url": _ELOG_URL,
                "web.panels.elog.label": "ELOG",
                "web.panels.elog.path": "/",
            },
        },
    )
    assert result.exit_code == 0, result.output

    built = _build_from(runner, repo)
    assert built.exit_code == 0, built.output

    config = yaml.safe_load((repo / "build" / "config.yml").read_text())
    assert config["web"]["panels"]["elog"]["url"] == _ELOG_URL


def test_fragment_web_panel_list_alone_is_refused(runner: CliRunner, tmp_path: Path) -> None:
    """The half the annotation warns about must fail loudly, not quietly.

    A list entry with no address behind it would otherwise produce a repo whose
    web workspace advertises a tab that goes nowhere.
    """
    _, result = _init_with_fragment(runner, tmp_path, {"web_panels": ["elog"]})

    assert result.exit_code != 0
    assert "Unknown web_panel 'elog'" in result.output
    assert "no 'web.panels.elog.url' config override" in result.output


def test_fragment_env_required_reaches_the_env_example(runner: CliRunner, tmp_path: Path) -> None:
    """A required variable is only documented once it is in .env.example."""
    repo, result = _init_with_fragment(runner, tmp_path, {"env": {"required": ["MY_TOKEN"]}})
    assert result.exit_code == 0, result.output

    assert "MY_TOKEN" in (repo / ".env.example").read_text()
    assert _build_from(runner, repo).exit_code == 0
