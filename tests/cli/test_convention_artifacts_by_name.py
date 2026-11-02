"""A deployment's own list-selected artifacts render only where a list names them.

A ``skills/<name>/``, ``agents/<name>.md``, ``rules/<name>.md`` or
``output-styles/<name>.md`` beside ``profile.yml`` is selected the way a library
artifact is: by name, in the render's resolved list (``skills:``, ``agents:``,
``rules:``, ``output_styles:``), spelled with its directory (``skills/house-diag``,
``output-styles/house-terse``) so it can never be mistaken for a library name.
``profile.yml``'s lists select for the deployment's own render and are the base
every tier adds to; an app's lists are its whole selection. A file no list names
reaches no render, and the build says so.

A ``hooks/<file>`` is listed only when it replaces an OSPREY hook. Any other
facility hook is wired by its ``claude_code.hooks`` declaration alone, and its
file is copied wherever that declaration is in force. Guards are a floor: every
``PreToolUse`` hook in force on the deployment's render, OSPREY's or the
facility's, reaches every tier and app render.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner, Result

from osprey.cli.build_cmd import build
from osprey.cli.build_profile_load import load_profile_document
from osprey.cli.profile_conventions import (
    convention_selection_errors,
    convention_selections,
    split_convention_selection,
)
from osprey_connectors.errors import BuildProfileError
from tests.fixtures.lifecycle_repo import build_exemplar_repo, preserved_environ

PROJECT = "als-exemplar"
RENDERS = (None, "readonly", "readwrite", "admin", "logbook", "knowledge", "channel-finder")

_SKILL = "---\nname: {name}\ndescription: A facility skill for the test.\n---\n\n# {name}\n"
_AGENT = (
    "---\nname: {name}\ndescription: A facility agent for the test.\n"
    "tools: Read\n---\n\nYou are {name}.\n"
)
_HOOK = "#!/usr/bin/env python3\nimport sys\n\nsys.exit(0)\n"
HOUSE_SAFETY = "# Safety\n\nHOUSE SAFETY: the deployment's own words.\n"


def _own_logins(repo: Path) -> list[Path]:
    """The repo's ``lists: own`` login files, which select their artifacts by name."""
    return [
        path
        for path in sorted((repo / "tiers").glob("*.yml"))
        if (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("lists") == "own"
    ]


def _insert_after(path: Path, anchor: str, line: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert anchor in text, f"{anchor!r} not in {path.name}"
    path.write_text(text.replace(anchor, f"{anchor}\n{line}", 1), encoding="utf-8")


def _insert_before_config(path: Path, block: str, config_lines: str = "") -> None:
    text = path.read_text(encoding="utf-8")
    assert "\nconfig:\n" in text, f"no config: block in {path.name}"
    path.write_text(
        text.replace("\nconfig:\n", f"\n{block}config:\n{config_lines}", 1), encoding="utf-8"
    )


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def write_house_artifacts(repo: Path) -> None:
    """One facility artifact of every list-selected kind, plus a second skill."""
    for name in ("house-report", "house-diag"):
        _write(repo / "skills" / name / "SKILL.md", _SKILL.format(name=name))
    _write(repo / "agents" / "house-search.md", _AGENT.format(name="house-search"))
    _write(repo / "rules" / "house-rule.md", "# House rule\n\nBe careful.\n")
    _write(repo / "output-styles" / "house-terse.md", "---\nname: house-terse\n---\n\nTerse.\n")
    _write(repo / "hooks" / "house_guard.py", _HOOK).chmod(0o755)


def _named_repo(dest: Path) -> Path:
    """The exemplar plus facility artifacts of every kind, each named by some lists only.

    - ``skills/house-report``: named by profile.yml, so the deployment and every
      tier carry it; no app names it.
    - ``output-styles/house-terse``: named by profile.yml.
    - ``skills/house-diag``: named by the admin tier and the logbook app.
    - ``agents/house-search``: named by the logbook app alone.
    - ``rules/house-rule``: named by the readonly tier alone.
    - ``hooks/house_guard.py``: wired by the admin tier alone, by declaration.
    - ``skills/house-unlisted``: named by nothing.
    """
    repo = build_exemplar_repo(dest)
    write_house_artifacts(repo)
    _write(repo / "skills" / "house-unlisted" / "SKILL.md", _SKILL.format(name="house-unlisted"))

    _insert_after(repo / "profile.yml", "skills:", "  - skills/house-report")
    _insert_after(repo / "profile.yml", "output_styles:", "  - output-styles/house-terse")
    _insert_before_config(repo / "tiers" / "readonly.yml", "rules:\n  - rules/house-rule\n")
    admin = repo / "tiers" / "admin.yml"
    _insert_after(admin, "skills:", "  - skills/house-diag")
    _insert_before_config(
        admin, "", "  claude_code.hooks.PreToolUse:\n    - hooks/house_guard.py\n"
    )
    logbook = repo / "tiers" / "logbook.yml"
    _insert_after(logbook, "skills:", "  - skills/house-diag")
    _insert_after(logbook, "agents:", "  - agents/house-search")
    # The deployment's own safety rule, named by its path in the logbook app
    # alone; every other render selects the library rule by its bare name.
    _write(repo / "rules" / "safety.md", HOUSE_SAFETY)
    _swap_entry(logbook, "safety", "rules/safety")
    return repo


def _swap_entry(path: Path, old: str, new: str) -> None:
    """Rename one list entry, keeping its line comment."""
    text = path.read_text(encoding="utf-8")
    assert text.count(f"\n  - {old} ") == 1, f"{old!r} not listed once in {path.name}"
    path.write_text(text.replace(f"\n  - {old} ", f"\n  - {new} ", 1), encoding="utf-8")


def run_build(repo: Path, cwd: Path) -> Result:
    """``osprey build --skip-deps --skip-lifecycle`` on *repo*, from *cwd*."""
    with pytest.MonkeyPatch.context() as patch, preserved_environ():
        patch.chdir(cwd)
        return CliRunner().invoke(build, ["--repo", str(repo), "--skip-deps", "--skip-lifecycle"])


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Result]:
    outside = tmp_path_factory.mktemp("by-name")
    repo = _named_repo(outside / PROJECT)
    result = run_build(repo, outside)
    assert result.exit_code == 0, result.output
    return repo, result


def render_dir(repo: Path, name: str | None) -> Path:
    """The deployment's render (``None``) or the named tier or app render."""
    return repo / "build" if name is None else repo / "build" / f"{PROJECT}-{name}"


def house_artifacts(render: Path) -> set[str]:
    """Every facility (``house``) artifact in a render, spelled the way a list names it."""
    claude = render / ".claude"
    # A render that selects nothing of a kind has no directory for it.
    found = {f"skills/{p.name}" for p in (claude / "skills").glob("house-*")}
    for directory in ("agents", "rules", "output-styles"):
        found |= {f"{directory}/{p.stem}" for p in (claude / directory).glob("house-*.md")}
    found |= {f"hooks/{p.name}" for p in (claude / "hooks").glob("house_*")}
    return found


def wired_house_hooks(render: Path) -> bool:
    """Whether the render's ``settings.json`` wires the facility hook."""
    settings = json.loads((render / ".claude" / "settings.json").read_text(encoding="utf-8"))
    return "house_guard.py" in json.dumps(settings.get("hooks", {}))


_REPORT, _TERSE = "skills/house-report", "output-styles/house-terse"


@pytest.mark.parametrize(
    ("render", "expected"),
    [
        (None, {_REPORT, _TERSE}),
        ("readonly", {_REPORT, _TERSE, "rules/house-rule"}),
        ("readwrite", {_REPORT, _TERSE}),
        ("admin", {_REPORT, _TERSE, "skills/house-diag", "hooks/house_guard.py"}),
        ("logbook", {"skills/house-diag", "agents/house-search"}),
        ("knowledge", set()),
        ("channel-finder", set()),
    ],
)
def test_each_render_carries_exactly_the_facility_artifacts_its_lists_name(
    built: tuple[Path, Result], render: str | None, expected: set[str]
) -> None:
    assert house_artifacts(render_dir(built[0], render)) == expected


@pytest.mark.parametrize("render", RENDERS)
def test_a_facility_hook_is_copied_and_wired_only_where_a_tier_declares_it(
    built: tuple[Path, Result], render: str | None
) -> None:
    root = render_dir(built[0], render)
    assert wired_house_hooks(root) is (render == "admin")
    assert (root / ".claude" / "hooks" / "house_guard.py").is_file() is (render == "admin")


@pytest.mark.parametrize("render", RENDERS)
def test_a_shadow_replaces_the_library_artifact_only_where_its_path_is_named(
    built: tuple[Path, Result], render: str | None
) -> None:
    rule = (render_dir(built[0], render) / ".claude" / "rules" / "safety.md").read_text("utf-8")
    assert ("HOUSE SAFETY" in rule) is (render == "logbook")


def test_only_the_named_artifacts_register_as_user_owned(built: tuple[Path, Result]) -> None:
    config = yaml.safe_load((render_dir(built[0], "logbook") / "config.yml").read_text("utf-8"))
    owned = set(config["scaffold"]["user_owned"])
    assert {"skills/house-diag", "agents/house-search"} <= owned
    assert not {o for o in owned if "house-report" in o or "house-unlisted" in o}


def test_an_artifact_no_list_names_is_reported_once(built: tuple[Path, Result]) -> None:
    output = built[1].output
    assert output.count("skills/house-unlisted") == 1
    assert "house-report" not in output


# ── the spelling ─────────────────────────────────────────────────────────────


def test_one_list_splits_into_library_names_and_own_selections() -> None:
    assert split_convention_selection("skills", ["diagnose", "skills/beam-check"]) == (
        ["diagnose"],
        ["skills/beam-check"],
    )
    # The directory spelling, not the field's: output-styles/, hyphenated.
    assert split_convention_selection("output_styles", ["output-styles/terse", "terse"]) == (
        ["terse"],
        ["output-styles/terse"],
    )
    assert split_convention_selection("web_panels", ["a/b"]) == (["a/b"], [])


def test_a_bare_name_selects_the_library_artifact_only() -> None:
    assert convention_selections({"rules": ["facility"], "hooks": ["writes-check"]}) == frozenset()


def test_a_replacing_path_selects_its_library_artifact_for_library_readers() -> None:
    from osprey.cli.profile_conventions import library_selection

    assert library_selection("hooks", ["approval", "hooks/osprey_writes_check.py"]) == [
        "approval",
        "writes-check",
    ]
    assert library_selection("rules", ["rules/safety", "rules/house-rule"]) == ["safety"]
    assert library_selection("skills", ["skills/house-diag"]) == []


def test_a_list_naming_a_library_artifact_and_its_shadow_is_refused(tmp_path: Path) -> None:
    _write(tmp_path / "rules" / "facility.md", "x")
    _write(tmp_path / "hooks" / "osprey_writes_check.py", "x")
    errors = convention_selection_errors(
        tmp_path,
        {
            "rules": ["facility", "rules/facility"],
            "hooks": ["writes-check", "hooks/osprey_writes_check.py"],
        },
    )
    assert sorted(errors) == sorted(
        [
            "rules: 'facility' and 'rules/facility' both render .claude/rules/facility.md. "
            "Keep one: 'facility' for OSPREY's, 'rules/facility' for the deployment's own.",
            "hooks: 'writes-check' and 'hooks/osprey_writes_check.py' both render "
            ".claude/hooks/osprey_writes_check.py. Keep one: 'writes-check' for OSPREY's, "
            "'hooks/osprey_writes_check.py' for the deployment's own.",
        ]
    )


def test_a_prefixed_entry_must_name_an_artifact_beside_the_profile(tmp_path: Path) -> None:
    _write(tmp_path / "skills" / "here" / "SKILL.md", "x")
    _write(tmp_path / "hooks" / "osprey_limits.py", "x")
    errors = convention_selection_errors(
        tmp_path,
        {
            "skills": ["skills/here", "skills/gone", "agents/elsewhere"],
            "hooks": ["hooks/osprey_limits.py", "hooks/guard"],
            "output_styles": ["output-styles/missing"],
        },
    )
    assert sorted(errors) == sorted(
        [
            "hooks: 'hooks/guard' — there is no hooks/guard beside the profile.",
            "skills: 'skills/gone' — there is no skills/gone/ beside the profile.",
            "skills: 'agents/elsewhere' names a agents/ artifact. List it under agents: instead.",
            "output_styles: 'output-styles/missing' — there is no "
            "output-styles/missing.md beside the profile.",
        ]
    )


def test_a_listed_facility_hook_that_replaces_no_osprey_hook_is_refused(tmp_path: Path) -> None:
    _write(tmp_path / "hooks" / "house.py", _HOOK)
    errors = convention_selection_errors(tmp_path, {"hooks": ["hooks/house.py"]})
    assert errors == [
        "hooks: hooks/house.py replaces no OSPREY hook; wire it with a claude_code.hooks "
        "declaration instead (it is copied wherever that declaration applies)"
    ]


def test_loading_a_profile_refuses_a_prefixed_entry_naming_nothing(tmp_path: Path) -> None:
    repo = build_exemplar_repo(tmp_path / PROJECT)
    _insert_after(repo / "profile.yml", "skills:", "  - skills/nowhere")
    with pytest.raises(BuildProfileError, match="there is no skills/nowhere/ beside the profile"):
        load_profile_document(repo / "profile.yml")


def test_a_qualified_exclusion_drops_the_selection_too(tmp_path: Path) -> None:
    repo = build_exemplar_repo(tmp_path / PROJECT)
    _write(repo / "skills" / "house-report" / "SKILL.md", _SKILL.format(name="house-report"))
    _insert_after(repo / "profile.yml", "skills:", "  - skills/house-report")
    child = _write(
        tmp_path / "child" / "profile.yml",
        f"extends: {repo / 'profile.yml'}\nexclude:\n  skills: [skills/house-report]\n",
    )
    from osprey.cli.build_profile_merge import resolve_profile_document

    raw = yaml.safe_load(child.read_text(encoding="utf-8"))
    skills = resolve_profile_document(raw, child, warn=False).raw["skills"]
    assert "skills/house-report" not in skills
    assert "diagnose" in skills


# ── the migration ────────────────────────────────────────────────────────────

#: What an existing deployment's renders all carried before the by-name rule:
#: every facility artifact, in every render.
_EVERYTHING = {
    "skills/house-report",
    "skills/house-diag",
    "agents/house-search",
    "rules/house-rule",
    "output-styles/house-terse",
    "hooks/house_guard.py",
}


def _revalidate(repo: Path) -> Result:
    from osprey.cli.init_cmd import init

    with preserved_environ():
        return CliRunner().invoke(init, [str(repo), "--revalidate"], catch_exceptions=False)


@pytest.fixture(scope="module")
def migrated(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, str]]:
    """A deployment that ships one artifact of every kind and names none, revalidated once.

    The hook is wired in profile.yml's ``config:``, the way a deployment wired
    one before: every render inherits the declaration, and before the by-name
    rule every render carried the file, so every render ran it.
    """
    outside = tmp_path_factory.mktemp("migrated")
    repo = build_exemplar_repo(outside / PROJECT)
    write_house_artifacts(repo)
    # A shadow of the library `safety` rule, which profile.yml and every app
    # select by its bare name: before, it replaced the library rule everywhere.
    _write(repo / "rules" / "safety.md", HOUSE_SAFETY)
    _insert_before_config(
        repo / "profile.yml", "", "  claude_code.hooks.PreToolUse:\n    - hooks/house_guard.py\n"
    )
    first = _revalidate(repo)
    assert first.exit_code == 0, first.output
    after_first = {
        rel: (repo / rel).read_text(encoding="utf-8")
        for rel in ("profile.yml", *(f"tiers/{p.name}" for p in _own_logins(repo)))
    }
    result = run_build(repo, outside)
    assert result.exit_code == 0, result.output
    return repo, after_first


@pytest.mark.parametrize("render", RENDERS)
def test_revalidate_keeps_every_render_carrying_what_it_carried(
    migrated: tuple[Path, dict[str, str]], render: str | None
) -> None:
    root = render_dir(migrated[0], render)
    assert house_artifacts(root) == _EVERYTHING
    assert wired_house_hooks(root)
    assert "HOUSE SAFETY" in (root / ".claude" / "rules" / "safety.md").read_text("utf-8")


def test_revalidate_names_them_in_profile_and_own_list_logins_but_not_tiers(
    migrated: tuple[Path, dict[str, str]],
) -> None:
    repo, _ = migrated
    profile = yaml.safe_load((repo / "profile.yml").read_text(encoding="utf-8"))
    assert {"skills/house-report", "skills/house-diag"} <= set(profile["skills"])
    assert "output-styles/house-terse" in profile["output_styles"]
    for tier in (repo / "tiers").glob("*.yml"):
        if tier in _own_logins(repo):
            continue
        assert "house" not in tier.read_text(encoding="utf-8"), tier.name


def test_revalidate_lists_no_facility_hook_its_declaration_wires(
    migrated: tuple[Path, dict[str, str]],
) -> None:
    repo, _ = migrated
    for rel in ("profile.yml", *(f"tiers/{p.name}" for p in _own_logins(repo))):
        hooks = yaml.safe_load((repo / rel).read_text(encoding="utf-8")).get("hooks") or []
        assert not [h for h in hooks if "house" in h], rel


def test_revalidate_names_a_shipped_shadow_by_its_path_instead_of_the_bare_name(
    migrated: tuple[Path, dict[str, str]],
) -> None:
    repo, _ = migrated
    for rel in ("profile.yml", *(f"tiers/{p.name}" for p in _own_logins(repo))):
        rules = yaml.safe_load((repo / rel).read_text(encoding="utf-8"))["rules"]
        assert "rules/safety" in rules and "safety" not in rules, rel


def test_a_second_revalidate_changes_nothing(migrated: tuple[Path, dict[str, str]]) -> None:
    repo, after_first = migrated
    assert _revalidate(repo).exit_code == 0
    for rel, text in after_first.items():
        assert (repo / rel).read_text(encoding="utf-8") == text, rel


# ── a fresh repo ─────────────────────────────────────────────────────────────


def test_init_names_the_facility_description_it_seeds_by_its_path(tmp_path: Path) -> None:
    from osprey.cli.init_cmd import init
    from osprey.cli.validate_cmd import validate

    repo = tmp_path / "fresh"
    with preserved_environ():
        made = CliRunner().invoke(init, [str(repo), "--preset", "control-assistant", "--no-git"])
    assert made.exit_code == 0, made.output
    assert (repo / "rules" / "facility.md").is_file()
    for rel in ("profile.yml", *(f"tiers/{p.name}" for p in _own_logins(repo))):
        rules = yaml.safe_load((repo / rel).read_text(encoding="utf-8")).get("rules") or []
        assert "facility" not in rules, rel
    assert "rules/facility" in yaml.safe_load((repo / "profile.yml").read_text("utf-8"))["rules"]

    # The path covers the preset's bare entry: a fresh repo has no drift.
    with preserved_environ():
        validated = CliRunner().invoke(validate, ["--repo", str(repo)])
    assert validated.exit_code == 0, validated.output

    built = run_build(repo, tmp_path)
    assert built.exit_code == 0, built.output
    assert (repo / "build" / ".claude" / "rules" / "facility.md").read_text("utf-8") == (
        repo / "rules" / "facility.md"
    ).read_text("utf-8")


def test_validate_does_not_count_the_deployments_own_entries_as_drift(
    built: tuple[Path, Result],
) -> None:
    from osprey.cli.validate_cmd import validate

    with preserved_environ():
        validated = CliRunner().invoke(validate, ["--repo", str(built[0])])
    assert "house" not in validated.output, validated.output


# ── an app and the host's hook wiring ────────────────────────────────────────


_NOTE_HOOK = "house_note.py"


def _wired(render: Path, filename: str) -> bool:
    return filename in _hook_wiring(render)


@pytest.fixture(scope="module")
def host_hook(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Result]:
    """A host that wires a facility guard and a facility session hook; no list names either."""
    outside = tmp_path_factory.mktemp("host-hook")
    repo = build_exemplar_repo(outside / PROJECT)
    _write(repo / "hooks" / "house_guard.py", _HOOK).chmod(0o755)
    _write(repo / "hooks" / _NOTE_HOOK, _HOOK).chmod(0o755)
    _insert_before_config(
        repo / "profile.yml",
        "",
        "  claude_code.hooks.PreToolUse:\n    - hooks/house_guard.py\n"
        f"  claude_code.hooks.SessionStart:\n    - hooks/{_NOTE_HOOK}\n",
    )
    result = run_build(repo, outside)
    return repo, result


def test_a_host_wiring_hooks_no_list_names_builds(host_hook: tuple[Path, Result]) -> None:
    assert host_hook[1].exit_code == 0, host_hook[1].output
    assert "house_guard.py reaches no render" not in host_hook[1].output


@pytest.mark.parametrize("render", RENDERS)
def test_a_host_guard_reaches_every_tier_and_app(
    host_hook: tuple[Path, Result], render: str | None
) -> None:
    root = render_dir(host_hook[0], render)
    assert wired_house_hooks(root)
    assert (root / ".claude" / "hooks" / "house_guard.py").is_file()


@pytest.mark.parametrize("render", RENDERS)
def test_a_host_session_hook_reaches_tiers_but_no_app_that_does_not_declare_it(
    host_hook: tuple[Path, Result], render: str | None
) -> None:
    root = render_dir(host_hook[0], render)
    in_tier = render in (None, "readonly", "readwrite", "admin")
    assert _wired(root, _NOTE_HOOK) is in_tier
    assert (root / ".claude" / "hooks" / _NOTE_HOOK).is_file() is in_tier


@pytest.mark.parametrize("render", ("logbook", "knowledge", "channel-finder"))
def test_a_host_library_guard_reaches_an_app_whose_list_omits_it(
    host_hook: tuple[Path, Result], render: str
) -> None:
    # It gates a controls tool, so it is wired on the host and ships inert in an
    # app that has no controls server: present, ready the moment one is added.
    root = render_dir(host_hook[0], render)
    listed = yaml.safe_load((host_hook[0] / "tiers" / f"{render}.yml").read_text("utf-8"))
    assert "limits" not in listed["hooks"]
    assert _wired(render_dir(host_hook[0], None), "osprey_limits.py")
    assert (root / ".claude" / "hooks" / "osprey_limits.py").is_file()


def test_revalidate_keeps_a_host_session_hook_in_each_app_by_its_own_declaration(
    tmp_path: Path,
) -> None:
    """Before by-name selection an app inherited every host declaration; revalidate keeps it."""
    repo = build_exemplar_repo(tmp_path / PROJECT)
    _write(repo / "hooks" / _NOTE_HOOK, _HOOK).chmod(0o755)
    _insert_before_config(
        repo / "profile.yml", "", f"  claude_code.hooks.SessionStart:\n    - hooks/{_NOTE_HOOK}\n"
    )
    assert _revalidate(repo).exit_code == 0
    for app in _own_logins(repo):
        config = yaml.safe_load(app.read_text(encoding="utf-8"))["config"]
        assert config["claude_code.hooks.SessionStart"] == [f"hooks/{_NOTE_HOOK}"], app.name
        assert "hooks/" not in "".join(yaml.safe_load(app.read_text("utf-8"))["hooks"])
    once = {p.name: p.read_text(encoding="utf-8") for p in _own_logins(repo)}
    assert _revalidate(repo).exit_code == 0
    assert {p.name: p.read_text(encoding="utf-8") for p in _own_logins(repo)} == once

    built = run_build(repo, tmp_path)
    assert built.exit_code == 0, built.output
    for render in RENDERS:
        root = render_dir(repo, render)
        assert _wired(root, _NOTE_HOOK), render
        assert (root / ".claude" / "hooks" / _NOTE_HOOK).is_file(), render


def test_a_claimed_library_artifact_is_named_by_its_path_where_it_was_selected(
    tmp_path: Path,
) -> None:
    from osprey.cli.scaffold_cmd import claim_into_profile

    repo = build_exemplar_repo(tmp_path / PROJECT)
    assert run_build(repo, tmp_path).exit_code == 0
    claimed = claim_into_profile(repo / "build", "rules/safety")
    claimed.profile_path.write_text(HOUSE_SAFETY, encoding="utf-8")

    for rel in ("profile.yml", *(f"tiers/{p.name}" for p in _own_logins(repo))):
        rules = yaml.safe_load((repo / rel).read_text(encoding="utf-8"))["rules"]
        assert "rules/safety" in rules and "safety" not in rules, rel
    rebuilt = run_build(repo, tmp_path)
    assert rebuilt.exit_code == 0, rebuilt.output
    for render in RENDERS:
        rule = render_dir(repo, render) / ".claude" / "rules" / "safety.md"
        assert rule.read_text(encoding="utf-8") == HOUSE_SAFETY, render


def test_revalidate_appends_to_an_app_file_and_changes_no_other_byte(tmp_path: Path) -> None:
    import difflib

    repo = build_exemplar_repo(tmp_path / PROJECT)
    write_house_artifacts(repo)
    app = repo / "tiers" / "logbook.yml"
    text = app.read_text(encoding="utf-8")
    # Spellings a re-dump does not keep: quoting, an explicit null, an empty
    # value, a flow list with a comment, and comments around the lists.
    text = text.replace(
        "default_panel: ariel", "default_panel: 'ariel'   # quoted, single\n# a lone comment"
    )
    text = text.replace(
        "claude_md_template: CLAUDE.logbook.md.j2", 'claude_md_template: "CLAUDE.logbook.md.j2"'
    )
    text = text.replace(
        "output_styles:\n  - control-operator  # Terse, actionable output style",
        "output_styles: [control-operator]  # a flow list, with a comment",
    )
    text += "image:\nseed_base: null\n"
    app.write_text(text, encoding="utf-8")

    assert _revalidate(repo).exit_code == 0
    after = app.read_text(encoding="utf-8")

    changes = [
        line
        for line in difflib.ndiff(text.splitlines(), after.splitlines())
        if line[:2] in ("- ", "+ ")
    ]
    removed = [line[2:] for line in changes if line.startswith("- ")]
    added = [line[2:].strip() for line in changes if line.startswith("+ ")]
    # The one-line flow list grows in place; every other change is an added entry.
    assert removed == ["output_styles: [control-operator]  # a flow list, with a comment"]
    assert (
        "output_styles: [control-operator, output-styles/house-terse]  # a flow list, with a comment"
        in added
    )
    # A facility hook is wired by declaration, never listed.
    assert sorted(a for a in added if a.startswith("- ")) == sorted(
        f"- {name}" for name in _EVERYTHING - {"output-styles/house-terse", "hooks/house_guard.py"}
    )


# ── a shadow of a library artifact stands in for it everywhere ──────────────

_HOUSE_MARK = "# HOUSE COPY of the kill switch\n"
_WRITES_CHECK_FILE = "osprey_writes_check.py"


def _ship_writes_check(repo: Path) -> None:
    """The library kill switch, copied into the deployment's hooks/ and marked."""
    import osprey

    library = next(Path(osprey.__file__).parent.rglob(_WRITES_CHECK_FILE))
    _write(repo / "hooks" / _WRITES_CHECK_FILE, library.read_text(encoding="utf-8") + _HOUSE_MARK)


def _hook_wiring(render: Path) -> str:
    settings = json.loads((render / ".claude" / "settings.json").read_text(encoding="utf-8"))
    return json.dumps(settings.get("hooks", {}), sort_keys=True)


@pytest.fixture(scope="module")
def library_wiring(tmp_path_factory: pytest.TempPathFactory) -> dict[str | None, str]:
    """Each render's hook wiring with OSPREY's own kill switch, no shadow."""
    outside = tmp_path_factory.mktemp("library-wiring")
    repo = build_exemplar_repo(outside / PROJECT)
    result = run_build(repo, outside)
    assert result.exit_code == 0, result.output
    return {render: _hook_wiring(render_dir(repo, render)) for render in RENDERS}


def _assert_shadow_wired(repo: Path, wiring: dict[str | None, str], renders) -> None:
    for render in renders:
        root = render_dir(repo, render)
        shipped = root / ".claude" / "hooks" / _WRITES_CHECK_FILE
        assert shipped.read_text(encoding="utf-8").endswith(_HOUSE_MARK), render
        assert _hook_wiring(root) == wiring[render], render


def test_a_migrated_shadow_of_the_kill_switch_is_wired_where_the_library_one_was(
    tmp_path: Path, library_wiring: dict[str | None, str]
) -> None:
    repo = build_exemplar_repo(tmp_path / PROJECT)
    _ship_writes_check(repo)

    assert _revalidate(repo).exit_code == 0
    hooks = yaml.safe_load((repo / "profile.yml").read_text(encoding="utf-8"))["hooks"]
    assert f"hooks/{_WRITES_CHECK_FILE}" in hooks and "writes-check" not in hooks
    built = run_build(repo, tmp_path)
    assert built.exit_code == 0, built.output

    _assert_shadow_wired(repo, library_wiring, RENDERS)


@pytest.mark.parametrize("armed", [True, False])
def test_a_fresh_profile_naming_the_kill_switch_by_path_wires_it(
    tmp_path: Path, library_wiring: dict[str | None, str], armed: bool
) -> None:
    repo = build_exemplar_repo(tmp_path / PROJECT)
    _ship_writes_check(repo)
    _swap_entry(repo / "profile.yml", "writes-check", f"hooks/{_WRITES_CHECK_FILE}")
    if not armed:
        profile = repo / "profile.yml"
        text = profile.read_text(encoding="utf-8")
        assert text.count("  control_system.writes: asked\n") == 1
        profile.write_text(
            text.replace(
                "  control_system.writes: asked\n",
                "  control_system.writes: never\n",
            ),
            encoding="utf-8",
        )

    built = run_build(repo, tmp_path)
    assert built.exit_code == 0, built.output

    root = render_dir(repo, None)
    assert (
        (root / ".claude" / "hooks" / _WRITES_CHECK_FILE).read_text("utf-8").endswith(_HOUSE_MARK)
    )
    assert _WRITES_CHECK_FILE in _hook_wiring(root)
    if armed:
        _assert_shadow_wired(repo, library_wiring, [None, "readonly"])


def test_the_guard_floor_carries_the_hosts_spelling_and_pins_the_kill_switch() -> None:
    from osprey.cli.build_profile_merge import _guard_floor

    host = ["approval", "hooks/osprey_limits.py", "memory-guard", "error-guidance"]
    assert _guard_floor(host, ["approval", "turn-state"], writes_check=True) == [
        "approval",
        "turn-state",
        "hooks/osprey_limits.py",
        "memory-guard",
        "writes-check",
    ]
    # A narrow login no `shared:` names gets the floor, but no kill-switch pin.
    assert _guard_floor(host, ["approval", "turn-state"], writes_check=False) == [
        "approval",
        "turn-state",
        "hooks/osprey_limits.py",
        "memory-guard",
    ]
    # Selected already, in either spelling: nothing is added twice.
    assert _guard_floor(
        host, ["limits", f"hooks/{_WRITES_CHECK_FILE}", "memory-guard"], writes_check=True
    ) == [
        "limits",
        f"hooks/{_WRITES_CHECK_FILE}",
        "memory-guard",
        "approval",
    ]
