"""An application ships no hook code: every way it could bring one is refused by the check."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from osprey.apps.errors import AppManifestError
from osprey.apps.loader import check_deployment
from osprey.apps.manifest import parse_manifest
from tests.apps import _installed
from tests.fixtures.lifecycle_repo import (  # noqa: F401
    lifecycle_repo,
    lifecycle_repo_factory,
)

_NAMES_WHAT_TO_DECLARE = ("tools.approval", "tools.deny_in_jobs", "tools.errors")


def _data(name: str = "knob", **extra: object) -> dict[str, object]:
    return {
        "name": name,
        "containers": {name: {"runtime": "osprey", "health": "/health"}},
        **extra,
    }


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _app(repo: Path, name: str = "knob", **extra: object) -> Path:
    """List one application in the deployment, its manifest valid unless ``extra`` says not."""
    app_dir = repo / "apps" / name
    _write(app_dir / "app.yml", yaml.safe_dump(_data(name, **extra), sort_keys=False))
    profile = yaml.safe_load((repo / "profile.yml").read_text(encoding="utf-8"))
    profile["apps"] = {name: {}}
    (repo / "profile.yml").write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")
    return app_dir


def _no_hooks(exc: pytest.ExceptionInfo[AppManifestError]) -> list[tuple[str, str]]:
    return [(v.field, v.message) for v in exc.value.violations if v.rule == "no_hooks"]


def test_a_hooks_key_in_app_yml_is_refused_with_what_to_declare_instead() -> None:
    with pytest.raises(AppManifestError) as exc:
        parse_manifest(_data(hooks={"PreToolUse": []}), app="knob")

    [(field, message)] = _no_hooks(exc)
    assert field == "hooks"
    assert "apps ship no hook code" in message
    assert all(key in message for key in _NAMES_WHAT_TO_DECLARE)


@pytest.mark.parametrize("relative", ["hooks/x.py", "panel/.claude/settings.json"])
def test_a_hook_carrying_directory_in_the_app_is_refused(
    lifecycle_repo: Path,  # noqa: F811
    relative: str,
) -> None:
    app_dir = _app(lifecycle_repo)
    _write(app_dir / relative, "{}\n")

    with pytest.raises(AppManifestError) as exc:
        check_deployment(lifecycle_repo)

    [(field, message)] = _no_hooks(exc)
    assert field == relative.rsplit("/", 1)[0]
    assert "apps ship no hook code" in message


def test_hooks_in_a_listed_skills_frontmatter_are_refused(
    lifecycle_repo: Path,  # noqa: F811
) -> None:
    app_dir = _app(lifecycle_repo, skills=["knob"])
    _write(
        app_dir / "skills" / "knob" / "SKILL.md",
        "---\nname: knob\ndescription: d\nhooks:\n  PreToolUse: []\n---\nbody\n",
    )

    with pytest.raises(AppManifestError) as exc:
        check_deployment(lifecycle_repo)

    [(field, _message)] = _no_hooks(exc)
    assert field == "skills.knob.hooks"


def test_an_unlisted_skill_directory_is_not_checked(lifecycle_repo: Path) -> None:  # noqa: F811
    """Only declared skills are installed (``_copy_app_skills``), so only they are read."""
    app_dir = _app(lifecycle_repo)
    _write(
        app_dir / "skills" / "stray" / "SKILL.md",
        "---\nname: stray\ndescription: d\nhooks:\n  PreToolUse: []\n---\n",
    )

    check_deployment(lifecycle_repo)


def _installed_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A copy of the fixture installed application, declared by an entry point; its directory."""
    site = tmp_path / "site"
    package = site / "hooked_installed_app"
    shutil.copytree(_installed.DIRECTORY, package)
    manifest = package / "app.yml"
    text = manifest.read_text(encoding="utf-8")
    manifest.write_text(text.replace(_installed.MODULE, package.name), encoding="utf-8")
    monkeypatch.syspath_prepend(str(site))
    _installed.install(monkeypatch, _installed.entry_point(module="hooked_installed_app"))
    return package


def _list_installed(repo: Path) -> None:
    profile = yaml.safe_load((repo / "profile.yml").read_text(encoding="utf-8"))
    profile["apps"] = {_installed.NAME: {}}
    profile["dependencies"] = [*(profile.get("dependencies") or []), _installed.DIST]
    (repo / "profile.yml").write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")


def test_a_hooks_directory_in_an_installed_application_is_refused(
    lifecycle_repo: Path,  # noqa: F811
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = _installed_copy(tmp_path, monkeypatch)
    _write(package / "hooks" / "x.py", "\n")
    _list_installed(lifecycle_repo)

    with pytest.raises(AppManifestError) as exc:
        check_deployment(lifecycle_repo)

    [(field, _message)] = _no_hooks(exc)
    assert field == "hooks"


def test_hooks_in_an_installed_applications_listed_skill_are_refused(
    lifecycle_repo: Path,  # noqa: F811
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = _installed_copy(tmp_path, monkeypatch)
    _write(
        package / "skills" / _installed.NAME / "SKILL.md",
        f"---\nname: {_installed.NAME}\ndescription: d\nhooks:\n  PreToolUse: []\n---\nbody\n",
    )
    _list_installed(lifecycle_repo)

    with pytest.raises(AppManifestError) as exc:
        check_deployment(lifecycle_repo)

    [(field, _message)] = _no_hooks(exc)
    assert field == f"skills.{_installed.NAME}.hooks"


_HOOKED_SKILL = "---\nname: knob\ndescription: d\nhooks:\n  PreToolUse: []\n---\nbody\n"


def _skill_frontmatter(exc: pytest.ExceptionInfo[AppManifestError]) -> list[str]:
    return [v.field for v in exc.value.violations if v.rule == "skill_frontmatter"]


@pytest.mark.parametrize(
    "raw",
    [
        _HOOKED_SKILL.replace("\n", "\r\n").encode("utf-8"),
        b"\xef\xbb\xbf" + _HOOKED_SKILL.encode("utf-8"),
    ],
    ids=["crlf", "byte-order-mark"],
)
def test_hooks_in_a_listed_skill_are_refused_whatever_the_line_endings_or_bom(
    lifecycle_repo: Path,  # noqa: F811
    raw: bytes,
) -> None:
    app_dir = _app(lifecycle_repo, skills=["knob"])
    skill_file = app_dir / "skills" / "knob" / "SKILL.md"
    skill_file.parent.mkdir(parents=True, exist_ok=True)
    skill_file.write_bytes(raw)

    with pytest.raises(AppManifestError) as exc:
        check_deployment(lifecycle_repo)

    [(field, _message)] = _no_hooks(exc)
    assert field == "skills.knob.hooks"


@pytest.mark.parametrize(
    "frontmatter",
    ["name: knob\nhooks: [unclosed\n", "- name\n- hooks\n"],
    ids=["unparseable-yaml", "a-list"],
)
def test_a_listed_skill_whose_frontmatter_is_not_a_mapping_is_refused(
    lifecycle_repo: Path,  # noqa: F811
    frontmatter: str,
) -> None:
    app_dir = _app(lifecycle_repo, skills=["knob"])
    _write(app_dir / "skills" / "knob" / "SKILL.md", f"---\n{frontmatter}---\nbody\n")

    with pytest.raises(AppManifestError) as exc:
        check_deployment(lifecycle_repo)

    assert _skill_frontmatter(exc) == ["skills.knob.frontmatter"]
    [message] = [v.message for v in exc.value.violations if v.rule == "skill_frontmatter"]
    assert "does not parse as a YAML mapping" in message


def test_a_listed_skill_with_no_frontmatter_passes_the_hooks_check(
    lifecycle_repo: Path,  # noqa: F811
) -> None:
    app_dir = _app(lifecycle_repo, skills=["knob"])
    _write(app_dir / "skills" / "knob" / "SKILL.md", "# knob\n\nbody\n")

    check_deployment(lifecycle_repo)


# --- the block closes where the session closes it: on a line that is exactly --- --


_HIDDEN_HOOKS = "---\nname: knob\ndescription: d\n{line}\nhooks:\n  PreToolUse: []\n---\nbody\n"


@pytest.mark.parametrize(
    "raw",
    [
        _HIDDEN_HOOKS.format(line="---x: 1").encode("utf-8"),
        _HIDDEN_HOOKS.format(line="--- x: 1").encode("utf-8"),
        _HIDDEN_HOOKS.format(line="----").encode("utf-8"),
        _HIDDEN_HOOKS.format(line="---x: 1").replace("\n", "\r\n").encode("utf-8"),
        b"\xef\xbb\xbf" + _HIDDEN_HOOKS.format(line="---x: 1").encode("utf-8"),
        b"---\nname: knob\ndescription: d\nhooks:\n  PreToolUse: []\n",
        "---\u00a0\nname: knob\ndescription: d\nhooks:\n  PreToolUse: []\n---\nbody\n".encode(),
        _HIDDEN_HOOKS.format(line=" ---").encode("utf-8"),
    ],
    ids=[
        "dashes-then-text",
        "dashes-space-text",
        "four-dashes",
        "crlf",
        "byte-order-mark",
        "unclosed",
        "no-break-space-opening",
        "closing-candidate",
    ],
)
def test_a_line_starting_with_dashes_never_hides_a_skills_hooks(
    lifecycle_repo: Path,  # noqa: F811
    raw: bytes,
) -> None:
    app_dir = _app(lifecycle_repo, skills=["knob"])
    skill_file = app_dir / "skills" / "knob" / "SKILL.md"
    skill_file.parent.mkdir(parents=True, exist_ok=True)
    skill_file.write_bytes(raw)

    with pytest.raises(AppManifestError) as exc:
        check_deployment(lifecycle_repo)

    assert _skill_frontmatter(exc) == ["skills.knob.frontmatter"]


@pytest.mark.parametrize(
    "text",
    [
        "---\nname: knob\ndescription: d\n---\nbody\n",
        "--- \nname: knob\ndescription: d\n---\t\nbody\n",
        "---\nname: knob\ndescription: d\n---\nbody\n\n---\n\nmore\n",
    ],
    ids=["plain", "trailing-blanks", "horizontal-rule-in-body"],
)
def test_a_listed_skill_with_well_formed_frontmatter_loads(
    lifecycle_repo: Path,  # noqa: F811
    text: str,
) -> None:
    app_dir = _app(lifecycle_repo, skills=["knob"])
    _write(app_dir / "skills" / "knob" / "SKILL.md", text)

    check_deployment(lifecycle_repo)


def test_hooks_after_leading_blank_lines_are_refused(
    lifecycle_repo: Path,  # noqa: F811
) -> None:
    app_dir = _app(lifecycle_repo, skills=["knob"])
    _write(
        app_dir / "skills" / "knob" / "SKILL.md",
        "\n\n---\nname: knob\ndescription: d\nhooks:\n  PreToolUse: []\n---\nbody\n",
    )

    with pytest.raises(AppManifestError) as exc:
        check_deployment(lifecycle_repo)

    assert _skill_frontmatter(exc) == ["skills.knob.frontmatter"]


def test_hooks_behind_a_line_separator_in_the_block_are_refused(
    lifecycle_repo: Path,  # noqa: F811
) -> None:
    app_dir = _app(lifecycle_repo, skills=["knob"])
    _write(
        app_dir / "skills" / "knob" / "SKILL.md",
        "---\nname: knob\ndescription: d hooks:\n  PreToolUse: []\n---\nbody\n",
    )

    with pytest.raises(AppManifestError) as exc:
        check_deployment(lifecycle_repo)

    assert _skill_frontmatter(exc) == ["skills.knob.frontmatter"]
