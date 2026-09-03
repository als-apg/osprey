"""Static gate over the shipped plugin marketplaces, plugin manifests and skills.

OSPREY's agent skills ship as one plugin, ``osprey``, offered from two
marketplace manifests that describe the same tree to two different agent
runtimes:

* ``.claude-plugin/marketplace.json`` — Claude Code. A plugin entry's ``source``
  is the bare relative path string.
* ``.agents/plugins/marketplace.json`` — Codex. The same entry's ``source`` is an
  object ``{"source": "local", "path": ...}``, and the entry additionally carries
  ``policy`` and ``category``.

Because the two files are written and edited independently, the interesting
failure is *skew*: a version bumped on one side only, a plugin renamed in one
manifest, a source path that no longer resolves. Every check below is a function
over a root directory rather than a hard-coded repo path, so the same code that
guards the real tree also drives the negative tests, which mutate a throwaway
copy and assert the corresponding check notices.

The Codex field values (``policy.installation``, the ``policy.authentication``
enum, ``category``, ``skills``) are pinned to the Codex plugin specification at
https://developers.openai.com/plugins/build/plugins, retrieved 2026-09-01. That
spec enumerates no categories, so ``category`` is only required to be a
non-empty string.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Marketplace manifests, relative to a root directory.
_CLAUDE_MARKETPLACE = ".claude-plugin/marketplace.json"
_CODEX_MARKETPLACE = ".agents/plugins/marketplace.json"

#: Plugin manifests, relative to a plugin directory.
_CLAUDE_PLUGIN_MANIFEST = ".claude-plugin/plugin.json"
_CODEX_PLUGIN_MANIFEST = ".codex-plugin/plugin.json"

#: Both marketplaces and the single plugin are all named this.
_MARKETPLACE_NAME = "osprey"

_SKILL_NAME_RE = re.compile(r"^[a-z0-9-]{1,64}$")
_CALVER_RE = re.compile(r"^\d{4}\.\d{1,2}\.\d+$")
_MAX_DESCRIPTION_CHARS = 1024
_ALLOWED_FRONTMATTER_KEYS = frozenset(
    {"name", "description", "allowed-tools", "license", "metadata"}
)

_CODEX_INSTALLATION = "AVAILABLE"
_CODEX_AUTHENTICATION = frozenset({"ON_INSTALL", "ON_USE"})
_CODEX_SKILLS_VALUE = "./skills/"

#: Substrings that betray a developer machine or a chat session in shipped text.
#: The sweep never reaches ``tests/``, so spelling them here is safe.
_FORBIDDEN_LITERALS = ("/Users/", "/home/", "claude.ai/code")

#: Directories the leak sweep walks — both marketplaces and the plugin tree.
_LEAK_SWEEP_ROOTS = (".claude-plugin", ".agents/plugins", "plugins")

#: Copied wholesale into the throwaway tree the negative tests mutate.
_MUTABLE_ROOTS = (".claude-plugin", ".agents/plugins", "plugins")

#: Single files the throwaway tree also needs, because checks read them.
_MUTABLE_FILES = ("pyproject.toml",)


# =====================================================================
# Loading helpers
# =====================================================================


def _load_json(path: Path) -> Any:
    """Return the parsed JSON at ``path``, or ``None`` when it is absent."""
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _frontmatter(text: str) -> dict[str, Any] | None:
    """Return the YAML frontmatter mapping of a SKILL.md body.

    Args:
        text: Full file contents, expected to open with a ``---`` fence.

    Returns:
        The parsed mapping, or ``None`` when there is no closed frontmatter
        block or it does not parse to a mapping.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            block = "\n".join(lines[1:index])
            loaded = yaml.safe_load(block) if block.strip() else None
            return loaded if isinstance(loaded, dict) else None
    return None


def _skill_dirs(root: Path) -> list[Path]:
    """Return every skill directory under ``<root>/plugins/*/skills/``, sorted."""
    return sorted(p for p in (root / "plugins").glob("*/skills/*") if p.is_dir())


def _plugin_dirs(root: Path) -> list[Path]:
    """Return every plugin directory named by the Claude marketplace."""
    manifest = _load_json(root / _CLAUDE_MARKETPLACE) or {}
    dirs: list[Path] = []
    for entry in manifest.get("plugins", []):
        source = _source_path(entry)
        if source:
            dirs.append(root / source)
    return dirs


def _source_path(entry: dict[str, Any]) -> str | None:
    """Normalise a marketplace entry's ``source`` to a plain relative path.

    Claude spells it as a bare string; Codex wraps it in
    ``{"source": "local", "path": ...}``. Both reduce to the same path.
    """
    source = entry.get("source")
    if isinstance(source, str):
        return source
    if isinstance(source, dict):
        path = source.get("path")
        return path if isinstance(path, str) else None
    return None


def _entry_identity(manifest: dict[str, Any]) -> list[tuple[str | None, str | None]]:
    """Return the ``(name, source path)`` pairs of a marketplace's plugin list."""
    return sorted((entry.get("name"), _source_path(entry)) for entry in manifest.get("plugins", []))


# =====================================================================
# Checks — each returns a list of human-readable problems
# =====================================================================


def check_skill_frontmatter(root: Path) -> list[str]:
    """Check every skill under ``<root>/plugins/*/skills/`` against FR3.

    Each skill directory must hold a ``SKILL.md`` whose frontmatter ``name``
    equals the directory name and matches the slug pattern, whose
    ``description`` is non-empty and within the length cap, and whose keys stay
    inside the supported set.
    """
    problems: list[str] = []
    skill_dirs = _skill_dirs(root)
    if not skill_dirs:
        return [f"no skill directories found under {root / 'plugins'}"]

    for skill_dir in skill_dirs:
        rel = skill_dir.relative_to(root)
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.is_file():
            problems.append(f"{rel}: missing SKILL.md")
            continue

        data = _frontmatter(skill_md.read_text(encoding="utf-8"))
        if data is None:
            problems.append(f"{rel}/SKILL.md: no parsable YAML frontmatter block")
            continue

        extra = sorted(set(data) - _ALLOWED_FRONTMATTER_KEYS)
        if extra:
            problems.append(f"{rel}/SKILL.md: unsupported frontmatter keys {extra}")

        name = data.get("name")
        if name != skill_dir.name:
            problems.append(f"{rel}/SKILL.md: name {name!r} does not equal directory name")
        if not isinstance(name, str) or not _SKILL_NAME_RE.match(name):
            problems.append(
                f"{rel}/SKILL.md: name {name!r} does not match {_SKILL_NAME_RE.pattern}"
            )

        description = data.get("description")
        if not isinstance(description, str) or not description.strip():
            problems.append(f"{rel}/SKILL.md: description is missing or empty")
        elif len(description) > _MAX_DESCRIPTION_CHARS:
            problems.append(
                f"{rel}/SKILL.md: description is {len(description)} chars, "
                f"over the {_MAX_DESCRIPTION_CHARS} cap"
            )

    return problems


def check_marketplace_sources(root: Path) -> list[str]:
    """Check that every plugin source resolves to a dir holding both manifests."""
    problems: list[str] = []
    for marketplace in (_CLAUDE_MARKETPLACE, _CODEX_MARKETPLACE):
        manifest = _load_json(root / marketplace)
        if manifest is None:
            problems.append(f"{marketplace}: missing")
            continue
        entries = manifest.get("plugins")
        if not entries:
            problems.append(f"{marketplace}: no plugin entries")
            continue
        for entry in entries:
            source = _source_path(entry)
            if not source:
                problems.append(f"{marketplace}: entry {entry.get('name')!r} has no source path")
                continue
            plugin_dir = root / source
            if not plugin_dir.is_dir():
                problems.append(f"{marketplace}: source {source!r} is not a directory")
                continue
            for manifest_rel in (_CLAUDE_PLUGIN_MANIFEST, _CODEX_PLUGIN_MANIFEST):
                if not (plugin_dir / manifest_rel).is_file():
                    problems.append(f"{marketplace}: {source!r} is missing {manifest_rel}")
    return problems


def check_marketplace_agreement(root: Path) -> list[str]:
    """Check the two marketplaces describe the same plugin names and paths."""
    claude = _load_json(root / _CLAUDE_MARKETPLACE)
    codex = _load_json(root / _CODEX_MARKETPLACE)
    if claude is None or codex is None:
        return ["cannot compare marketplaces: one of the two manifests is missing"]

    problems: list[str] = []
    for label, manifest in (("Claude", claude), ("Codex", codex)):
        if manifest.get("name") != _MARKETPLACE_NAME:
            problems.append(
                f"{label} marketplace name {manifest.get('name')!r} is not {_MARKETPLACE_NAME!r}"
            )
    claude_entries = _entry_identity(claude)
    codex_entries = _entry_identity(codex)
    if claude_entries != codex_entries:
        problems.append(
            f"plugin (name, source path) skew: Claude {claude_entries} vs Codex {codex_entries}"
        )
    return problems


def check_plugin_versions(root: Path) -> list[str]:
    """Check both plugin manifests carry the same CalVer version string."""
    plugin_dirs = _plugin_dirs(root)
    if not plugin_dirs:
        return ["no plugin directories resolved from the Claude marketplace"]

    problems: list[str] = []
    for plugin_dir in plugin_dirs:
        versions: dict[str, Any] = {}
        for manifest_rel in (_CLAUDE_PLUGIN_MANIFEST, _CODEX_PLUGIN_MANIFEST):
            manifest = _load_json(plugin_dir / manifest_rel)
            if manifest is None:
                problems.append(f"{plugin_dir.name}: missing {manifest_rel}")
                continue
            versions[manifest_rel] = manifest.get("version")
        if len(versions) != 2:
            continue
        claude_version, codex_version = (
            versions[_CLAUDE_PLUGIN_MANIFEST],
            versions[_CODEX_PLUGIN_MANIFEST],
        )
        if claude_version != codex_version:
            problems.append(
                f"{plugin_dir.name}: version skew — Claude {claude_version!r} vs "
                f"Codex {codex_version!r}; run scripts/plugin_version.py bump"
            )
        for label, version in (("Claude", claude_version), ("Codex", codex_version)):
            if not isinstance(version, str) or not _CALVER_RE.match(version):
                problems.append(
                    f"{plugin_dir.name}: {label} version {version!r} is not CalVer "
                    f"({_CALVER_RE.pattern})"
                )
    return problems


def check_codex_fields(root: Path) -> list[str]:
    """Check the Codex-only fields against the Codex plugin specification.

    Pinned to https://developers.openai.com/plugins/build/plugins, retrieved
    2026-09-01: ``policy.installation`` is ``AVAILABLE``,
    ``policy.authentication`` is one of ``ON_INSTALL`` / ``ON_USE``,
    ``category`` is a non-empty string, and a skills-only plugin points
    ``skills`` at its skills directory.
    """
    manifest = _load_json(root / _CODEX_MARKETPLACE)
    if manifest is None:
        return [f"{_CODEX_MARKETPLACE}: missing"]

    problems: list[str] = []
    for entry in manifest.get("plugins", []):
        name = entry.get("name")
        policy = entry.get("policy")
        if not isinstance(policy, dict):
            problems.append(f"{name}: policy is missing or not an object")
        else:
            installation = policy.get("installation")
            if installation != _CODEX_INSTALLATION:
                problems.append(
                    f"{name}: policy.installation {installation!r} is not {_CODEX_INSTALLATION!r}"
                )
            authentication = policy.get("authentication")
            if authentication not in _CODEX_AUTHENTICATION:
                problems.append(
                    f"{name}: policy.authentication {authentication!r} is outside "
                    f"{sorted(_CODEX_AUTHENTICATION)}"
                )
        category = entry.get("category")
        if not isinstance(category, str) or not category.strip():
            problems.append(f"{name}: category {category!r} is missing or empty")

        source = _source_path(entry)
        if not source:
            continue
        plugin_manifest = _load_json(root / source / _CODEX_PLUGIN_MANIFEST)
        if plugin_manifest is None:
            problems.append(f"{name}: missing {_CODEX_PLUGIN_MANIFEST}")
            continue
        skills = plugin_manifest.get("skills")
        if skills != _CODEX_SKILLS_VALUE:
            problems.append(f"{name}: skills {skills!r} is not {_CODEX_SKILLS_VALUE!r}")
    return problems


def check_owner_and_author(root: Path) -> list[str]:
    """Check the marketplace owner and the plugin author are one identity, minus email.

    The reference marketplace this pair was modelled on carries a personal
    email address. Publishing one from a repository manifest is a contact-detail
    leak nobody asked for, so both objects are required to stay email-free, and
    the two must not drift apart into two different-looking identities.
    """
    marketplace = _load_json(root / _CLAUDE_MARKETPLACE)
    if marketplace is None:
        return [f"{_CLAUDE_MARKETPLACE}: missing"]

    problems: list[str] = []
    owner = marketplace.get("owner")
    if not isinstance(owner, dict):
        problems.append(f"{_CLAUDE_MARKETPLACE}: owner is missing or not an object")
        owner = {}
    if "email" in owner:
        problems.append(f"{_CLAUDE_MARKETPLACE}: owner carries an email key")

    for plugin_dir in _plugin_dirs(root):
        plugin = _load_json(plugin_dir / _CLAUDE_PLUGIN_MANIFEST)
        if plugin is None:
            problems.append(f"{plugin_dir.name}: missing {_CLAUDE_PLUGIN_MANIFEST}")
            continue
        author = plugin.get("author")
        if not isinstance(author, dict):
            problems.append(f"{plugin_dir.name}: author is missing or not an object")
            continue
        if "email" in author:
            problems.append(f"{plugin_dir.name}: author carries an email key")
        if author != owner:
            problems.append(
                f"{plugin_dir.name}: author {author} does not equal the marketplace owner {owner}"
            )
    return problems


def _pyproject_license(raw: Any) -> str | None:
    """Normalise the pyproject ``license`` field to its SPDX string.

    PEP 621 allows both a bare string and a table. This repository spells it
    ``license = {text = "BSD-3-Clause"}``; both shapes are accepted here so a
    later migration to the bare-string form does not read as a licence change.
    """
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        text = raw.get("text")
        return text if isinstance(text, str) else None
    return None


def check_project_metadata(root: Path) -> list[str]:
    """Check the plugin manifest's URLs and licence match ``pyproject.toml``.

    The plugin manifest restates project metadata that already has a single
    source of truth in ``[project]`` and ``[project.urls]``. Restating it is
    fine; letting the two drift is not.
    """
    pyproject_path = root / "pyproject.toml"
    if not pyproject_path.is_file():
        return [f"{pyproject_path} does not exist"]
    project = tomllib.loads(pyproject_path.read_text(encoding="utf-8")).get("project", {})
    urls = project.get("urls", {})

    expected = {
        "homepage": urls.get("Homepage"),
        "repository": urls.get("Repository"),
        "license": _pyproject_license(project.get("license")),
    }
    problems = [
        f"pyproject.toml has no value for {key}" for key, value in expected.items() if not value
    ]

    for plugin_dir in _plugin_dirs(root):
        plugin = _load_json(plugin_dir / _CLAUDE_PLUGIN_MANIFEST)
        if plugin is None:
            problems.append(f"{plugin_dir.name}: missing {_CLAUDE_PLUGIN_MANIFEST}")
            continue
        for key, value in expected.items():
            if plugin.get(key) != value:
                problems.append(
                    f"{plugin_dir.name}: {key} {plugin.get(key)!r} does not match "
                    f"pyproject.toml {value!r}"
                )
    return problems


def check_no_machine_paths(root: Path) -> list[str]:
    """Check no manifest or shipped plugin file leaks a machine path or session URL.

    The sweep covers both marketplace directories as well as ``plugins/``: a
    hand-written manifest is exactly where an absolute developer path ends up.
    """
    problems: list[str] = []
    bases = [root / rel for rel in _LEAK_SWEEP_ROOTS]
    missing = [str(base) for base in bases if not base.is_dir()]
    if missing:
        return [f"leak sweep root does not exist: {name}" for name in missing]
    for path in sorted(p for base in bases for p in base.rglob("*")):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for literal in _FORBIDDEN_LITERALS:
            if literal in text:
                problems.append(f"{path.relative_to(root)}: contains {literal!r}")
    return problems


# =====================================================================
# Positive tests — the real tree
# =====================================================================


def test_skill_frontmatter_is_well_formed() -> None:
    """Every shipped skill satisfies the frontmatter contract."""
    assert check_skill_frontmatter(_REPO_ROOT) == []


def test_marketplace_sources_resolve() -> None:
    """Both marketplaces point at a directory holding both plugin manifests."""
    assert check_marketplace_sources(_REPO_ROOT) == []


def test_marketplaces_agree_on_plugins() -> None:
    """The two marketplaces name the same plugins at the same paths."""
    assert check_marketplace_agreement(_REPO_ROOT) == []


def test_plugin_versions_match_and_are_calver() -> None:
    """Both plugin manifests carry one identical CalVer version."""
    assert check_plugin_versions(_REPO_ROOT) == []


def test_codex_fields_follow_the_spec() -> None:
    """The Codex policy, category and skills fields hold their pinned values."""
    assert check_codex_fields(_REPO_ROOT) == []


def test_owner_and_author_are_one_identity() -> None:
    """The marketplace owner equals the plugin author, and neither carries an email."""
    assert check_owner_and_author(_REPO_ROOT) == []


def test_project_metadata_matches_pyproject() -> None:
    """The plugin manifest's homepage, repository and licence match pyproject.toml."""
    assert check_project_metadata(_REPO_ROOT) == []


def test_no_machine_paths_in_manifests_or_plugins() -> None:
    """No manifest or shipped plugin file leaks an absolute path or session URL."""
    assert check_no_machine_paths(_REPO_ROOT) == []


# =====================================================================
# Negative tests — mutate a throwaway copy, assert the check notices
# =====================================================================


@pytest.fixture
def mutable_tree(tmp_path: Path) -> Path:
    """Copy the manifest and plugin trees into ``tmp_path`` for mutation."""
    for rel in _MUTABLE_ROOTS:
        destination = tmp_path / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(_REPO_ROOT / rel, destination)
    for rel in _MUTABLE_FILES:
        shutil.copy2(_REPO_ROOT / rel, tmp_path / rel)
    return tmp_path


def _a_skill_dir(root: Path) -> Path:
    """Return one skill directory from the copied tree, deterministically."""
    return _skill_dirs(root)[0]


def _set_frontmatter(skill_dir: Path, lines: list[str]) -> None:
    """Replace a SKILL.md frontmatter block with ``lines``, keeping the body."""
    path = skill_dir / "SKILL.md"
    _, _, body = path.read_text(encoding="utf-8").partition("---\n")
    _, _, body = body.partition("---\n")
    path.write_text("---\n" + "\n".join(lines) + "\n---\n" + body, encoding="utf-8")


def _edit_json(path: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    """Load, mutate in place, and rewrite the JSON document at ``path``."""
    data = json.loads(path.read_text(encoding="utf-8"))
    mutate(data)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _mut_extra_frontmatter_key(root: Path) -> None:
    skill_dir = _a_skill_dir(root)
    _set_frontmatter(skill_dir, [f"name: {skill_dir.name}", "description: ok", "summary: extra"])


def _mut_name_mismatch(root: Path) -> None:
    _set_frontmatter(_a_skill_dir(root), ["name: not-the-directory", "description: ok"])


def _mut_bad_name_pattern(root: Path) -> None:
    skill_dir = _a_skill_dir(root)
    renamed = skill_dir.with_name("Bad_Name")
    skill_dir.rename(renamed)
    _set_frontmatter(renamed, ["name: Bad_Name", "description: ok"])


def _mut_empty_description(root: Path) -> None:
    skill_dir = _a_skill_dir(root)
    _set_frontmatter(skill_dir, [f"name: {skill_dir.name}", 'description: ""'])


def _mut_overlong_description(root: Path) -> None:
    skill_dir = _a_skill_dir(root)
    filler = "x" * (_MAX_DESCRIPTION_CHARS + 1)
    _set_frontmatter(skill_dir, [f"name: {skill_dir.name}", f"description: {filler}"])


def _mut_missing_skill_md(root: Path) -> None:
    (_a_skill_dir(root) / "SKILL.md").unlink()


def _mut_unresolvable_source(root: Path) -> None:
    def mutate(data: dict[str, Any]) -> None:
        data["plugins"][0]["source"] = "./plugins/does-not-exist"

    _edit_json(root / _CLAUDE_MARKETPLACE, mutate)


def _mut_source_path_skew(root: Path) -> None:
    def mutate(data: dict[str, Any]) -> None:
        data["plugins"][0]["source"]["path"] = "./plugins/elsewhere"

    _edit_json(root / _CODEX_MARKETPLACE, mutate)


def _mut_plugin_name_skew(root: Path) -> None:
    def mutate(data: dict[str, Any]) -> None:
        data["plugins"][0]["name"] = "renamed"

    _edit_json(root / _CODEX_MARKETPLACE, mutate)


def _mut_version_skew(root: Path) -> None:
    def mutate(data: dict[str, Any]) -> None:
        data["version"] = "2026.9.99"

    _edit_json(_plugin_dirs(root)[0] / _CODEX_PLUGIN_MANIFEST, mutate)


def _mut_non_calver_version(root: Path) -> None:
    def mutate(data: dict[str, Any]) -> None:
        data["version"] = "1.0.0-rc1"

    for manifest_rel in (_CLAUDE_PLUGIN_MANIFEST, _CODEX_PLUGIN_MANIFEST):
        _edit_json(_plugin_dirs(root)[0] / manifest_rel, mutate)


def _mut_bad_installation(root: Path) -> None:
    def mutate(data: dict[str, Any]) -> None:
        data["plugins"][0]["policy"]["installation"] = "HIDDEN"

    _edit_json(root / _CODEX_MARKETPLACE, mutate)


def _mut_bad_authentication(root: Path) -> None:
    def mutate(data: dict[str, Any]) -> None:
        data["plugins"][0]["policy"]["authentication"] = "NEVER"

    _edit_json(root / _CODEX_MARKETPLACE, mutate)


def _mut_empty_category(root: Path) -> None:
    def mutate(data: dict[str, Any]) -> None:
        data["plugins"][0]["category"] = ""

    _edit_json(root / _CODEX_MARKETPLACE, mutate)


def _mut_wrong_skills_value(root: Path) -> None:
    def mutate(data: dict[str, Any]) -> None:
        data["skills"] = "skills"

    _edit_json(_plugin_dirs(root)[0] / _CODEX_PLUGIN_MANIFEST, mutate)


def _mut_marketplace_name(root: Path) -> None:
    def mutate(data: dict[str, Any]) -> None:
        data["name"] = "not-osprey"

    _edit_json(root / _CLAUDE_MARKETPLACE, mutate)
    _edit_json(root / _CODEX_MARKETPLACE, mutate)


def _mut_owner_email(root: Path) -> None:
    def mutate(data: dict[str, Any]) -> None:
        data["owner"]["email"] = "someone@example.org"

    _edit_json(root / _CLAUDE_MARKETPLACE, mutate)


def _mut_author_skew(root: Path) -> None:
    def mutate(data: dict[str, Any]) -> None:
        data["author"]["name"] = "Someone Else"

    _edit_json(_plugin_dirs(root)[0] / _CLAUDE_PLUGIN_MANIFEST, mutate)


def _mut_homepage_skew(root: Path) -> None:
    def mutate(data: dict[str, Any]) -> None:
        data["homepage"] = "https://example.org/elsewhere"

    _edit_json(_plugin_dirs(root)[0] / _CLAUDE_PLUGIN_MANIFEST, mutate)


def _mut_repository_skew(root: Path) -> None:
    def mutate(data: dict[str, Any]) -> None:
        data["repository"] = "https://example.org/fork"

    _edit_json(_plugin_dirs(root)[0] / _CLAUDE_PLUGIN_MANIFEST, mutate)


def _mut_license_skew(root: Path) -> None:
    def mutate(data: dict[str, Any]) -> None:
        data["license"] = "MIT"

    _edit_json(_plugin_dirs(root)[0] / _CLAUDE_PLUGIN_MANIFEST, mutate)


def _mut_machine_path(root: Path) -> None:
    skill_md = _a_skill_dir(root) / "SKILL.md"
    skill_md.write_text(
        skill_md.read_text(encoding="utf-8") + "\nRun it from /Users/someone/code/osprey.\n",
        encoding="utf-8",
    )


def _mut_machine_path_in_manifest(root: Path) -> None:
    def mutate(data: dict[str, Any]) -> None:
        data["plugins"][0]["description"] = "Skills, checked out at /home/someone/osprey."

    _edit_json(root / _CODEX_MARKETPLACE, mutate)


_MUTATIONS: list[tuple[str, Callable[[Path], None], Callable[[Path], list[str]], str]] = [
    ("extra_frontmatter_key", _mut_extra_frontmatter_key, check_skill_frontmatter, "unsupported"),
    ("name_mismatch", _mut_name_mismatch, check_skill_frontmatter, "directory name"),
    ("bad_name_pattern", _mut_bad_name_pattern, check_skill_frontmatter, "does not match"),
    ("empty_description", _mut_empty_description, check_skill_frontmatter, "missing or empty"),
    ("overlong_description", _mut_overlong_description, check_skill_frontmatter, "over the 1024"),
    ("missing_skill_md", _mut_missing_skill_md, check_skill_frontmatter, "missing SKILL.md"),
    ("unresolvable_source", _mut_unresolvable_source, check_marketplace_sources, "not a directory"),
    ("source_path_skew", _mut_source_path_skew, check_marketplace_agreement, "./plugins/elsewhere"),
    ("plugin_name_skew", _mut_plugin_name_skew, check_marketplace_agreement, "'renamed'"),
    ("version_skew", _mut_version_skew, check_plugin_versions, "version skew"),
    ("non_calver_version", _mut_non_calver_version, check_plugin_versions, "is not CalVer"),
    ("bad_installation", _mut_bad_installation, check_codex_fields, "policy.installation"),
    ("bad_authentication", _mut_bad_authentication, check_codex_fields, "policy.authentication"),
    ("empty_category", _mut_empty_category, check_codex_fields, "category"),
    ("wrong_skills_value", _mut_wrong_skills_value, check_codex_fields, "skills"),
    ("marketplace_name", _mut_marketplace_name, check_marketplace_agreement, "not-osprey"),
    ("owner_email", _mut_owner_email, check_owner_and_author, "email key"),
    ("author_skew", _mut_author_skew, check_owner_and_author, "does not equal"),
    ("homepage_skew", _mut_homepage_skew, check_project_metadata, "homepage"),
    ("repository_skew", _mut_repository_skew, check_project_metadata, "repository"),
    ("license_skew", _mut_license_skew, check_project_metadata, "license"),
    ("machine_path", _mut_machine_path, check_no_machine_paths, "contains"),
    (
        "machine_path_in_manifest",
        _mut_machine_path_in_manifest,
        check_no_machine_paths,
        "marketplace.json",
    ),
]


@pytest.mark.parametrize(
    ("mutate", "check", "expected"),
    [pytest.param(m, c, e, id=name) for name, m, c, e in _MUTATIONS],
)
def test_mutated_tree_is_rejected(
    mutable_tree: Path,
    mutate: Callable[[Path], None],
    check: Callable[[Path], list[str]],
    expected: str,
) -> None:
    """A deliberately broken copy of the tree fails the check that owns it."""
    assert check(mutable_tree) == [], "the unmutated copy should already be clean"
    mutate(mutable_tree)
    problems = check(mutable_tree)
    assert problems, "the mutation went unnoticed"
    assert any(expected in problem for problem in problems), problems


# =====================================================================
# Ignore rules and the Claude CLI validator
# =====================================================================


def _run_git(*args: str) -> subprocess.CompletedProcess[str]:
    """Run git in the repo root, failing hard on any exit outside 0/1.

    ``check-ignore`` uses exit 1 to mean "not ignored", so 0 and 1 are both
    answers. Anything else is a broken invocation and must not be read as one.
    """
    result = subprocess.run(
        ["git", *args],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode not in (0, 1):
        pytest.fail(f"git {' '.join(args)} exited {result.returncode}: {result.stderr.strip()}")
    return result


def test_git_ignore_shape() -> None:
    """The Codex marketplace manifest is tracked; installed plugins are ignored.

    ``.gitignore`` carves exactly one file out of an otherwise ignored
    ``.agents/`` tree. A probe file proves the ignore still bites, and
    ``ls-files`` proves the carve-out is not merely un-ignored but actually
    tracked.
    """
    probe = _REPO_ROOT / ".agents" / "plugins" / "installed" / "probe.bin"
    installed_dir = probe.parent
    created_dir = not installed_dir.exists()
    installed_dir.mkdir(parents=True, exist_ok=True)
    probe.write_bytes(b"probe")
    try:
        ignored = _run_git("check-ignore", "-q", ".agents/plugins/installed/probe.bin")
        assert ignored.returncode == 0, "installed plugins under .agents/ must stay ignored"

        carved_out = _run_git("check-ignore", "-q", ".agents/plugins/marketplace.json")
        assert carved_out.returncode == 1, "the Codex marketplace manifest must not be ignored"

        tracked = _run_git("ls-files", ".agents/plugins/marketplace.json")
        assert tracked.returncode == 0
        assert tracked.stdout.strip(), "the Codex marketplace manifest is not tracked"
    finally:
        probe.unlink(missing_ok=True)
        if created_dir:
            shutil.rmtree(installed_dir, ignore_errors=True)


def test_validate_strict(tmp_path: Path) -> None:
    """The bundled Claude CLI validates the marketplace manifest in strict mode.

    This covers the plugin manifest transitively: the validator resolves every
    plugin source. It never skips — a missing CLI is a real failure, because the
    binary ships with the SDK this project depends on.
    """
    try:
        import claude_agent_sdk
    except ImportError as exc:  # pragma: no cover - dependency is declared
        pytest.fail(f"claude_agent_sdk is not importable, so the bundled CLI is unreachable: {exc}")

    binary = Path(claude_agent_sdk.__file__).parent / "_bundled" / "claude"
    assert binary.is_file(), f"bundled Claude CLI not found at {binary}"

    result = subprocess.run(
        [str(binary), "plugin", "validate", "--strict", _CLAUDE_MARKETPLACE],
        cwd=_REPO_ROOT,
        env={**os.environ, "CLAUDE_CONFIG_DIR": str(tmp_path)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"plugin validate --strict exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
