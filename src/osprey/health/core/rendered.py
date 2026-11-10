"""Core ``rendered`` health category.

Compares the rendered ``.claude/`` tree and ``.mcp.json`` with what ``osprey
build`` wrote there. The build already records that: every file it renders is
checksummed into the render's ``.osprey-manifest.json`` (``file_checksums``,
:func:`osprey.cli.templates.manifest.calculate_file_checksums`). This category
reads that record rather than re-rendering — the manifest is the one producer
of "what the build wrote", and a re-render would be a second, slower one that
could disagree with it. A file edited, replaced or deleted since the build is
drift; a file the build never wrote (an agent's own addition) is not counted,
and neither is a graph agent prompt whose snapshot region ``osprey up`` baked
after the build (that writer owns the region by design).

Runs against the render directory (``cwd``: the directory holding the rendered
``config.yml``). With no ``.claude/`` and no ``.mcp.json`` there — nothing was
rendered — the category contributes no rows. Otherwise:

* ``rendered.claude_dir`` — tracked ``.claude/`` files that differ from their
  manifest checksum or are missing: ``ok`` at zero, else ``warning`` naming
  them in ``details``;
* ``rendered.settings_json`` — ``.claude/settings.json`` against its checksum;
* ``rendered.mcp_json`` — ``.mcp.json`` against its checksum, with its server
  count; ``warning`` when it cannot be parsed;
* ``rendered.agents`` — sub-agent files in ``.claude/agents/`` whose checksum
  matches the manifest, out of those the manifest tracks;
* ``rendered.skills`` — ``SKILL.md`` files under ``.claude/skills/`` and how
  many have no parseable YAML frontmatter (``warning`` when any).

Without a manifest the four comparisons are ``skip`` rows ("no build
manifest"); the skill count still reports. Every drift row is advisory
(``warning``) with ``osprey build`` as its fix: a drifted render is a stale
build, not a broken deployment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from osprey.health.models import CheckResult, Status

if TYPE_CHECKING:
    from collections.abc import Mapping

    from osprey.health.core import CategoryCallable
    from osprey.health.runtime import HealthRuntime

CATEGORY = "rendered"

_FIX = "osprey build"
_NO_MANIFEST = "no build manifest"
_CLAUDE = ".claude/"
_SETTINGS = ".claude/settings.json"
_AGENTS = ".claude/agents/"
_MCP_JSON = ".mcp.json"

#: How many drifted paths ``details`` names before it stops listing.
_LISTED = 20


def rendered(
    config: Mapping[str, Any] | None = None,  # noqa: ARG001 - health category factory signature; this category reads the render, not the config
    context: HealthRuntime | None = None,  # noqa: ARG001 - health category factory signature; categories that probe a runtime read the context
    *,
    cwd: Path | None = None,
) -> CategoryCallable:
    """Build the ``rendered`` category callable.

    Args:
        config: Parsed config mapping. Unused.
        context: Health runtime. Unused.
        cwd: The render directory (holding ``config.yml``, ``.claude/``,
            ``.mcp.json`` and ``.osprey-manifest.json``). ``None`` yields no
            rows.

    Returns:
        A no-argument callable returning the category's check results.
    """

    def _run() -> list[CheckResult]:
        if cwd is None:
            return []
        root = Path(cwd)
        if not (root / _CLAUDE).is_dir() and not (root / _MCP_JSON).is_file():
            return []
        checksums = _manifest_checksums(root)
        return [
            _claude_dir_row(root, checksums),
            _file_row("rendered.settings_json", root, _SETTINGS, checksums),
            _mcp_json_row(root, checksums),
            _agents_row(root, checksums),
            _skills_row(root),
        ]

    return _run


def _manifest_checksums(root: Path) -> dict[str, str] | None:
    """The manifest's ``file_checksums``, or ``None`` when there is no usable manifest."""
    from osprey.cli.templates.manifest import load_project_manifest

    manifest = load_project_manifest(root)
    checksums = manifest.get("file_checksums") if manifest else None
    return checksums if isinstance(checksums, dict) else None


def _matches(root: Path, rel: str, expected: str) -> bool:
    """Whether ``root/rel`` exists and carries the manifest's checksum."""
    from osprey.build.manifest import sha256_file

    path = root / rel
    try:
        return path.is_file() and f"sha256:{sha256_file(path)}" == expected
    except OSError:
        return False


def _baked(root: Path, rel: str) -> bool:
    """Whether ``root/rel`` is a graph agent file the seeder baked after the build.

    ``osprey up`` rewrites the snapshot region of these agent prompts in place
    (:mod:`osprey.services.graph_store.prompt_snapshot`), so their
    checksum differs from the build's by design. A file carrying its complete
    marker pair is that writer's output, not drift.
    """
    from osprey.services.graph_store.prompt_snapshot import (
        SNAPSHOT_BEGIN,
        SNAPSHOT_END,
        TARGET_FILENAMES,
    )

    if Path(rel).name not in TARGET_FILENAMES:
        return False
    try:
        text = (root / rel).read_text(encoding="utf-8")
    except OSError:
        return False
    begin = text.find(SNAPSHOT_BEGIN)
    return begin != -1 and text.find(SNAPSHOT_END) > begin


def _differs(root: Path, rel: str, expected: str) -> bool:
    """Whether a tracked file drifted: checksum differs and no deploy-time writer owns it."""
    return not _matches(root, rel, expected) and not _baked(root, rel)


def _drifted(root: Path, checksums: Mapping[str, str], prefix: str) -> tuple[list[str], int]:
    """Tracked paths under *prefix* that differ or are missing, and how many are tracked."""
    tracked = sorted(rel for rel in checksums if rel.startswith(prefix))
    return [rel for rel in tracked if _differs(root, rel, checksums[rel])], len(tracked)


def _skip(name: str, message: str) -> CheckResult:
    return CheckResult(name, CATEGORY, Status.SKIP, message, value=_NO_MANIFEST)


def _claude_dir_row(root: Path, checksums: Mapping[str, str] | None) -> CheckResult:
    name = "rendered.claude_dir"
    if checksums is None:
        return _skip(name, "No build manifest to compare .claude/ against")
    drifted, tracked = _drifted(root, checksums, _CLAUDE)
    value = f"{len(drifted)} / {tracked} files differ"
    if not drifted:
        return CheckResult(name, CATEGORY, Status.OK, ".claude/ matches the build", value=value)
    listed = "\n".join(drifted[:_LISTED])
    more = f"\n… and {len(drifted) - _LISTED} more" if len(drifted) > _LISTED else ""
    return CheckResult(
        name,
        CATEGORY,
        Status.WARNING,
        f".claude/ differs from the build in {len(drifted)} file(s)",
        value=value,
        details=listed + more,
        fix=_FIX,
    )


def _file_row(name: str, root: Path, rel: str, checksums: Mapping[str, str] | None) -> CheckResult:
    if checksums is None:
        return _skip(name, f"No build manifest to compare {rel} against")
    expected = checksums.get(rel)
    if expected is None:
        return CheckResult(
            name, CATEGORY, Status.SKIP, f"{rel} is not tracked by the build", value="not tracked"
        )
    if _matches(root, rel, expected):
        return CheckResult(name, CATEGORY, Status.OK, f"{rel} matches the build", value="= build")
    return CheckResult(
        name,
        CATEGORY,
        Status.WARNING,
        f"{rel} differs from the build",
        value="differs from build",
        fix=_FIX,
    )


def _mcp_json_row(root: Path, checksums: Mapping[str, str] | None) -> CheckResult:
    name = "rendered.mcp_json"
    try:
        servers = json.loads((root / _MCP_JSON).read_text(encoding="utf-8")).get("mcpServers")
        count = len(servers) if isinstance(servers, dict) else 0
    except (OSError, ValueError, AttributeError) as exc:
        return CheckResult(
            name,
            CATEGORY,
            Status.WARNING,
            f"{_MCP_JSON} cannot be read",
            value="unreadable",
            details=str(exc),
            fix=_FIX,
        )
    noun = "server" if count == 1 else "servers"
    row = _file_row(name, root, _MCP_JSON, checksums)
    if row.status is Status.SKIP:
        return row
    row.value = f"{count} {noun} · {row.value}"
    return row


def _agents_row(root: Path, checksums: Mapping[str, str] | None) -> CheckResult:
    name = "rendered.agents"
    if checksums is None:
        return _skip(name, "No build manifest to compare the sub-agents against")
    tracked = sorted(rel for rel in checksums if rel.startswith(_AGENTS) and rel.count("/") == 2)
    drifted = [rel for rel in tracked if _differs(root, rel, checksums[rel])]
    matching = len(tracked) - len(drifted)
    if not drifted:
        return CheckResult(
            name,
            CATEGORY,
            Status.OK,
            f"{matching} sub-agent file(s) match the build",
            value=f"{matching} / {len(tracked)} · sha = render",
        )
    return CheckResult(
        name,
        CATEGORY,
        Status.WARNING,
        f"{len(drifted)} sub-agent file(s) differ from the build",
        value=f"{matching} / {len(tracked)} · {len(drifted)} differs from render",
        details="\n".join(drifted),
        fix=_FIX,
    )


def _frontmatter_ok(path: Path) -> bool:
    """Whether a ``SKILL.md`` opens with a YAML frontmatter mapping that parses."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False
    if not text.startswith("---"):
        return False
    parts = text.split("---", 2)
    if len(parts) < 3:
        return False
    try:
        return isinstance(yaml.safe_load(parts[1]), dict)
    except yaml.YAMLError:
        return False


def _skills_row(root: Path) -> CheckResult:
    skills = sorted((root / _CLAUDE / "skills").glob("*/SKILL.md"))
    bad = [p.parent.name for p in skills if not _frontmatter_ok(p)]
    value = f"{len(skills)} · {len(bad)} parse errors"
    if not bad:
        return CheckResult(
            "rendered.skills",
            CATEGORY,
            Status.OK,
            f"{len(skills)} skill(s) with parseable frontmatter",
            value=value,
        )
    return CheckResult(
        "rendered.skills",
        CATEGORY,
        Status.WARNING,
        f"{len(bad)} skill(s) have no parseable YAML frontmatter",
        value=value,
        details="\n".join(bad),
    )
