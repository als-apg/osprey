"""The ``rendered`` health category compares the render against its build manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from osprey.build.manifest import MANIFEST_FILENAME
from osprey.health.core.rendered import rendered
from osprey.health.models import Status

_SKILL = "---\nname: {name}\ndescription: does {name}\n---\nbody\n"


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write(root: Path, rel: str, text: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _render(root: Path) -> Path:
    """A small render: settings, two agents, two skills, one hook, .mcp.json."""
    _write(root, ".claude/settings.json", json.dumps({"hooks": {}}))
    _write(root, ".claude/agents/a.md", "---\nname: a\n---\nA\n")
    _write(root, ".claude/agents/b.md", "---\nname: b\n---\nB\n")
    _write(root, ".claude/skills/one/SKILL.md", _SKILL.format(name="one"))
    _write(root, ".claude/skills/two/SKILL.md", _SKILL.format(name="two"))
    _write(root, ".claude/hooks/osprey_limits.py", "print('limits')\n")
    _write(root, ".mcp.json", json.dumps({"mcpServers": {"controls": {}, "workspace": {}}}))
    return root


def _stamp(root: Path) -> None:
    """Write the manifest the build would: a checksum of every file as it is now."""
    checksums = {
        str(p.relative_to(root)): _sha(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != MANIFEST_FILENAME
    }
    (root / MANIFEST_FILENAME).write_text(json.dumps({"file_checksums": checksums}))


def _rows(root: Path | None) -> dict:
    return {row.name: row for row in rendered({}, cwd=root)()}


def test_no_rows_without_a_render_dir(tmp_path):
    assert rendered({}, cwd=None)() == []
    assert rendered({}, cwd=tmp_path)() == []


def test_a_render_matching_its_manifest(tmp_path):
    _stamp(_render(tmp_path))
    rows = _rows(tmp_path)

    assert rows["rendered.claude_dir"].status is Status.OK
    assert rows["rendered.claude_dir"].value == "0 / 6 files differ"
    assert rows["rendered.settings_json"].value == "= build"
    assert rows["rendered.mcp_json"].value == "2 servers · = build"
    assert rows["rendered.agents"].value == "2 / 2 · sha = render"
    assert rows["rendered.skills"].value == "2 · 0 parse errors"
    assert all(row.status is Status.OK for row in rows.values())


def test_an_edited_file_is_drift_with_the_build_as_fix(tmp_path):
    _stamp(_render(tmp_path))
    _write(tmp_path, ".claude/agents/b.md", "---\nname: b\n---\nedited\n")
    _write(tmp_path, ".claude/settings.json", "{}")
    rows = _rows(tmp_path)

    drift = rows["rendered.claude_dir"]
    assert drift.status is Status.WARNING
    assert drift.value == "2 / 6 files differ"
    assert drift.fix == "osprey build"
    assert ".claude/agents/b.md" in drift.details
    assert rows["rendered.settings_json"].status is Status.WARNING
    assert rows["rendered.settings_json"].value == "differs from build"
    assert rows["rendered.agents"].status is Status.WARNING
    assert rows["rendered.agents"].value == "1 / 2 · 1 differs from render"


def test_a_deleted_tracked_file_counts_as_drift(tmp_path):
    _stamp(_render(tmp_path))
    (tmp_path / ".claude/hooks/osprey_limits.py").unlink()

    assert _rows(tmp_path)["rendered.claude_dir"].value == "1 / 6 files differ"


def test_mcp_json_drift(tmp_path):
    _stamp(_render(tmp_path))
    _write(tmp_path, ".mcp.json", json.dumps({"mcpServers": {"controls": {}}}))
    row = _rows(tmp_path)["rendered.mcp_json"]

    assert row.status is Status.WARNING
    assert row.value == "1 server · differs from build"


def test_skill_frontmatter_errors_warn(tmp_path):
    _stamp(_render(tmp_path))
    _write(tmp_path, ".claude/skills/bad/SKILL.md", "---\nname: [unclosed\n---\n")
    _write(tmp_path, ".claude/skills/bare/SKILL.md", "no frontmatter\n")
    row = _rows(tmp_path)["rendered.skills"]

    assert row.status is Status.WARNING
    assert row.value == "4 · 2 parse errors"
    assert "bad" in row.details and "bare" in row.details


def test_without_a_manifest_comparisons_skip_and_counts_still_report(tmp_path):
    _render(tmp_path)
    rows = _rows(tmp_path)

    for name in ("rendered.claude_dir", "rendered.settings_json", "rendered.mcp_json"):
        assert rows[name].status is Status.SKIP
        assert rows[name].value == "no build manifest"
    assert rows["rendered.agents"].status is Status.SKIP
    assert rows["rendered.skills"].value == "2 · 0 parse errors"


def test_unreadable_mcp_json_warns(tmp_path):
    _stamp(_render(tmp_path))
    _write(tmp_path, ".mcp.json", "{not json")
    row = _rows(tmp_path)["rendered.mcp_json"]

    assert row.status is Status.WARNING
    assert "unreadable" in row.value


def test_a_graph_agent_baked_by_the_seeder_is_not_drift(tmp_path):
    from osprey.services.facility_knowledge.seeder.prompt_snapshot import (
        CHANNEL_FINDER_FILENAME,
        SNAPSHOT_BEGIN,
        SNAPSHOT_END,
    )

    rel = f".claude/agents/{CHANNEL_FINDER_FILENAME}"
    _render(tmp_path)
    _write(tmp_path, rel, f"prompt\n{SNAPSHOT_BEGIN}\nplaceholder\n{SNAPSHOT_END}\n")
    _stamp(tmp_path)
    # `osprey up` rewrites the region marker to marker after the build.
    _write(tmp_path, rel, f"prompt\n{SNAPSHOT_BEGIN}\nlabels: Device\n{SNAPSHOT_END}\n")
    rows = _rows(tmp_path)

    assert rows["rendered.claude_dir"].status is Status.OK
    assert rows["rendered.agents"].status is Status.OK


def test_a_graph_agent_with_its_markers_removed_is_drift(tmp_path):
    from osprey.services.facility_knowledge.seeder.prompt_snapshot import (
        CHANNEL_FINDER_FILENAME,
        SNAPSHOT_BEGIN,
        SNAPSHOT_END,
    )

    rel = f".claude/agents/{CHANNEL_FINDER_FILENAME}"
    _render(tmp_path)
    _write(tmp_path, rel, f"prompt\n{SNAPSHOT_BEGIN}\nplaceholder\n{SNAPSHOT_END}\n")
    _stamp(tmp_path)
    _write(tmp_path, rel, "prompt edited by hand\n")

    assert _rows(tmp_path)["rendered.agents"].status is Status.WARNING
