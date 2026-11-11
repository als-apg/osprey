"""Byte-for-byte pin of the Knowledge surface an operator session is rendered with.

The demo preset, built at its defaults, renders three files that carry Knowledge into
a session: the root operator login's brief (``CLAUDE.md``), its ``facility-knowledge``
agent (``.claude/agents/facility-knowledge.md``), and the brief of the shared
``knowledge`` login, which opens on the Knowledge tab. Each is compared whole with its
golden under ``golden/knowledge_surface/``, so a change anywhere in it — a bullet, a
routing line, the agent list — shows.

Each golden leads with an HTML comment block naming this test and the update rule; the
test strips it and compares the rest with the render byte for byte.

A failure means the operator session reads differently. A later change to a golden is a
declared delta: the PR that makes it names the delta. Regenerate from the build, in the
same change as the template or code that moved it, then read the diff::

    PYTHONPATH=src:.:packages/osprey-connectors/tests ./.venv/bin/python \\
        tests/cli/test_knowledge_surface_golden.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from tests._builds import BuiltProject, build_demo
from tests.cli._lifecycle_build import rendered_config

_GOLDEN_DIR = Path(__file__).parent / "golden" / "knowledge_surface"

#: The comment block every golden leads with; stripped before comparison.
_HEADER = (
    "<!--\n"
    "Golden owned by tests/cli/test_knowledge_surface_golden.py.\n"
    "A change to this file is a declared delta: the PR that makes it names the delta.\n"
    "Regenerate with: PYTHONPATH=src:.:packages/osprey-connectors/tests ./.venv/bin/python\n"
    "tests/cli/test_knowledge_surface_golden.py\n"
    "-->\n"
)

#: Golden path (under the golden directory) for each pinned render.
_ROOT_BRIEF = "root/CLAUDE.md"
_ROOT_AGENT = "root/facility-knowledge.md"
_KNOWLEDGE_BRIEF = "knowledge/CLAUDE.md"


def _renders(built: BuiltProject) -> dict[str, str]:
    """The pinned renders of *built*, keyed by golden path."""
    renders = rendered_config(built.repo)["modules"]["web_terminals"]["renders"]
    knowledge = built.repo / renders["knowledge"]["project_path"]
    return {
        _ROOT_BRIEF: (built.build_dir / "CLAUDE.md").read_text(encoding="utf-8"),
        _ROOT_AGENT: (built.build_dir / ".claude" / "agents" / "facility-knowledge.md").read_text(
            encoding="utf-8"
        ),
        _KNOWLEDGE_BRIEF: (knowledge / "CLAUDE.md").read_text(encoding="utf-8"),
    }


def _golden_body(name: str) -> str:
    """The golden *name* without its header; fails when the header is missing."""
    text = (_GOLDEN_DIR / name).read_text(encoding="utf-8")
    assert text.startswith(_HEADER), f"{name} must lead with the golden header"
    return text[len(_HEADER) :]


@pytest.mark.parametrize("name", [_ROOT_BRIEF, _ROOT_AGENT, _KNOWLEDGE_BRIEF])
def test_render_equals_the_golden(built_demo: BuiltProject, name: str) -> None:
    assert _renders(built_demo)[name] == _golden_body(name)


def _regenerate() -> None:
    """Overwrite every golden from a fresh demo build."""
    with tempfile.TemporaryDirectory() as parent:
        renders = _renders(build_demo(Path(parent)))
    for name, text in sorted(renders.items()):
        path = _GOLDEN_DIR / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_HEADER + text, encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    _regenerate()
