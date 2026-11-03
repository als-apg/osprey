"""The bundled CANOPY skills are CANOPY's own, byte for byte.

CANOPY ships its skills as package data (``canopy_mcp.skill.SKILLS`` maps each
name to its file) and is their single source. OSPREY carries a copy of each
under the Claude Code templates so a build can render it into every project
that shows the CANOPY panel, without CANOPY installed on the build host. The
copies are never edited: these tests compare each with the installed package's
bytes, so a CANOPY release that changes a skill fails here until the file is
copied again, and a CANOPY release that adds a skill fails here until OSPREY
carries it.

The build writes a template that is not a ``.j2`` file with ``shutil.copy2``,
so each catalog entry is checked to point at the plain file: a rendered
template would no longer be the same bytes.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from canopy_mcp import skill

from osprey.agent_runner.build_artifacts.catalog import BuildArtifactCatalog
from osprey.cli.templates.artifact_library import list_artifacts

CLAUDE_TEMPLATES = (
    Path(__file__).resolve().parents[3] / "src" / "osprey" / "templates" / "claude_code" / "claude"
)
SKILLS_DIR = CLAUDE_TEMPLATES / "skills"

#: The file name every skill has in a project, whatever its source path in CANOPY.
INSTALLED_FILE = "SKILL.md"

SKILL_NAMES = sorted(skill.SKILLS)


def _bundled(name: str) -> Path:
    return SKILLS_DIR / name / INSTALLED_FILE


def test_osprey_carries_the_skills_canopy_ships() -> None:
    assert SKILL_NAMES == ["canopy-operator", "canopy-panel-builder"]


@pytest.mark.parametrize("name", SKILL_NAMES)
def test_bundled_skill_equals_the_installed_package_data(name: str) -> None:
    assert _bundled(name).read_bytes() == skill.skill_bytes(name)


@pytest.mark.parametrize("name", SKILL_NAMES)
def test_skill_directory_is_named_as_canopy_names_the_skill(name: str) -> None:
    assert name in list_artifacts("skills")
    assert skill.skill_path(Path("project"), name).parent.name == _bundled(name).parent.name


@pytest.mark.parametrize("name", SKILL_NAMES)
def test_the_skill_directory_holds_nothing_else(name: str) -> None:
    """A second file, or a ``.j2`` twin, would render beside or over the copy."""
    assert [entry.name for entry in _bundled(name).parent.iterdir()] == [INSTALLED_FILE]


@pytest.mark.parametrize("name", SKILL_NAMES)
def test_catalog_copies_the_file_and_does_not_render_it(name: str) -> None:
    artifact = BuildArtifactCatalog.default().get(f"skills/{name}")
    assert artifact is not None
    assert artifact.template_root == "claude_code"
    assert CLAUDE_TEMPLATES.parent / artifact.template_path == _bundled(name)
    assert not artifact.template_path.endswith(".j2")
    assert artifact.output_path == f".claude/skills/{name}/{INSTALLED_FILE}"


def test_a_project_gets_the_paths_canopy_installs_to(tmp_path: Path) -> None:
    """``python -m canopy_mcp.skill install`` and a build agree on every path."""
    catalog = BuildArtifactCatalog.default()
    installed = skill.install(tmp_path)
    assert len(installed) == len(skill.SKILLS)
    for name, path in zip(skill.SKILLS, installed, strict=True):
        artifact = catalog.get(f"skills/{name}")
        assert artifact is not None
        assert path == tmp_path / artifact.output_path
        assert path.read_bytes() == _bundled(name).read_bytes()
