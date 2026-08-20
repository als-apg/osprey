"""Tests for the ``bluesky-plans`` routing skill: install wiring and trigger separation.

Part A mirrors ``test_operating_bluesky_plans_skill_install.py``'s wiring shape
-- catalog registration, the template file, the preset directives, the
regen-tracked-files list -- and then adds the two assertions that are specific
to this skill being a Jinja template rather than a static ``SKILL.md``:

* it renders to the empty string when the Bluesky server is off, and the render
  pass then deletes the file *and its parent directory*, so a disabled build
  must leave no ``.claude/skills/bluesky-plans/`` behind at all;
* its plan listing has to survive into the artifact that actually ships.
  ``regenerate_claude_code`` runs last in a build and rebuilds the context from
  ``config.yml`` alone, so a check that stops at ``build_claude_code_context``
  proves nothing about the installed file. The build fixture here goes through
  the regeneration path and asserts against the rendered ``SKILL.md`` on disk --
  the same reason
  ``tests/cli/templates/test_plan_index_build_wiring.py::test_regenerate_path_carries_plan_index``
  captures that path rather than ``create_project``'s.

Part B is the over-triggering guard, and is the reason this file exists.
``bluesky-plans`` is a broad family name sitting beside two narrower action
skills; if its description re-states their trigger phrases, an agent picking a
skill by description has no way to tell the routing decision apart from the
work. ``operating-bluesky-plans`` keeps the multi-setting-measurement triggers;
``bluesky-plans`` deliberately surrendered them and covers only the routing
decision. The guard pins that separation across *every* bundled skill, not just
these three, so the same collision cannot be introduced from the other side.
"""

import itertools
import re
from pathlib import Path

import pytest
import yaml

from osprey.cli.templates import claude_code, manifest
from osprey.cli.templates.manager import TemplateManager
from osprey.services.build_artifacts.catalog import BuildArtifactCatalog

REPO_ROOT = Path(__file__).parent.parent.parent
TEMPLATE_ROOT = REPO_ROOT / "src" / "osprey" / "templates" / "claude_code"
PRESETS_DIR = REPO_ROOT / "src" / "osprey" / "profiles" / "presets"
SKILLS_DIR = TEMPLATE_ROOT / "claude" / "skills"

CANONICAL_NAME = "skills/bluesky-plans"
SKILL_REL = "claude/skills/bluesky-plans/SKILL.md.j2"
OUTPUT_REL = ".claude/skills/bluesky-plans/SKILL.md"

SKILL_TEMPLATE = SKILLS_DIR / "bluesky-plans" / "SKILL.md.j2"


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------


def _plan_source(name: str, description: str, writes: bool) -> str:
    """A facility plan file's source -- metadata only; the index never imports it."""
    return (
        '"""A plan fixture."""\n\n'
        "PLAN_METADATA = {\n"
        f"    'name': {name!r},\n"
        f"    'description': {description!r},\n"
        f"    'writes': {writes!r},\n"
        "}\n"
    )


def _set_bluesky(project_dir: Path, *, enabled: bool, plan_dir: Path | None = None) -> None:
    """Rewrite the built project's ``config.yml`` the way an operator would.

    Regeneration reads ``config.yml`` and nothing else, so editing it here is
    what a deployment with (or without) the Bluesky server actually looks like
    to the next render pass.
    """
    config_path = project_dir / "config.yml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config.setdefault("claude_code", {}).setdefault("servers", {}).setdefault("bluesky", {})[
        "enabled"
    ] = enabled
    if plan_dir is not None:
        config.setdefault("services", {}).setdefault("bluesky", {})["plan_dir"] = str(plan_dir)
    config_path.write_text(yaml.dump(config), encoding="utf-8")


def _regenerate(manager: TemplateManager, project_dir: Path) -> None:
    claude_code.regenerate_claude_code(manager.template_root, manager.jinja_env, project_dir)


def _build_project(output_dir: Path, name: str) -> tuple[TemplateManager, Path]:
    manager = TemplateManager()
    project_dir = manager.create_project(
        project_name=name,
        output_dir=output_dir,
        data_bundle="control_assistant",
        context={"channel_finder_mode": "hierarchical"},
    )
    return manager, project_dir


@pytest.fixture(scope="module")
def built_project(tmp_path_factory) -> Path:
    """A finished build with Bluesky enabled and one facility plan on disk.

    Module-scoped because the build is the expensive part and every consumer
    below only reads the result. The returned path is the *project* directory,
    so tests assert against the shipped artifact rather than a render context.
    """
    tmp_path = tmp_path_factory.mktemp("bluesky-plans-install")
    plans = tmp_path / "facility_plans"
    plans.mkdir()
    (plans / "sweep.py").write_text(
        _plan_source("facility_sweep", "Sweep the thing.", True), encoding="utf-8"
    )

    manager, project_dir = _build_project(tmp_path, "bluesky-plans-install-test")
    _set_bluesky(project_dir, enabled=True, plan_dir=plans)
    _regenerate(manager, project_dir)
    return project_dir


# ---------------------------------------------------------------------------
# Part A -- install and wiring
# ---------------------------------------------------------------------------


class TestBlueskyPlansRegistry:
    """Wiring point 1: BuildArtifactCatalog registration."""

    def test_registered(self):
        art = BuildArtifactCatalog.default().get(CANONICAL_NAME)
        assert art is not None, f"{CANONICAL_NAME} is not in the build artifact catalog"
        assert art.canonical_name == CANONICAL_NAME
        assert art.template_path == SKILL_REL
        assert art.output_path == OUTPUT_REL


class TestBlueskyPlansTemplateExists:
    """Wiring point 2: the skill bundle template file itself."""

    def test_skill_template_exists(self):
        assert SKILL_TEMPLATE.exists(), f"SKILL.md.j2 not found at {SKILL_TEMPLATE}"

    def test_body_is_wrapped_in_the_enabled_servers_guard(self):
        """The whole template is conditional; that is what makes the empty
        render -- and therefore the directory deletion below -- possible."""
        text = SKILL_TEMPLATE.read_text(encoding="utf-8")
        assert text.startswith('{% if "bluesky" in enabled_servers'), (
            "the template must open with the enabled_servers guard so a build "
            "without the Bluesky server renders nothing"
        )


class TestBlueskyPlansPresetWiring:
    """Wiring point 3: the presets' ``skills:`` and ``exclude:`` directives."""

    def test_control_assistant_lists_the_skill(self):
        profile = yaml.safe_load(
            (PRESETS_DIR / "control-assistant.yml").read_text(encoding="utf-8")
        )
        assert "bluesky-plans" in profile["skills"]

    def test_ariel_preset_excludes_the_skill(self):
        """The ARIEL preset ships without the Bluesky server, so it drops the
        routing skill explicitly rather than inheriting it."""
        profile = yaml.safe_load(
            (PRESETS_DIR / "control-assistant-ariel.yml").read_text(encoding="utf-8")
        )
        assert "bluesky-plans" in profile["exclude"]["skills"]
        assert "bluesky-plans" not in profile.get("skills", [])


class TestBlueskyPlansManifestWiring:
    """Wiring point 4: the regen-tracked-files fallback list."""

    def test_in_regen_tracked_files(self):
        assert OUTPUT_REL in manifest.REGEN_TRACKED_FILES

    def test_resolve_manifest_outputs_includes_the_skill(self):
        outputs = manifest.resolve_manifest_outputs({"artifacts": {"skills": ["bluesky-plans"]}})
        assert OUTPUT_REL in outputs


class TestBlueskyPlansInstall:
    """End-to-end: what the build actually leaves on disk."""

    def test_skill_installs(self, built_project):
        installed = built_project / OUTPUT_REL
        assert installed.exists(), f"Skill not installed at {installed}"

    def test_frontmatter_parses(self, built_project):
        """The rendered file must be a valid skill: parseable YAML frontmatter
        carrying the name Claude Code triggers on, plus its two prose fields."""
        text = (built_project / OUTPUT_REL).read_text(encoding="utf-8")
        assert text.startswith("---"), "the Jinja guard must not leak into the rendered file"

        front = yaml.safe_load(_frontmatter_block(text))
        assert front["name"] == "bluesky-plans"
        assert isinstance(front["description"], str) and front["description"].strip()
        assert isinstance(front["summary"], str) and front["summary"].strip()

    def test_installed_skill_carries_the_facility_plan_row(self, built_project):
        """The listing has to reach the *shipped* file.

        ``regenerate_claude_code`` rebuilds the render context from
        ``config.yml`` and runs last, so anything handed only to
        ``create_project`` is overwritten. Asserting on the file this build left
        behind is the only check that covers what an operator would read.
        """
        text = (built_project / OUTPUT_REL).read_text(encoding="utf-8")

        assert "facility_sweep" in text, "the facility plan is missing from the shipped listing"
        row = next(line for line in text.splitlines() if "facility_sweep" in line)
        assert "Sweep the thing." in row, "the plan's description must reach the row"
        assert "| yes |" in row, "a writing plan must be marked as moving the machine"
        assert "facility" in row, "the row must carry the trust tier the file was found in"

    def test_disabled_bluesky_leaves_no_skill_directory(self, tmp_path):
        """Off means gone -- the file *and* the directory that held it.

        The render pass deletes an empty rendered file and then removes the
        parent directory if that leaves it empty. Asserting only that the file
        is absent would pass on a build that stranded an empty
        ``.claude/skills/bluesky-plans/`` in every non-Bluesky deployment, which
        reads to an operator as a skill that exists and says nothing. The
        enabled render runs first so the absence afterwards is a deletion, not a
        file that was never written.
        """
        manager, project_dir = _build_project(tmp_path, "bluesky-plans-disabled-test")
        skill_dir = project_dir / ".claude" / "skills" / "bluesky-plans"

        _set_bluesky(project_dir, enabled=True)
        _regenerate(manager, project_dir)
        assert (skill_dir / "SKILL.md").exists(), (
            "the enabled render must produce the skill, or the disabled case below is vacuous"
        )

        _set_bluesky(project_dir, enabled=False)
        _regenerate(manager, project_dir)

        assert not skill_dir.exists(), (
            f"a build without the Bluesky server left {skill_dir} behind; the empty "
            f"render must take the parent directory with it"
        )


# ---------------------------------------------------------------------------
# Part B -- the over-triggering guard
# ---------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(r"\A---\r?\n(.*?)\r?\n---\s*$", re.DOTALL | re.MULTILINE)
_LEADING_JINJA_RE = re.compile(r"\A(?:\s*\{%.*?%\}\s*)+", re.DOTALL)
_WORD_RE = re.compile(r"[a-z0-9]+")

SHINGLE_LENGTH = 6


def _frontmatter_block(text: str) -> str:
    """The YAML between the opening and closing ``---`` fences.

    A bundled ``.md.j2`` may open with a Jinja guard on the same line as the
    fence (``bluesky-plans`` does), so any leading tags are stripped first. The
    frontmatter itself sits before any body branching in every bundled skill,
    which is why it can be read as plain YAML without rendering the template.
    """
    stripped = _LEADING_JINJA_RE.sub("", text)
    match = _FRONTMATTER_RE.search(stripped)
    assert match is not None, "no YAML frontmatter fence found"
    return match.group(1)


def _bundled_skill_descriptions() -> dict[str, str]:
    """Every bundled skill's frontmatter ``description``, keyed by skill name."""
    descriptions = {}
    for path in sorted(SKILLS_DIR.glob("*/SKILL.md*")):
        if path.suffix not in (".md", ".j2"):
            continue
        front = yaml.safe_load(_frontmatter_block(path.read_text(encoding="utf-8")))
        descriptions[front["name"]] = front["description"]
    return descriptions


def _shingles(description: str, length: int = SHINGLE_LENGTH) -> set[str]:
    """Normalized word n-grams of a description.

    Normalization is lowercasing plus discarding everything that is not a
    letter or digit, then re-joining on single spaces. That folds away the
    things two descriptions can differ by without differing as *triggers* --
    case, punctuation, backticks, em dashes, and the line wrapping a folded
    YAML block introduces -- so a phrase copied between two skills is still
    caught after a re-wrap or a comma. It deliberately does not stem or drop
    stop words: a shared six-word run of ordinary English is exactly the kind
    of phrase that makes two skills look interchangeable to a model choosing
    between them.
    """
    words = _WORD_RE.findall(description.lower())
    return {" ".join(words[i : i + length]) for i in range(len(words) - length + 1)}


class TestBundledSkillDescriptionsDoNotCollide:
    """No trigger phrase may be shared by two bundled skills.

    ``bluesky-plans`` is a family name covering a subsystem that two narrower
    skills already act on. A model selects a skill by reading descriptions, so
    a phrase appearing in two of them makes the choice arbitrary at exactly the
    point where it matters: whether to route, to author, or to run something on
    the machine. Six words is the threshold because shorter runs collide by
    coincidence in any set of descriptions about the same subsystem, while a
    six-word run is a copied phrase.
    """

    @pytest.fixture(scope="class")
    def descriptions(self) -> dict[str, str]:
        return _bundled_skill_descriptions()

    def test_the_scan_sees_the_bluesky_family(self, descriptions):
        """Non-vacuity: the guard below is meaningless if the scan finds
        nothing, or if a description is too short to yield any shingle."""
        for name in ("bluesky-plans", "operating-bluesky-plans", "writing-bluesky-plans"):
            assert name in descriptions, f"{name} was not picked up by the skill scan"
            assert _shingles(descriptions[name]), (
                f"{name}'s description is shorter than {SHINGLE_LENGTH} words, so it "
                f"cannot collide with anything -- the guard would pass vacuously"
            )

    def test_no_shared_trigger_phrase(self, descriptions):
        shingles = {name: _shingles(text) for name, text in descriptions.items()}

        collisions = []
        for first, second in itertools.combinations(sorted(shingles), 2):
            shared = sorted(shingles[first] & shingles[second])
            if shared:
                collisions.append(f"{first} <-> {second}: {shared}")

        assert not collisions, (
            f"bundled skill descriptions share {SHINGLE_LENGTH}-word trigger phrases, "
            f"so a model choosing between them has nothing to go on:\n" + "\n".join(collisions)
        )
