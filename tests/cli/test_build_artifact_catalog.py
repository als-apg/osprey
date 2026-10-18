"""Tests for BuildArtifactCatalog — the declarative catalog of prompt artifacts."""

import ast
import importlib.util
from pathlib import Path

import pytest

from osprey.agent_runner.build_artifacts.catalog import (
    DEFAULT_CLAUDE_MD_TEMPLATE,
    INSTRUCTIONS_OUTPUT,
    LONE_SCRIPTS,
    BuildArtifact,
    BuildArtifactCatalog,
    LoneScript,
)


class TestBuildArtifact:
    """Basic BuildArtifact dataclass tests."""

    def test_frozen(self):
        art = BuildArtifact("a", "b", "c", "d")
        with pytest.raises(AttributeError):
            art.canonical_name = "x"

    def test_fields(self):
        art = BuildArtifact("a", "b", "c", "d")
        assert art.canonical_name == "a"
        assert art.template_path == "b"
        assert art.output_path == "c"
        assert art.description == "d"


class TestBuildArtifactCatalogDefault:
    """Tests for the default registry contents."""

    @pytest.fixture()
    def registry(self):
        return BuildArtifactCatalog.default()

    def test_has_artifacts(self, registry):
        assert len(registry.all_artifacts()) > 0

    def test_get_known_artifact(self, registry):
        art = registry.get("claude-md")
        assert art is not None
        assert art.output_path == "CLAUDE.md"
        assert art.template_path == "CLAUDE.md.j2"

    def test_get_unknown_returns_none(self, registry):
        assert registry.get("nonexistent/artifact") is None

    def test_all_names_sorted(self, registry):
        names = registry.all_names()
        assert names == sorted(names)

    def test_all_names_nonempty(self, registry):
        assert len(registry.all_names()) > 0

    def test_get_by_output(self, registry):
        art = registry.get_by_output("CLAUDE.md")
        assert art is not None
        assert art.canonical_name == "claude-md"

    def test_get_by_output_unknown(self, registry):
        assert registry.get_by_output("nonexistent.md") is None

    # ── Known artifacts ────────────────────────────────────────────
    @pytest.mark.parametrize(
        "name",
        [
            "claude-md",
            "mcp-json",
            "settings-json",
            "agents/channel-finder",
            "agents/data-visualizer",
            "agents/logbook-search",
            "agents/logbook-deep-research",
            "agents/pyat-specialist",
            "rules/safety",
            "rules/error-handling",
            "rules/artifacts",
            "rules/facility",
            "hooks/approval",
            "hooks/writes-check",
            "hooks/error-guidance",
            "hooks/limits",
            "hooks/notebook-update",
            "hooks/cf-feedback-capture",
            "hooks/hook-log",
            "hooks/target-state",
            "hooks/safety-file",
            "hooks/config-drift",
            "skills/session-report",
            "skills/session-report/reference",
            "skills/diagnose",
            "skills/setup-mode",
            "output-styles/control-operator",
        ],
    )
    def test_known_artifact_exists(self, registry, name):
        art = registry.get(name)
        assert art is not None, f"Missing artifact: {name}"
        assert art.canonical_name == name

    def test_categories_derived_from_canonical_names(self, registry):
        cats = registry.categories
        assert isinstance(cats, set)
        assert "agents" in cats
        assert "rules" in cats
        assert "hooks" in cats
        assert "skills" in cats
        assert "commands" not in cats
        assert "output-styles" in cats
        assert "config" in cats  # top-level artifacts without "/"

    def test_facility_exists(self, registry):
        art = registry.get("rules/facility")
        assert art is not None
        assert art.description == "Facility identity & context"


class TestRegistryMatchesTemplateDirectory:
    """Verify that the registry matches the actual template files on disk."""

    def test_all_template_paths_exist(self):
        """Every artifact's template_path must exist under its template_root."""
        registry = BuildArtifactCatalog.default()
        templates_dir = Path(__file__).parent.parent.parent / "src" / "osprey" / "templates"

        for art in registry.all_artifacts():
            template_file = templates_dir / art.template_root / art.template_path
            assert template_file.exists(), (
                f"Template {art.template_path} for artifact "
                f"'{art.canonical_name}' not found at {template_file}"
            )
            if art.is_directory:
                assert template_file.is_dir(), (
                    f"Directory artifact '{art.canonical_name}' must point at a directory"
                )

    def test_no_unregistered_templates(self):
        """All files in the template directory should be registered.

        Exemptions: __pycache__, .pyc, directories-only, __init__.py, partials
        under an underscore-prefixed directory (``_terminology``, ``_shared`` —
        included by the agent templates, never rendered on their own; the same
        rule the renderer applies), underscore-prefixed files at the template
        root (``_facility_facts.md.j2`` — the facts view renders it into
        ``data/facility_facts.md``, never into ``.claude/``), and
        ``web-terminal-context/`` — the web-terminal persona baseline. The
        catalog governs artifacts rendered into ``.claude/`` and claimable via
        ``osprey scaffold claim``; base.md
        is copied verbatim to ``docker/web-terminal-context/`` for deploy-time
        seeding and never passes through the override machinery, so a catalog
        entry for it would advertise a claim that the build ignores.
        """
        registry = BuildArtifactCatalog.default()
        template_root = (
            Path(__file__).parent.parent.parent / "src" / "osprey" / "templates" / "claude_code"
        )

        registered_templates = {a.template_path for a in registry.all_artifacts()}

        for template_file in template_root.rglob("*"):
            if not template_file.is_file():
                continue
            if "__pycache__" in str(template_file):
                continue
            if template_file.name == "__init__.py":
                continue
            rel_path = template_file.relative_to(template_root)
            if any(part.startswith("_") for part in rel_path.parts[:-1]):
                continue
            if rel_path.name.startswith("_"):
                continue
            if "web-terminal-context" in rel_path.parts:
                continue

            rel = str(rel_path)
            assert rel in registered_templates, (
                f"Template file {rel} is not registered in the BuildArtifactCatalog"
            )


class TestCustomRegistry:
    """Tests for constructing a custom registry."""

    def test_empty_registry(self):
        reg = BuildArtifactCatalog([])
        assert reg.all_artifacts() == []
        assert reg.all_names() == []
        assert reg.get("anything") is None

    def test_custom_artifacts(self):
        art = BuildArtifact("my/thing", "my.j2", "my.md", "My thing")
        reg = BuildArtifactCatalog([art])
        assert reg.get("my/thing") is art
        assert reg.all_names() == ["my/thing"]


_PERSONAS = [
    ("CLAUDE.md.j2", "claude-md"),
    ("CLAUDE.ariel.md.j2", "claude-md-ariel"),
    ("CLAUDE.knowledge.md.j2", "claude-md-knowledge"),
    ("CLAUDE.channel-finder.md.j2", "claude-md-channel-finder"),
]


class TestInstructionsPersona:
    """CLAUDE.md resolves to the persona the catalog is built for."""

    def test_the_default_persona_is_the_control_system_instructions(self):
        catalog = BuildArtifactCatalog.default()
        art = catalog.get_by_output("CLAUDE.md")
        assert art is not None
        assert art.canonical_name == "claude-md"
        assert catalog.claude_md_template == DEFAULT_CLAUDE_MD_TEMPLATE

    @pytest.mark.parametrize(("template", "name"), _PERSONAS)
    def test_each_bundled_persona_resolves_claude_md(self, template, name):
        catalog = BuildArtifactCatalog.default(claude_md_template=template)
        art = catalog.get_by_output(INSTRUCTIONS_OUTPUT)
        assert art is not None
        assert art.canonical_name == name
        for _, persona in _PERSONAS:
            assert catalog.get(persona) is not None

    def test_an_unknown_persona_is_refused_by_name(self):
        with pytest.raises(ValueError, match=r"CLAUDE\.nope\.md\.j2") as excinfo:
            BuildArtifactCatalog.default(claude_md_template="CLAUDE.nope.md.j2")
        assert "CLAUDE.ariel.md.j2" in str(excinfo.value)

    def test_a_second_artifact_on_one_output_is_refused(self):
        first = BuildArtifact("my/first", "first.j2", "shared.md", "First")
        second = BuildArtifact("my/second", "second.j2", "shared.md", "Second")
        with pytest.raises(ValueError) as excinfo:
            BuildArtifactCatalog([first, second])
        assert "my/first" in str(excinfo.value)
        assert "my/second" in str(excinfo.value)

    def test_a_catalog_with_no_persona_takes_the_default_argument(self):
        catalog = BuildArtifactCatalog([BuildArtifact("my/thing", "my.j2", "my.md", "x")])
        assert catalog.get_by_output("CLAUDE.md") is None

    def test_only_the_instructions_output_is_shared(self):
        counts: dict[str, int] = {}
        for art in BuildArtifactCatalog.default().all_artifacts():
            counts[art.output_path] = counts.get(art.output_path, 0) + 1
        shared = {path: n for path, n in counts.items() if n > 1}
        assert shared == {INSTRUCTIONS_OUTPUT: 4}


# ---------------------------------------------------------------------------
# Hook helpers: each hook row declares the sibling files it needs
# ---------------------------------------------------------------------------

_TEMPLATES_DIR = Path(__file__).parent.parent.parent / "src" / "osprey" / "templates"


def _hook_rows() -> list[BuildArtifact]:
    """Every catalog row rendered into ``.claude/hooks/``."""
    return [
        art
        for art in BuildArtifactCatalog.default().all_artifacts()
        if art.template_root == "claude_code" and art.template_path.startswith("claude/hooks/")
    ]


def _lone_script_source(script: LoneScript) -> Path:
    """The file a lone script's module is loaded from."""
    origin = importlib.util.find_spec(script.module).origin
    assert origin is not None
    return Path(origin)


def _derived_requires(source: Path, own_name: str) -> set[str]:
    """The sibling rows the file at *source* imports or names, read from its AST."""
    siblings = _hook_rows()
    by_module = {
        Path(art.output_path).stem: art.canonical_name
        for art in siblings
        if art.output_path.endswith(".py")
    }
    by_file = {Path(art.output_path).name: art.canonical_name for art in siblings}

    tree = ast.parse(source.read_text())
    heads: set[str] = set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            heads.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            heads.add(node.module.split(".")[0])
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            names.add(node.value)
    heads.discard("osprey_connectors")

    derived = {by_module[head] for head in heads if head in by_module}
    derived |= {by_file[name] for name in names if name in by_file}
    derived.discard(own_name)
    return derived


def _requiring_files() -> list[tuple[str, Path, tuple[str, ...]]]:
    """Every file that declares its siblings: the Python hook rows and the lone scripts."""
    hooks = [
        (art.canonical_name, _TEMPLATES_DIR / art.template_root / art.template_path, art.requires)
        for art in _hook_rows()
        if art.template_path.endswith(".py")
    ]
    lone = [(script.name, _lone_script_source(script), script.requires) for script in LONE_SCRIPTS]
    return hooks + lone


@pytest.mark.parametrize(
    ("name", "source", "requires"),
    _requiring_files(),
    ids=[entry[0] for entry in _requiring_files()],
)
def test_each_file_requires_exactly_the_siblings_it_uses(
    name: str, source: Path, requires: tuple[str, ...]
):
    derived = _derived_requires(source, name)
    assert set(requires) == derived, (
        f"{name} ({source.name}) declares requires {sorted(requires)} but its file uses "
        f"{sorted(derived)}; set requires on that entry to {sorted(derived)}"
    )


def test_the_closure_is_transitive():
    closure = BuildArtifactCatalog.default().closure({"hooks/approval"})
    assert "hooks/approval" in closure
    assert "hooks/hook-config" in closure


def test_helpers_are_the_rows_something_requires():
    assert BuildArtifactCatalog.default().helpers() == {
        "hooks/hook-config",
        "hooks/hook-log",
        "hooks/safety-file",
        "hooks/target-state",
    }


def test_a_requires_entry_naming_no_row_is_refused():
    hook = BuildArtifact("hooks/mine", "mine.py", "mine.py", "x", requires=("hooks/nope",))
    with pytest.raises(ValueError) as excinfo:
        BuildArtifactCatalog([hook])
    assert "hooks/mine" in str(excinfo.value)
    assert "hooks/nope" in str(excinfo.value)


def test_a_lone_script_has_its_closure_beside_it():
    files = BuildArtifactCatalog.default().files_beside("control-identity", _TEMPLATES_DIR)
    assert set(files) == {"control_identity.py", "osprey_safety_file.py"}
    assert files["control_identity.py"] == _lone_script_source(LONE_SCRIPTS[0])
    assert files["osprey_safety_file.py"] == (
        _TEMPLATES_DIR / "claude_code" / "claude" / "hooks" / "osprey_safety_file.py"
    )
    assert all(path.is_file() for path in files.values())


def test_a_lone_script_requiring_no_row_is_refused():
    script = LoneScript("mine", "osprey.deployment.control_identity", ("hooks/nope",))
    with pytest.raises(ValueError) as excinfo:
        BuildArtifactCatalog([], lone_scripts=(script,))
    assert "mine" in str(excinfo.value)
    assert "hooks/nope" in str(excinfo.value)


def test_a_lone_script_whose_closure_reaches_a_template_is_refused():
    config = BuildArtifact("hooks/config", "config.json.j2", "config.json", "x")
    log = BuildArtifact("hooks/log", "log.py", "log.py", "x", requires=("hooks/config",))
    script = LoneScript("mine", "osprey.deployment.control_identity", ("hooks/log",))
    with pytest.raises(ValueError) as excinfo:
        BuildArtifactCatalog([config, log], lone_scripts=(script,))
    assert "mine" in str(excinfo.value)
    assert "hooks/config" in str(excinfo.value)
