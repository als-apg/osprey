"""The deployment-root tables hold these exact values.

Each table that names an entry at the deployment root is pinned here as a literal,
so a change to any of them, order included where the table is ordered, is a
deliberate edit to this file and never a side effect of restructuring the code
that derives it. Unordered tables are compared as sets; mappings are compared as
ordered item sequences; ``CONVENTION_DIRS`` rows are compared field by field on
the fields named here, so a field added to the row type does not break the pin.
"""

from __future__ import annotations

from osprey.cli import build_cmd, init_cmd, profile_cmd, profile_conventions
from osprey.mcp_server.python_executor import executor
from osprey.utils import workspace

# --- profile_conventions --------------------------------------------------------


def test_source_zone_entries() -> None:
    assert set(profile_conventions._SOURCE_ZONE_ENTRIES) == {
        ".dockerignore",
        ".osprey",
        "apps",
        "ci-extra.yml",
        "data",
        "osprey.service",
        "personas",  # retired-spelling
        "profile.yml",
        "profiles",
        "providers.yml",
        "scripts",
        "tiers",
        "triggers.yml",
        "watchdogs",
    }


def test_secrets_zone_entries() -> None:
    assert set(profile_conventions._SECRETS_ZONE_ENTRIES) == {".env", ".env.example"}


def test_generated_zone_entries() -> None:
    assert set(profile_conventions._GENERATED_ZONE_ENTRIES) == {"build", "var"}


def test_repo_claude_code_entries() -> None:
    assert set(profile_conventions.REPO_CLAUDE_CODE_ENTRIES) == {
        ".claude",
        ".mcp.json",
        "CLAUDE.local.md",
        "CLAUDE.md",
    }


def test_known_root_entries() -> None:
    assert set(profile_conventions.KNOWN_ROOT_ENTRIES) == {
        ".claude",
        ".dockerignore",
        ".env",
        ".env.example",
        ".mcp.json",
        ".osprey",
        "CLAUDE.local.md",
        "CLAUDE.md",
        "agents",
        "apps",
        "build",
        "ci-extra.yml",
        "commands",
        "data",
        "hooks",
        "mcp_servers",
        "osprey.service",
        "output-styles",
        "personas",  # retired-spelling
        "profile.yml",
        "profiles",
        "project",
        "providers.yml",
        "rules",
        "scripts",
        "services",
        "skills",
        "tiers",
        "triggers.yml",
        "var",
        "watchdogs",
        "web-terminal-context",
    }


def test_convention_sources() -> None:
    assert tuple(profile_conventions.CONVENTION_SOURCES) == (
        "rules",
        "skills",
        "agents",
        "commands",
        "output-styles",
        "hooks",
        "web-terminal-context",
        "mcp_servers",
        "services",
        "project",
    )


def test_selected_convention_sources() -> None:
    assert tuple(profile_conventions.SELECTED_CONVENTION_SOURCES.items()) == (
        ("rules", "rules"),
        ("skills", "skills"),
        ("agents", "agents"),
        ("output_styles", "output-styles"),
        ("hooks", "hooks"),
    )


_CONVENTION_DIR_ROWS: tuple[dict[str, object], ...] = (
    {
        "source": "rules",
        "destination": ".claude/rules",
        "shape": "markdown",
        "entry_noun": "rule",
        "description": "Agent behavior rules",
        "per_user": False,
        "list_field": "rules",
    },
    {
        "source": "skills",
        "destination": ".claude/skills",
        "shape": "directory",
        "entry_noun": "skill",
        "description": "Agent skills (one directory per skill)",
        "per_user": False,
        "list_field": "skills",
    },
    {
        "source": "agents",
        "destination": ".claude/agents",
        "shape": "markdown",
        "entry_noun": "agent",
        "description": "Subagent definitions",
        "per_user": False,
        "list_field": "agents",
    },
    {
        "source": "commands",
        "destination": ".claude/commands",
        "shape": "markdown",
        "entry_noun": "command",
        "description": "Slash commands",
        "per_user": False,
        "list_field": None,
    },
    {
        "source": "output-styles",
        "destination": ".claude/output-styles",
        "shape": "markdown",
        "entry_noun": "output style",
        "description": "Output styles",
        "per_user": False,
        "list_field": "output_styles",
    },
    {
        "source": "hooks",
        "destination": ".claude/hooks",
        "shape": "file",
        "entry_noun": "hook",
        "description": "Claude Code hook scripts",
        "per_user": False,
        "list_field": "hooks",
    },
    {
        "source": "web-terminal-context",
        "destination": "docker/web-terminal-context",
        "shape": "directory",
        "entry_noun": "web-terminal user",
        "description": "Per-user web-terminal context (one directory per user, plus a shared base.md)",
        "per_user": True,
        "list_field": None,
    },
    {
        "source": "mcp_servers",
        "destination": "_mcp_servers",
        "shape": "directory",
        "entry_noun": "MCP server",
        "description": "Facility MCP server sources (one directory per server)",
        "per_user": False,
        "list_field": None,
    },
    {
        "source": "services",
        "destination": "services",
        "shape": "directory",
        "entry_noun": "service",
        "description": "Facility-owned compose services (one directory per service)",
        "per_user": False,
        "list_field": None,
    },
    {
        "source": "project",
        "destination": "",
        "shape": "mirror",
        "entry_noun": "file",
        "description": "Verbatim mirror onto the project root (escape hatch)",
        "per_user": False,
        "list_field": None,
    },
)


def test_convention_dirs_field_by_field() -> None:
    rows = profile_conventions.CONVENTION_DIRS
    assert len(rows) == len(_CONVENTION_DIR_ROWS)
    for row, expected in zip(rows, _CONVENTION_DIR_ROWS, strict=True):
        for field, value in expected.items():
            actual = getattr(row, field)
            if field == "shape":
                actual = getattr(actual, "value", actual)
            assert actual == value, (expected["source"], field, actual)


# --- profile_cmd ----------------------------------------------------------------


def test_materialized_source_entries() -> None:
    assert tuple(profile_cmd.MATERIALIZED_SOURCE_ENTRIES) == (
        "profile.yml",
        "providers.yml",
        "data",
        "tiers",
        "apps",
        ".osprey",
        "triggers.yml",
        "web-terminal-context",
        ".env.example",
        "watchdogs",
    )


# --- init_cmd -------------------------------------------------------------------


def test_write_once_files_keys() -> None:
    assert tuple(init_cmd.WRITE_ONCE_FILES) == (
        ".gitignore",
        ".env.shared",
        "README.md",
        "ci-extra.yml",
    )


def test_write_once_dirs_keys() -> None:
    assert tuple(init_cmd.WRITE_ONCE_DIRS) == ("mcp_servers",)


def test_ci_emitted_paths() -> None:
    assert tuple(init_cmd.CI_EMITTED_PATHS) == (".gitlab-ci.yml", "scripts/verify.sh")


def test_facility_rule_dir() -> None:
    assert init_cmd.FACILITY_RULE_DIR == "rules"


def test_preserved_by_force() -> None:
    assert tuple(init_cmd.PRESERVED_BY_FORCE) == (
        ".env",
        ".git",
        "var/",
        "build/",
        ".gitignore",
        ".env.shared",
        "README.md",
        "ci-extra.yml",
        ".gitlab-ci.yml",
        "scripts/verify.sh",
        "mcp_servers/",
        "rules/",
    )


# --- build_cmd ------------------------------------------------------------------


def test_non_source_root_entries() -> None:
    assert set(build_cmd._NON_SOURCE_ROOT_ENTRIES) == {
        ".git",
        ".osprey-compose.yml",
        "build",
        "var",
    }


# --- workspace ------------------------------------------------------------------


def test_layer_dir_kinds() -> None:
    assert tuple(workspace.LAYER_DIR_KINDS.items()) == (("tiers", "tier"),)


def test_retired_layer_dirs() -> None:
    assert tuple(workspace.RETIRED_LAYER_DIRS.items()) == (
        ("personas", "tiers"),  # retired-spelling
    )


def test_layer_dirnames() -> None:
    assert tuple(workspace.LAYER_DIRNAMES) == (
        "tiers",
        "personas",  # retired-spelling
    )


# --- python executor ------------------------------------------------------------


def test_profile_source_entries() -> None:
    assert set(executor.PROFILE_SOURCE_ENTRIES) == {
        "profile.yml",
        "providers.yml",
        "triggers.yml",
        "ci-extra.yml",
        "osprey.service",
        ".dockerignore",
        "data",
        "tiers",
        "personas",  # retired-spelling
        "apps",
        "profiles",
        "scripts",
        "watchdogs",
        ".osprey",
        "rules",
        "skills",
        "agents",
        "commands",
        "output-styles",
        "hooks",
        "web-terminal-context",
        "mcp_servers",
        "services",
        "project",
    }
