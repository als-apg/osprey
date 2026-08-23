"""Tests for the canonical agent-invisible credential list.

Covers three things: that the declared name set is the one the rest of the
safety work depends on, that :func:`strip_sensitive` removes exactly those
names and nothing else, and that the module stays a leaf — importable without
dragging in the ``osprey`` package tree, which is what lets ``agent_runner``
and ``mcp_server`` both depend on it without an import cycle.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from osprey.utils.sensitive_env import (
    SENSITIVE_ENV_EXACT,
    SENSITIVE_ENV_SUFFIXES,
    is_sensitive,
    strip_sensitive,
)


class TestDeclaredNames:
    """The constants themselves — shape and membership."""

    def test_constants_are_tuples(self) -> None:
        """Immutable containers, so no consumer can append to the shared list."""
        assert isinstance(SENSITIVE_ENV_EXACT, tuple)
        assert isinstance(SENSITIVE_ENV_SUFFIXES, tuple)

    @pytest.mark.parametrize(
        "name",
        ["OSPREY_TERMINAL_SECRET", "OSPREY_PANEL_TOKEN", "EVENT_DISPATCHER_TOKEN"],
    )
    def test_required_exact_names_present(self, name: str) -> None:
        """Each credential the D2 scrub must cover is declared exactly."""
        assert name in SENSITIVE_ENV_EXACT

    def test_launch_token_suffix_present(self) -> None:
        """Bridge launch tokens are covered by convention, not per-name."""
        assert "_LAUNCH_TOKEN" in SENSITIVE_ENV_SUFFIXES

    def test_no_empty_entries(self) -> None:
        """An empty suffix would match every variable and strip the whole env."""
        assert all(SENSITIVE_ENV_EXACT)
        assert all(SENSITIVE_ENV_SUFFIXES)

    def test_no_duplicate_exact_names(self) -> None:
        """Duplicates would signal two people adding the same name blind."""
        assert len(set(SENSITIVE_ENV_EXACT)) == len(SENSITIVE_ENV_EXACT)


class TestIsSensitive:
    """The per-name predicate."""

    @pytest.mark.parametrize("name", SENSITIVE_ENV_EXACT)
    def test_exact_names_match(self, name: str) -> None:
        assert is_sensitive(name) is True

    @pytest.mark.parametrize(
        "name",
        ["BLUESKY_LAUNCH_TOKEN", "SOME_OTHER_BRIDGE_LAUNCH_TOKEN", "_LAUNCH_TOKEN"],
    )
    def test_suffix_names_match(self, name: str) -> None:
        """Any variable ending in a declared suffix matches, known or not."""
        assert is_sensitive(name) is True

    @pytest.mark.parametrize(
        "name",
        ["PATH", "HOME", "OSPREY_WEB_PORT", "OSPREY_CONFIG", "ANTHROPIC_API_KEY"],
    )
    def test_unrelated_names_do_not_match(self, name: str) -> None:
        assert is_sensitive(name) is False

    def test_matching_is_case_sensitive(self) -> None:
        """Lower-case spellings are different variables and are left alone."""
        assert is_sensitive("osprey_panel_token") is False
        assert is_sensitive("bluesky_launch_token") is False

    def test_prefix_match_is_not_enough(self) -> None:
        """A name merely starting with a sensitive name is not itself sensitive."""
        assert is_sensitive("OSPREY_PANEL_TOKEN_FILE") is False
        assert is_sensitive("EVENT_DISPATCHER_TOKEN_PATH") is False

    def test_suffix_must_be_at_the_end(self) -> None:
        assert is_sensitive("MY_LAUNCH_TOKEN_BACKUP") is False

    def test_empty_name_is_not_sensitive(self) -> None:
        assert is_sensitive("") is False


class TestStripSensitive:
    """The dict transform used at every spawn site."""

    def test_removes_exact_and_suffix_matches_keeps_the_rest(self) -> None:
        env = {
            "PATH": "/usr/bin",
            "OSPREY_TERMINAL_SECRET": "s1",
            "OSPREY_PANEL_TOKEN": "s2",
            "EVENT_DISPATCHER_TOKEN": "s3",
            "BLUESKY_LAUNCH_TOKEN": "s4",
            "OSPREY_WEB_PORT": "8000",
        }

        result = strip_sensitive(env)

        assert result == {"PATH": "/usr/bin", "OSPREY_WEB_PORT": "8000"}

    def test_does_not_mutate_the_input(self) -> None:
        """Callers may pass ``os.environ`` itself; their own env must survive."""
        env = {"PATH": "/usr/bin", "OSPREY_PANEL_TOKEN": "secret"}
        original = dict(env)

        strip_sensitive(env)

        assert env == original

    def test_returns_a_new_object(self) -> None:
        env = {"PATH": "/usr/bin"}

        result = strip_sensitive(env)

        assert result is not env
        assert result == env

    def test_preserves_values_verbatim(self) -> None:
        """Only keys are filtered — surviving values are untouched."""
        env = {"PATH": "/usr/bin:/bin", "EMPTY": ""}

        assert strip_sensitive(env) == env

    def test_empty_env(self) -> None:
        assert strip_sensitive({}) == {}

    def test_all_sensitive_env(self) -> None:
        env = dict.fromkeys(SENSITIVE_ENV_EXACT, "x")
        env["BLUESKY_LAUNCH_TOKEN"] = "x"

        assert strip_sensitive(env) == {}

    def test_accepts_any_mapping(self) -> None:
        """``os.environ`` is a Mapping, not a dict — the signature allows it."""
        from types import MappingProxyType

        env = MappingProxyType({"PATH": "/usr/bin", "OSPREY_PANEL_TOKEN": "s"})

        result = strip_sensitive(env)

        assert result == {"PATH": "/usr/bin"}
        assert isinstance(result, dict)

    def test_is_idempotent(self) -> None:
        env = {"PATH": "/usr/bin", "OSPREY_TERMINAL_SECRET": "s"}

        once = strip_sensitive(env)

        assert strip_sensitive(once) == once


class TestLeafModule:
    """The property that lets both agent_runner and mcp_server import this."""

    def test_source_imports_nothing_from_osprey(self) -> None:
        """Static check: no ``import osprey...`` of any spelling in the source."""
        import osprey.utils.sensitive_env as module

        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                # level > 0 is a relative import, which resolves inside osprey.
                assert node.level == 0, "relative import breaks the leaf property"
                if node.module:
                    imported.append(node.module)

        assert not [name for name in imported if name.split(".")[0] == "osprey"], imported

    def test_loading_the_file_pulls_in_no_osprey_modules(self) -> None:
        """Behavioural check: executing the module imports no osprey package."""
        module_path = Path(__file__).resolve().parents[2] / "src/osprey/utils/sensitive_env.py"
        probe = f"""
import importlib.util, sys, json
spec = importlib.util.spec_from_file_location("_probe_sensitive_env", {str(module_path)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert module.SENSITIVE_ENV_EXACT, "module did not define its constants"
print(json.dumps(sorted(k for k in sys.modules if k == "osprey" or k.startswith("osprey."))))
"""
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            check=True,
        )

        assert completed.stdout.strip() == "[]", completed.stdout
