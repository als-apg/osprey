"""Tests for the shared acting-identity ladder.

Covers four things: that the rungs are consulted in the declared order, that a
blank or path-unsafe value at one rung falls through instead of naming an actor
nothing can file under, that an unresolvable local account costs the record
nothing, and that the module stays a leaf — importable without dragging in the
``osprey`` package tree, which is what lets ``mcp_server``, the interface apps
and the services all depend on it without an import cycle.

Every test pins :func:`getpass.getuser`, including the ones that never reach
that rung: unpinned it reads ``$USER``/``$LOGNAME`` from the real environment,
so an assertion about the env rungs would otherwise depend on who ran pytest.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from osprey_connectors.identity import (
    AUDIT_IDENTITY_ENV,
    IDENTITY_ENV_LADDER,
    TERMINAL_USER_ENV,
    UNKNOWN_IDENTITY,
    acting_identity,
    is_path_segment,
    usable_identity,
)


@pytest.fixture
def no_identity_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both env rungs unset, so a test opts into the ones it wants."""
    for env_name in IDENTITY_ENV_LADDER:
        monkeypatch.delenv(env_name, raising=False)


@pytest.fixture
def local_account(monkeypatch: pytest.MonkeyPatch) -> str:
    """Pin the local-account rung to a value no real machine would produce."""
    account = "pinned-local-account"
    monkeypatch.setattr("getpass.getuser", lambda: account)
    return account


class TestDeclaredNames:
    """The constants themselves — the deployment contract other layers render."""

    def test_env_names_are_the_deployment_spelling(self) -> None:
        """These exact names are what compose and the entrypoint write."""
        assert TERMINAL_USER_ENV == "OSPREY_TERMINAL_USER"
        assert AUDIT_IDENTITY_ENV == "OSPREY_AUDIT_IDENTITY"

    def test_ladder_order_is_terminal_user_then_audit_identity(self) -> None:
        """A real person outranks the container's service name, never the reverse."""
        assert IDENTITY_ENV_LADDER == (TERMINAL_USER_ENV, AUDIT_IDENTITY_ENV)

    def test_unknown_is_a_plain_lowercase_word(self) -> None:
        """It becomes a directory name, so it must need no escaping itself."""
        assert UNKNOWN_IDENTITY == "unknown"


class TestEnvRungs:
    """Rungs 1 and 2 — the values a deployment renders."""

    @pytest.mark.usefixtures("no_identity_env", "local_account")
    def test_terminal_user_outranks_audit_identity(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(TERMINAL_USER_ENV, "alice")
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "channel_finder")

        assert acting_identity() == "alice"

    @pytest.mark.usefixtures("no_identity_env", "local_account")
    def test_audit_identity_used_when_no_terminal_user(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A framework service names itself rather than the process account."""
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "sidecar")

        assert acting_identity() == "sidecar"

    @pytest.mark.parametrize("env_name", IDENTITY_ENV_LADDER)
    @pytest.mark.usefixtures("no_identity_env", "local_account")
    def test_surrounding_whitespace_is_stripped(
        self, env_name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The stripped value is what names the file, so it is what names the actor."""
        monkeypatch.setenv(env_name, "  alice\n")

        assert acting_identity() == "alice"

    @pytest.mark.parametrize("blank", ["", "   ", "\t", "\n"])
    @pytest.mark.usefixtures("no_identity_env", "local_account")
    def test_blank_terminal_user_falls_through(
        self, blank: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A rendered-but-empty entry is the unset case spelled differently."""
        monkeypatch.setenv(TERMINAL_USER_ENV, blank)
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "channel_finder")

        assert acting_identity() == "channel_finder"

    @pytest.mark.parametrize("blank", ["", "   "])
    @pytest.mark.usefixtures("no_identity_env")
    def test_both_rungs_blank_falls_to_local_account(
        self, blank: str, monkeypatch: pytest.MonkeyPatch, local_account: str
    ) -> None:
        for env_name in IDENTITY_ENV_LADDER:
            monkeypatch.setenv(env_name, blank)

        assert acting_identity() == local_account

    @pytest.mark.usefixtures("no_identity_env")
    def test_environment_is_read_per_call(
        self, monkeypatch: pytest.MonkeyPatch, local_account: str
    ) -> None:
        """Nothing is cached at import: a later export changes the next record."""
        assert acting_identity() == local_account

        monkeypatch.setenv(TERMINAL_USER_ENV, "bob")

        assert acting_identity() == "bob"


class TestPathSafety:
    """The identity is one directory component; a value that is not, is not one."""

    @pytest.mark.parametrize(
        "unsafe",
        ["..", ".", "../elsewhere", "a/b", "/absolute", "back\\slash"],
    )
    @pytest.mark.usefixtures("no_identity_env", "local_account")
    def test_unsafe_terminal_user_falls_through(
        self, unsafe: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A traversal or split would file one service's records under another."""
        monkeypatch.setenv(TERMINAL_USER_ENV, unsafe)
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "channel_finder")

        assert acting_identity() == "channel_finder"

    @pytest.mark.parametrize("unsafe", ["..", "a/b", "/absolute"])
    @pytest.mark.usefixtures("no_identity_env")
    def test_unsafe_audit_identity_falls_through(
        self, unsafe: str, monkeypatch: pytest.MonkeyPatch, local_account: str
    ) -> None:
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, unsafe)

        assert acting_identity() == local_account

    def test_null_byte_is_rejected_by_the_rule_itself(self) -> None:
        """The OS refuses a NUL in ``os.environ``, so the rule is pinned directly.

        Unreachable through :func:`os.environ.setenv` today, but the rejection
        rule is what a future rung would be checked against, and a NUL splits a
        path at the syscall boundary.
        """
        from osprey_connectors.identity import usable_identity

        assert usable_identity("nul\0byte") == ""
        assert usable_identity("alice") == "alice"

    @pytest.mark.usefixtures("no_identity_env")
    def test_unsafe_local_account_becomes_unknown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The floor is honest, not a value that would escape the audit tree."""
        monkeypatch.setattr("getpass.getuser", lambda: "../root")

        assert acting_identity() == UNKNOWN_IDENTITY

    @pytest.mark.parametrize(
        "accepted",
        ["alice", "alice.smith", "als_operator", "svc-web-terminal", "user@example.org"],
    )
    @pytest.mark.usefixtures("no_identity_env", "local_account")
    def test_ordinary_names_are_accepted(
        self, accepted: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Rejection stays narrow: a strict allowlist would erase real accounts."""
        monkeypatch.setenv(TERMINAL_USER_ENV, accepted)

        assert acting_identity() == accepted


class TestLocalAccountRung:
    """Rung 3 and the floor — the single-user laptop, and the slim container."""

    @pytest.mark.usefixtures("no_identity_env")
    def test_local_account_when_no_env_rung_is_set(self, local_account: str) -> None:
        assert acting_identity() == local_account

    @pytest.mark.parametrize("failure", [KeyError("uid"), OSError("no passwd entry")])
    @pytest.mark.usefixtures("no_identity_env")
    def test_unresolvable_account_yields_unknown(
        self, failure: Exception, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A uid with no passwd entry is normal in a slim image, not an error."""

        def _raise() -> str:
            raise failure

        monkeypatch.setattr("getpass.getuser", _raise)

        assert acting_identity() == UNKNOWN_IDENTITY

    @pytest.mark.usefixtures("no_identity_env")
    def test_any_other_failure_also_yields_unknown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Resolving an identity must never be what breaks the audited operation."""

        def _raise() -> str:
            raise RuntimeError("something else entirely")

        monkeypatch.setattr("getpass.getuser", _raise)

        assert acting_identity() == UNKNOWN_IDENTITY

    @pytest.mark.parametrize("empty", ["", "   "])
    @pytest.mark.usefixtures("no_identity_env")
    def test_empty_account_name_yields_unknown(
        self, empty: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("getpass.getuser", lambda: empty)

        assert acting_identity() == UNKNOWN_IDENTITY

    @pytest.mark.usefixtures("no_identity_env")
    def test_env_rung_still_wins_over_a_broken_account(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The container case must not depend on the account resolving at all."""

        def _raise() -> str:
            raise OSError("no passwd entry")

        monkeypatch.setattr("getpass.getuser", _raise)
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "channel_finder")

        assert acting_identity() == "channel_finder"


class TestNeverTheHostname:
    """The one fallback that must never appear, however plausible it looks."""

    @pytest.mark.usefixtures("no_identity_env")
    def test_exhausted_ladder_yields_unknown_not_a_machine_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A host-network container reports the shared host; a bridge one, noise."""
        import socket

        def _raise() -> str:
            raise OSError("no passwd entry")

        monkeypatch.setattr("getpass.getuser", _raise)
        resolved = acting_identity()

        assert resolved == UNKNOWN_IDENTITY
        assert resolved != socket.gethostname()

    def test_source_names_no_hostname_api(self) -> None:
        """Static check: no socket/platform/uname call can creep in as a rung.

        Reads the module that holds the ladder, not the re-export: a source
        check pointed at the shim would have no rung to inspect and would pass
        on an empty set whatever the ladder had grown.

        Both call spellings are collected. ``socket.gethostname()`` reaches the
        tree as an attribute call, but ``from socket import gethostname``
        followed by a bare ``gethostname()`` reaches it as a plain name — a
        guard that looked only at attribute calls would wave the shorter
        spelling through, which is the one an editor reaching for a hostname is
        at least as likely to write.
        """
        import osprey_connectors.identity as module

        source = Path(module.__file__).read_text(encoding="utf-8")
        code = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("#"))
        tree = ast.parse(code)
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        called = {node.func.attr for node in calls if isinstance(node.func, ast.Attribute)} | {
            node.func.id for node in calls if isinstance(node.func, ast.Name)
        }

        assert not called & {"gethostname", "getfqdn", "uname", "node"}


class TestLeafModule:
    """The property that lets interfaces, services and mcp_server all import this."""

    def test_source_imports_nothing_from_osprey(self) -> None:
        """Static check: no ``import osprey...`` of any spelling in the source.

        Aimed at the ladder's own module, which is the one that is read where
        ``osprey`` is not importable.
        """
        import osprey_connectors.identity as module

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
        import osprey_connectors.identity

        module_path = Path(osprey_connectors.identity.__file__)
        probe = f"""
import importlib.util, sys, json
spec = importlib.util.spec_from_file_location("_probe_identity", {str(module_path)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert module.acting_identity(), "module did not resolve an identity"
print(json.dumps(sorted(k for k in sys.modules if k == "osprey" or k.startswith("osprey."))))
"""
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )

        assert completed.stdout.strip().endswith("[]"), completed.stdout


# The three segment rules ``is_path_segment`` replaced, as each stood: the
# identity ladder's (``usable_identity`` answering a non-empty name), the
# control-context tree's screen of an owner, and the grants directory's.
def _ladder_rule(value: object) -> bool:
    if not isinstance(value, str):
        return False
    candidate = value.strip()
    if not candidate or candidate in (".", ".."):
        return False
    return not any(separator in candidate for separator in ("/", "\\", "\0"))


def _tree_rule(owner: object) -> bool:
    unsafe = frozenset({"/", "\\", "\x00"})
    return not (
        not isinstance(owner, str) or owner.strip() in {"", ".", ".."} or unsafe & set(owner)
    )


def _grants_rule(name: object) -> bool:
    unsafe = frozenset({"/", "\\", "\x00"})
    return (
        isinstance(name, str) and name.strip() not in {"", ".", ".."} and not (unsafe & set(name))
    )


SEGMENTS = [
    ("alice", True),
    ("a.b", True),
    ("...", True),
    ("a b", True),
    (" alice ", True),
    ("\talice", True),
    (" .. x", True),
    ("", False),
    ("   ", False),
    ("\t\n", False),
    (".", False),
    ("..", False),
    (" .. ", False),
    ("\t.\n", False),
    ("a/b", False),
    ("/", False),
    ("a\\b", False),
    ("a\x00b", False),
    ("\x00", False),
    (" a/ ", False),
    (None, False),
    (7, False),
    (b"alice", False),
]


class TestPathSegment:
    @pytest.mark.parametrize(("value", "segment"), SEGMENTS, ids=repr)
    def test_one_rule_answers_as_each_of_the_three_did(self, value: object, segment: bool) -> None:
        assert is_path_segment(value) is segment
        assert (_ladder_rule(value), _tree_rule(value), _grants_rule(value)) == (segment,) * 3

    @pytest.mark.parametrize(("value", "segment"), SEGMENTS, ids=repr)
    def test_a_usable_identity_is_a_segment_stripped(self, value: object, segment: bool) -> None:
        expected = value.strip() if segment and isinstance(value, str) else ""
        assert usable_identity(value) == expected  # type: ignore[arg-type]

    def test_the_rule_judges_but_never_strips(self) -> None:
        name = " alice "
        assert is_path_segment(name)
        assert name == " alice "
