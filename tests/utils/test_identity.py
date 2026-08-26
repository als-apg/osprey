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

from osprey.utils.identity import (
    AUDIT_IDENTITY_ENV,
    IDENTITY_ENV_LADDER,
    TERMINAL_USER_ENV,
    UNKNOWN_IDENTITY,
    acting_identity,
)

pytestmark = pytest.mark.unit


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

    def test_terminal_user_outranks_audit_identity(
        self, monkeypatch: pytest.MonkeyPatch, no_identity_env: None, local_account: str
    ) -> None:
        monkeypatch.setenv(TERMINAL_USER_ENV, "alice")
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "channel_finder")

        assert acting_identity() == "alice"

    def test_audit_identity_used_when_no_terminal_user(
        self, monkeypatch: pytest.MonkeyPatch, no_identity_env: None, local_account: str
    ) -> None:
        """A framework service names itself rather than the process account."""
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "sidecar")

        assert acting_identity() == "sidecar"

    @pytest.mark.parametrize("env_name", IDENTITY_ENV_LADDER)
    def test_surrounding_whitespace_is_stripped(
        self,
        env_name: str,
        monkeypatch: pytest.MonkeyPatch,
        no_identity_env: None,
        local_account: str,
    ) -> None:
        """The stripped value is what names the file, so it is what names the actor."""
        monkeypatch.setenv(env_name, "  alice\n")

        assert acting_identity() == "alice"

    @pytest.mark.parametrize("blank", ["", "   ", "\t", "\n"])
    def test_blank_terminal_user_falls_through(
        self,
        blank: str,
        monkeypatch: pytest.MonkeyPatch,
        no_identity_env: None,
        local_account: str,
    ) -> None:
        """A rendered-but-empty entry is the unset case spelled differently."""
        monkeypatch.setenv(TERMINAL_USER_ENV, blank)
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "channel_finder")

        assert acting_identity() == "channel_finder"

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_both_rungs_blank_falls_to_local_account(
        self,
        blank: str,
        monkeypatch: pytest.MonkeyPatch,
        no_identity_env: None,
        local_account: str,
    ) -> None:
        for env_name in IDENTITY_ENV_LADDER:
            monkeypatch.setenv(env_name, blank)

        assert acting_identity() == local_account

    def test_environment_is_read_per_call(
        self, monkeypatch: pytest.MonkeyPatch, no_identity_env: None, local_account: str
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
    def test_unsafe_terminal_user_falls_through(
        self,
        unsafe: str,
        monkeypatch: pytest.MonkeyPatch,
        no_identity_env: None,
        local_account: str,
    ) -> None:
        """A traversal or split would file one service's records under another."""
        monkeypatch.setenv(TERMINAL_USER_ENV, unsafe)
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "channel_finder")

        assert acting_identity() == "channel_finder"

    @pytest.mark.parametrize("unsafe", ["..", "a/b", "/absolute"])
    def test_unsafe_audit_identity_falls_through(
        self,
        unsafe: str,
        monkeypatch: pytest.MonkeyPatch,
        no_identity_env: None,
        local_account: str,
    ) -> None:
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, unsafe)

        assert acting_identity() == local_account

    def test_null_byte_is_rejected_by_the_rule_itself(self) -> None:
        """The OS refuses a NUL in ``os.environ``, so the rule is pinned directly.

        Unreachable through :func:`os.environ.setenv` today, but the rejection
        rule is what a future rung would be checked against, and a NUL splits a
        path at the syscall boundary.
        """
        from osprey.utils.identity import _usable

        assert _usable("nul\0byte") == ""
        assert _usable("alice") == "alice"

    def test_unsafe_local_account_becomes_unknown(
        self, monkeypatch: pytest.MonkeyPatch, no_identity_env: None
    ) -> None:
        """The floor is honest, not a value that would escape the audit tree."""
        monkeypatch.setattr("getpass.getuser", lambda: "../root")

        assert acting_identity() == UNKNOWN_IDENTITY

    @pytest.mark.parametrize(
        "accepted",
        ["alice", "alice.smith", "als_operator", "svc-web-terminal", "user@example.org"],
    )
    def test_ordinary_names_are_accepted(
        self,
        accepted: str,
        monkeypatch: pytest.MonkeyPatch,
        no_identity_env: None,
        local_account: str,
    ) -> None:
        """Rejection stays narrow: a strict allowlist would erase real accounts."""
        monkeypatch.setenv(TERMINAL_USER_ENV, accepted)

        assert acting_identity() == accepted


class TestLocalAccountRung:
    """Rung 3 and the floor — the single-user laptop, and the slim container."""

    def test_local_account_when_no_env_rung_is_set(
        self, no_identity_env: None, local_account: str
    ) -> None:
        assert acting_identity() == local_account

    @pytest.mark.parametrize("failure", [KeyError("uid"), OSError("no passwd entry")])
    def test_unresolvable_account_yields_unknown(
        self,
        failure: Exception,
        monkeypatch: pytest.MonkeyPatch,
        no_identity_env: None,
    ) -> None:
        """A uid with no passwd entry is normal in a slim image, not an error."""

        def _raise() -> str:
            raise failure

        monkeypatch.setattr("getpass.getuser", _raise)

        assert acting_identity() == UNKNOWN_IDENTITY

    def test_any_other_failure_also_yields_unknown(
        self, monkeypatch: pytest.MonkeyPatch, no_identity_env: None
    ) -> None:
        """Resolving an identity must never be what breaks the audited operation."""

        def _raise() -> str:
            raise RuntimeError("something else entirely")

        monkeypatch.setattr("getpass.getuser", _raise)

        assert acting_identity() == UNKNOWN_IDENTITY

    @pytest.mark.parametrize("empty", ["", "   "])
    def test_empty_account_name_yields_unknown(
        self, empty: str, monkeypatch: pytest.MonkeyPatch, no_identity_env: None
    ) -> None:
        monkeypatch.setattr("getpass.getuser", lambda: empty)

        assert acting_identity() == UNKNOWN_IDENTITY

    def test_env_rung_still_wins_over_a_broken_account(
        self, monkeypatch: pytest.MonkeyPatch, no_identity_env: None
    ) -> None:
        """The container case must not depend on the account resolving at all."""

        def _raise() -> str:
            raise OSError("no passwd entry")

        monkeypatch.setattr("getpass.getuser", _raise)
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "channel_finder")

        assert acting_identity() == "channel_finder"


class TestNeverTheHostname:
    """The one fallback that must never appear, however plausible it looks."""

    def test_exhausted_ladder_yields_unknown_not_a_machine_name(
        self, monkeypatch: pytest.MonkeyPatch, no_identity_env: None
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
        """Static check: no socket/platform/uname call can creep in as a rung."""
        import osprey.utils.identity as module

        source = Path(module.__file__).read_text(encoding="utf-8")
        code = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("#"))
        tree = ast.parse(code)
        called = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }

        assert not called & {"gethostname", "getfqdn", "uname", "node"}


class TestLeafModule:
    """The property that lets interfaces, services and mcp_server all import this."""

    def test_source_imports_nothing_from_osprey(self) -> None:
        """Static check: no ``import osprey...`` of any spelling in the source."""
        import osprey.utils.identity as module

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
        module_path = Path(__file__).resolve().parents[2] / "src/osprey/utils/identity.py"
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
        )

        assert completed.stdout.strip().endswith("[]"), completed.stdout
