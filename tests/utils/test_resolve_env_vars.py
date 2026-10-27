"""Tests for the public resolve_env_vars() function."""

import pytest

from osprey_connectors.config import is_unresolved_placeholder, resolve_env_vars


class TestResolveEnvVars:
    """Unit tests for resolve_env_vars()."""

    @pytest.mark.parametrize(
        ("env", "template", "expected"),
        [
            pytest.param({"TZ": None}, "${TZ:-UTC}", "UTC", id="default_value_when_env_unset"),
            pytest.param(
                {"TZ": "Europe/Berlin"},
                "${TZ:-UTC}",
                "Europe/Berlin",
                id="env_value_overrides_default",
            ),
            pytest.param({"MY_VAR": "hello"}, "${MY_VAR}", "hello", id="simple_env_var"),
            pytest.param({"MY_VAR": "hello"}, "$MY_VAR", "hello", id="simple_dollar_var"),
            # Bash ``:-`` semantics: a set-but-empty variable takes the default.
            pytest.param(
                {"TZ": ""}, "${TZ:-UTC}", "UTC", id="empty_string_with_default_uses_default"
            ),
            # Set, with no default: the empty value stands.
            pytest.param(
                {"EMPTY_VAR": ""},
                "${EMPTY_VAR}",
                "",
                id="empty_string_without_default_returns_empty",
            ),
            pytest.param(
                {"EMPTY_VAR": ""},
                "${EMPTY_VAR:-}",
                "",
                id="empty_string_with_empty_default_returns_empty",
            ),
            # Unset with no default keeps the original placeholder.
            pytest.param(
                {"UNSET_VAR": None, "OSPREY_QUIET": "1"},
                "${UNSET_VAR}",
                "${UNSET_VAR}",
                id="unset_var_no_default_preserves_original",
            ),
        ],
    )
    def test_resolves_a_reference(self, monkeypatch, env, template, expected):
        """A ``$VAR``/``${VAR}``/``${VAR:-default}`` string resolves against the environment."""
        for name, value in env.items():
            if value is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, value)
        assert resolve_env_vars(template) == expected

    def test_nested_dict(self, monkeypatch):
        """Resolves env vars in nested dicts."""
        monkeypatch.delenv("TZ", raising=False)
        data = {"system": {"timezone": "${TZ:-UTC}", "name": "test"}}
        result = resolve_env_vars(data)
        assert result == {"system": {"timezone": "UTC", "name": "test"}}

    def test_list_values(self, monkeypatch):
        """Resolves env vars inside lists."""
        monkeypatch.setenv("ITEM", "resolved")
        data = ["${ITEM}", "literal"]
        result = resolve_env_vars(data)
        assert result == ["resolved", "literal"]

    def test_non_string_passthrough(self):
        """Non-string values (int, bool, None) pass through unchanged."""
        assert resolve_env_vars(42) == 42
        assert resolve_env_vars(True) is True
        assert resolve_env_vars(None) is None
        assert resolve_env_vars(3.14) == 3.14

    def test_mixed_nested_structure(self, monkeypatch):
        """Complex nested structure with mixed types."""
        monkeypatch.setenv("PORT", "8080")
        data = {
            "servers": [
                {"port": "${PORT:-3000}", "enabled": True},
                {"port": 9090, "enabled": False},
            ],
            "count": 2,
        }
        result = resolve_env_vars(data)
        assert result["servers"][0]["port"] == "8080"
        assert result["servers"][0]["enabled"] is True
        assert result["servers"][1]["port"] == 9090
        assert result["count"] == 2


class TestResolveEnvVarsWithEnviron:
    """resolve_env_vars(data, environ=...) expands against a supplied mapping."""

    def test_resolves_brace_var_from_environ(self):
        """${VAR} resolves from the passed mapping, not os.environ."""
        result = resolve_env_vars("${ARGO_PROD_URL}", environ={"ARGO_PROD_URL": "https://argo"})
        assert result == "https://argo"

    def test_resolves_bare_dollar_var_from_environ(self):
        """$VAR resolves from the passed mapping."""
        result = resolve_env_vars("$ARGO", environ={"ARGO": "https://argo"})
        assert result == "https://argo"

    def test_resolves_nested_dict_from_environ(self):
        """Nested-dict ${VAR} keys resolve from the passed mapping."""
        data = {"api": {"providers": {"argo": {"base_url": "${ARGO_PROD_URL}"}}}}
        result = resolve_env_vars(data, environ={"ARGO_PROD_URL": "https://argo"})
        assert result == {"api": {"providers": {"argo": {"base_url": "https://argo"}}}}

    def test_default_check_uses_environ(self):
        """${VAR:-default} with empty value in the mapping uses the default."""
        assert resolve_env_vars("${X:-fallback}", environ={"X": ""}) == "fallback"
        assert resolve_env_vars("${X:-fallback}", environ={"X": "set"}) == "set"

    def test_does_not_read_os_environ_when_environ_given(self, monkeypatch):
        """A var present in os.environ but absent from the mapping is NOT used."""
        monkeypatch.setenv("ONLY_IN_OS", "from-os")
        monkeypatch.setenv("OSPREY_QUIET", "1")
        result = resolve_env_vars("${ONLY_IN_OS}", environ={})
        assert result == "${ONLY_IN_OS}"

    def test_preserves_server_env_blocks_with_environ(self):
        """claude_code.servers.*.env stays verbatim even when environ is given."""
        data = {
            "claude_code": {
                "servers": {"controls": {"env": {"KEY": "${SHOULD_NOT_EXPAND}"}}},
                "provider": "${P}",
            }
        }
        result = resolve_env_vars(data, environ={"SHOULD_NOT_EXPAND": "leaked", "P": "argo"})
        assert result["claude_code"]["servers"]["controls"]["env"]["KEY"] == "${SHOULD_NOT_EXPAND}"
        assert result["claude_code"]["provider"] == "argo"


class TestIsUnresolvedPlaceholder:
    """The recogniser and the resolver must agree on what a placeholder is.

    ``is_unresolved_placeholder`` exists to answer one question — "is this what
    :func:`resolve_env_vars` leaves behind when the variable is unset?" — so its
    accepted syntax is the resolver's, not a narrower re-statement of it.
    """

    @pytest.mark.parametrize(
        "reference",
        ["${MISSING}", "$MISSING", "${SOME-VAR}", "${lowercase}", "${VAR.WITH.DOTS}"],
    )
    def test_what_the_resolver_leaves_verbatim_is_recognised(self, reference, monkeypatch):
        monkeypatch.setenv("OSPREY_QUIET", "1")
        # The premise: with nothing to substitute, the resolver hands the
        # reference back unchanged.
        assert resolve_env_vars(reference, environ={}) == reference
        assert is_unresolved_placeholder(reference) is True

    @pytest.mark.parametrize(
        "value",
        [
            "https://gateway.example.org/v1",
            "${VAR:-https://default.example.org}",  # a default always substitutes
            "prefix-${VAR}",  # the rest of the string is a value
            "",
            None,
            42,
        ],
    )
    def test_anything_that_is_not_a_lone_reference_is_not_a_placeholder(self, value):
        assert is_unresolved_placeholder(value) is False
