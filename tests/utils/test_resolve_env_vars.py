"""Tests for the public resolve_env_vars() function."""

from osprey.utils.config import resolve_env_vars


class TestResolveEnvVars:
    """Unit tests for resolve_env_vars()."""

    def test_default_value_when_env_unset(self, monkeypatch):
        """${VAR:-default} resolves to default when VAR is not set."""
        monkeypatch.delenv("TZ", raising=False)
        result = resolve_env_vars("${TZ:-UTC}")
        assert result == "UTC"

    def test_env_value_overrides_default(self, monkeypatch):
        """${VAR:-default} resolves to env value when VAR is set."""
        monkeypatch.setenv("TZ", "Europe/Berlin")
        result = resolve_env_vars("${TZ:-UTC}")
        assert result == "Europe/Berlin"

    def test_simple_env_var(self, monkeypatch):
        """${VAR} resolves to env value."""
        monkeypatch.setenv("MY_VAR", "hello")
        result = resolve_env_vars("${MY_VAR}")
        assert result == "hello"

    def test_simple_dollar_var(self, monkeypatch):
        """$VAR resolves to env value."""
        monkeypatch.setenv("MY_VAR", "hello")
        result = resolve_env_vars("$MY_VAR")
        assert result == "hello"

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

    def test_empty_string_with_default_uses_default(self, monkeypatch):
        """${VAR:-default} with VAR="" uses default (bash :- semantics)."""
        monkeypatch.setenv("TZ", "")
        result = resolve_env_vars("${TZ:-UTC}")
        assert result == "UTC"

    def test_empty_string_without_default_returns_empty(self, monkeypatch):
        """${VAR} with VAR="" returns empty string (var is set, no default)."""
        monkeypatch.setenv("EMPTY_VAR", "")
        result = resolve_env_vars("${EMPTY_VAR}")
        assert result == ""

    def test_empty_string_with_empty_default_returns_empty(self, monkeypatch):
        """${VAR:-} with VAR="" returns empty string (empty default = empty)."""
        monkeypatch.setenv("EMPTY_VAR", "")
        result = resolve_env_vars("${EMPTY_VAR:-}")
        assert result == ""

    def test_unset_var_no_default_preserves_original(self, monkeypatch):
        """${VAR} with no default and VAR unset keeps the original placeholder."""
        monkeypatch.delenv("UNSET_VAR", raising=False)
        monkeypatch.setenv("OSPREY_QUIET", "1")
        result = resolve_env_vars("${UNSET_VAR}")
        assert result == "${UNSET_VAR}"


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
