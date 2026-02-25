"""Tests for the public resolve_env_vars() function."""

from osprey.utils.config import resolve_env_vars


class TestResolveEnvVars:
    """Unit tests for resolve_env_vars()."""

    def test_default_value_when_env_unset(self, monkeypatch):
        """${VAR:-default} resolves to default when VAR is not set."""
        monkeypatch.delenv("TZ", raising=False)
        result = resolve_env_vars("${TZ:-America/Los_Angeles}")
        assert result == "America/Los_Angeles"

    def test_env_value_overrides_default(self, monkeypatch):
        """${VAR:-default} resolves to env value when VAR is set."""
        monkeypatch.setenv("TZ", "Europe/Berlin")
        result = resolve_env_vars("${TZ:-America/Los_Angeles}")
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
        data = {"system": {"timezone": "${TZ:-America/Los_Angeles}", "name": "test"}}
        result = resolve_env_vars(data)
        assert result == {"system": {"timezone": "America/Los_Angeles", "name": "test"}}

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
        result = resolve_env_vars("${TZ:-America/Los_Angeles}")
        assert result == "America/Los_Angeles"

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
