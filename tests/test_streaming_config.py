"""Tests for streaming display mode configuration."""

from unittest.mock import patch


class TestGetStreamingMode:
    """Tests for get_streaming_mode() resolution logic."""

    def _call(self, interface: str, node: str, config_overrides: dict | None = None):
        """Helper to call get_streaming_mode with optional config overrides."""
        config_overrides = config_overrides or {}

        def mock_get_config_value(path, default=None, config_path=None):
            return config_overrides.get(path)

        with patch("osprey.utils.config.get_config_value", side_effect=mock_get_config_value):
            from osprey.utils.config import get_streaming_mode

            return get_streaming_mode(interface, node)

    # --- Default resolution (no config set) ---

    def test_respond_defaults_to_show(self):
        assert self._call("openwebui", "respond") == "show"

    def test_respond_defaults_to_show_for_cli(self):
        assert self._call("cli", "respond") == "show"

    def test_respond_defaults_to_show_for_tui(self):
        assert self._call("tui", "respond") == "show"

    def test_non_respond_defaults_to_hide(self):
        assert self._call("openwebui", "python_code_generator") == "hide"

    def test_non_respond_defaults_to_hide_for_tui(self):
        assert self._call("tui", "python_code_generator") == "hide"

    def test_cli_non_respond_defaults_to_disabled(self):
        """CLI has no collapse mechanism, so hide maps to disabled."""
        assert self._call("cli", "python_code_generator") == "disabled"

    def test_unknown_node_defaults_to_hide(self):
        assert self._call("openwebui", "some_future_node") == "hide"

    # --- Node-specific config overrides ---

    def test_node_specific_override(self):
        config = {"openwebui.streaming.python_code_generator": "show"}
        assert self._call("openwebui", "python_code_generator", config) == "show"

    def test_node_specific_disabled(self):
        config = {"openwebui.streaming.python_code_generator": "disabled"}
        assert self._call("openwebui", "python_code_generator", config) == "disabled"

    def test_respond_disabled(self):
        config = {"tui.streaming.respond": "disabled"}
        assert self._call("tui", "respond", config) == "disabled"

    # --- Interface default override ---

    def test_interface_default_overrides_hardcoded(self):
        config = {"openwebui.streaming.default": "disabled"}
        assert self._call("openwebui", "python_code_generator", config) == "disabled"

    def test_node_specific_overrides_interface_default(self):
        config = {
            "openwebui.streaming.default": "disabled",
            "openwebui.streaming.python_code_generator": "show",
        }
        assert self._call("openwebui", "python_code_generator", config) == "show"

    def test_interface_default_does_not_affect_respond_hardcoded(self):
        """Interface default shouldn't apply to respond — respond has its own hardcoded default."""
        # When interface default is "disabled" but no node-specific config for respond,
        # the interface default takes effect (it IS a valid respond mode)
        config = {"openwebui.streaming.default": "disabled"}
        assert self._call("openwebui", "respond", config) == "disabled"

    # --- Validation: respond can't be "hide" ---

    def test_respond_hide_corrected_to_show(self):
        config = {"openwebui.streaming.respond": "hide"}
        assert self._call("openwebui", "respond", config) == "show"

    def test_respond_hide_corrected_for_tui(self):
        config = {"tui.streaming.respond": "hide"}
        assert self._call("tui", "respond", config) == "show"

    def test_respond_hide_corrected_for_cli(self):
        config = {"cli.streaming.respond": "hide"}
        assert self._call("cli", "respond", config) == "show"

    # --- CLI hide → disabled coercion ---

    def test_cli_hide_becomes_disabled(self):
        config = {"cli.streaming.python_code_generator": "hide"}
        assert self._call("cli", "python_code_generator", config) == "disabled"

    def test_cli_show_stays_show(self):
        config = {"cli.streaming.python_code_generator": "show"}
        assert self._call("cli", "python_code_generator", config) == "show"

    def test_cli_disabled_stays_disabled(self):
        config = {"cli.streaming.python_code_generator": "disabled"}
        assert self._call("cli", "python_code_generator", config) == "disabled"
