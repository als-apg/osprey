"""Every artifact-gallery address consumer honours ``OSPREY_ARTIFACT_SERVER_PORT``.

Multi-user deployments run each user's containers on the host network namespace
and export ``OSPREY_ARTIFACT_SERVER_PORT`` per user, so the config file's port
is not the port anything listens on. A consumer that reads the config value
alone builds a URL for a dead port: the gallery link 404s and the focus POSTs
fail closed with ``gallery_unreachable``.

The fix is one resolver — ``registry.web.resolve_web_server_address`` — applied
by every consumer. These tests pin that: the resolver's own resolution order,
and then each consumer separately, driven only through its public entry point
so a consumer that quietly re-derives the address again would fail here.

The env var is set with ``monkeypatch.setenv`` throughout; nothing here mutates
the real process environment.
"""

from __future__ import annotations

import logging

import pytest

from osprey.registry.web import resolve_web_server_address, resolve_web_server_base_url
from tests.mcp_server.conftest import extract_response_dict, get_tool_fn

ENV_VAR = "OSPREY_ARTIFACT_SERVER_PORT"


@pytest.fixture
def artifact_config(monkeypatch):
    """Install a config.yml mapping for every loader call, and hand it back.

    Patches the loader itself rather than writing a file, so the module-level
    config cache in ``osprey.utils.workspace`` cannot serve a stale answer.
    """
    config: dict = {"artifact_server": {"host": "127.0.0.1", "port": 8086}}

    def _fake_loader(*_args, **_kwargs):
        return config

    monkeypatch.setattr("osprey.utils.workspace.load_osprey_config", _fake_loader)
    return config


@pytest.fixture
def approval_hook():
    """The rendered ``osprey_approval`` hook, imported through the hooks seam.

    ``import_hook`` is what strips the ``sys.path`` entry the hook prepends for
    its sibling imports; importing the module directly would leave it behind for
    the rest of the worker.
    """
    from tests.hooks.conftest import import_hook

    return import_hook("osprey_approval")


class TestResolver:
    """``resolve_web_server_address`` — the single derivation."""

    def test_config_value_used_when_env_unset(self, artifact_config, monkeypatch):
        monkeypatch.delenv(ENV_VAR, raising=False)
        artifact_config["artifact_server"]["port"] = 8500

        assert resolve_web_server_address("artifact") == ("127.0.0.1", 8500)

    def test_definition_defaults_when_section_absent(self, artifact_config, monkeypatch):
        monkeypatch.delenv(ENV_VAR, raising=False)
        artifact_config.pop("artifact_server")

        assert resolve_web_server_address("artifact") == ("127.0.0.1", 8086)

    def test_env_overrides_config_port(self, artifact_config, monkeypatch):
        monkeypatch.setenv(ENV_VAR, "9291")

        host, port = resolve_web_server_address("artifact")

        assert (host, port) == ("127.0.0.1", 9291)

    def test_env_overrides_the_default_port_too(self, artifact_config, monkeypatch):
        """A deployment that never writes the section still follows the env."""
        artifact_config.pop("artifact_server")
        monkeypatch.setenv(ENV_VAR, "9292")

        assert resolve_web_server_address("artifact")[1] == 9292

    def test_empty_env_value_is_not_an_override(self, artifact_config, monkeypatch):
        monkeypatch.setenv(ENV_VAR, "")

        assert resolve_web_server_address("artifact")[1] == 8086

    def test_non_integer_env_value_is_logged_and_ignored(
        self, artifact_config, monkeypatch, caplog
    ):
        """A typo must not raise out of a URL builder on a tool's hot path."""
        monkeypatch.setenv(ENV_VAR, "not-a-port")

        with caplog.at_level(logging.WARNING, logger="osprey.registry.web"):
            assert resolve_web_server_address("artifact")[1] == 8086

        assert ENV_VAR in caplog.text
        assert "not-a-port" in caplog.text

    def test_nested_web_subkey_and_its_own_env_var(self, monkeypatch):
        """Servers configured under a ``web`` subkey resolve the same way."""
        monkeypatch.setattr(
            "osprey.utils.workspace.load_osprey_config",
            lambda *a, **k: {"ariel": {"web": {"host": "0.0.0.0", "port": 8085}}},
        )
        monkeypatch.setenv("OSPREY_ARIEL_PORT", "9391")
        monkeypatch.delenv(ENV_VAR, raising=False)

        assert resolve_web_server_address("ariel") == ("0.0.0.0", 9391)

    def test_preloaded_config_is_used_without_loading(self, monkeypatch):
        """Callers holding a config mapping pass it; the loader is not called."""

        def _explode(*_args, **_kwargs):
            raise AssertionError("loader called despite a config being supplied")

        monkeypatch.setattr("osprey.utils.workspace.load_osprey_config", _explode)
        monkeypatch.setenv(ENV_VAR, "9291")

        address = resolve_web_server_address(
            "artifact", {"artifact_server": {"host": "10.0.0.1", "port": 8086}}
        )

        assert address == ("10.0.0.1", 9291)

    def test_unknown_server_key_raises(self, artifact_config):
        with pytest.raises(KeyError):
            resolve_web_server_address("no-such-server")

    def test_base_url_wraps_the_address(self, artifact_config, monkeypatch):
        monkeypatch.setenv(ENV_VAR, "9291")

        assert resolve_web_server_base_url("artifact") == "http://127.0.0.1:9291"


class TestConsumers:
    """Each consumer's public entry point, with the env var set."""

    def test_gallery_url_follows_env(self, artifact_config, monkeypatch):
        from osprey.mcp_server.http import gallery_url

        monkeypatch.setenv(ENV_VAR, "9291")

        assert gallery_url() == "http://127.0.0.1:9291"

    @pytest.mark.asyncio
    async def test_focus_tool_posts_to_env_port(self, artifact_config, monkeypatch, tmp_path):
        """The focus POST — the consumer that fails closed on a wrong port."""
        from osprey.mcp_server.workspace.tools import focus_tools
        from osprey.mcp_server.workspace.tools.artifact_register import artifact_register

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv(ENV_VAR, "9291")

        saved = extract_response_dict(
            await get_tool_fn(artifact_register)(
                title="Port probe", content="# hi", content_type="markdown"
            )
        )

        posted: list[str] = []

        def _record(url, _payload):
            posted.append(url)
            return 200, {"status": "ok"}

        monkeypatch.setattr(focus_tools, "_post_json_with_response", _record)

        result = extract_response_dict(
            await get_tool_fn(focus_tools.artifact_focus)(artifact_id=saved["artifact_id"])
        )

        assert posted == ["http://127.0.0.1:9291/api/focus"]
        assert result["gallery_url"] == "http://127.0.0.1:9291#focus"

    def test_artifacts_cli_binds_env_port(self, artifact_config, monkeypatch):
        from click.testing import CliRunner

        from osprey.cli.artifacts_cmd import web

        monkeypatch.setenv(ENV_VAR, "9291")
        served: list[tuple[str, int]] = []
        monkeypatch.setattr(
            "osprey.interfaces.artifacts.run_server",
            lambda host, port: served.append((host, port)),
        )

        result = CliRunner().invoke(web, [])

        assert result.exit_code == 0, result.output
        assert served == [("127.0.0.1", 9291)]

    def test_artifacts_cli_port_flag_still_wins(self, artifact_config, monkeypatch):
        """The override chain ends at the operator's explicit flag."""
        from click.testing import CliRunner

        from osprey.cli.artifacts_cmd import web

        monkeypatch.setenv(ENV_VAR, "9291")
        served: list[tuple[str, int]] = []
        monkeypatch.setattr(
            "osprey.interfaces.artifacts.run_server",
            lambda host, port: served.append((host, port)),
        )

        result = CliRunner().invoke(web, ["--port", "8123"])

        assert result.exit_code == 0, result.output
        assert served == [("127.0.0.1", 8123)]

    def test_approval_hook_follows_env(self, approval_hook, monkeypatch):
        """The hook builds the review-notebook link from the same resolution."""
        monkeypatch.setenv(ENV_VAR, "9291")

        url = approval_hook._gallery_base_url({"artifact_server": {"port": 8086}})

        assert url == "http://127.0.0.1:9291"

    def test_approval_hook_survives_a_missing_resolver(self, approval_hook, monkeypatch):
        """Rendered projects may run an osprey install without the resolver.

        The hook must still produce a URL — and still honour the env var, which
        is the whole point of routing it through the resolver.
        """
        import builtins

        real_import = builtins.__import__

        def _no_registry_web(name, *args, **kwargs):
            if name == "osprey.registry.web":
                raise ImportError("simulated: older osprey install")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_registry_web)
        monkeypatch.setenv(ENV_VAR, "9291")

        url = approval_hook._gallery_base_url({"artifact_server": {"port": 8086}})

        assert url == "http://127.0.0.1:9291"
