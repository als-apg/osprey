"""Spend attribution through a LiteLLM gateway — the identity every request carries.

Covers the pure helpers in :mod:`osprey.models.spend_attribution`: which
providers are LiteLLM-fronted, what surface a process is, the headers a
request carries, and how they merge into an operator's own
``ANTHROPIC_CUSTOM_HEADERS`` without clobbering it.
"""

from __future__ import annotations

import pytest

from osprey.models.spend_attribution import (
    CUSTOM_HEADERS_ENV,
    END_USER_HEADER,
    LITELLM_GATEWAY,
    TAGS_HEADER,
    acting_surface,
    apply_attribution_env,
    attribution_headers,
    gateway_for,
    merge_custom_headers,
    render_custom_headers,
)
from osprey.utils.identity import AUDIT_IDENTITY_ENV, TERMINAL_USER_ENV


@pytest.fixture
def no_identity(monkeypatch):
    """Neither identity rung is set — the single-user laptop case."""
    monkeypatch.delenv(TERMINAL_USER_ENV, raising=False)
    monkeypatch.delenv(AUDIT_IDENTITY_ENV, raising=False)


class TestGatewayFor:
    def test_builtin_litellm_providers(self):
        assert gateway_for("als-apg") == LITELLM_GATEWAY
        assert gateway_for("cborg") == LITELLM_GATEWAY

    def test_direct_providers_have_no_gateway(self):
        assert gateway_for("anthropic") is None
        assert gateway_for("openai") is None

    def test_custom_provider_declares_gateway_in_config(self):
        providers = {"my-gw": {"base_url": "https://gw.example/v1", "gateway": "litellm"}}
        assert gateway_for("my-gw", providers) == LITELLM_GATEWAY

    def test_custom_provider_without_key_is_unattributed(self):
        providers = {"my-gw": {"base_url": "https://gw.example/v1"}}
        assert gateway_for("my-gw", providers) is None

    def test_unknown_provider_is_unattributed(self):
        assert gateway_for("no-such-provider") is None


class TestActingSurface:
    def test_terminal_user_wins(self, monkeypatch):
        monkeypatch.setenv(TERMINAL_USER_ENV, "thellert")
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "dispatch-worker-0")
        assert acting_surface() == "terminal"

    def test_dispatch_worker(self, monkeypatch):
        monkeypatch.delenv(TERMINAL_USER_ENV, raising=False)
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "dispatch-worker-0")
        assert acting_surface() == "dispatch"

    def test_other_service_container(self, monkeypatch):
        monkeypatch.delenv(TERMINAL_USER_ENV, raising=False)
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "sidecar")
        assert acting_surface() == "service"

    def test_local(self, no_identity):
        assert acting_surface() == "local"


class TestAttributionHeaders:
    def test_terminal_user_is_the_end_user(self, monkeypatch):
        monkeypatch.setenv(TERMINAL_USER_ENV, "thellert")
        headers = attribution_headers()
        assert headers[END_USER_HEADER] == "thellert"
        assert headers[TAGS_HEADER] == "osprey,surface:terminal"

    def test_dispatch_worker_identity(self, monkeypatch):
        monkeypatch.delenv(TERMINAL_USER_ENV, raising=False)
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "dispatch-worker-1")
        headers = attribution_headers()
        assert headers[END_USER_HEADER] == "dispatch-worker-1"
        assert headers[TAGS_HEADER] == "osprey,surface:dispatch"

    def test_exactly_two_headers(self, no_identity):
        assert set(attribution_headers()) == {END_USER_HEADER, TAGS_HEADER}


class TestCustomHeadersRendering:
    def test_render_is_newline_separated_name_colon_value(self):
        text = render_custom_headers({"A": "1", "B": "two"})
        assert text == "A: 1\nB: two"

    def test_merge_keeps_operator_headers(self):
        merged = merge_custom_headers("X-Corp-Trace: abc123", {"x-litellm-tags": "osprey"})
        assert merged == "X-Corp-Trace: abc123\nx-litellm-tags: osprey"

    def test_merge_replaces_same_name_case_insensitively(self):
        merged = merge_custom_headers(
            "X-LiteLLM-Tags: stale\nX-Corp-Trace: abc",
            {"x-litellm-tags": "osprey"},
        )
        assert merged == "X-Corp-Trace: abc\nx-litellm-tags: osprey"

    def test_merge_from_empty(self):
        assert merge_custom_headers(None, {"a": "1"}) == "a: 1"
        assert merge_custom_headers("", {"a": "1"}) == "a: 1"

    def test_merge_drops_blank_lines(self):
        merged = merge_custom_headers("X-A: 1\n\n  \n", {"b": "2"})
        assert merged == "X-A: 1\nb: 2"


class TestApplyAttributionEnv:
    def test_litellm_gateway_sets_custom_headers(self, monkeypatch):
        monkeypatch.setenv(TERMINAL_USER_ENV, "thellert")
        environ: dict[str, str] = {}
        apply_attribution_env(environ, LITELLM_GATEWAY)
        assert environ[CUSTOM_HEADERS_ENV] == (
            f"{END_USER_HEADER}: thellert\n{TAGS_HEADER}: osprey,surface:terminal"
        )

    def test_merges_into_existing_operator_headers(self, monkeypatch):
        monkeypatch.setenv(TERMINAL_USER_ENV, "thellert")
        environ = {CUSTOM_HEADERS_ENV: "X-Corp-Trace: abc123"}
        apply_attribution_env(environ, LITELLM_GATEWAY)
        assert environ[CUSTOM_HEADERS_ENV].startswith("X-Corp-Trace: abc123\n")
        assert f"{END_USER_HEADER}: thellert" in environ[CUSTOM_HEADERS_ENV]

    def test_no_gateway_leaves_environ_untouched(self, no_identity):
        environ = {CUSTOM_HEADERS_ENV: "X-Corp-Trace: abc123"}
        apply_attribution_env(environ, None)
        assert environ == {CUSTOM_HEADERS_ENV: "X-Corp-Trace: abc123"}

        environ2: dict[str, str] = {}
        apply_attribution_env(environ2, None)
        assert environ2 == {}


class TestBuiltinTableAgreesWithAdapters:
    """The resolver-side built-in table and the adapters' ``gateway`` attribute
    are two spellings of one fact; the table exists so the launch paths need not
    import litellm. Pin them together."""

    def test_every_table_entry_matches_its_adapter(self):
        from osprey.models.provider_registry import get_provider_registry
        from osprey.models.spend_attribution import _BUILTIN_GATEWAYS

        registry = get_provider_registry()
        for name, gateway in _BUILTIN_GATEWAYS.items():
            assert registry.get_provider(name).gateway == gateway, name

    def test_every_adapter_gateway_is_in_the_table(self):
        from osprey.models.provider_registry import get_provider_registry
        from osprey.models.spend_attribution import _BUILTIN_GATEWAYS

        registry = get_provider_registry()
        declared = {
            name: registry.get_provider(name).gateway
            for name in registry.list_providers()
            if getattr(registry.get_provider(name), "gateway", None)
        }
        assert declared == dict(_BUILTIN_GATEWAYS)
