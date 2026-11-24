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
    declared_header_names,
    gateway_for,
    merge_custom_headers,
    render_custom_headers,
)
from osprey_connectors.identity import AUDIT_IDENTITY_ENV, TERMINAL_USER_ENV


@pytest.fixture
def no_identity(monkeypatch):
    """Neither identity rung is set — the single-user laptop case."""
    monkeypatch.delenv(TERMINAL_USER_ENV, raising=False)
    monkeypatch.delenv(AUDIT_IDENTITY_ENV, raising=False)


class TestGatewayFor:
    @pytest.mark.parametrize(
        ("provider", "providers", "expected"),
        [
            pytest.param("als-apg", None, LITELLM_GATEWAY, id="builtin_litellm_providers-als-apg"),
            pytest.param("cborg", None, LITELLM_GATEWAY, id="builtin_litellm_providers-cborg"),
            pytest.param("anthropic", None, None, id="direct_providers_have_no_gateway-anthropic"),
            pytest.param("openai", None, None, id="direct_providers_have_no_gateway-openai"),
            pytest.param(
                "my-gw",
                {"my-gw": {"base_url": "https://gw.example/v1", "gateway": "litellm"}},
                LITELLM_GATEWAY,
                id="custom_provider_declares_gateway_in_config",
            ),
            pytest.param(
                "my-gw",
                {"my-gw": {"base_url": "https://gw.example/v1"}},
                None,
                id="custom_provider_without_key_is_unattributed",
            ),
            pytest.param("no-such-provider", None, None, id="unknown_provider_is_unattributed"),
            # The table is a default for a name with nothing said about it. A
            # deployment that declares what fronts its own endpoint is not overruled
            # by a framework table that cannot know.
            pytest.param(
                "cborg",
                {"cborg": {"base_url": "https://gw.example/v1", "gateway": "none"}},
                None,
                id="a_declared_gateway_wins_over_the_builtin_table",
            ),
            pytest.param(
                "als-apg",
                {"als-apg": {"base_url": "https://gw.example/v1", "gateway": "house-gw"}},
                "house-gw",
                id="a_builtin_can_be_pointed_at_a_different_gateway",
            ),
            pytest.param(
                "cborg",
                {"cborg": {"base_url": "https://gw.example/v1"}},
                LITELLM_GATEWAY,
                id="a_block_that_says_nothing_falls_back_to_the_table",
            ),
            pytest.param(
                "cborg", {"cborg": {"gateway": ""}}, None, id="an_empty_declaration_means_direct"
            ),
        ],
    )
    def test_gateway_for(self, provider, providers, expected):
        assert gateway_for(provider, providers) == expected


class TestActingSurface:
    @pytest.mark.parametrize(
        ("terminal_user", "audit_identity", "expected"),
        [
            pytest.param("alice", "dispatch-worker-0", "terminal", id="terminal_user_wins"),
            pytest.param(None, "dispatch-worker-0", "dispatch", id="dispatch_worker"),
            pytest.param(None, "sidecar", "service", id="other_service_container"),
            # Neither identity rung is set: the single-user laptop case.
            pytest.param(None, None, "local", id="local"),
        ],
    )
    def test_acting_surface(self, monkeypatch, terminal_user, audit_identity, expected):
        """``None`` means the variable is unset."""
        for name, value in (
            (TERMINAL_USER_ENV, terminal_user),
            (AUDIT_IDENTITY_ENV, audit_identity),
        ):
            if value is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, value)
        assert acting_surface() == expected


class TestAttributionHeaders:
    def test_terminal_user_is_the_end_user(self, monkeypatch):
        monkeypatch.setenv(TERMINAL_USER_ENV, "alice")
        headers = attribution_headers()
        assert headers[END_USER_HEADER] == "alice"
        assert headers[TAGS_HEADER] == "osprey,surface:terminal"

    def test_dispatch_worker_identity(self, monkeypatch):
        monkeypatch.delenv(TERMINAL_USER_ENV, raising=False)
        monkeypatch.setenv(AUDIT_IDENTITY_ENV, "dispatch-worker-1")
        headers = attribution_headers()
        assert headers[END_USER_HEADER] == "dispatch-worker-1"
        assert headers[TAGS_HEADER] == "osprey,surface:dispatch"

    @pytest.mark.usefixtures("no_identity")
    def test_exactly_two_headers(self):
        assert set(attribution_headers()) == {END_USER_HEADER, TAGS_HEADER}


class TestCustomHeadersRendering:
    def test_render_is_newline_separated_name_colon_value(self):
        text = render_custom_headers({"A": "1", "B": "two"})
        assert text == "A: 1\nB: two"

    @pytest.mark.parametrize(
        ("existing", "added", "expected"),
        [
            pytest.param(
                "X-Corp-Trace: abc123",
                {"x-litellm-tags": "osprey"},
                "X-Corp-Trace: abc123\nx-litellm-tags: osprey",
                id="merge_keeps_operator_headers",
            ),
            pytest.param(
                "X-LiteLLM-Tags: stale\nX-Corp-Trace: abc",
                {"x-litellm-tags": "osprey"},
                "X-Corp-Trace: abc\nx-litellm-tags: osprey",
                id="merge_replaces_same_name_case_insensitively",
            ),
            pytest.param(None, {"a": "1"}, "a: 1", id="merge_from_empty-none"),
            pytest.param("", {"a": "1"}, "a: 1", id="merge_from_empty-empty_string"),
            pytest.param(
                "X-A: 1\n\n  \n", {"b": "2"}, "X-A: 1\nb: 2", id="merge_drops_blank_lines"
            ),
        ],
    )
    def test_merge_custom_headers(self, existing, added, expected):
        assert merge_custom_headers(existing, added) == expected


class TestApplyAttributionEnv:
    def test_litellm_gateway_sets_custom_headers(self, monkeypatch):
        monkeypatch.setenv(TERMINAL_USER_ENV, "alice")
        environ: dict[str, str] = {}
        apply_attribution_env(environ, LITELLM_GATEWAY)
        assert environ[CUSTOM_HEADERS_ENV] == (
            f"{END_USER_HEADER}: alice\n{TAGS_HEADER}: osprey,surface:terminal"
        )

    def test_merges_into_existing_operator_headers(self, monkeypatch):
        monkeypatch.setenv(TERMINAL_USER_ENV, "alice")
        environ = {CUSTOM_HEADERS_ENV: "X-Corp-Trace: abc123"}
        apply_attribution_env(environ, LITELLM_GATEWAY)
        assert environ[CUSTOM_HEADERS_ENV].startswith("X-Corp-Trace: abc123\n")
        assert f"{END_USER_HEADER}: alice" in environ[CUSTOM_HEADERS_ENV]

    @pytest.mark.usefixtures("no_identity")
    def test_no_gateway_leaves_environ_untouched(self):
        environ = {CUSTOM_HEADERS_ENV: "X-Corp-Trace: abc123"}
        apply_attribution_env(environ, None)
        assert environ == {CUSTOM_HEADERS_ENV: "X-Corp-Trace: abc123"}

        environ2: dict[str, str] = {}
        apply_attribution_env(environ2, None)
        assert environ2 == {}


class TestDeclaredHeaderNames:
    def test_names_every_header_the_variable_carries(self):
        environ = {CUSTOM_HEADERS_ENV: "X-Corp-Trace: abc\nx-litellm-tags: osprey"}
        assert declared_header_names(environ) == frozenset({"x-corp-trace", "x-litellm-tags"})

    def test_an_unset_or_blank_variable_declares_nothing(self):
        assert declared_header_names({}) == frozenset()
        assert declared_header_names({CUSTOM_HEADERS_ENV: "\n  \n"}) == frozenset()

    def test_a_line_without_a_header_name_declares_nothing(self):
        environ = {CUSTOM_HEADERS_ENV: "no-colon-here\n: value\nX-A: 1"}
        assert declared_header_names(environ) == frozenset({"x-a"})

    def test_the_attribution_headers_are_declared_once_applied(self, monkeypatch):
        monkeypatch.setenv(TERMINAL_USER_ENV, "alice")
        environ = {CUSTOM_HEADERS_ENV: "X-Corp-Trace: abc123"}
        apply_attribution_env(environ, LITELLM_GATEWAY)
        assert declared_header_names(environ) == {END_USER_HEADER, TAGS_HEADER, "x-corp-trace"}


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
