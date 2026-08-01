"""Tests that the env-file templates derive providers from the registry.

``project/env.j2`` and ``project/env.example.j2`` iterate the
``provider_api_keys`` context entry (built from
``osprey.models.provider_registry.PROVIDER_API_KEYS``) instead of
hand-listing providers, so a provider added to the registry automatically
appears in scaffolded env files — and a detected key's real value is always
written (the old hand-list dropped ALS_APG_API_KEY and discarded the
detected ARGO_API_KEY value).
"""

from __future__ import annotations

from osprey.cli.templates.manager import TemplateManager
from osprey.cli.templates.scaffolding import provider_api_key_entries
from osprey.models.provider_registry import PROVIDER_API_KEYS


def _render(template_name: str, ctx: dict) -> str:
    manager = TemplateManager()
    return manager.jinja_env.get_template(template_name).render(**ctx)


def _base_ctx(env: dict) -> dict:
    return {
        "project_name": "test-project",
        "project_root": "/tmp/test-project",
        "current_python_env": "/usr/bin/python3",
        "env": env,
        "provider_api_keys": provider_api_key_entries(),
    }


class TestProviderApiKeyEntries:
    def test_matches_registry_keyed_providers(self):
        entries = provider_api_key_entries()
        expected = {v for v in PROVIDER_API_KEYS.values() if v is not None}
        assert {e["var"] for e in entries} == expected

    def test_keyless_providers_excluded(self):
        providers = {e["provider"] for e in provider_api_key_entries()}
        assert "ollama" not in providers
        assert "vllm" not in providers


class TestEnvJ2:
    def test_detected_keys_render_with_real_values(self):
        rendered = _render(
            "project/env.j2",
            _base_ctx({"ALS_APG_API_KEY": "als-secret", "ARGO_API_KEY": "argo-secret"}),
        )
        # ALS_APG was missing from the old hand-list entirely.
        assert "ALS_APG_API_KEY=als-secret" in rendered
        # The old template wrote a literal $${USER} instead of the detected value.
        assert "ARGO_API_KEY=argo-secret" in rendered
        assert "$${USER}" not in rendered

    def test_undetected_keys_are_omitted(self):
        rendered = _render("project/env.j2", _base_ctx({"CBORG_API_KEY": "x"}))
        assert "CBORG_API_KEY=x" in rendered
        assert "OPENAI_API_KEY" not in rendered


class TestEnvExampleJ2:
    def test_lists_every_registry_provider(self):
        rendered = _render("project/env.example.j2", _base_ctx({}))
        for entry in provider_api_key_entries():
            assert f"{entry['var']}=" in rendered

    def test_no_stale_langfuse_block(self):
        rendered = _render("project/env.example.j2", _base_ctx({}))
        assert "LANGFUSE" not in rendered

    def test_documents_minted_service_credentials(self):
        rendered = _render("project/env.example.j2", _base_ctx({}))
        assert "ARIEL_DB_PASSWORD" in rendered
        assert "ZO_ROOT_USER_PASSWORD" in rendered
