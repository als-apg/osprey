"""Tests that ``.env.example`` derives its providers from the registry.

``project/env.example.j2`` iterates the ``provider_api_keys`` context entry
(built from ``osprey.models.provider_registry.PROVIDER_API_KEYS``) instead of
hand-listing providers, so a provider added to the registry automatically
appears in the emitted example.

Its former sibling ``project/env.j2`` is gone. The render writes no ``.env`` at
all now — the deployment's one secret store is the repo-root ``.env``, written
by ``osprey init``'s shell harvest and ``osprey up``'s token mint, neither of
which renders a template.
"""

from __future__ import annotations

from osprey.cli.templates.manager import TemplateManager
from osprey.cli.templates.scaffolding import provider_api_key_entries, service_token_var_entries
from osprey.deployment.container_lifecycle import _SERVICE_TOKEN_VARS
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
        # The same derivations the real context supplies. Rendering without
        # them would silently produce an empty section — a Jinja loop over an
        # undefined name emits nothing — and the assertions below would then be
        # testing the fixture rather than the template.
        "service_token_vars": service_token_var_entries(),
        "env_required": [],
        "env_defaults": {},
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


class TestEnvExampleJ2:
    def test_lists_every_registry_provider(self):
        rendered = _render("project/env.example.j2", _base_ctx({}))
        for entry in provider_api_key_entries():
            assert f"{entry['var']}=" in rendered

    def test_no_stale_langfuse_block(self):
        rendered = _render("project/env.example.j2", _base_ctx({}))
        assert "LANGFUSE" not in rendered

    def test_documents_every_minted_service_credential(self):
        """Completeness is the point of deriving the list rather than writing it.

        Asserted against ``_SERVICE_TOKEN_VARS`` itself — the map the deploy
        path mints from — rather than against
        :func:`service_token_var_entries`, which would only prove the
        derivation agrees with itself. Every declared variable appears whatever
        services a given deployment enables, because a variable an operator
        cannot see documented is one they cannot pin.
        """
        rendered = _render("project/env.example.j2", _base_ctx({}))

        declared = {var for token_vars in _SERVICE_TOKEN_VARS.values() for var in token_vars}
        assert declared  # the loop below must not pass vacuously
        for var in declared:
            assert var in rendered, f"{var} is minted by deploy but undocumented in .env.example"

    def test_minted_credentials_are_commented_out(self):
        """They are minted, not guessed: an uncommented ``VAR=`` would make a
        copied ``.env`` claim an empty value the deploy would then honour."""
        rendered = _render("project/env.example.j2", _base_ctx({}))

        declared = {var for token_vars in _SERVICE_TOKEN_VARS.values() for var in token_vars}
        for line in rendered.splitlines():
            stripped = line.strip()
            for var in declared:
                if stripped.startswith(f"{var}="):
                    raise AssertionError(f"{var} must be commented out, got: {line!r}")
