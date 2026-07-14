"""Guard: shipped config templates expose the same LLM providers, and every
provider they declare is actually routable by the Claude Code harness.

Regression for the ds4 gap (2026-06): the ``ds4`` (local DeepSeek V4) provider
was added to the model registry and the *generic* ``project/config.yml.j2`` but
not to the per-app preset configs. Building ``--preset control-assistant
--set provider=ds4`` therefore produced ``claude_code.provider: ds4`` with no
matching ``api.providers.ds4`` stanza, and the resolver raised
``Unknown Claude Code provider 'ds4'`` — every agentic call died at fixture
setup. The root cause is structural: the ``api.providers`` block is duplicated
across templates instead of inherited from one framework default, so "add a
provider" silently means "edit N files" and one got missed.

These tests encode the invariant that prevents the next provider from drifting:
the full-featured presets carry an identical provider set, and each provider in
any shipped template resolves through ``ClaudeCodeModelResolver`` (no
half-wired provider that builds but can't route).
"""

import re
from pathlib import Path

import pytest
import yaml

import osprey
from osprey.cli.claude_code_resolver import ClaudeCodeModelResolver

TEMPLATES = Path(osprey.__file__).parent / "templates"

# Presets that intentionally expose the *full* provider menu. ariel_standalone
# is deliberately excluded — it ships a curated subset (logbook-only preset);
# see test_ariel_standalone_is_curated_subset for the weaker invariant it holds.
FULL_TEMPLATES = (
    "project/config.yml.j2",
    "apps/control_assistant/config.yml.j2",
    "apps/hello_world/config.yml.j2",
)
CURATED_TEMPLATE = "apps/ariel_standalone/config.yml.j2"


def _api_providers(rel: str) -> dict:
    """Parse the ``api.providers`` block out of a config template.

    The ``api:`` section is literal YAML (no Jinja), so we slice from the
    top-level ``api:`` line up to the next ``# ===`` banner and parse it. This
    reads the shipped template exactly as ``osprey build`` renders it.
    """
    text = (TEMPLATES / rel).read_text()
    match = re.search(r"(?m)^api:\n(?:.*\n)*?(?=^# =)", text)
    assert match, f"no top-level `api:` section found in {rel}"
    block = yaml.safe_load(match.group(0)) or {}
    return block.get("api", {}).get("providers", {}) or {}


def test_full_presets_include_ds4():
    """ds4 must be present in every full preset (the specific regression)."""
    for rel in FULL_TEMPLATES:
        providers = _api_providers(rel)
        assert "ds4" in providers, (
            f"{rel} is missing the `ds4` api.providers stanza — building with "
            f"provider=ds4 would resolve to an Unknown Claude Code provider."
        )


def test_full_presets_have_identical_provider_sets():
    """The full presets must expose the same provider menu.

    This is the invariant that would have caught the ds4 omission: a provider
    added to one full template but not the others fails here.
    """
    sets = {rel: frozenset(_api_providers(rel)) for rel in FULL_TEMPLATES}
    reference = sets[FULL_TEMPLATES[0]]
    mismatches = {rel: ps ^ reference for rel, ps in sets.items() if ps != reference}
    assert not mismatches, (
        "full preset templates disagree on api.providers; symmetric differences "
        f"vs {FULL_TEMPLATES[0]}: "
        + ", ".join(f"{rel}: {sorted(diff)}" for rel, diff in mismatches.items())
    )


def test_ariel_standalone_is_curated_subset():
    """ariel_standalone ships a curated subset — but never a stray provider.

    It may omit providers (it's the lean logbook preset), yet every provider it
    *does* declare must also exist in the full menu, so it can't reference a
    provider the framework doesn't otherwise ship.
    """
    curated = frozenset(_api_providers(CURATED_TEMPLATE))
    full = frozenset(_api_providers(FULL_TEMPLATES[1]))  # control_assistant
    extra = curated - full
    assert not extra, (
        f"{CURATED_TEMPLATE} declares providers absent from the full menu: {sorted(extra)}"
    )


@pytest.mark.parametrize("rel", FULL_TEMPLATES + (CURATED_TEMPLATE,))
def test_every_declared_provider_resolves(rel):
    """No half-wired provider: each api.providers entry must route via the
    Claude Code resolver (built-in or custom-proxy with a base_url)."""
    providers = _api_providers(rel)
    for name in providers:
        spec = ClaudeCodeModelResolver.resolve({"provider": name}, providers)
        assert spec is not None, f"{rel}: provider {name!r} resolved to None"


def test_ds4_stanza_resolves_to_deepseek_tiers():
    """The ds4 stanza must resolve to the DeepSeek tier models end-to-end."""
    providers = _api_providers("apps/control_assistant/config.yml.j2")
    spec = ClaudeCodeModelResolver.resolve({"provider": "ds4"}, providers)
    assert spec.tier_to_model["haiku"] == "deepseek-v4-flash"
    assert spec.tier_to_model["sonnet"] == "deepseek-v4-pro"
    assert spec.tier_to_model["opus"] == "deepseek-v4-pro"
    # Claude-Code-facing var is stripped of /v1 (issue #312); the proxy upstream
    # keeps it for /chat/completions forwarding.
    assert spec.env_block["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8000"
    assert spec.upstream_base_url == "http://127.0.0.1:8000/v1"
