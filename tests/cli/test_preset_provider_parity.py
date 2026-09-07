"""Guard: one provider catalog, and every provider in it is actually routable.

Regression for the ds4 gap (2026-06): the ``ds4`` (local DeepSeek V4) provider
was added to the model registry and the generic ``project/config.yml.j2`` but
not to the per-app config templates. Building ``--preset control-assistant
--set provider=ds4`` therefore produced ``claude_code.provider: ds4`` with no
matching ``api.providers.ds4`` stanza, and the resolver raised ``Unknown Claude
Code provider 'ds4'`` — every agentic call died at fixture setup. The root cause
was structural: the ``api.providers`` block was duplicated across templates, so
"add a provider" silently meant "edit N files" and one got missed.

The duplication is gone. One packaged ``providers.yml`` ships beside the
presets, ``osprey init`` writes it into the deployment, and the build renders it
into ``api.providers`` — so the sets cannot disagree, because there is one set.
What is still worth pinning is that nothing has grown a second home for the
fact, that ds4 is in the one catalog, and that every entry in it resolves
through ``ClaudeCodeModelResolver`` (no half-wired provider that builds but
cannot route).
"""

import re
from pathlib import Path

import pytest
import yaml

import osprey
from osprey.build.claude_code_resolver import (
    ClaudeCodeModelResolver,
    _without_unresolved_base_urls,
)
from osprey.profiles.providers import packaged_catalog_path
from osprey_connectors.config import resolve_env_vars

TEMPLATES = Path(osprey.__file__).parent / "templates"


def _api_providers() -> dict:
    """The provider stanzas a build renders into ``api.providers``."""
    catalog = yaml.safe_load(packaged_catalog_path().read_text(encoding="utf-8")) or {}
    return catalog.get("providers") or {}


#: ``${VAR}`` / ``${VAR:-default}`` — the two forms ``resolve_env_vars`` accepts,
#: read here only to name the variables a catalog entry's endpoint comes from.
_ENDPOINT_VAR = re.compile(r"\$\{([^}:]+)(?::-[^}]*)?\}")

#: Any endpoint will do: these tests ask whether a stanza routes, not where to.
PLACEHOLDER_ENDPOINT = "https://gateway.example.org/v1"


def _endpoint_env_vars() -> set[str]:
    """The env vars the catalog's ``base_url`` values are spelled in terms of.

    A gateway that each site hosts itself ships as a reference rather than a
    host (``base_url: ${ALS_APG_BASE_URL}``), so a deployment that exports
    nothing has no endpoint for it. Naming the variables here lets the routing
    test supply one, instead of resolving against a literal ``"${VAR}"`` that
    is not a URL at all.
    """
    return {
        var
        for entry in _api_providers().values()
        if isinstance(entry, dict)
        for var in _ENDPOINT_VAR.findall(str(entry.get("base_url") or ""))
    }


def test_the_catalog_includes_ds4():
    """ds4 must be in the catalog (the specific regression)."""
    assert "ds4" in _api_providers(), (
        "the packaged catalog is missing the `ds4` stanza — building with "
        "provider=ds4 would resolve to an Unknown Claude Code provider."
    )


def test_only_one_template_renders_the_catalog_and_it_restates_no_entry():
    """The structural fix: one home for the fact, so nothing can drift from it.

    This replaces the old set-comparison across templates. Two things make the
    ds4 gap unrepeatable: exactly one template renders ``api.providers`` at all,
    and it renders the catalog whole rather than spelling entries, so there is
    no second place a provider could be added to and no first place it could be
    left out of.
    """
    renderers = [
        str(path.relative_to(TEMPLATES))
        for path in sorted(TEMPLATES.rglob("config.yml.j2"))
        if "\n  providers:" in path.read_text(encoding="utf-8")
    ]
    assert renderers == ["project/config.yml.j2"], (
        f"more than one template renders api.providers: {renderers}. The catalog "
        "is the one source; a template must render it, not restate it."
    )

    body = (TEMPLATES / "project" / "config.yml.j2").read_text(encoding="utf-8")
    rendered = body.split("\n  providers:", 1)[1].strip().splitlines()[0]
    assert rendered.startswith("{{"), (
        f"the providers block spells content of its own: {rendered!r}. It has to "
        "interpolate the catalog, or an entry could be added here and nowhere else."
    )
    for name in _api_providers():
        assert f"\n    {name}:" not in body, (
            f"provider {name!r} is spelled literally in the template as well as "
            "in providers.yml — two homes for one fact"
        )


def test_every_declared_provider_resolves(monkeypatch):
    """No half-wired provider: each entry must route via the Claude Code
    resolver (built-in or custom-proxy with a base_url).

    The catalog is expanded first, with every endpoint variable exported, so
    what each stanza is judged on is the config a deployment actually runs. On
    the raw text a stanza that ships ``base_url: ${VAR}`` resolves either way —
    the literal is a non-empty string, so it passes for a URL and the guard
    goes quiet exactly where an unexported variable would have broken the
    launch.
    """
    for var in _endpoint_env_vars():
        monkeypatch.setenv(var, PLACEHOLDER_ENDPOINT)
    providers = resolve_env_vars(_api_providers())
    assert providers, "the packaged catalog declares no providers"
    for name in providers:
        spec = ClaudeCodeModelResolver.resolve({"provider": name}, providers)
        assert spec is not None, f"provider {name!r} resolved to None"
        endpoints = (spec.upstream_base_url, spec.env_block.get("ANTHROPIC_BASE_URL"))
        for endpoint in endpoints:
            assert "${" not in (endpoint or ""), (
                f"provider {name!r} routes to {endpoint!r} — an expanded "
                "catalog still carries a variable reference where the endpoint "
                "belongs"
            )


def test_a_gateway_with_no_endpoint_exported_is_refused_rather_than_routed(monkeypatch):
    """The other half: nothing exported must not become a hostname.

    ``resolve_env_vars`` keeps ``${VAR}`` verbatim when the variable is unset,
    so the value a launch reads is the reference itself. The launch path blanks
    it before resolving, which turns "nobody named an endpoint" into the
    refusal that names the variable — rather than a spec that would hand
    Claude Code the string ``"${ALS_APG_BASE_URL}"`` as its base URL.
    """
    for var in _endpoint_env_vars():
        monkeypatch.delenv(var, raising=False)
    providers = resolve_env_vars(_api_providers())
    assert providers["als-apg"]["base_url"] == "${ALS_APG_BASE_URL}", (
        "the catalog no longer spells the als-apg endpoint as a variable "
        "reference — this guard is aimed at the wrong shape"
    )

    with pytest.raises(ValueError, match="ALS_APG_BASE_URL"):
        ClaudeCodeModelResolver.resolve(
            {"provider": "als-apg"}, _without_unresolved_base_urls(providers)
        )


def test_ds4_stanza_resolves_to_deepseek_tiers():
    """The ds4 stanza must resolve to the DeepSeek tier models end-to-end."""
    spec = ClaudeCodeModelResolver.resolve({"provider": "ds4"}, _api_providers())
    assert spec.tier_to_model["haiku"] == "deepseek-v4-flash"
    assert spec.tier_to_model["sonnet"] == "deepseek-v4-pro"
    assert spec.tier_to_model["opus"] == "deepseek-v4-pro"
    # Claude-Code-facing var is stripped of /v1 (issue #312); the proxy upstream
    # keeps it for /chat/completions forwarding.
    assert spec.env_block["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8000"
    assert spec.upstream_base_url == "http://127.0.0.1:8000/v1"
