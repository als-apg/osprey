"""Model tier resolution — maps abstract tier names to provider-specific model IDs."""

from __future__ import annotations

VALID_TIERS = frozenset(("haiku", "sonnet", "opus"))


def resolve_model_id(provider: str, model_id_or_tier: str, config_path: str | None = None) -> str:
    """Resolve a tier name to a provider-specific model ID.

    If model_id_or_tier is a tier (haiku/sonnet/opus), looks up the concrete
    model ID from api.providers[provider].models[tier]. Non-tier values pass
    through unchanged for backward compatibility.
    """
    if model_id_or_tier not in VALID_TIERS:
        return model_id_or_tier
    from osprey.models.config import get_provider_config  # deferred: avoid import cycle

    provider_cfg = get_provider_config(provider, config_path)
    models = provider_cfg.get("models", {})
    return models.get(model_id_or_tier, model_id_or_tier)
