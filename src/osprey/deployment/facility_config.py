"""Facility-config validation for the canonical `ci:` shape.

`facility-config.yml` uses a provider-tagged `ci:` block (`ci: {provider:
"gitlab", ...}`) for the project's CI/CD source. An older GitLab-specific
`gitlab:` block has been removed: a config still carrying it is rejected with a
clear error naming the replacement. This module is the single normalization
chokepoint applied at every facility-config load site so the rest of the
codebase only ever sees the canonical `ci:` shape.
"""

import copy
from typing import Any

from osprey.errors import ConfigurationError


def normalize_facility_config(config: dict[str, Any]) -> dict[str, Any]:
    """Normalize a facility-config dict to the canonical `ci:` shape.

    - A legacy `gitlab:` block is rejected outright: rename it to
      `ci: {provider: "gitlab", ...}`.
    - `registry.token_env_var` defaults to `ci.token_env_var` when the
      `registry:` block doesn't already set its own.
    - A config with a `ci:` block (or neither block) otherwise passes through
      unchanged.

    Args:
        config: Parsed facility-config dict, in the canonical (`ci:`) shape.

    Returns:
        A new, normalized dict. The input `config` is never mutated.

    Raises:
        ConfigurationError: If the config carries the removed `gitlab:` block.
    """
    if config.get("gitlab") is not None:
        raise ConfigurationError(
            "facility-config uses the removed `gitlab:` block; rename it to "
            '`ci: {provider: "gitlab", ...}` (see facility-config-schema.md).'
        )

    normalized = dict(config)
    ci_block = config.get("ci")

    if ci_block is not None:
        registry_block = normalized.get("registry")
        token_env_var = ci_block.get("token_env_var")
        if registry_block is not None and token_env_var is not None:
            if "token_env_var" not in registry_block:
                registry_block = dict(registry_block)
                registry_block["token_env_var"] = token_env_var
                normalized["registry"] = registry_block

    return copy.deepcopy(normalized)
