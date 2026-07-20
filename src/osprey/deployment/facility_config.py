"""Facility-config normalization: alias legacy `gitlab:` to canonical `ci:`.

`facility-config.yml` originally had a GitLab-specific `gitlab:` block for the
project's CI/CD source. To support non-GitLab CI providers, that block is
being replaced by a provider-tagged `ci:` block (`ci: {provider: "gitlab",
...}`). This module is the single normalization chokepoint applied at every
facility-config load site so the rest of the codebase only ever sees the
canonical `ci:` shape, regardless of which shape the on-disk config uses.
"""

import copy
import warnings
from typing import Any

# Module-level warn-once flag: a single CLI invocation may call
# normalize_facility_config() at several independent load sites (scaffold
# lint, deploy, container lifecycle, seeding, ...). Without this flag each
# load site would re-emit the same deprecation warning.
_gitlab_alias_warned = False


def _reset_warn_state() -> None:
    """Reset the module-level warn-once flag.

    Test-only hook: production code never needs to warn more than once per
    process, but tests that assert "exactly one warning" need to start each
    case from a clean slate.
    """
    global _gitlab_alias_warned
    _gitlab_alias_warned = False


def _warn_once(message: str) -> None:
    global _gitlab_alias_warned
    if not _gitlab_alias_warned:
        warnings.warn(message, DeprecationWarning, stacklevel=3)
        _gitlab_alias_warned = True


def normalize_facility_config(config: dict[str, Any]) -> dict[str, Any]:
    """Normalize a facility-config dict to the canonical `ci:` shape.

    - A `gitlab:` block with no `ci:` block is aliased to
      `ci: {provider: "gitlab", **gitlab_fields}`.
    - If both `gitlab:` and `ci:` are present, `ci:` wins (the `gitlab:`
      block is dropped) and a warning is emitted.
    - A `ci:` block with no `gitlab:` block passes through unchanged.
    - `registry.token_env_var` defaults to `ci.token_env_var` when the
      `registry:` block doesn't already set its own.
    - At most one deprecation warning is emitted per process (see
      `_reset_warn_state` for tests that need to reset this).

    Args:
        config: Parsed facility-config dict, in either legacy (`gitlab:`) or
            canonical (`ci:`) shape.

    Returns:
        A new, normalized dict. The input `config` is never mutated.
    """
    normalized = dict(config)

    gitlab_block = config.get("gitlab")
    ci_block = config.get("ci")

    if gitlab_block is not None:
        if ci_block is not None:
            _warn_once(
                "facility-config has both `gitlab:` and `ci:` blocks; `ci:` "
                "takes precedence. Remove the deprecated `gitlab:` block."
            )
        else:
            _warn_once(
                "facility-config `gitlab:` is deprecated; rename it to "
                '`ci: {provider: "gitlab", ...}`.'
            )
            ci_block = {"provider": "gitlab", **gitlab_block}
            normalized["ci"] = ci_block
        del normalized["gitlab"]

    if ci_block is not None:
        registry_block = normalized.get("registry")
        token_env_var = ci_block.get("token_env_var")
        if registry_block is not None and token_env_var is not None:
            if "token_env_var" not in registry_block:
                registry_block = dict(registry_block)
                registry_block["token_env_var"] = token_env_var
                normalized["registry"] = registry_block

    return copy.deepcopy(normalized)
