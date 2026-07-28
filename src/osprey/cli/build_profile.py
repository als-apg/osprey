"""Build profile — compatibility facade over the profile pipeline's concern modules.

A build profile is a YAML file describing how to assemble a facility-specific
assistant project from an OSPREY template plus overlay files, config overrides,
and MCP server definitions. The pipeline that turns such a file (or a bundled
``--preset``) into a validated :class:`BuildProfile` lives in sibling modules,
one per concern:

- :mod:`osprey.cli.build_profile_schema` — the nested config dataclasses a
  ``profile.yml`` deserializes into (``mcp_servers``, ``lifecycle``, ``env``,
  ``services``, ``dispatch``, ``bluesky``, ``virtual_accelerator``,
  ``bluesky_panels``).
- :mod:`osprey.cli.build_profile_presets` — discovery of the bundled preset and
  trigger YAMLs, plus CLI-spelling normalization.
- :mod:`osprey.cli.build_profile_merge` — ``extends`` chain resolution, deep
  merging, ``exclude:`` subtraction, and the manifest content hashes.
- :mod:`osprey.cli.build_profile_model` — the :class:`BuildProfile` dataclass,
  its tier default, and the consistency checks a profile must pass.
- :mod:`osprey.cli.build_profile_load` — raw-mapping-to-dataclass parsing and
  the single-file :func:`load_profile` entry point.
- :mod:`osprey.cli.build_profile_resolve` — the multi-source resolution the
  ``build`` command calls: preset or file, plus ``-O`` overlays and ``--set``.

This module holds no logic of its own. The names those modules export are
re-exported below so ``from osprey.cli.build_profile import <helper>`` keeps
working.
"""

from __future__ import annotations

from .build_profile_load import _KNOWN_PROFILE_KEYS, _parse_profile, load_profile
from .build_profile_merge import (
    _apply_exclude,
    _deep_merge,
    _merge_lists,
    _resolve_extends,
    compute_preset_hash,
    compute_profile_hash,
)
from .build_profile_model import BuildProfile
from .build_profile_presets import (
    _load_preset_raw,
    _normalize_preset_name,
    _preset_exists,
    _presets_dir,
    _triggers_dir,
    list_presets,
)
from .build_profile_resolve import explicit_model_override_keys, resolve_build_profile
from .build_profile_schema import (
    BlueskyConfig,
    BlueskyPanelsConfig,
    DispatchConfig,
    EnvConfig,
    LifecycleConfig,
    LifecycleStep,
    McpServerDef,
    ServiceDef,
    VAConfig,
)

__all__ = [
    "BlueskyConfig",
    "BlueskyPanelsConfig",
    "BuildProfile",
    "DispatchConfig",
    "EnvConfig",
    "LifecycleConfig",
    "LifecycleStep",
    "McpServerDef",
    "ServiceDef",
    "VAConfig",
    "_KNOWN_PROFILE_KEYS",
    "_apply_exclude",
    "_deep_merge",
    "_load_preset_raw",
    "_merge_lists",
    "_normalize_preset_name",
    "_parse_profile",
    "_preset_exists",
    "_presets_dir",
    "_resolve_extends",
    "_triggers_dir",
    "compute_preset_hash",
    "compute_profile_hash",
    "explicit_model_override_keys",
    "list_presets",
    "load_profile",
    "resolve_build_profile",
]
