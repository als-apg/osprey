"""ARIEL enhancement module factory.

This module provides factory functions for creating enhancement modules.
"""

from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal

EnhancerStage = Literal["inline", "catchup", "all"]

if TYPE_CHECKING:
    from osprey.services.ariel_search.config import ARIELConfig
    from osprey.services.ariel_search.enhancement.base import BaseEnhancementModule


def create_enhancers_from_config(
    config: "ARIELConfig",
    *,
    stage: EnhancerStage = "inline",
    names: Iterable[str] | None = None,
) -> list["BaseEnhancementModule"]:
    """Create enhancement module instances for enabled modules in execution order.

    Uses the central Osprey registry for module discovery with explicit
    execution ordering. The stage filter reads ``runs_inline`` from the class
    before it is instantiated or configured, so a module outside the requested
    stage never runs, nor fails ``configure()``, in that caller.

    Args:
        config: ARIEL configuration with enhancement_modules settings
        stage: ``inline`` keeps modules with ``runs_inline=True`` (the ingest
            pipeline), ``catchup`` keeps those with ``runs_inline=False``, and
            ``all`` keeps both.
        names: When given, only these module names are considered; execution
            order and the enabled check still apply.

    Returns:
        List of configured enhancement module instances, in execution order

    Raises:
        ValueError: If ``stage`` is not one of ``inline``, ``catchup``, ``all``.
    """
    if stage not in ("inline", "catchup", "all"):
        raise ValueError(f"unknown enhancer stage {stage!r}")
    wanted = None if names is None else set(names)
    from osprey.registry import get_registry

    registry = get_registry()
    registry.initialize(silent=True)
    ordered_names = registry.list_ariel_enhancement_modules()
    enhancers: list[BaseEnhancementModule] = []
    for name in ordered_names:
        if wanted is not None and name not in wanted:
            continue
        if not config.is_enhancement_module_enabled(name):
            continue
        result = registry.get_ariel_enhancement_module(name)
        if result is None:
            continue
        cls, _reg = result
        if not _in_stage(cls, stage):
            continue
        enhancer = cls()
        if hasattr(enhancer, "configure"):
            module_config = config.get_enhancement_module_config(name)
            if module_config:
                enhancer.configure(module_config)
        enhancers.append(enhancer)
    return enhancers


def _in_stage(cls: type, stage: EnhancerStage) -> bool:
    """Return whether a module class belongs to ``stage``, without instantiating it."""
    if stage == "all":
        return True
    runs_inline = bool(getattr(cls, "runs_inline", True))
    return runs_inline if stage == "inline" else not runs_inline


def get_enhancer_names() -> list[str]:
    """Return list of available enhancer names.

    Returns:
        List of enhancer names in execution order
    """
    from osprey.registry import get_registry

    registry = get_registry()
    registry.initialize(silent=True)
    return registry.list_ariel_enhancement_modules()
