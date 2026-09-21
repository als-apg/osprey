"""System resolution and merging for MML imports.

Every input to ``osprey mml import`` ends up under one or more system tokens.
The token of an input is resolved by the first rule that applies:

1. A system-keyed input contributes its own top-level keys and refuses an
   explicit ``--system``.
2. A flat input takes the explicit ``--system`` token when one is given.
3. Otherwise the exporter's ``_export.submachine``, then ``SubMachine`` from
   the input's AD (the paired ``.ad.json`` or the ``.mat``'s AD variable).
4. Otherwise the import is a usage error naming the file.

:func:`merge_inputs` then folds every input into one AO keyed
``{system: {family: body}}`` and one AD keyed ``{system: ad}``. The AO also
carries two top-level bookkeeping keys, which every reader skips because they
start with ``_``: ``_exports`` (each system's ``_export`` block) and
``_import_order`` (the systems in encounter order, which a key-sorting writer
would otherwise lose).
"""

from __future__ import annotations

from typing import Any

import click

from osprey.services.mml.loaders import LoadedInput
from osprey.services.mml.normalize import normalize_family

__all__ = ["EXPORTS_KEY", "IMPORT_ORDER_KEY", "input_systems", "merge_inputs", "resolve_system"]

#: Top-level AO key holding ``{system: export block}``.
EXPORTS_KEY = "_exports"

#: Top-level AO key holding the system tokens in encounter order.
IMPORT_ORDER_KEY = "_import_order"


def resolve_system(loaded: LoadedInput, explicit: str | None) -> str | None:
    """Resolve the system token of one input.

    Args:
        loaded: The decoded input.
        explicit: The ``--system`` token given for this input, if any.

    Returns:
        The system token of a flat input, or ``None`` for a system-keyed input,
        whose systems are its own top-level keys (see :func:`input_systems`).

    Raises:
        click.UsageError: A system-keyed input was given ``--system``, a token
            starts with ``_``, or no rule yields a token for a flat input.
    """
    if loaded.system_keyed:
        if explicit is not None:
            raise click.UsageError(
                f"{loaded.source} is keyed by system and names its own systems; "
                "drop --system for this input."
            )
        return None

    if explicit is not None:
        return _checked_token(explicit, loaded, "--system")

    submachine = _text(loaded.export.get("submachine")) if loaded.export else None
    if submachine is not None:
        return _checked_token(submachine, loaded, "_export.submachine")

    ad_submachine = _text(loaded.ad.get("SubMachine")) if loaded.ad else None
    if ad_submachine is not None:
        return _checked_token(ad_submachine, loaded, "AD.SubMachine")

    raise click.UsageError(
        f"Cannot tell which system {loaded.source} belongs to: it has no "
        "_export.submachine and no AD.SubMachine; pass --system TOKEN for it."
    )


def input_systems(loaded: LoadedInput, explicit: str | None) -> list[str]:
    """The system tokens one input contributes, in file order.

    Args:
        loaded: The decoded input.
        explicit: The ``--system`` token given for this input, if any.

    Returns:
        The input's own non-``_`` top-level keys when it is system-keyed,
        otherwise the single resolved token.

    Raises:
        click.UsageError: As :func:`resolve_system`.
    """
    token = resolve_system(loaded, explicit)
    if token is not None:
        return [token]
    systems = []
    for key in loaded.ao:
        if not isinstance(key, str) or key.startswith("_"):
            continue
        if _text(key) != key:
            raise click.UsageError(
                f"Invalid system key {key!r} in {loaded.source}: a system token is a "
                "non-empty name without surrounding whitespace."
            )
        systems.append(key)
    return systems


def merge_inputs(
    inputs: list[tuple[LoadedInput, str | None]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Merge decoded inputs into one AO and one AD keyed by system.

    Args:
        inputs: Each decoded input with its ``--system`` token (or its already
            resolved token, which resolves to itself), in command-line order.

    Returns:
        ``(ao, ad)``. ``ao`` is ``{system: {family: normalised body}}`` plus
        ``_exports`` and ``_import_order``; ``ad`` is ``{system: ad}`` for the
        systems that carry an AD. Keys starting with ``_`` inside a system
        (such as ``_description``) are copied unchanged, every other map is
        normalised as a family body whether or not it holds a channel key.

    Raises:
        click.UsageError: A system token is resolved twice, or as
            :func:`resolve_system`.
    """
    ao: dict[str, Any] = {}
    ad: dict[str, Any] = {}
    exports: dict[str, Any] = {}
    order: list[str] = []
    seen: dict[str, LoadedInput] = {}

    for loaded, explicit in inputs:
        for system in input_systems(loaded, explicit):
            if system in seen:
                raise click.UsageError(
                    f"System {system!r} is imported twice, from {seen[system].source} "
                    f"and from {loaded.source}; give each input a distinct system."
                )
            seen[system] = loaded
            order.append(system)

            families = loaded.ao[system] if loaded.system_keyed else loaded.ao
            ao[system] = _merge_system(families)

            system_ad = _system_ad(loaded, system)
            if system_ad is not None:
                ad[system] = system_ad
            if loaded.export is not None:
                exports[system] = loaded.export

    ao[EXPORTS_KEY] = exports
    ao[IMPORT_ORDER_KEY] = order
    return ao, ad


def _merge_system(families: dict) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, value in families.items():
        if isinstance(name, str) and name.startswith("_"):
            out[name] = value
        elif isinstance(value, dict):
            out[name] = normalize_family(value)
        else:
            out[name] = value
    return out


def _system_ad(loaded: LoadedInput, system: str) -> dict | None:
    """The AD of *system*: a per-system entry of a system-keyed AD, else the whole AD."""
    if loaded.ad is None:
        return None
    if loaded.system_keyed and isinstance(loaded.ad.get(system), dict):
        return _without_export(loaded.ad[system])
    return _without_export(loaded.ad)


def _without_export(ad: dict) -> dict:
    return {key: value for key, value in ad.items() if key != "_export"}


def _text(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _checked_token(token: object, loaded: LoadedInput, origin: str) -> str:
    text = _text(token)
    if text is None or text.startswith("_"):
        raise click.UsageError(
            f"Invalid system token {token!r} for {loaded.source} (from {origin}): "
            "a system token is a non-empty name that does not start with '_'."
        )
    return text
