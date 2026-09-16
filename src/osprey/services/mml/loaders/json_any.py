"""Loader for family-keyed JSON exports.

Accepts every JSON form an MML export takes in the wild:

* **flat** --- one map of families (``{"BPM": {...}, "HCM": {...}}``);
* **wrapped** --- the same map under an ``ao`` or ``AO`` key, optionally with
  an ``ad``/``AD`` sibling;
* **system-keyed** --- top-level keys are systems, each a map of families.

A family is recognised by the structural rule
:func:`osprey.services.channel_finder.tools.preview_database.is_family_dict`.
The loader only decodes: bare ``NaN``/``Infinity``/``-Infinity`` tokens become
the normaliser's ``"NaN"``/``"Inf"``/``"-Inf"`` strings, and every other value,
including misspelled keys and nested cell arrays, is returned as parsed.
"""

from __future__ import annotations

import json
from pathlib import Path

import click

from osprey.services.channel_finder.tools.preview_database import is_family_dict
from osprey.services.mml.loaders import LoadedInput

__all__ = ["load_json"]

#: Bare JSON constants mapped to the normaliser's canonical non-finite strings.
_NON_FINITE = {"NaN": "NaN", "Infinity": "Inf", "-Infinity": "-Inf"}

_WRAPPER_KEYS = ("ao", "AO")
_AD_KEYS = ("ad", "AD")
_EXPORT_KEY = "_export"
_AO_SUFFIX = ".ao.json"
_AD_SUFFIX = ".ad.json"


def load_json(path: str | Path) -> LoadedInput:
    """Load a family-keyed JSON export.

    Args:
        path: The JSON file. When its name ends in ``.ao.json`` and a
            ``.ad.json`` file with the same stem sits beside it, that file is
            read as the AD.

    Returns:
        The decoded input. A top-level ``_export`` block is moved out of
        ``ao`` into ``export``.

    Raises:
        click.ClickException: When the file cannot be read or parsed, or holds
            no family in any accepted form; the message names the file.
    """
    source = Path(path)
    data = _read(source)
    if not isinstance(data, dict):
        raise _no_family(source)

    body = data
    ad: dict | None = None
    export = data.get(_EXPORT_KEY) if isinstance(data.get(_EXPORT_KEY), dict) else None

    wrapper = _wrapper_key(data)
    if wrapper is not None:
        body = data[wrapper]
        ad = next((data[k] for k in _AD_KEYS if isinstance(data.get(k), dict)), None)
        if export is None and isinstance(body.get(_EXPORT_KEY), dict):
            export = body[_EXPORT_KEY]

    if _has_family(body):
        system_keyed = False
    elif _is_system_keyed(body):
        system_keyed = True
    else:
        raise _no_family(source)

    ao = {key: value for key, value in body.items() if key != _EXPORT_KEY or export is None}

    if ad is None:
        ad = _paired_ad(source)
    if export is None and ad is not None and isinstance(ad.get(_EXPORT_KEY), dict):
        export = ad[_EXPORT_KEY]

    return LoadedInput(ao=ao, ad=ad, export=export, system_keyed=system_keyed, source=source)


def _read(source: Path) -> object:
    """Parse *source*, mapping bare non-finite tokens to canonical strings."""
    try:
        with source.open(encoding="utf-8") as handle:
            return json.load(handle, parse_constant=_NON_FINITE.__getitem__)
    except (OSError, ValueError) as exc:
        raise click.ClickException(f"Cannot read JSON from {source}: {exc}") from exc


def _no_family(source: Path) -> click.ClickException:
    return click.ClickException(
        f"No middle-layer family found in {source}: expected a map of families, "
        "an 'ao'/'AO' wrapper around one, or a map of systems holding families."
    )


def _wrapper_key(data: dict) -> str | None:
    """The ``ao``/``AO`` key whose value is a map rather than a family, if any."""
    for key in _WRAPPER_KEYS:
        value = data.get(key)
        if isinstance(value, dict) and not is_family_dict(key, value):
            return key
    return None


def _has_family(body: dict) -> bool:
    return any(isinstance(key, str) and is_family_dict(key, value) for key, value in body.items())


def _is_system_keyed(body: dict) -> bool:
    """True when every non-underscore top-level value is a map holding families."""
    systems = [
        value for key, value in body.items() if isinstance(key, str) and not key.startswith("_")
    ]
    return bool(systems) and all(
        isinstance(value, dict) and _has_family(value) for value in systems
    )


def _paired_ad(source: Path) -> dict | None:
    """Read ``<stem>.ad.json`` beside ``<stem>.ao.json``, when it exists."""
    if not source.name.endswith(_AO_SUFFIX):
        return None
    sibling = source.with_name(source.name[: -len(_AO_SUFFIX)] + _AD_SUFFIX)
    if not sibling.is_file():
        return None
    data = _read(sibling)
    return data if isinstance(data, dict) else None
