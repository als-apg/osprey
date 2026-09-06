"""Spelling a lane's profile pins as ``osprey init --set`` edits.

An E2E lane starts from a bundled preset and then pins the handful of keys the
lane's assertions stand on -- a port moved out of the way of the developer's
own stack, a service the lane must not deploy, a control target it must talk
to. Those pins are EDITS of the resolved preset, not another layer of
inheritance: each one states the value at the key it names, replacing whatever
the preset put there, so a lane that pins ``config.deployed_services`` to a
three-entry list gets three services and not the union with the preset's four.

That is what ``--set KEY=VALUE`` means at ``osprey init``, so the lanes state
their pins there. This module holds the one spelling of the translation, so a
lane describes its pins as the mapping they are and no lane hand-rolls the
argv.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from typing import Any


def set_pairs(edits: Mapping[str, Any]) -> list[str]:
    """Spell a mapping of profile edits as ``--set KEY=VALUE`` argv elements.

    One pair per LEAF, so a pin reaches the field it names and nothing else.
    Mappings are descended -- ``{"va_archiver": {"retention_days": 2}}`` becomes
    ``va_archiver.retention_days=2``, leaving the preset's ``host:`` and its
    cadences where they were. Scalars, lists, ``None`` and empty mappings are
    leaves and are stated whole: a list a lane pins is the list the profile
    ends up holding, which is the point of stating an edit rather than layering
    one.

    Under ``config:`` the leaf path is joined with dots and stays ONE key, which
    is how a profile's config block is written -- a flat bag of dotted paths
    into the rendered config, the spelling the presets themselves use. Stating
    a whole ``config:`` subtree as a single pair would not do the same thing: it
    replaces that subtree, and the preset's other fields under it are gone.

    Values are spelled as JSON. JSON is valid YAML, which is what the CLI parses
    the right-hand side as, and it quotes what needs quoting -- ``null``, ``[]``,
    ``true`` and strings that would otherwise read as something else all survive
    the round trip.

    Args:
        edits: Profile keys to pin, shaped like the profile itself.

    Returns:
        ``["--set", "k=v", "--set", ...]``, ready to splice into an
        ``osprey init`` argv. Each pair is one argv element, so nothing here
        depends on shell quoting.
    """
    argv: list[str] = []
    for key, value in edits.items():
        prefix = f"{key}." if key == "config" else ""
        for path, leaf in _leaves({key: value} if not prefix else value):
            argv += ["--set", f"{prefix}{path}={json.dumps(leaf)}"]
    return argv


def _leaves(mapping: Mapping[str, Any], prefix: str = "") -> Iterator[tuple[str, Any]]:
    """Every leaf of ``mapping`` as a ``(dotted path, value)`` pair.

    A non-empty mapping is descended; everything else -- scalar, list, ``None``,
    empty mapping -- is a leaf. Mirrors how ``osprey init`` itself walks a
    ``--set`` value, so a nested pin and the pairs spelled from it address the
    same fields.
    """
    for key, value in mapping.items():
        path = f"{prefix}{key}"
        if isinstance(value, Mapping) and value:
            yield from _leaves(value, f"{path}.")
        else:
            yield path, value
