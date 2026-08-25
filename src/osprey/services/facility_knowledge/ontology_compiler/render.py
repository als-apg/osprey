"""Deterministic JSON rendering of a compiled ontology payload.

The compiler turns an authored LinkML schema into the same table shape
:func:`~osprey.services.facility_knowledge.ttl_generator.ontology_map.parse_ontology`
already reads.  That table is committed to the repository, so the bytes have to
be reproducible: recompiling an unchanged schema must produce a file ``git
diff`` reports as unchanged, and a ``--check`` run must be able to compare
rendered text against the file on disk without normalising anything first.

This module is the single place that decides those bytes.  It sorts every key,
indents by two spaces, keeps non-ASCII characters literal, and ends the text
with exactly one newline.  It also stamps a ``_generated`` header naming the
source schema, so an operator who opens the artifact learns not to edit it by
hand.  The header carries :attr:`~pathlib.Path.name` only — never a full path —
because the artifact is committed and an absolute path would differ per
checkout.

This module is **pure**: standard library only, no ``linkml_runtime``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

#: Provenance line stamped into every rendered artifact under ``_generated``.
#: Formatted with the source schema's file name only, so the rendered bytes do
#: not depend on where the repository is checked out.
GENERATED_HEADER = "Generated from {name} by `osprey knowledge compile-ontology`. Do not edit."


def render_json(payload: Mapping[str, object], source: Path) -> str:
    """Render a compiled ontology payload as deterministic JSON text.

    Args:
        payload: The compiled table, carrying the ``root``, ``family_to_class``
            and ``classes`` keys ``parse_ontology`` accepts.  A ``_generated``
            key is added.  Compiled payloads never carry one themselves; should
            one appear, the stamped header replaces it.
        source: The authored schema the payload was compiled from.  Only its
            file name reaches the output.

    Returns:
        The artifact text: keys sorted, indented by two spaces, non-ASCII
        characters kept literal, ending in exactly one newline.
    """
    document = {**payload, "_generated": GENERATED_HEADER.format(name=source.name)}
    return json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
