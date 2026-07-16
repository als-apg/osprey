"""Per-user host port allocation for multi-user web terminal deployments."""

from typing import Any

_PORT_FAMILIES = ("web", "artifact", "ariel", "lattice")

# Maps each facility-config base-port field to the family key allocate_ports()
# expects. Shared by lint.py (Rule 11 port-overlap / family-completeness checks)
# and render.py (per-service port construction) so the two can't drift on which
# config field maps to which family.
FAMILY_BASE_FIELDS = {
    "web_base_port": "web",
    "artifact_base_port": "artifact",
    "ariel_base_port": "ariel",
    "lattice_base_port": "lattice",
}


def base_ports_from_config(web_terminals: dict[str, Any]) -> dict[str, int]:
    """Build the ``{family: base_port}`` dict :func:`allocate_ports` expects from a
    facility config's ``modules.web_terminals`` stanza.

    Drops any family whose base-port field is missing or not an int, so
    :func:`allocate_ports` reports it as a clear missing-family ``ValueError``
    instead of a ``TypeError``.

    Args:
        web_terminals: The already-dict-coerced ``modules.web_terminals`` section.

    Returns:
        Mapping of family name to its configured base port, containing only the
        families whose base-port field was present and an int.
    """
    base_ports: dict[str, int] = {}
    for base_field, family in FAMILY_BASE_FIELDS.items():
        value = web_terminals.get(base_field)
        if isinstance(value, int):
            base_ports[family] = value
    return base_ports


def normalize_users(users_raw: Any) -> list[dict[str, Any]]:
    """Normalize ``modules.web_terminals.users`` into explicit ``{"name", "index"}`` dicts.

    This is the canonical users-normalizer for the module; render.py delegates to it
    directly rather than re-parsing the raw roster itself. Its purpose is to let a
    decommission migration *freeze* a bare roster's positional indices onto explicit
    ``index`` fields before removing a user — once frozen, deleting an earlier entry
    can no longer shift a later user's port allocation, because the survivor's index
    no longer depends on its position in the list.

    Legacy bare strings (``"alice"``) normalize to ``{"name": "alice", "index":
    <raw list position>}``, matching the fallback ``render.py`` already uses so
    ports stay identical to what a pre-existing all-strings roster produces today.
    Already-explicit object entries (``{"name": ..., "index": ...}``) pass through
    with their explicit index preserved, regardless of list position. This makes
    the function idempotent: normalizing an already-normalized list is a no-op,
    which is what lets the freeze step be applied without checking whether a
    roster has already been frozen.

    Malformed entries — anything that isn't a string, and any dict missing a
    string ``name`` or an int ``index`` — are dropped rather than raising
    (well-formedness is lint.py's job). ``bool`` is a subclass of ``int`` in
    Python, so a dict entry
    like ``{"name": "alice", "index": True}`` would technically satisfy
    ``isinstance(index, int)``; this function deliberately treats ``bool`` as an
    invalid index type and drops such entries, since ``index: true/false`` in a
    facility config is a config typo (e.g. a YAML boolean where an int was meant),
    not a meaningful index.

    Args:
        users_raw: The raw ``modules.web_terminals.users`` value. Anything other
            than a list (including ``None``) is treated as an empty roster.

    Returns:
        New ``{"name": str, "index": int}`` dicts in config-declaration order.
        Input dicts are never mutated or returned by reference.
    """
    if not isinstance(users_raw, list):
        return []
    normalized: list[dict[str, Any]] = []
    for position, entry in enumerate(users_raw):
        if isinstance(entry, str):
            normalized.append({"name": entry, "index": position})
        elif isinstance(entry, dict):
            name = entry.get("name")
            index = entry.get("index")
            if isinstance(name, str) and isinstance(index, int) and not isinstance(index, bool):
                normalized.append({"name": name, "index": index})
    return normalized


def allocate_ports(base_ports: dict[str, int], index: int) -> dict[str, int]:
    """Allocate per-user host ports for the four web terminal port families.

    Args:
        base_ports: Facility-configured base port for each family. Must contain
            all of ``web``, ``artifact``, ``ariel``, and ``lattice``.
        index: Zero-based user index; added to each family's base port.

    Returns:
        Mapping of family name to allocated host port (``base_ports[family] + index``).

    Raises:
        ValueError: If a required family key is missing from ``base_ports``.
    """
    missing = [family for family in _PORT_FAMILIES if family not in base_ports]
    if missing:
        raise ValueError(f"base_ports is missing required family key(s): {missing}")
    return {family: base_ports[family] + index for family in _PORT_FAMILIES}
