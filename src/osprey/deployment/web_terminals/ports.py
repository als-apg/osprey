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
