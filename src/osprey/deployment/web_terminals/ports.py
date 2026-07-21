"""Per-user host port allocation for multi-user web terminal deployments."""

from typing import Any

from osprey.registry.web import FRAMEWORK_WEB_SERVERS


def _family_name(key: str) -> str:
    """A registry entry's port-family name (its key unless overridden)."""
    return FRAMEWORK_WEB_SERVERS[key].port_family or key


# Maps each facility-config base-port field to the family key allocate_ports()
# expects: the fixed "web" family (the terminal itself) plus ONE family per
# FRAMEWORK_WEB_SERVERS companion server. Derived, never hand-listed — the
# per-user containers share the host network namespace, so a companion server
# without its own per-user family collides with itself across users;
# registering a server in registry/web.py is what wires it here, by
# construction. Shared by lint.py (Rule 11 port-overlap / family-completeness
# checks) and render.py (per-service port construction) so the two can't drift
# on which config field maps to which family.
FAMILY_BASE_FIELDS = {
    "web_base_port": "web",
    **{f"{_family_name(key)}_base_port": _family_name(key) for key in FRAMEWORK_WEB_SERVERS},
}

# Registry-declared default base port per companion family ("web" has none: the
# terminal's own base port is facility-chosen, required in config). These are
# what keep a facility config written before a companion server existed
# deploying unchanged — the new family allocates from its registry default
# unless the config overrides `<family>_base_port`.
DEFAULT_BASE_PORTS = {
    _family_name(key): defn.multi_user_base_port
    for key, defn in FRAMEWORK_WEB_SERVERS.items()
    if isinstance(defn.multi_user_base_port, int)
}

# Env var each companion family's allocated port is exported under in every
# per-user container (the compose template's per-service environment block).
# Same `port_env_var` derivation server_launcher's config reader consumes —
# the two ends of one contract. The "web" family is not here: the terminal
# itself exports the OSPREY_TERMINAL_WEB_PORT/OSPREY_WEB_PORT pair, handled
# explicitly by the compose template.
PANEL_ENV_VARS = {
    _family_name(key): defn.port_env_var for key, defn in FRAMEWORK_WEB_SERVERS.items()
}

_PORT_FAMILIES = tuple(FAMILY_BASE_FIELDS.values())


def base_ports_from_config(web_terminals: dict[str, Any]) -> dict[str, int]:
    """Build the ``{family: base_port}`` dict :func:`allocate_ports` expects from a
    facility config's ``modules.web_terminals`` stanza.

    A companion family whose base-port field is missing (or not an int) falls
    back to its registry default (:data:`DEFAULT_BASE_PORTS`) — the
    zero-migration path for configs written before that companion server
    existed. ``web`` has no default: a missing/malformed ``web_base_port`` is
    dropped, so :func:`allocate_ports` reports it as a clear missing-family
    ``ValueError`` instead of a ``TypeError``.

    Args:
        web_terminals: The already-dict-coerced ``modules.web_terminals`` section.

    Returns:
        Mapping of family name to its effective base port (config value, else
        registry default), containing only families that resolved one.
    """
    base_ports: dict[str, int] = {}
    for base_field, family in FAMILY_BASE_FIELDS.items():
        value = web_terminals.get(base_field)
        if isinstance(value, int):
            base_ports[family] = value
        elif family in DEFAULT_BASE_PORTS:
            base_ports[family] = DEFAULT_BASE_PORTS[family]
    return base_ports


def allocate_ports(base_ports: dict[str, int], index: int) -> dict[str, int]:
    """Allocate per-user host ports for every web terminal port family.

    Args:
        base_ports: Effective base port for each family (config value or
            registry default — see :func:`base_ports_from_config`). Must
            contain every :data:`FAMILY_BASE_FIELDS` family.
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
