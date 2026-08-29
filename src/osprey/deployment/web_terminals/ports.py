"""Per-user host port allocation for multi-user web terminal deployments.

Every port here is a band of the deployment's block: family *F* starts at
``port_base + LAYOUT[F].offset`` and user *i* takes ``+ i``, one hundred users
per family. The base is never looked up in this module — it is handed down by
the caller that resolved it (:func:`osprey.port_layout.resolve_port_base`), so
a deployment on a non-default base is never described in terms of the default
one.
"""

from collections.abc import Mapping
from typing import Any

from osprey.port_layout import INDEX_MAX, default_port, resolve_port_base
from osprey.registry.web import FRAMEWORK_WEB_SERVERS


def _family_name(key: str) -> str:
    """A registry entry's port-family name (its key unless overridden)."""
    return FRAMEWORK_WEB_SERVERS[key].port_family or key


# Maps each `modules.web_terminals` base-port field to the family key allocate_ports()
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


def default_base_ports(base: int) -> dict[str, int]:
    """Return each port family's layout base at one deployment base.

    Every family — ``web`` included — has a default, because every family is a
    per-index slot of :data:`osprey.port_layout.LAYOUT` and the layout gives it
    an offset. That is what makes ``modules.web_terminals`` deployable with no
    port keys at all: the stanza names an override, never a requirement.

    Args:
        base: The base the deployment resolved, from
            :func:`osprey.port_layout.resolve_port_base`. There is deliberately
            no default — a caller that guessed the base would publish ports the
            rest of the deployment does not use.

    Returns:
        Mapping of family name to the first port of its band at ``base``.

    Raises:
        ValueError: If ``base`` is outside the range a block can start at.
    """
    return {family: default_port(family, 0, base=base) for family in _PORT_FAMILIES}


def base_ports_from_config(web_terminals: dict[str, Any], *, base: int) -> dict[str, int]:
    """Build the ``{family: base_port}`` dict :func:`allocate_ports` expects from a
    facility config's ``modules.web_terminals`` stanza.

    A family whose base-port field is missing (or not an int) falls back to its
    layout band at ``base`` (:func:`default_base_ports`), so the result always
    carries every family and :func:`allocate_ports` can never fail for a
    missing one. A config value is an absolute port and wins outright, which is
    how a family is moved out of the block entirely.

    Args:
        web_terminals: The already-dict-coerced ``modules.web_terminals`` section.
        base: The base the deployment resolved. Keyword-only and without a
            default so that no caller can silently allocate against the layout
            default while the rest of the deployment sits elsewhere.

    Returns:
        Mapping of family name to its effective base port (config value, else
        the family's layout band at ``base``), containing every family.

    Raises:
        ValueError: If ``base`` is outside the range a block can start at.
    """
    defaults = default_base_ports(base)
    base_ports: dict[str, int] = {}
    for base_field, family in FAMILY_BASE_FIELDS.items():
        value = web_terminals.get(base_field)
        if isinstance(value, int) and not isinstance(value, bool):
            base_ports[family] = value
        else:
            base_ports[family] = defaults[family]
    return base_ports


#: The layout slot nginx's own listener takes — the gateway of the block, the
#: one port a deployment is reached on from off-host.
NGINX_SLOT = "nginx"


def resolve_nginx_port(config: Mapping[str, Any] | None) -> int:
    """Return the host port nginx listens on for this deployment.

    ``modules.web_terminals.nginx_port`` when it is set, and the gateway slot
    of the deployment's own port block when it is not. An unset key is not a
    missing value: the layout gives nginx an offset, so a config that never
    mentions the port still names one, derived from the base this deployment
    resolved rather than from the framework default. That is what lets a
    ``modules.web_terminals`` stanza be written with no port keys at all.

    A value that is present but is not a port number is a different thing
    entirely — an author's intent that cannot be honoured — so it refuses
    rather than falling back. ``True`` is not a port here: :class:`bool` is a
    subclass of :class:`int`, and ``nginx_port: true`` is a typo, not port 1.

    Args:
        config: The rendered deployment config (``build/config.yml`` as
            loaded), or ``None``. The whole mapping, not the
            ``modules.web_terminals`` subtree, because the base comes from
            ``deployment.port_base`` in the same document — reading the port
            without the base is what publishes a default-block port on a
            deployment that lives elsewhere.

    Returns:
        The configured port, else ``resolve_port_base(config)`` plus the
        ``nginx`` slot's offset.

    Raises:
        ValueError: If ``modules.web_terminals.nginx_port`` is set to anything
            other than an integer, or if ``deployment.port_base`` is outside
            the range a block can start at.
    """
    root: Mapping[str, Any] = config if isinstance(config, Mapping) else {}
    modules = root.get("modules")
    web_terminals = modules.get("web_terminals") if isinstance(modules, Mapping) else None
    value = web_terminals.get("nginx_port") if isinstance(web_terminals, Mapping) else None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if value is None:
        return default_port(NGINX_SLOT, base=resolve_port_base(root))
    raise ValueError(
        f"modules.web_terminals.nginx_port is {value!r}, which is not a port number. "
        "Set it to an integer, or remove the key entirely to publish nginx on the "
        "gateway slot of this deployment's port block (deployment.port_base + 0)."
    )


def _index_refusal(index: Any) -> str:
    """Compose the message for a user index past the end of every family band.

    Args:
        index: The user index that was asked for.

    Returns:
        A message naming the band, the family whose band the index would run
        into, and the ``<family>_base_port`` escape that moves a family out of
        the block.
    """
    escapes = ", ".join(f"modules.web_terminals.{field}" for field in FAMILY_BASE_FIELDS)
    return (
        f"user index {index!r} is out of range: each port family holds indices "
        f"0..{INDEX_MAX}, so every family — {', '.join(_PORT_FAMILIES)} — would take a "
        f"port belonging to the next family's band. A deployment's block holds "
        f"{INDEX_MAX + 1} users; run a second deployment on its own "
        f"deployment.port_base, or move a family out of the block by setting its base "
        f"port to an absolute value ({escapes})."
    )


def allocate_ports(base_ports: dict[str, int], index: int) -> dict[str, int]:
    """Allocate per-user host ports for every web terminal port family.

    Args:
        base_ports: Effective base port for each family (config value or layout
            band — see :func:`base_ports_from_config`). Must contain every
            :data:`FAMILY_BASE_FIELDS` family.
        index: Zero-based user index; added to each family's base port. A
            family band is one hundred ports wide, so this runs
            0..:data:`osprey.port_layout.INDEX_MAX`.

    Returns:
        Mapping of family name to allocated host port (``base_ports[family] + index``).

    Raises:
        ValueError: If a required family key is missing from ``base_ports``, or
            if ``index`` is negative or past :data:`osprey.port_layout.INDEX_MAX`,
            where the user's ports would land in the next family's band.
    """
    missing = [family for family in _PORT_FAMILIES if family not in base_ports]
    if missing:
        raise ValueError(f"base_ports is missing required family key(s): {missing}")
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index <= INDEX_MAX:
        raise ValueError(_index_refusal(index))
    return {family: base_ports[family] + index for family in _PORT_FAMILIES}
