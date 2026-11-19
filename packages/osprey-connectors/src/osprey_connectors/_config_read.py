"""Readers of a rendered ``config.yml`` shared by the endpoint predicates.

:mod:`osprey_connectors.deployment.standin` and :mod:`osprey_connectors.deployment.simulator` each
ask whether an endpoint is a container this deployment stood up. Both read a
port the build projected into the config and compare it with a loopback
endpoint, so the three readers they need live here once.

The config is read as nested sections only, because that is how the rendered
``config.yml`` is read by everything that acts on it.
"""

from __future__ import annotations

import ipaddress
import re
from collections.abc import Mapping

from osprey_connectors.errors import ConfigurationError

#: The only host name that is loopback without being an address. Compared
#: case-insensitively; every other spelling has to parse as an IP address, so a
#: resolvable name that merely *happens* to point at this host is not accepted.
LOOPBACK_HOSTNAME = "localhost"


def is_loopback(host: object) -> bool:
    """Whether *host* names this machine's loopback interface; a non-string names none."""
    candidate = host.strip() if isinstance(host, str) else ""
    if not candidate:
        return False
    if candidate.lower() == LOOPBACK_HOSTNAME:
        return True
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        # A name this module cannot read is a machine it cannot vouch for.
        return False


def coerce_port(value: object) -> int | None:
    """*value* as a port number, or ``None`` when it does not name one.

    ``bool`` is rejected ahead of ``int`` on purpose: ``live_standin: true``
    says a stand-in is wanted without saying where it is, and Python would
    otherwise read it as port 1.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def nested_value(config: object, dotted: str) -> object:
    """The value of *dotted* walked as nested sections, or ``None`` when absent."""
    node: object = config
    for part in dotted.split("."):
        if not isinstance(node, Mapping):
            return None
        node = node.get(part)
    return node


#: A string that is nothing but a single unresolved env-var placeholder, e.g.
#: ``"${MISSING}"`` or ``"$MISSING"``. :func:`~osprey_connectors.config.resolve_env_vars` leaves such a
#: value verbatim when the variable is unset (there is nothing to substitute and
#: no declared default), so every consumer that reads a config value has to know
#: this shape to tell "not configured" from a value.
#:
#: The two branches are the two branches of the resolver's own pattern above,
#: character class included: the braced form takes any name without ``}`` or
#: ``:`` (a ``:-`` default always substitutes, so it never survives), the bare
#: form the shell-style identifier. Narrowing either half here would fail to
#: recognise a reference the resolver did leave verbatim.
_LONE_PLACEHOLDER = re.compile(r"^\$\{[^}:]+\}$|^\$[A-Za-z_][A-Za-z0-9_]*$")


def is_unresolved_placeholder(value: object) -> bool:
    """Whether *value* is an env-var reference :func:`~osprey_connectors.config.resolve_env_vars` left alone.

    The counterpart to :func:`~osprey_connectors.config.resolve_env_vars`: it keeps ``${VAR}`` verbatim
    when ``VAR`` is unset, so a consumer that treats what it reads as data sees
    the literal string ``"${VAR}"`` where a URL or a key belongs. This is the
    one producer of that test, so a caller never has to re-state the syntax the
    resolver accepts.

    Args:
        value: A config value, of any type. Only a ``str`` can be a placeholder.

    Returns:
        True when the value is exactly one unsubstituted reference.
    """
    return isinstance(value, str) and bool(_LONE_PLACEHOLDER.match(value))


_FLAG_ON = frozenset({"true", "yes", "on", "1"})
_FLAG_OFF = frozenset({"false", "no", "off", "0", ""})


def config_flag(value: object, *, key: str) -> bool:
    """Read an on/off config value for what it spells.

    :func:`~osprey_connectors.config.resolve_env_vars` substitutes text, so ``${USE_NS:-false}`` reaches a
    consumer as the *string* ``"false"`` — and ``bool("false")`` is True. Every
    consumer of such a switch reads it through here, so they all agree on one
    rule: a bool is itself, ``None`` is off, ``0``/``1`` are off/on, and a string
    is matched case-insensitively against ``true/yes/on/1`` and
    ``false/no/off/0`` (blank is off). Anything else — an unresolved
    ``${VAR}``, a typo, another number — is refused rather than guessed, since
    either guess would silently pick a behaviour nobody configured.

    Args:
        value: The raw config value.
        key: The dotted config key, named in the refusal.

    Returns:
        The switch's state.

    Raises:
        ConfigurationError: When the value spells neither on nor off, naming *key*.
    """
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        word = value.strip().lower()
        if word in _FLAG_ON:
            return True
        if word in _FLAG_OFF:
            return False
    raise ConfigurationError(
        f"{key} must be true or false, got {value!r}"
        + (" (an unset environment variable)" if is_unresolved_placeholder(value) else ""),
        key=key,
    )
