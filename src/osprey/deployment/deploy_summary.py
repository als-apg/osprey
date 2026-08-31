"""Post-deploy endpoint summary.

Every ``osprey up`` ends by saying what is reachable where, derived
from the published host ports in the rendered compose files (the same source
:mod:`osprey.deployment.host_ports` preflights) — so the summary needs no
per-facility knowledge. A service on the host's network namespace publishes no
port and appears in no ``ports:`` block, so its bound ports are derived from the
rendered config the same way the preflight derives them; leaving them out would
have made a host-mode deploy's summary silently short of the very services the
operator asked for. A web-terminal line is always included: a project
*without* a web tier says "(not configured)" explicitly, turning "nothing
listens on the landing port" from a silent absence into a stated fact.

The block, not an alphabet
--------------------------
Every framework port is ``deployment.port_base`` plus a fixed offset
(:mod:`osprey.port_layout`), so the summary is laid out the way the block is:
the heading names the thousand ports this deployment reserved, and the rows are
grouped into the layout's own tiers — gateway, dispatch, services, panels,
stores, facility — in ascending offset order. Sorting the same addresses by
service name would have hidden that shape behind a spelling, and left an
operator with no way to see that the port they do not recognise is simply the
next slot along.

Reserved is not served
----------------------
The panel bands are the exception the block's own arithmetic creates: every
roster user is allocated a port in every family whether or not the persona
they run ever starts that server. Listing the whole roster beside every band —
and listing a band no persona serves at all — told an operator that people
answer at addresses nothing listens on. So each band names only the users whose
rendered project declares that panel, a family nobody serves gets no row, and
one closing note says which bands the block still reserves. Where the personas
cannot be read the wide, pre-persona reading comes back: it is the honest
answer when nothing on disk says otherwise, and this module never fails a
deploy that otherwise succeeded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from osprey.cli import output
from osprey.deployment.compose_generator import resolve_project_name
from osprey.deployment.graphdb_service import (
    CONTAINER_BOLT_PORT,
    CONTAINER_HTTP_PORT,
    GRAPHDB_SERVICE_NAME,
)
from osprey.deployment.host_ports import (
    _WILDCARD_HOSTS,
    _WORKER_SERVICE_PREFIX,
    HostPortBinding,
    derive_host_network_bindings,
    parse_host_port_bindings,
)
from osprey.deployment.web_terminals.ports import resolve_nginx_port
from osprey.port_layout import (
    LAYOUT,
    PORT_BASE_CONFIG_KEY,
    SLOTS_BY_NAME,
    block_range,
    index_bounds,
    resolve_port_base,
)
from osprey.utils.logger import get_logger

if TYPE_CHECKING:
    from osprey.deployment.web_terminals.auth_credentials import SeededLoginsReport

logger = get_logger("deployment.summary")

# Framework services fronted by HTTP on every port they bind, shown as
# clickable URLs. Everything else (databases, channel-access gateways, ...) is
# shown as a bare address. Keyed on compose service names, mirroring
# host_ports._SERVICE_REMEDY_KEYS.
_HTTP_SERVICES = {
    "openobserve",
    "event-dispatcher",
    "bluesky-bridge",
    "tiled",
    "bluesky-web",
}

# What each port of a multi-port service IS: the word a merged row puts in front
# of the address, and whether that address opens in a browser. The graph store
# is the first single compose service to bind two host ports, and its name alone
# cannot say which of them speaks HTTP — prefixing bolt with ``http://`` would
# hand the operator a link that cannot open, and a binary protocol behind a URL
# is worse than no link. Keyed on the CONTAINER port, which the image fixes, so
# a project that moved either published port still resolves the right vocabulary
# for each; the same split host_ports._SERVICE_PORT_REMEDY_KEYS makes.
#
# THIS DICT'S ORDER IS THE PRINTED ORDER of a merged row, so keep each service's
# ports in the order their LAYOUT slots sit — bolt (graphdb_bolt, +802) before
# browser (graphdb_http, +803). Sorting by the slots themselves would need a
# third table mapping container port to slot name for one service; the pin is
# test_the_graph_store_is_one_row_that_says_which_port_is_which, which fails
# loudly if these two lines are ever swapped.
_MULTI_PORT_ROLES: dict[str, dict[int, tuple[str, bool]]] = {
    GRAPHDB_SERVICE_NAME: {
        CONTAINER_BOLT_PORT: ("bolt", False),
        CONTAINER_HTTP_PORT: ("browser", True),
    },
}

# Which layout slot each compose service publishes, so a row's display tier and
# its place in the block are read off `osprey.port_layout.LAYOUT` rather than off
# a second ordering kept here. A service naming two slots (a Bluesky lane, the
# graph store) takes the first: both members of every such pair share a tier, and
# the pair's first member is the lower offset.
#
# host_ports._SERVICE_LAYOUT_SLOTS answers a neighbouring question — which slots
# a binding might be sitting on, so the preflight can tell "moves with the block"
# from "was hand-placed" — and `test_deploy_summary.py` pins the two against each
# other. Three entries here are display-only and deliberately absent there:
# `nginx` and `auth` publish the gateway a deployment is reached on but are never
# preflighted by name, and `virtual-accelerator` is out of the block by design
# (instance 1 serves Channel Access on 5064) yet still has to be shown under a
# tier. It reads as the machine it stands in for, so it shares the stand-in slot.
_SERVICE_SLOTS = {
    "nginx": "nginx",
    "auth": "auth",
    "event-dispatcher": "dispatcher",
    _WORKER_SERVICE_PREFIX: "worker",
    "openobserve": "openobserve",
    "qmd": "qmd",
    "tiled": "tiled",
    "bluesky-web": "bluesky_web",
    "bluesky-bridge": "bluesky",
    "bluesky-va-bridge": "bluesky",
    "bluesky-live-bridge": "bluesky",
    "virtual-accelerator": "va_standin",
    "live-standin": "va_standin",
    GRAPHDB_SERVICE_NAME: "graphdb_bolt",
    "postgresql": "postgres",
    "mongodb": "mongo",
}

#: Compose service names that carry a per-member suffix off one shared name, so
#: every member of the band resolves to the band's one slot. Workers are
#: ``dispatch-worker-<i>``; the per-user terminals are ``web-<user>`` (the
#: companion families run inside that same container and publish no service of
#: their own).
_INDEXED_SERVICE_PREFIXES = (_WORKER_SERVICE_PREFIX, "web")

#: The tiers in display order, taken from the layout's own ordering rather than
#: listed again: :data:`~osprey.port_layout.LAYOUT` is in ascending offset order,
#: so the first appearance of each tier IS the order the sections print in.
_TIER_ORDER = tuple(dict.fromkeys(entry.tier for entry in LAYOUT))

#: Where a service the layout does not name is shown. The facility band exists
#: for exactly this — a deployment's own services, which the framework does not
#: place — so a row nothing here recognises lands there rather than inventing a
#: seventh section or being dropped from the summary.
_UNPLACED_TIER = "facility"

#: Sort position of the landing-page row inside its tier. Below any real host
#: port, so the front door heads the gateway section it shares with nginx's own
#: binding — a reader looking for "where do I start" finds it first.
_LANDING_SORT = -1

#: Appended to an address a service binds directly on the host namespace, where
#: nothing is published and ``docker ps`` shows no mapping to read it off.
_HOST_NETWORK_NOTE = "  (host network)"

#: Separator between the addresses of one service that answers at several.
_ADDRESS_SEPARATOR = " · "


def _http_service(service: str, container_port: int | None = None) -> bool:
    """Whether a binding is fronted by HTTP, so its address is shown as a URL.

    Multi-port services resolve per binding, on the port inside the container:
    that is the number the image fixes, so a published binding and a
    host-network derived one — which :mod:`osprey.deployment.host_ports` labels
    with the same canonical port — cannot describe the same endpoint two ways.
    An unrecognised port of such a service stays a bare address, since nothing
    here knows it answers HTTP.

    Workers are indexed (``dispatch-worker-1``, ``-2``, …) off one shared name,
    so the index is dropped before the lookup — the same reduction
    :mod:`osprey.deployment.host_ports` makes to find their remedy key. They
    reach the summary only on the host network, where they bind directly.

    Args:
        service: Compose service name.
        container_port: Port inside the container this binding maps to, when
            known. ``None`` reads as "not one of the HTTP ports" for a
            multi-port service, and is ignored for every single-port one.

    Returns:
        True when the address should carry an ``http://`` scheme.
    """
    roles = _MULTI_PORT_ROLES.get(service)
    if roles is not None:
        return roles.get(container_port or 0, ("", False))[1]
    return service in _HTTP_SERVICES or service.startswith(f"{_WORKER_SERVICE_PREFIX}-")


def _slot_name(service: str) -> str:
    """The layout slot ``service`` publishes, or ``""`` if the layout has none.

    Three ways a name resolves, in order: an indexed service reduced to the one
    name its whole band shares (``dispatch-worker-2`` -> ``dispatch-worker``,
    ``web-alice`` -> ``web``), the compose-service table, and finally the slot
    names themselves — a panel row is emitted under its port family, and a
    family name IS its slot name, so those need no table entry at all.

    Args:
        service: Compose service name, index or user suffix included.

    Returns:
        A key of :data:`~osprey.port_layout.SLOTS_BY_NAME`, or the empty string
        for a service the framework does not place.
    """
    for prefix in _INDEXED_SERVICE_PREFIXES:
        if service.startswith(f"{prefix}-"):
            service = prefix
            break
    slot = _SERVICE_SLOTS.get(service, "")
    if slot:
        return slot
    return service if service in SLOTS_BY_NAME else ""


def _slot_holding(host_port: int, base: int | None) -> str:
    """The layout slot whose band ``host_port`` falls in, or ``""``.

    The fallback for a service no name resolves: a port still sitting inside
    this deployment's block was placed by the layout whatever the service is
    called, and saying which band it landed in beats filing a framework-placed
    port under the facility's own section. :data:`~osprey.port_layout.LAYOUT` is
    in ascending offset order, so walking it backwards finds the last band that
    starts at or below the offset — which is the band that holds it.

    Args:
        host_port: The published or derived host port.
        base: The base this deployment resolved, or ``None`` when it could not
            be resolved at all, in which case no port can be placed.

    Returns:
        A key of :data:`~osprey.port_layout.SLOTS_BY_NAME`, or the empty string
        when the port is outside the block or lands in no band.
    """
    if base is None:
        return ""
    first, last = block_range(base)
    if not first <= host_port <= last:
        return ""
    offset = host_port - base
    for entry in reversed(LAYOUT):
        low, high = index_bounds(entry)
        if entry.offset + low <= offset <= entry.offset + high:
            return entry.name
    return ""


def _placement(service: str, host_port: int = 0, base: int | None = None) -> tuple[str, int]:
    """The tier ``service`` is shown under, and its offset inside the block.

    Both come from one slot lookup, so a service cannot be sorted into one
    section and printed under another. The name decides it wherever the name is
    known, because a service's identity outlives whichever port it was moved to;
    only a service nothing recognises is placed by where it actually binds.

    Args:
        service: Compose service name, or a port-family name for a panel row.
        host_port: A port the service binds, used only for the fallback.
        base: The base this deployment resolved, for the same fallback.

    Returns:
        ``(tier, offset)``. A service that neither the name table nor the block
        can place takes the facility band, which is the section a deployment's
        own services belong to.
    """
    name = _slot_name(service) or _slot_holding(host_port, base)
    slot = SLOTS_BY_NAME.get(name) or SLOTS_BY_NAME[_UNPLACED_TIER]
    return (slot.tier, slot.offset)


def _binding_address(binding: HostPortBinding) -> str:
    """The address one published or derived binding answers at.

    Args:
        binding: One host port, parsed from a compose file or derived from the
            rendered config.

    Returns:
        ``host:port``, carrying an ``http://`` scheme when the service speaks
        HTTP on that container port. A wildcard bind is shown on loopback,
        where a service bound to every interface always answers.
    """
    host = "127.0.0.1" if binding.host_ip in _WILDCARD_HOSTS else binding.host_ip
    address = f"{host}:{binding.host_port}"
    if _http_service(binding.service, binding.container_port):
        return f"http://{address}"
    return address


def _service_address(service: str, bindings: list[HostPortBinding]) -> str:
    """One service's whole address, however many ports it answers at.

    A service is one thing, so it gets one row: the graph store's bolt port and
    its Browser port belong to the same store, and two rows carrying one name
    invite the reader to treat them as two services with a spare port between
    them. Each address is prefixed with what that port IS whenever the service
    declares roles for its ports, since "127.0.0.1:10802 · 127.0.0.1:10803" says
    nothing about which one a browser can open.

    Args:
        service: Compose service name.
        bindings: Every binding that service publishes or derives. A single
            binding renders exactly as it did before roles existed.

    Returns:
        The addresses joined by :data:`_ADDRESS_SEPARATOR`, with the
        host-network note appended once when every binding carries it.
    """
    roles = _MULTI_PORT_ROLES.get(service, {})
    order = list(roles)
    ordered = sorted(
        bindings,
        key=lambda b: (
            order.index(b.container_port) if b.container_port in order else len(order),
            b.host_port,
        ),
    )
    all_host_network = all(binding.host_network for binding in ordered)
    parts: list[str] = []
    for binding in ordered:
        address = _binding_address(binding)
        label = roles.get(binding.container_port or 0, ("", False))[0]
        if label and len(ordered) > 1:
            address = f"{label} {address}"
        if binding.host_network and not all_host_network:
            address = f"{address}{_HOST_NETWORK_NOTE}"
        parts.append(address)
    joined = _ADDRESS_SEPARATOR.join(parts)
    return f"{joined}{_HOST_NETWORK_NOTE}" if all_host_network and parts else joined


#: Roster names listed in full on a panel row before it gives up and counts
#: them instead. Four fits a terminal line beside the address; a fifty-user
#: facility would push the row past any width and say nothing a count does not.
_ROSTER_NAMES_SHOWN = 4

#: The port family the terminal itself takes, as
#: :data:`~osprey.deployment.web_terminals.ports.FAMILY_BASE_FIELDS` names it.
#: The one family with no companion server behind it and no panel id to declare:
#: every roster user has a terminal, which is what having a roster entry MEANS,
#: so no persona's ``web.panels`` can add or remove it.
_TERMINAL_FAMILY = "web"

#: The service column of the one row that reports the bands nothing serves.
#: Public because the tests key on it rather than on its spelling, and because
#: a reader filtering the entries (the summary card keeps only ``http://``
#: addresses) should be able to name it rather than pattern-match the text.
RESERVED_BANDS_LABEL = "(reserved)"

#: Sort position of that row inside the panels tier. Above every real family
#: offset, so the note trails the bands it is a footnote to rather than landing
#: among them.
_RESERVED_SORT = 1_000_000_000


def _roster(config: dict) -> list[tuple[str, int]]:
    """This deployment's web-terminal roster as ``(name, index)``, in order.

    A deployment with the module on and no roster is single-user, which is
    index 0 — the same reading :func:`allocate_ports` is given everywhere else,
    and the reason a single-user deployment cannot share a base with a
    multi-user one on the same host.

    Args:
        config: Loaded configuration dictionary.

    Returns:
        The roster, or an empty list when the module is off.
    """
    from osprey.deployment.web_terminals.personas import normalize_users

    web_terminals = (config.get("modules") or {}).get("web_terminals") or {}
    if not web_terminals.get("enabled"):
        return []
    entries = [
        (str(entry.get("name") or ""), entry.get("index"))
        for entry in normalize_users(web_terminals.get("users"))
    ]
    roster = [(name, index) for name, index in entries if isinstance(index, int)]
    return roster or [("", 0)]


def _who_serves(names: list[str]) -> str:
    """The ``(who)`` suffix of one band: the names, or how many there are.

    Args:
        names: The roster names on this band, in roster order. Empty for the
            single-user deployment, whose one entry has no name to print.

    Returns:
        The names joined, a count past :data:`_ROSTER_NAMES_SHOWN`, or ``""``.
    """
    if len(names) > _ROSTER_NAMES_SHOWN:
        return f"{len(names)} users"
    return ", ".join(names)


def _families_a_project_serves(project_config: object) -> set[str]:
    """The port families one rendered project's panels actually listen on.

    Read off ``web.panels`` exactly as the terminal reads it
    (``web_terminal.app._load_panel_config``, and the health probe's
    :func:`~osprey.health.core.web_panels._resolve_targets` beside it): a
    built-in id is on when it is ``true`` or a mapping that has not switched
    itself off, and :data:`~osprey.profiles.web_panels.UNIVERSAL_PANELS` is on
    whatever the config says. A fourth reading of that block here would let the
    summary claim a panel the container never starts.

    Panel ids and port families are two namespaces (``artifacts`` is served by
    registry key ``artifact`` on family ``artifact``; ``lattice`` by
    ``lattice_dashboard`` on family ``lattice``), so the crossing goes through
    :data:`~osprey.registry.web.PANEL_ID_TO_REGISTRY_KEY` and the definition's
    own ``port_family`` rather than through a table kept here.

    Args:
        project_config: A persona's parsed rendered ``config.yml``. Anything
            that is not a mapping reads as a project declaring no panels.

    Returns:
        Family names, always including :data:`_TERMINAL_FAMILY` — a project
        that declares nothing still has the terminal the roster entry IS.
    """
    from osprey.profiles.web_panels import BUILTIN_PANELS, UNIVERSAL_PANELS
    from osprey.registry.web import FRAMEWORK_WEB_SERVERS, PANEL_ID_TO_REGISTRY_KEY

    web = project_config.get("web") if isinstance(project_config, dict) else None
    declared = web.get("panels") if isinstance(web, dict) else None

    enabled = set(UNIVERSAL_PANELS)
    for panel_id, spec in (declared if isinstance(declared, dict) else {}).items():
        if panel_id in BUILTIN_PANELS and (
            spec is True or (isinstance(spec, dict) and spec.get("enabled", True))
        ):
            enabled.add(panel_id)

    families = {_TERMINAL_FAMILY}
    for panel_id in enabled:
        key = PANEL_ID_TO_REGISTRY_KEY.get(panel_id)
        if key is not None:
            definition = FRAMEWORK_WEB_SERVERS[key]
            families.add(definition.port_family or key)
    return families


def _families_by_user(config: dict, project_root: Path | str | None) -> dict[str, set[str]] | None:
    """Which port families each roster user actually serves, or ``None``.

    The narrowing behind :func:`_panel_entries`: a user's persona names a
    rendered project, and that project's ``web.panels`` says which companion
    servers its container starts. Every user is allocated a port in every
    family regardless — the allocator reserves the whole block whatever the
    roster runs — so without this walk the summary names every user beside
    every band and tells an operator that people answer where they do not.

    ALL OR NOTHING, deliberately. ``None`` means "nothing on disk answers this
    question", and the caller then reports the wide, pre-persona bands. A
    partial answer would be worse than either: a user whose persona could not
    be resolved would silently vanish from every band he might be on, which
    reads as "he serves nothing" rather than as "nothing here knows".

    Advisory at every step, like everything else this module derives — an
    unbuilt persona project, a catalog entry with no ``project_path``, an
    ``authorization`` stanza that does not parse, a roster entry
    :func:`~osprey.deployment.web_terminals.personas.normalize_users` cannot
    place — each degrades to ``None`` rather than failing the summary that a
    successful deploy ends with. Nothing here raises: two of this module's
    three entry points call it without a guard, and a summary that crashes
    after a deploy has already succeeded reports a failure that did not happen.

    Args:
        config: Loaded configuration dictionary.
        project_root: The deployment repo, against which a catalog entry's
            ``project_path`` resolves. ``None`` falls back to the rendered
            config's own ``project_root``, which is the only anchor
            ``osprey status`` has — it is handed a config and compose files and
            no root at all.

    Returns:
        ``{user: families}`` covering the whole roster, or ``None`` when any
        part of the roster cannot be placed.
    """
    from osprey.deployment.web_terminals.personas import (
        effective_persona,
        freeze_user_indices,
        rendered_persona_configs,
        resolve_authorization_roles,
    )

    root = project_root if project_root is not None else config.get("project_root")
    if not root:
        return None
    web_terminals = (config.get("modules") or {}).get("web_terminals") or {}
    users = web_terminals.get("users")
    if not isinstance(users, list) or not users:
        return None

    # ONE reader of this roster's shape, shared with `_roster`.
    # `freeze_user_indices` IS `normalize_users`' output with the keys the
    # author wrote re-attached, which is what this walk needs and the bare
    # normalizer cannot give it: `normalize_users` carries `role` through but
    # drops `persona:`, and `effective_persona` reads both. Walking the raw
    # list here instead would be a second spelling of "what a roster entry
    # looks like" — one that could drift from the normalizer, and did: an
    # entry the normalizer drops but a raw `.get("name")` chokes on (`- 42`)
    # raised out of an advisory summary instead of degrading.
    entries = freeze_user_indices(users)
    if len(entries) != len(users):
        # Some entry the file declares could not be placed at all. The roster
        # this walk can describe is then not the roster that was written, and
        # the all-or-nothing rule above says so rather than narrowing around
        # the gap.
        logger.debug("Panel bands not narrowed to their personas: unplaceable roster entry")
        return None

    try:
        roles = resolve_authorization_roles(web_terminals)
        serves = {
            persona: _families_a_project_serves(project_config)
            for persona, project_config in rendered_persona_configs(config, root).items()
        }
    except Exception as exc:
        logger.debug(f"Panel bands not narrowed to their personas: {exc}")
        return None

    default_persona = web_terminals.get("default_persona")
    by_user: dict[str, set[str]] = {}
    for entry in entries:
        name = entry["name"]
        if not name:
            # A nameless entry is the single-user shape, which reaches this
            # walk only as a roster that spells it out; nothing here can say
            # whose panels those are.
            return None
        try:
            persona = effective_persona(entry, roles, default_persona, strict=False)
        except Exception as exc:
            logger.debug(f"Panel bands not narrowed to their personas: {exc}")
            return None
        if persona not in serves:
            # No persona in effect (every pre-catalog roster), or one whose
            # project is unset, unrendered or unreadable. Either way nothing
            # says what this user serves, and a guess is not an answer.
            return None
        if name in by_user:
            # Two entries under one name (lint refuses it; reachable past
            # --no-lint) cannot be placed on their bands as written.
            return None
        by_user[name] = serves[persona]
    return by_user


def _panel_entries(
    config: dict, project_root: Path | str | None = None
) -> list[tuple[tuple[int, int, int, str], str, str, str]]:
    """One row per port family, covering the band of the users who serve it.

    The panels are the largest stretch of the block and reach neither of the
    two binding sources: the per-user containers run on the host namespace, so
    they publish nothing a compose file records, and their ports live under
    ``modules.web_terminals`` rather than under ``services.<name>``, so the
    preflight's config derivation does not walk them either. They are derived
    here instead, through the allocator the render itself uses.

    One row per FAMILY, not per family and user. A family is a band, and its
    band is what an operator needs; a four-user deployment would otherwise
    contribute twenty-eight rows to a list whose whole point is to show the
    shape of the block at a glance.

    But only the users who SERVE it, wherever the personas can be read
    (:func:`_families_by_user`). A band's address is a thing to open, and the
    names beside it are the claim about who answers there; a persona that never
    starts the LATTICE server has a port reserved in that band and nothing
    listening on it. A family no persona serves gets no row at all, because a
    row hedged with "reserved" would still carry the address that does not
    answer — the same claim in smaller type. What the whole tier owes instead is
    one closing note naming those bands, so a reader who counts six families
    where the layout has seven knows the seventh was not lost.

    Args:
        config: Loaded configuration dictionary.
        project_root: The deployment repo, for resolving the persona projects;
            see :func:`_families_by_user`.

    Returns:
        Sortable rows in :func:`endpoint_entries`' own shape. Empty when the
        module is off or the ports cannot be resolved.
    """
    from osprey.deployment.web_terminals.ports import allocate_ports, base_ports_from_config

    roster = _roster(config)
    if not roster:
        return []
    web_terminals = (config.get("modules") or {}).get("web_terminals") or {}
    try:
        base_ports = base_ports_from_config(web_terminals, base=resolve_port_base(config))
        allocated = [allocate_ports(base_ports, index) for _name, index in roster]
    except ValueError:
        # Advisory, like everything else this module derives: a roster index
        # past the end of its band is lint's refusal to raise, and costs the
        # panel rows rather than the whole summary.
        return []

    served = _families_by_user(config, project_root)
    if served is not None and any(name not in served for name, _index in roster):
        # `_roster` normalizes the raw list this walk read; a name that survived
        # one and not the other leaves part of the roster unplaced, which is the
        # all-or-nothing case _families_by_user documents.
        served = None

    rows: list[tuple[tuple[int, int, int, str], str, str, str]] = []
    unserved: list[str] = []
    for family in base_ports:
        members = [
            (name, ports[family])
            for (name, _index), ports in zip(roster, allocated, strict=True)
            if served is None or family in served[name]
        ]
        if not members:
            unserved.append(family)
            continue
        ports_on_band = sorted(port for _name, port in members)
        first, last = ports_on_band[0], ports_on_band[-1]
        # A range is not an address anything can open, and the renderer would
        # linkify it into a URL that 404s on the dash. So the scheme goes on
        # only where there is one port to open — the same rule that keeps
        # `http://` off the graph store's bolt address.
        address = f"http://127.0.0.1:{first}" if first == last else f"127.0.0.1:{first}-{last}"
        who = _who_serves([name for name, _port in members if name])
        if who:
            address = f"{address}  ({who})"
        tier, offset = _placement(family, first, None)
        rows.append(((_TIER_ORDER.index(tier), offset, first, family), tier, family, address))

    if unserved:
        tier, _offset = _placement(_TERMINAL_FAMILY)
        rows.append(
            (
                (_TIER_ORDER.index(tier), _RESERVED_SORT, _RESERVED_SORT, ""),
                tier,
                RESERVED_BANDS_LABEL,
                f"{', '.join(unserved)} — reserved by the block, served by no user",
            )
        )
    return rows


def endpoint_entries(
    config: dict, compose_files: list[str], *, project_root: Path | str | None = None
) -> list[tuple[str, str, str]]:
    """Where a deployment's services are reachable, as ``(tier, service, address)``.

    The single derivation of "what answers where": the text summary below lays
    these out, ``osprey status`` prints the same rows, and the closing summary
    card takes the URLs off the same list, so no reader of a deployment can
    describe it differently from another. It says where a service *would*
    answer — the published ports are read out of the rendered compose files, not
    probed.

    Three sources, because a deployment has three ways to reach a host port.
    What is published lives in the compose files; what a host-namespace service
    binds directly lives only in the rendered config, and is derived from it by
    the same function the port preflight uses, so the summary and the preflight
    cannot name different ports for the same service; and the per-user panels
    live under ``modules.web_terminals``, out of reach of both, and are derived
    through the render's own allocator (:func:`_panel_entries`).

    Ordered by the block, not by name. A deployment's ports are a layout — the
    gateway, then dispatch, then the shared services, then the panels, then the
    stores, then whatever the facility placed itself — and a list sorted
    alphabetically hides that shape behind a spelling. Each row carries the tier
    it belongs to so the surfaces can section the list without deriving the
    grouping a second time.

    Args:
        config: Loaded configuration dictionary.
        compose_files: Rendered compose file paths, spelled absolutely or
            resolvable from the working directory — they are opened here.
        project_root: The deployment repo, for resolving the persona projects
            that say which panel bands each roster user actually serves
            (:func:`_families_by_user`). Optional because the caller that has
            it (:func:`as_built_endpoint_entries`) and the caller that does not
            (``osprey status``, which is handed a config and compose files)
            must reach the same rows: unset falls back to the rendered config's
            own ``project_root``, so both surfaces narrow the bands the same
            way.

    Returns:
        One row per service, in block order, each ``(tier, service, address)``.
    """
    try:
        bindings = parse_host_port_bindings(compose_files)
    except Exception:
        bindings = []
    try:
        bindings = bindings + derive_host_network_bindings(config)
    except Exception:
        pass

    by_service: dict[str, list[HostPortBinding]] = {}
    for binding in bindings:
        by_service.setdefault(binding.service, []).append(binding)

    try:
        base: int | None = resolve_port_base(config)
    except ValueError:
        # A base no block can start at places nothing, which leaves the name
        # table as the only placement there is. Refusing it is the preflight's
        # job; see `summary_title`.
        base = None

    rows: list[tuple[tuple[int, int, int, str], str, str, str]] = _panel_entries(
        config, project_root
    )
    for service, service_bindings in by_service.items():
        first_port = min(binding.host_port for binding in service_bindings)
        tier, offset = _placement(service, first_port, base)
        rows.append(
            (
                (_TIER_ORDER.index(tier), offset, first_port, service),
                tier,
                service,
                _service_address(service, service_bindings),
            )
        )

    landing_url = landing_page_url(config)
    landing = (
        f"{landing_url}  (landing page)" if landing_url else "(not configured in this project)"
    )
    landing_tier, landing_offset = _placement("nginx")
    rows.append(
        (
            (_TIER_ORDER.index(landing_tier), landing_offset, _LANDING_SORT, ""),
            landing_tier,
            "web terminal",
            landing,
        )
    )

    return [(tier, service, address) for _key, tier, service, address in sorted(rows)]


def landing_page_url(config: dict) -> str | None:
    """Where this deployment's landing page answers, or ``None`` if it has none.

    The one derivation of that address: the endpoint list above spells it into a
    row, and the closing call to action hands it to the operator as the thing to
    open. A deployment with ``modules.web_terminals`` disabled has no landing
    page at all, which is what ``None`` says.

    THE EXTERNAL ORIGIN, not the loopback address, whenever one can be derived.
    Every terminal checks a mutating request's ``Origin`` against exactly that
    value, so a browser that arrived on ``http://127.0.0.1:<port>`` loads every
    page and then has each write, approval and chat message refused with a 403
    the operator can only see inside a container. The loopback address is the
    fallback for a deployment that declares no origin at all — one with the
    module enabled and no roster, which has no terminal to be wrong about —
    and :func:`as_built_closing_facts` records which of the two this is so the
    closing line can say so.

    :param config: Loaded configuration dictionary
    """
    web_terminals = (config.get("modules") or {}).get("web_terminals") or {}
    if not web_terminals.get("enabled"):
        return None
    try:
        nginx_port = resolve_nginx_port(config)
    except ValueError:
        # Advisory, like everything else this module derives: a port key that is
        # not a port is render's refusal to raise, and a summary card that
        # cannot name the address says so by naming none.
        return None
    try:
        from osprey.deployment.web_terminals.render import deployment_external_origin

        return deployment_external_origin(config)
    except Exception as exc:
        # Advisory, like everything else this module derives: a config with no
        # `deploy.fqdn` and no `external_origin` still has a landing page on the
        # deploy host, and the closing card marks the address as local-only.
        logger.debug(f"External origin unavailable, falling back to loopback: {exc}")
        return f"http://127.0.0.1:{nginx_port}"


def as_built_endpoint_entries(repo_root: Path | str) -> list[tuple[str, str, str]]:
    """The endpoint entries of the deployment ``repo_root`` has BUILT.

    :func:`endpoint_entries`' own rows, in its own shape —
    ``(tier, service, address)`` in block order — for a caller holding nothing
    but the repo.

    The same two facts every other reader of a deployment starts from — the
    rendered ``build/config.yml`` and the compose files it names — so a caller
    holding nothing but the repo (the summary card at the end of a verb) reaches
    the same answer as one that was handed the config. A repo with nothing
    rendered has no endpoints to declare and answers with an empty list.

    Compose paths come back relative to the repo root and are anchored on it
    here, because :func:`endpoint_entries` opens them itself.

    The repo is handed down as the persona root too, rather than left to the
    ``project_root`` the rendered config declares: this caller was POINTED at a
    deployment, and a build that ran somewhere else — a repo copied to another
    host, a checkout moved — wrote a path belonging to a machine this one is
    not. The live answer wins over the recorded one.
    """
    from osprey.deployment.container_lifecycle import as_built_compose_files

    root = Path(repo_root)
    config = _as_built_config(root)
    if config is None:
        return []
    files = [
        str(path if path.is_absolute() else root / path)
        for path in (Path(name) for name in as_built_compose_files(config, root))
    ]
    return endpoint_entries(config, files, project_root=root)


def _as_built_config(root: Path) -> dict | None:
    """The rendered ``build/config.yml`` of the deployment at ``root``, or ``None``.

    The one load the ``as_built_*`` readers start from, so a caller that wants
    two facts about a built deployment pays for one parse and cannot get the two
    answers out of two different renders.
    """
    from osprey.deployment.container_lifecycle import as_built_config_path
    from osprey.utils.config import load_project_config

    config_path = as_built_config_path(root)
    if not config_path.is_file():
        return None
    return load_project_config(str(config_path), wrap_errors=True)


def as_built_dangerously_allows_bash(repo_root: Path | str) -> bool:
    """Whether the deployment ``repo_root`` has BUILT runs under the Bash waiver.

    Read off the same rendered ``build/config.yml`` as every other
    ``as_built_*`` reader. Advisory like them: no build, an unreadable config or
    a mis-set key answer ``False`` here -- the preflight is where a mis-set key
    is refused, and the card that follows a successful verb must not fail it.
    """
    from osprey.deployment.web_terminals.artifacts import (
        DangerouslyAllowBashValueError,
        dangerously_allow_bash,
    )

    try:
        config = _as_built_config(Path(repo_root))
        return config is not None and dangerously_allow_bash(config)
    except (DangerouslyAllowBashValueError, OSError, ValueError) as exc:
        logger.debug("Bash waiver not read from the build: %s", exc)
        return False


@dataclass(frozen=True)
class ClosingFacts:
    """What a finished ``up`` needs in order to say where to go next.

    :param landing_url: The landing page's address, or ``None`` for a
        deployment that runs no web terminals.
    :param logins: ``(username, password)`` pairs safe to print -- the roster
        logins still carrying the password ``profile.yml`` declared. Empty
        whenever the operator owns the credentials; see
        :func:`~osprey.deployment.web_terminals.auth_credentials.seeded_logins`.
    :param token_login_users: Roster users who can only be reached by opening
        their own ``?token=`` URL, in roster order. These are the entries no
        login wall stands in front of -- the whole roster when ``auth.method``
        is ``token`` (the default), and the ``login: false`` entries under
        ``password``, ``oidc``, or ``none`` -- so the terminal's own gate is
        the only one and the URL is the only way through it. NOT the URLs
        themselves: each carries that user's
        operator secret, and a deploy's closing output is read over shoulders,
        pasted into tickets and captured by CI logs. The verb that prints one
        (``osprey users login-url <user>``) is what goes here instead.
    :param landing_url_is_external_origin: Whether ``landing_url`` is the origin
        the terminals actually check a mutating request against, or the loopback
        fallback :func:`landing_page_url` uses when no origin can be derived.
        False means the address serves the landing page and nothing beyond it --
        a browser that arrives there is refused on every write -- and the closing
        line has to say so rather than present it as the way in.
    """

    landing_url: str | None
    logins: tuple[tuple[str, str], ...]
    token_login_users: tuple[str, ...] = ()
    landing_url_is_external_origin: bool = True
    #: Roster users whose ``.env`` still carries the profile-declared password
    #: while the deployed ``.env.auth`` hash was minted from something else.
    #: Their login line would be a lie, so the card prints the rotate verb for
    #: them instead. See
    #: :func:`~osprey.deployment.web_terminals.auth_credentials.seeded_logins_report`.
    stale_logins: tuple[str, ...] = ()


def as_built_closing_facts(repo_root: Path | str) -> ClosingFacts:
    """The closing facts of the deployment ``repo_root`` has BUILT.

    Read off the same rendered ``build/config.yml`` every other ``as_built_*``
    reader uses, so the address printed at the end of a deploy is the address
    the endpoint list printed a few lines above it.

    Advisory: anything unreadable -- no build, a config that will not parse, an
    ``auth`` stanza naming a method that does not exist -- comes back as "no
    landing page, no logins" rather than as an error. This is the last thing a
    successful deploy prints, and it must not be what fails it.
    """
    try:
        root = Path(repo_root)
        config = _as_built_config(root)
        if config is None:
            return ClosingFacts(landing_url=None, logins=())
        seeded = _seeded_logins(root, config)
        return ClosingFacts(
            landing_url=landing_page_url(config),
            logins=seeded.printable,
            token_login_users=tuple(token_login_users(config)),
            landing_url_is_external_origin=_external_origin_is_derivable(config),
            stale_logins=seeded.stale,
        )
    except Exception as exc:
        logger.debug(f"Closing facts skipped: {exc}")
        return ClosingFacts(landing_url=None, logins=())


def _external_origin_is_derivable(config: dict) -> bool:
    """Whether this deployment declares the origin its terminals check against.

    The same call :func:`landing_page_url` makes, asked as a yes/no so the
    closing line can tell "open this" from "this shows you the landing page and
    nothing else works from it". Kept as its own reader rather than inferred
    from the URL's shape: a deployment whose ``deploy.fqdn`` genuinely IS a
    loopback address would be misread by any string test.
    """
    try:
        from osprey.deployment.web_terminals.render import deployment_external_origin

        return bool(deployment_external_origin(config))
    except Exception:
        return False


def _seeded_logins(root: Path, config: dict) -> SeededLoginsReport:
    """The profile-seeded logins of this deployment's roster, in roster order.

    Gated on there being a login to have: with ``auth.method`` at anything but
    ``password`` no OSPREY-held credential exists (``none``/``token`` have no
    login at all, and under ``oidc`` the facility's identity provider holds
    them), so a password
    named here would be one nothing ever checks. Roster entries carrying
    ``login: false`` sit outside the wall and are skipped for the same reason --
    the same predicate credential provisioning itself uses.
    """
    from osprey.deployment.web_terminals.auth_credentials import (
        SeededLoginsReport,
        seeded_logins_report,
    )
    from osprey.deployment.web_terminals.personas import entry_requires_login, normalize_users
    from osprey.deployment.web_terminals.render import _auth_tls_context

    web_terminals = (config.get("modules") or {}).get("web_terminals") or {}
    if not web_terminals.get("enabled"):
        return SeededLoginsReport()
    if _auth_tls_context(web_terminals).get("auth_method") != "password":
        return SeededLoginsReport()
    names = [
        entry["name"]
        for entry in normalize_users(web_terminals.get("users"))
        if entry_requires_login(entry)
    ]
    return seeded_logins_report(root, names)


def token_login_users(config: dict) -> list[str]:
    """The roster users whose terminal is entered through its ``?token=`` URL.

    The complement of :func:`_seeded_logins`, and derived from the same
    predicates nginx renders under so the two cannot disagree about who needs a
    login URL. Public because ``osprey users login-url`` asks the same question
    before it prints anything: the URL is inert for a user nginx vouches for, so
    the verb refuses there, and a second spelling of "who has a login page"
    could send an operator a live secret the deployment would then ignore.
    Exactly the users whose location nginx injects no operator secret into:
    with ``auth.method`` at ``token`` that is everybody — nginx runs no login
    flow and injects nothing, so the per-user app's own token->cookie gate is
    the only one and a browser gets past it exactly once, by opening that
    user's ``?token=`` URL. Under every other method it is the ``login: false``
    entries, which sit outside nginx's vouching for the same reason and reach
    their terminal the same way; under ``none`` (open) a non-exempt terminal
    needs no URL at all, which is the point of that posture.

    Returns names, never URLs. The URL carries a live credential; the closing
    card is not a place to put one.
    """
    from osprey.deployment.web_terminals.personas import entry_requires_login, normalize_users
    from osprey.deployment.web_terminals.render import _auth_tls_context

    web_terminals = (config.get("modules") or {}).get("web_terminals") or {}
    if not web_terminals.get("enabled"):
        return []
    inject_secret = _auth_tls_context(web_terminals)["inject_secret"]
    return [
        entry["name"]
        for entry in normalize_users(web_terminals.get("users"))
        if not (inject_secret and entry_requires_login(entry))
    ]


def summary_title(config: dict) -> str:
    """The heading every surface that prints these endpoints carries.

    Names the deployment's whole block, not just the ports it happens to use:
    the numbers below the heading are one thousand consecutive ports starting at
    ``deployment.port_base``, and an operator reading a port they do not
    recognise needs to know it came from that one knob rather than from the
    service's own config. ``osprey status`` prints the same line, so the block a
    deploy reported is the block status reports.

    The key is spelled in full, from :data:`~osprey.port_layout.PORT_BASE_CONFIG_KEY`
    rather than by hand, because the port preflight's own block line
    (``This deployment's block is ports <first>-<last> (deployment.port_base
    <base>).``) names it the same way. An operator who reads the key on one
    surface and greps for it after reading the other must find one string, not
    two spellings of it.

    Args:
        config: Loaded configuration dictionary. The base is resolved from the
            config in hand — never from the framework default, which is only
            right when there is no config at all.

    Returns:
        The heading, with the block named whenever the base resolves. A base
        outside the range a block can start at drops the block phrase rather
        than failing a summary: refusing it is the preflight's job, not this
        line's.
    """
    name = resolve_project_name(config)
    try:
        first, last = block_range(resolve_port_base(config))
    except ValueError:
        return f"Service endpoints ({name}):"
    return f"Service endpoints ({name}) — ports {first}-{last} ({PORT_BASE_CONFIG_KEY} {first}):"


def summary_rows(entries: list[tuple[str, str, str]]) -> list[tuple[str, str]]:
    """Lay tiered entries out as the ``(label, value)`` rows a section prints.

    One heading row per tier, then that tier's services indented under it, so a
    reader sees the shape of the block rather than a flat list of addresses. A
    tier with nothing in it is not printed: a deployment that runs no stores has
    no store section, which is a fact, not a gap.

    Public because ``osprey status`` prints the same block from the same rows
    (:func:`osprey.cli.output.section`), and a second grouping loop there could
    section the same endpoints differently.

    Args:
        entries: Rows from :func:`endpoint_entries`, in block order.

    Returns:
        ``(label, value)`` pairs — a tier heading carries an empty value.
    """
    return [
        (heading, "") if heading else (f"  {service}", address)
        for heading, service, address in _sectioned(entries)
    ]


def _sectioned(entries: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    """Insert a heading before each new tier, in printing order.

    The one place that decides WHERE a section starts. Both laid-out forms —
    the terminal's :func:`summary_rows` and the record's :func:`_summary_text` —
    read their rows from here and differ only in how they pad them, so the block
    a log aggregator holds is sectioned exactly like the one on screen.

    Args:
        entries: Rows from :func:`endpoint_entries`, in block order.

    Returns:
        ``(heading, service, address)`` rows. A heading row carries the tier name
        and no service; a service row carries no heading.
    """
    rows: list[tuple[str, str, str]] = []
    tier = ""
    for entry_tier, service, address in entries:
        if entry_tier != tier:
            rows.append((entry_tier, "", ""))
            tier = entry_tier
        rows.append(("", service, address))
    return rows


#: Width the logged block pads its service column to. Fixed rather than measured
#: so one deployment's long service name cannot re-flow every other line of the
#: record a log aggregator holds.
_LOGGED_SERVICE_WIDTH = 20


def _summary_text(title: str, entries: list[tuple[str, str, str]]) -> str:
    """Lay ``entries`` out under ``title`` as the block a caller wants as text.

    The sections :func:`_sectioned` decides, exactly as :func:`summary_rows`
    takes them, at the record's own fixed column width. Only the padding is this
    function's own: the grouping is shared, so the logged block and the printed
    one cannot section the same endpoints differently.

    Args:
        title: The heading, from :func:`summary_title`.
        entries: Rows from :func:`endpoint_entries`, in block order.

    Returns:
        The multi-line block.
    """
    lines = [title]
    for heading, service, address in _sectioned(entries):
        if heading:
            lines.append(f"  {heading}")
        else:
            lines.append(f"    {service:<{_LOGGED_SERVICE_WIDTH}} {address}")
    return "\n".join(lines)


def format_endpoint_summary(config: dict, compose_files: list[str]) -> str:
    """Render :func:`endpoint_entries` as the text block a deploy or report prints.

    :param config: Loaded configuration dictionary
    :param compose_files: Rendered compose file paths, spelled absolutely or
        resolvable from the working directory — they are opened here
    :return: Multi-line summary text
    """
    return _summary_text(summary_title(config), endpoint_entries(config, compose_files))


def log_endpoint_summary(config: dict, compose_files: list[str]) -> None:
    """Print the endpoint summary, and keep it in the record for sinks.

    Where a deployment answers is what the operator ran ``osprey up`` to find
    out, so it is printed as the verb's own output rather than logged: an
    INFO record is not rendered on a normal run, and a fact that only a
    ``--verbose`` run shows is a fact the deploy did not report. Every exit
    path of ``deploy_up`` calls this function, so all of them inherit the
    printed form from here.

    The logged block stays as it was, at the same level, so file and
    aggregation sinks keep the whole summary in one record. It reaches a
    terminal only on a ``--verbose`` run, where the transcript is what was
    asked for.

    Advisory: a summary that cannot be derived is reported to the debug record
    and never fails a deploy that otherwise succeeded.

    :param config: Loaded configuration dictionary
    :param compose_files: Rendered compose file paths, spelled absolutely or
        resolvable from the working directory — they are opened here
    """
    try:
        title = summary_title(config)
        entries = endpoint_entries(config, compose_files)
        output.section(title, summary_rows(entries))
        logger.key_info(_summary_text(title, entries))
    except Exception as exc:
        logger.debug(f"Endpoint summary skipped: {exc}")
