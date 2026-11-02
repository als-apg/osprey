"""The composition card ``osprey init`` prints under its report.

One glance at what the deployment that was just materialized consists of,
before anything is built or started: who can sign in and with what rights,
what the agent runs on, what machine it talks to, and what else runs beside
it. Everything on the card is read off the resolved
:class:`~osprey.cli.build_profile_model.BuildProfile` (plus the emitted
layers), so the card is a pure function of the profile — a build with
no web tier simply has no ``web terminal`` group, and a profile that declares
no extra services has no ``services`` group. Nothing here probes the host or
the container runtime.

The card is echo-class: it is part of what ``init`` owes its operator, so it
prints under every reporter, through
:meth:`~osprey.cli.phase_reporter.PhaseReporter.echo_segments` — styled on a
terminal, byte-identical to its plain rendering through a pipe. The one
derivation feeds both the printer and :func:`format_profile_card`, the
plain-text twin the tests pin, so the card an operator reads cannot differ
from the card a test reads (:mod:`osprey.cli.summary_card`'s precedent).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from osprey_connectors.logger import get_logger

from .styles import Styles

if TYPE_CHECKING:
    from .build_profile_model import BuildProfile

logger = get_logger("cli.profile_card")

#: One styled run of text: ``(text, style token or None)``.
Segment = tuple[str, str | None]

#: One cell of a row — a list of segments, so a single cell can carry a plain
#: value with a dim separator or an accent port inside it.
Cell = list[Segment]

#: The list separator, always dim: lists should read as items, not dot soup.
_SEP: Segment = (" · ", Styles.DIM)

_INDENT = "  "
_GUTTER = "   "


@dataclass
class CardGroup:
    """One titled block of the card."""

    title: str
    """The group anchor, rendered in the header token."""

    suffix: str = ""
    """Text after the title (the web tier's entry port), rendered accent."""

    rows: list[list[Cell]] = field(default_factory=list)
    """Rows of cells. The first cell is the row's label; within a group every
    column is padded to a common width, except each row's last non-empty cell,
    which flows free (so a long panels list never pushes the user columns)."""


# ---------------------------------------------------------------------------
# Derivation — profile in, groups out
# ---------------------------------------------------------------------------


def _dotted_lookup(config: Mapping[str, Any], wanted: tuple[str, ...]) -> Any:
    """What a flat dotted ``config:`` bag sets at ``wanted``, or ``None``.

    Reads every spelling — the dotted key itself or an ancestor carrying the
    path nested inside its value — and later keys win, matching the order the
    build applies them in.
    """
    from .profile_cmd import _config_node

    found: Any = None
    for key, value in config.items():
        if not isinstance(key, str):
            continue
        node = _config_node(tuple(key.split(".")), value, wanted)
        if node is not None:
            found = node
    return found


def _document_tier(config: Any) -> str | None:
    """The tier one ``config:`` block states, read as the build reads it.

    ``control_system.writes`` when the block spells it (any spelling, deeper
    key winning), through :func:`~osprey_connectors.types.tier_writes` so only
    the literal ``asked`` and ``allowed`` arm; else what its retired ``writes_enabled``
    leaves amount to (:func:`~.build_profile_document.retired_write_answer`,
    the rewrite the normalizer applies — a layer handed to the card may be a
    raw parse); else ``None``, a block that says nothing about the posture.
    """
    from osprey_connectors.types import tier_writes

    from .build_profile_document import retired_write_answer
    from .build_profile_emit import effective_config_subtree

    if not isinstance(config, Mapping):
        return None
    section = effective_config_subtree(config, ("control_system",))
    if "writes" in section:
        return tier_writes({"writes": section["writes"]})
    return retired_write_answer(dict(config))


def _write_rights(
    profile: BuildProfile, layer_deltas: Mapping[str, Mapping[str, Any]], render: str
) -> list[str]:
    """``render``'s write posture: ``rights <ceiling word>`` when its tier arms writes, else none.

    The tier — ``control_system.writes`` — is the one posture a login has. The
    render's layer states it or inherits the host profile's (silence is
    ``never``), each document read on its own (:func:`_document_tier`) so a
    retired spelling in either answers as the build rewrites it. It is
    target-blind: a tier that arms writes does so for every machine the render
    reaches, and the person's switch picks which one at the moment of each
    write, so there is no per-target answer to print. The word is the ceiling
    the slider stops at: ``asked`` or ``allowed``.
    """
    from osprey_connectors.types import tier_allows

    tier = _document_tier(layer_deltas.get(render, {}).get("config"))
    if tier is None:
        tier = _document_tier(profile.config)
    return [f"rights {tier}"] if tier_allows({"writes": tier}) else []


def _panel_labels(
    profile: BuildProfile, layer_deltas: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    """Display labels for every panel any login of this deployment gets.

    The union across the host profile and the layers, in declaration
    order — panel sets are per-render (the write tier adds its own), and the
    card summarizes the deployment. Labels come from the same sources the web
    terminal reads: :data:`~osprey.profiles.web_panels.BUILTIN_PANEL_LABELS`
    for built-ins, ``web.panels.<id>.label`` for URL-backed custom panels, and
    the id itself, uppercased, when neither says.
    """
    from osprey.profiles.web_panels import BUILTIN_PANEL_LABELS

    ids: list[str] = list(dict.fromkeys(profile.web_panels))
    for delta in layer_deltas.values():
        extra = delta.get("web_panels")
        if not isinstance(extra, list):
            continue
        for panel_id in extra:
            if isinstance(panel_id, str) and panel_id not in ids:
                ids.append(panel_id)

    return [
        BUILTIN_PANEL_LABELS.get(panel_id)
        or _custom_panel_label(profile, layer_deltas, panel_id)
        or panel_id.upper()
        for panel_id in ids
    ]


def _custom_panel_label(
    profile: BuildProfile, layer_deltas: Mapping[str, Mapping[str, Any]], panel_id: str
) -> str | None:
    """The ``web.panels.<id>.label`` any layer of this deployment declares."""
    wanted = ("web", "panels", panel_id, "label")
    value = _dotted_lookup(profile.config, wanted)
    for delta in layer_deltas.values():
        config = delta.get("config")
        if isinstance(config, Mapping):
            value = _dotted_lookup(config, wanted) or value
    return str(value) if value else None


def _joined(parts: Sequence[Cell]) -> Cell:
    """One cell from several, separated by the dim list separator."""
    cell: Cell = []
    for part in parts:
        if cell:
            cell.append(_SEP)
        cell.extend(part)
    return cell


def _dotted_list(items: Sequence[str]) -> Cell:
    """Plain items joined by the dim separator."""
    return _joined([[(item, None)] for item in items])


def _allocated(base_ports: Mapping[str, int], index: Any) -> dict[str, int]:
    """The host ports web-terminal user ``index`` gets, or ``{}``.

    The render's own allocator, so the card cannot describe a user as reachable
    somewhere the deployment does not put them.

    Args:
        base_ports: Each family's effective base port, from
            :func:`~osprey.deployment.web_terminals.ports.base_ports_from_config`.
            Empty when the profile's ports could not be resolved at all.
        index: The user's roster index, as the profile spells it — anything that
            is not a usable index costs this user's port cell, which is lint's
            finding to raise rather than the card's.

    Returns:
        ``{family: port}``, or an empty mapping when nothing can be allocated.
    """
    from osprey.deployment.web_terminals.ports import allocate_ports

    try:
        return allocate_ports(dict(base_ports), index)
    except (TypeError, ValueError):
        return {}


def _family_ports(base_ports: Mapping[str, int]) -> Cell:
    """Every port family's first port, as one cell.

    Args:
        base_ports: Each family's effective base port.

    Returns:
        ``web :10100 · artifact :10200 · …`` as styled segments, or an empty
        cell when the ports could not be resolved. Ascending, so the row reads
        as the stretch of the block it is — the registry's own order is by
        family name, which would print a higher band before a lower one. Family
        names are shown as words, since the underscore in ``channel_finder`` is
        a config spelling and this row is prose.
    """
    allocated = sorted(_allocated(base_ports, 0).items(), key=lambda item: item[1])
    return _joined(
        [
            [(family.replace("_", " "), None), (f" :{port}", Styles.ACCENT)]
            for family, port in allocated
        ]
    )


def _web_terminal_group(
    profile: BuildProfile, layer_deltas: Mapping[str, Mapping[str, Any]]
) -> CardGroup | None:
    """Who gets in, with what rights, how, and where — plus what they see."""
    from osprey.deployment.web_terminals.ports import base_ports_from_config, resolve_nginx_port
    from osprey.deployment.web_terminals.roster import (
        effective_tier,
        entry_is_shared,
        resolve_authorization_roles,
    )
    from osprey.port_layout import resolve_port_base

    from .build_profile_emit import effective_config_subtree, effective_web_terminals

    web_tier = effective_web_terminals(profile.config)
    if not web_tier.get("enabled"):
        return None

    # The shape the port resolvers take: they read `deployment.port_base` and
    # `modules.web_terminals` out of ONE document, because a family's port is
    # the base plus its band and reading either half alone describes a
    # deployment that lives somewhere else. A profile carries the two in
    # separate places, so they are re-wrapped here once for every port on this
    # group. Report, not gate (see `roles` below): a base or a port key that is
    # not a number is lint's finding to raise, and costs a cell rather than the
    # card.
    rendered_shape = {
        "deployment": effective_config_subtree(profile.config, ("deployment",)),
        "modules": {"web_terminals": web_tier},
    }
    try:
        base_ports = base_ports_from_config(web_tier, base=resolve_port_base(rendered_shape))
    except ValueError:
        base_ports = {}

    # The card is a REPORT of a profile, not a gate on one: an `authorization`
    # stanza that does not parse, or an entry whose binding does not resolve,
    # is lint's finding to raise (the same lint belt runs at profile altitude).
    # Both degrade here to the render the entry would have shown before roles
    # existed, so a bad binding costs one wrong cell rather than the whole card.
    try:
        roles = resolve_authorization_roles(web_tier)
    except ValueError:
        roles = {}

    rows: list[list[Cell]] = []
    auth = web_tier.get("auth")
    auth_method = auth.get("method") if isinstance(auth, Mapping) else None
    default_tier = web_tier.get("default_tier")
    users = web_tier.get("users")
    for position, user in enumerate(users if isinstance(users, list) else []):
        if not isinstance(user, Mapping):
            continue
        name = str(user.get("name") or "")
        if not name:
            continue
        render = effective_tier(user, roles, default_tier, strict=False) or ""
        rights: list[str] = []
        if render:
            rights.append(render)
            rights.extend(_write_rights(profile, layer_deltas, render))
        auth_cell: Cell
        if not auth_method:
            auth_cell = []
        else:
            try:
                shared = entry_is_shared(dict(user))
            except ValueError:
                # An `access` value the vocabulary does not recognise. The card
                # cannot say whether the entry is shared, and the plain method
                # cell reads as "not shared" — the one answer that is certainly
                # wrong for a value lint refuses. Name it instead, so the card
                # and the refusal the operator is about to get agree.
                auth_cell = _joined(
                    [[(str(auth_method), None)], [("access invalid", Styles.WARNING)]]
                )
            else:
                # A shared card: behind the wall, opened with an admitted
                # login's own credential rather than one of its own.
                auth_cell = (
                    _dotted_list([str(auth_method), "shared"])
                    if shared
                    else [(str(auth_method), Styles.DIM)]
                )
        index = user.get("index", position)
        # The allocator the render itself uses, rather than `base + index` spelled
        # again: it is what falls a family back to its layout band when the
        # profile sets no base port, and what refuses an index past the end of
        # the band instead of quietly placing a user in the next family's ports.
        web_port = _allocated(base_ports, index).get("web")
        port_cell: Cell = [(f":{web_port}", Styles.ACCENT)] if web_port else []
        rows.append([[(name, Styles.BOLD)], _dotted_list(rights), auth_cell, port_cell])

    panels = _panel_labels(profile, layer_deltas)
    if panels:
        rows.append([[("panels", Styles.DIM)], _dotted_list(panels)])

    if not rows:
        return None

    # Every family, at the first index of the block. The rows above give each
    # user the one port they open — their terminal — but a deployment publishes
    # a whole band per family, and the panels row directly above says what those
    # bands serve without saying where any of them answers. One row settles both
    # halves, and states the index it is showing rather than leaving a reader to
    # infer whose ports these are.
    #
    # Below the guard on purpose: this row is derivable for any enabled web tier,
    # including one with no roster and no panels, and a group that consisted of
    # nothing but a port table would say a deployment has terminals when nobody
    # can sign into one.
    families = _family_ports(base_ports)
    if families:
        rows.append([[("ports (user 0)", Styles.DIM)], families])

    # The address the card puts beside the group title, resolved rather than
    # read: a profile that sets no `nginx_port` still lands on the gateway slot
    # of its own block, and a card that dropped the suffix there would report a
    # deployment with no front door.
    try:
        suffix = f":{resolve_nginx_port(rendered_shape)}"
    except ValueError:
        suffix = ""
    return CardGroup("web terminal", suffix, rows)


def _render_config(profile: BuildProfile, delta: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The rendered ``config.yml`` facts the registry's two gates read, for one render.

    ``delta`` is a tier or app file laid over the host, as the build lays it:
    its flat dotted ``config:`` bag folded over the host's, deeper key winning,
    and its ``web_panels`` added to the host's selection. Only the subtrees the
    gates read are assembled — ``services`` (the graph store), ``deployment``
    (the port base that store's defaults resolve against) and ``web.panels``.

    The panel blocks are projected the way the build projects them
    (:func:`~osprey.cli.build_profile_panels.panel_selection_overrides`): a
    selected panel always has a block (the template's or an injector's), and
    every block carries ``enabled`` saying whether this render selects it — so
    a block the config spells for a tab no layer selects reads as off here,
    exactly as it does in the render.
    """
    from .build_profile_emit import _merge_subtree, effective_config_subtree
    from .build_profile_panels import panel_selection_overrides

    layers: list[Mapping[str, Any]] = [profile.config]
    selected: list[str] = list(profile.web_panels)
    if delta:
        config = delta.get("config")
        if isinstance(config, Mapping):
            layers.append(config)
        extra = delta.get("web_panels")
        if isinstance(extra, list):
            selected.extend(panel for panel in extra if isinstance(panel, str))

    rendered: dict[str, Any] = {}
    for subtree in (("services",), ("deployment",), ("web",)):
        section: dict[str, Any] = {}
        for layer in layers:
            _merge_subtree(section, effective_config_subtree(layer, subtree))
        rendered[subtree[0]] = section

    web = rendered["web"]
    panels = web.get("panels")
    blocks: dict[str, Any] = dict(panels) if isinstance(panels, Mapping) else {}
    for panel_id in selected:
        blocks.setdefault(panel_id, {})
    web["panels"] = blocks
    for panel_id, shown in panel_selection_overrides(selected, rendered).items():
        block = blocks.get(panel_id)
        blocks[panel_id] = {**(dict(block) if isinstance(block, Mapping) else {}), "enabled": shown}
    return rendered


def _registry_context(profile: BuildProfile, rendered: Mapping[str, Any]) -> dict[str, Any]:
    """The context :func:`~osprey.registry.mcp.resolve_servers` gates on, for one render.

    The two config-derived gates are the render's own predicates, imported
    rather than restated, so the card and the render cannot disagree about
    whether a render ships the ``graph`` or the ``event_dispatcher`` server.
    """
    from osprey.deployment.web_terminals.roster import config_needs_dispatcher_token

    from .profile_conventions import library_selection
    from .templates.claude_code import _graphdb_configured

    ctx: dict[str, Any] = {
        "graphdb_configured": _graphdb_configured(dict(rendered)),
        "event_dispatcher_wired": config_needs_dispatcher_token(rendered),
    }
    if profile.channel_finder_mode and "channel-finder" in library_selection(
        "agents", profile.agents
    ):
        ctx["channel_finder_pipeline"] = profile.channel_finder_mode
    return ctx


def _enabled_mcp_servers(
    profile: BuildProfile, layer_deltas: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    """The MCP servers the agent gets, resolved as the render resolves them.

    :func:`~osprey.registry.mcp.resolve_servers` over the effective
    ``claude_code`` subtree — the registry's own defaults plus the
    ``claude_code.servers.<name>.enabled`` overrides — with the registry
    context the render builds (:func:`_registry_context`), plus the profile's
    own ``mcp_servers:`` declarations.

    Resolved once for the host and once per tier or app render, and listed as
    the union in first-seen order: the deployment summary the panels row
    gives, so a server only the write-armed tiers ship (the event dispatcher,
    gated on the EVENTS panel those tiers add) is on the card.
    """
    from osprey.registry.mcp import resolve_servers

    from .build_profile_emit import _merge_subtree, effective_config_subtree

    names: list[str] = []
    for delta in (None, *layer_deltas.values()):
        claude_code = effective_config_subtree(profile.config, ("claude_code",))
        config = delta.get("config") if delta else None
        if isinstance(config, Mapping):
            _merge_subtree(claude_code, effective_config_subtree(config, ("claude_code",)))
        ctx = _registry_context(profile, _render_config(profile, delta))
        for server in resolve_servers(claude_code, ctx):
            if server["enabled"] and server["name"] not in names:
                names.append(server["name"])
    for name in profile.mcp_servers:
        if name not in names:
            names.append(name)
    return names


def _model_label(profile: BuildProfile) -> str | None:
    """The model as a person reads it: its display name, then the id it is sent as.

    A profile that names no model runs its provider's ``default_model``; the
    card says so, from the packaged catalog.
    """
    from osprey.models.display import display_model_name

    if profile.model:
        name = display_model_name(profile.model)
        return profile.model if name == profile.model else f"{name} ({profile.model})"
    if not profile.provider:
        return None
    from osprey.profiles.providers import load_provider_catalog

    default = (load_provider_catalog(None).entries.get(profile.provider) or {}).get("default_model")
    return f"(provider default) {default}" if default else None


def _agent_group(
    profile: BuildProfile, layer_deltas: Mapping[str, Mapping[str, Any]]
) -> CardGroup | None:
    """What thinks: the model, its tool surface, and its bundled toolkit."""
    rows: list[list[Cell]] = []
    model_bits = [bit for bit in (profile.provider, _model_label(profile)) if bit]
    if model_bits:
        rows.append([[("model", Styles.DIM)], _dotted_list(model_bits)])
    servers = _enabled_mcp_servers(profile, layer_deltas)
    if servers:
        rows.append([[("mcp", Styles.DIM)], _dotted_list(servers)])
    toolkit = [
        f"{len(entries)} {noun}"
        for entries, noun in (
            (profile.hooks, "hooks"),
            (profile.rules, "rules"),
            (profile.skills, "skills"),
            (profile.agents, "agents"),
        )
        if entries
    ]
    if toolkit:
        rows.append([[("toolkit", Styles.DIM)], _dotted_list(toolkit)])
    return CardGroup("agent", rows=rows) if rows else None


def _machine_group(profile: BuildProfile) -> CardGroup | None:
    """What the agent talks to: connector, archive, channel database."""
    rows: list[list[Cell]] = []

    control: list[Cell] = []
    connector = _dotted_lookup(profile.config, ("control_system", "type"))
    if isinstance(connector, str) and connector:
        control.append([(connector.replace("_", " "), None)])
    if profile.virtual_accelerator is not None:
        port = profile.virtual_accelerator.port
        control.append([("EPICS ", None), (f":{port}", Styles.ACCENT)])
        standin_port = profile.virtual_accelerator.live_standin
        if standin_port is not None:
            control.append([("live stand-in ", None), (f":{standin_port}", Styles.ACCENT)])
    if control:
        rows.append([[("control", Styles.DIM)], _joined(control)])

    archiver: list[str] = []
    archiver_type = _dotted_lookup(profile.config, ("archiver", "type"))
    if isinstance(archiver_type, str) and archiver_type:
        archiver.append(archiver_type.removesuffix("_archiver").replace("_", " "))
    if profile.va_archiver is not None:
        archiver.append(f"{profile.va_archiver.retention_days} d retention")
    if archiver:
        rows.append([[("archiver", Styles.DIM)], _dotted_list(archiver)])

    if profile.channel_finder_mode:
        finder = f"{profile.channel_finder_mode.replace('_', ' ')} finder"
        rows.append([[("channels", Styles.DIM)], _dotted_list([finder])])

    return CardGroup("machine", rows=rows) if rows else None


def _services_group(profile: BuildProfile) -> CardGroup | None:
    """What else runs beside the terminals, named as the build names it."""
    rows: list[list[Cell]] = []

    if profile.bluesky is not None:
        parts: list[Cell] = [[(f":{profile.bluesky.port}", Styles.ACCENT)]]
        if profile.bluesky.tiled_enabled:
            parts.append([("tiled ", None), (f":{profile.bluesky.tiled_port}", Styles.ACCENT)])
        if profile.bluesky_web is not None:
            parts.append([("web ", None), (f":{profile.bluesky_web.port}", Styles.ACCENT)])
        rows.append([[("bluesky", Styles.DIM)], _joined(parts)])
    elif profile.bluesky_web is not None:
        rows.append(
            [[("bluesky web", Styles.DIM)], [(f":{profile.bluesky_web.port}", Styles.ACCENT)]]
        )

    if profile.dispatch is not None:
        dispatch = profile.dispatch
        noun = "worker" if dispatch.worker_count == 1 else "workers"
        workers: Cell = [(f"{dispatch.worker_count} {noun}", None)]
        triggers: Cell = [("triggers ", None), (dispatch.triggers, Styles.PATH)]
        rows.append([[("dispatch", Styles.DIM)], _joined([workers, triggers])])

    if profile.nextcloud_bridge is not None:
        rows.append([[("bridge", Styles.DIM)], [("Nextcloud Talk", None)]])
    if profile.gchat_bridge is not None:
        rows.append([[("bridge", Styles.DIM)], [("Google Chat", None)]])
    if profile.teams_bridge is not None:
        rows.append([[("bridge", Styles.DIM)], [("Microsoft Teams", None)]])
    rows.extend([[(name, Styles.DIM)], [("profile service", None)]] for name in profile.services)

    return CardGroup("services", rows=rows) if rows else None


def _card_groups(
    profile: BuildProfile, layer_deltas: Mapping[str, Mapping[str, Any]]
) -> list[CardGroup]:
    """The card's groups, in the fixed order the reader learns once."""
    candidates = (
        _web_terminal_group(profile, layer_deltas),
        _agent_group(profile, layer_deltas),
        _machine_group(profile),
        _services_group(profile),
    )
    return [group for group in candidates if group is not None]


# ---------------------------------------------------------------------------
# Layout — groups in, lines out
# ---------------------------------------------------------------------------


def _plain_text(segments: Sequence[Segment]) -> str:
    """The text of a run of segments, with the styles dropped."""
    return "".join(part for part, _ in segments)


def _last_filled(row: list[Cell]) -> int:
    """The index of ``row``'s last non-empty cell, or 0 for an empty row."""
    return max((column for column, cell in enumerate(row) if cell), default=0)


def _group_lines(group: CardGroup) -> list[list[Segment]]:
    """Lay one group out as segment lines: title, then padded rows."""
    title: list[Segment] = [(_INDENT, None), (group.title, Styles.HEADER)]
    if group.suffix:
        title += [("  ", None), (group.suffix, Styles.ACCENT)]
    lines = [title]

    # Each row is cut at its last non-empty cell, so that cell flows free and
    # only the columns before it are padded to the group's common width.
    rows = [row[: _last_filled(row) + 1] for row in group.rows]
    widths: dict[int, int] = {}
    for row in rows:
        for column, cell in enumerate(row[:-1]):
            widths[column] = max(widths.get(column, 0), len(_plain_text(cell)))

    for row in rows:
        line: list[Segment] = [(_INDENT * 2, None)]
        for column, cell in enumerate(row):
            line.extend(cell)
            if column < len(row) - 1:
                line.append((" " * (widths[column] - len(_plain_text(cell))) + _GUTTER, None))
        lines.append(line)
    return lines


def _card_segment_lines(
    profile: BuildProfile, layer_deltas: Mapping[str, Mapping[str, Any]]
) -> list[list[Segment]]:
    """The whole card as segment lines; an empty line separates the groups.

    Starts with its own separator line when there is anything to say, so the
    card always stands one blank line off whatever printed above it.
    """
    lines: list[list[Segment]] = []
    for group in _card_groups(profile, layer_deltas):
        lines.append([])
        lines.extend(_group_lines(group))
    return lines


def format_profile_card(
    profile: BuildProfile, layer_deltas: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    """The card as plain lines — what a pipe reads, and what the tests pin."""
    return [_plain_text(line) for line in _card_segment_lines(profile, layer_deltas)]


def card_rows(
    profile: BuildProfile, layer_deltas: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, str]]:
    """The card as flat rows — the twin a machine reads, in card order.

    One entry per row of every group: the group it stands under, the row's
    label, and the rest of the row as one value. These are the very cells
    :func:`_group_lines` lays out, flattened with the styles dropped, so a
    reader of these rows and a reader of the printed card cannot be told
    different things about the same deployment. Nothing is derived a second
    time here, and nothing is added: a fact that is not on the card is not in
    these rows either.

    Cells that carry nothing contribute nothing, the way the layout gives them
    no width — so a row whose middle column is empty reads as one value rather
    than as a value with a hole in it.

    Returns:
        ``[{"group": …, "label": …, "value": …}, …]``, in the order the card
        prints, with every value a string.
    """
    rows: list[dict[str, str]] = []
    for group in _card_groups(profile, layer_deltas):
        for row in group.rows:
            values = [text for text in (_plain_text(cell) for cell in row[1:]) if text]
            rows.append(
                {
                    "group": group.title,
                    "label": _plain_text(row[0]) if row else "",
                    "value": " ".join(values).strip(),
                }
            )
    return rows


def print_profile_card(
    profile: BuildProfile, layer_deltas: Mapping[str, Mapping[str, Any]]
) -> None:
    """Print the card through the reporter; advisory, never raises.

    ``init`` has created the repo by the time this runs, and a card that
    cannot be derived — a config shape this reader never met — must not turn
    that into a failure.
    """
    from .phase_reporter import current_reporter

    try:
        lines = _card_segment_lines(profile, layer_deltas)
    except Exception as exc:  # see docstring: the card is advisory
        logger.debug("Profile card skipped: %s", exc)
        return
    reporter = current_reporter()
    for line in lines:
        if line:
            reporter.echo_segments(line)
        else:
            reporter.echo("")
