"""The composition card ``osprey init`` prints under its report.

One glance at what the deployment that was just materialized consists of,
before anything is built or started: who can sign in and with what rights,
what the agent runs on, what machine it talks to, and what else runs beside
it. Everything on the card is read off the resolved
:class:`~osprey.cli.build_profile_model.BuildProfile` (plus the emitted
persona deltas), so the card is a pure function of the profile — a build with
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

from osprey.utils.logger import get_logger

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


def _persona_writes_enabled(
    profile: BuildProfile, persona_deltas: Mapping[str, Mapping[str, Any]], persona: str
) -> bool:
    """Whether ``persona``'s render arms the control system's write surface.

    The persona delta's own ``config:`` wins over the host profile's — the
    bundled tiers pin ``control_system.writes_enabled`` on both sides of the
    boundary, and a facility persona that says nothing inherits the host's
    posture.
    """
    wanted = ("control_system", "writes_enabled")
    config = persona_deltas.get(persona, {}).get("config")
    if isinstance(config, Mapping):
        value = _dotted_lookup(config, wanted)
        if value is not None:
            return bool(value)
    return bool(_dotted_lookup(profile.config, wanted))


def _panel_labels(
    profile: BuildProfile, persona_deltas: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    """Display labels for every panel any login of this deployment gets.

    The union across the host profile and the persona deltas, in declaration
    order — panel sets are per-persona (the write tier adds its own), and the
    card summarizes the deployment. Labels come from the same sources the web
    terminal reads: :data:`~osprey.profiles.web_panels.BUILTIN_PANEL_LABELS`
    for built-ins, ``web.panels.<id>.label`` for URL-backed custom panels, and
    the id itself, uppercased, when neither says.
    """
    from osprey.profiles.web_panels import BUILTIN_PANEL_LABELS

    ids: list[str] = list(dict.fromkeys(profile.web_panels))
    for delta in persona_deltas.values():
        extra = delta.get("web_panels")
        if not isinstance(extra, list):
            continue
        for panel_id in extra:
            if isinstance(panel_id, str) and panel_id not in ids:
                ids.append(panel_id)

    return [
        BUILTIN_PANEL_LABELS.get(panel_id)
        or _custom_panel_label(profile, persona_deltas, panel_id)
        or panel_id.upper()
        for panel_id in ids
    ]


def _custom_panel_label(
    profile: BuildProfile, persona_deltas: Mapping[str, Mapping[str, Any]], panel_id: str
) -> str | None:
    """The ``web.panels.<id>.label`` any layer of this deployment declares."""
    wanted = ("web", "panels", panel_id, "label")
    value = _dotted_lookup(profile.config, wanted)
    for delta in persona_deltas.values():
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


def _web_terminal_group(
    profile: BuildProfile, persona_deltas: Mapping[str, Mapping[str, Any]]
) -> CardGroup | None:
    """Who gets in, with what rights, how, and where — plus what they see."""
    from osprey.deployment.web_terminals.personas import (
        effective_persona,
        resolve_authorization_roles,
    )

    from .build_profile_emit import effective_web_terminals

    web_tier = effective_web_terminals(profile.config)
    if not web_tier.get("enabled"):
        return None

    # The card is a REPORT of a profile, not a gate on one: an `authorization`
    # stanza that does not parse, or an entry whose binding does not resolve,
    # is lint's finding to raise (the same lint belt runs at profile altitude).
    # Both degrade here to the persona the entry would have shown before roles
    # existed, so a bad binding costs one wrong cell rather than the whole card.
    try:
        roles = resolve_authorization_roles(web_tier)
    except ValueError:
        roles = {}

    rows: list[list[Cell]] = []
    auth = web_tier.get("auth")
    auth_method = auth.get("method") if isinstance(auth, Mapping) else None
    base_port = web_tier.get("web_base_port")
    default_persona = web_tier.get("default_persona")
    users = web_tier.get("users")
    for position, user in enumerate(users if isinstance(users, list) else []):
        if not isinstance(user, Mapping):
            continue
        name = str(user.get("name") or "")
        if not name:
            continue
        persona = effective_persona(user, roles, default_persona, strict=False) or ""
        rights: list[str] = []
        if persona:
            rights.append(persona)
            if _persona_writes_enabled(profile, persona_deltas, persona):
                rights.append("rights approval-gated")
        if user.get("login") is False:
            # The one warning tone on the card: an open surface must not be
            # skimmed past, so `password` stays dim precisely to keep it alone.
            auth_cell: Cell = [("no login", Styles.WARNING)]
        elif auth_method:
            auth_cell = [(str(auth_method), Styles.DIM)]
        else:
            auth_cell = []
        index = user.get("index", position)
        port_cell: Cell = []
        if isinstance(base_port, int) and isinstance(index, int):
            port_cell = [(f":{base_port + index}", Styles.ACCENT)]
        rows.append([[(name, Styles.BOLD)], _dotted_list(rights), auth_cell, port_cell])

    panels = _panel_labels(profile, persona_deltas)
    if panels:
        rows.append([[("panels", Styles.DIM)], _dotted_list(panels)])

    if not rows:
        return None
    nginx_port = web_tier.get("nginx_port")
    suffix = f":{nginx_port}" if isinstance(nginx_port, int) else ""
    return CardGroup("web terminal", suffix, rows)


def _enabled_mcp_servers(profile: BuildProfile) -> list[str]:
    """The MCP servers the agent gets, resolved as the render resolves them.

    :func:`~osprey.registry.mcp.resolve_servers` over the profile's effective
    ``claude_code`` subtree — the registry's own defaults plus the profile's
    ``claude_code.servers.<name>.enabled`` overrides — with the channel-finder
    condition keyed the way the render keys it, plus the profile's own
    ``mcp_servers:`` declarations.
    """
    from osprey.registry.mcp import resolve_servers

    from .build_profile_emit import effective_config_subtree

    claude_code = effective_config_subtree(profile.config, ("claude_code",))
    ctx: dict[str, Any] = {}
    if profile.channel_finder_mode and "channel-finder" in profile.agents:
        ctx["channel_finder_pipeline"] = profile.channel_finder_mode
    names = [server["name"] for server in resolve_servers(claude_code, ctx) if server["enabled"]]
    for name in profile.mcp_servers:
        if name not in names:
            names.append(name)
    return names


def _agent_group(profile: BuildProfile) -> CardGroup | None:
    """What thinks: the model, its tool surface, and its bundled toolkit."""
    rows: list[list[Cell]] = []
    model_bits = [bit for bit in (profile.provider, profile.model) if bit]
    if model_bits:
        rows.append([[("model", Styles.DIM)], _dotted_list(model_bits)])
    servers = _enabled_mcp_servers(profile)
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
        tier = f"tier {profile.resolved_tier()}"
        rows.append([[("channels", Styles.DIM)], _dotted_list([finder, tier])])

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
    for name in profile.services:
        rows.append([[(name, Styles.DIM)], [("profile service", None)]])

    return CardGroup("services", rows=rows) if rows else None


def _card_groups(
    profile: BuildProfile, persona_deltas: Mapping[str, Mapping[str, Any]]
) -> list[CardGroup]:
    """The card's groups, in the fixed order the reader learns once."""
    candidates = (
        _web_terminal_group(profile, persona_deltas),
        _agent_group(profile),
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
    profile: BuildProfile, persona_deltas: Mapping[str, Mapping[str, Any]]
) -> list[list[Segment]]:
    """The whole card as segment lines; an empty line separates the groups.

    Starts with its own separator line when there is anything to say, so the
    card always stands one blank line off whatever printed above it.
    """
    lines: list[list[Segment]] = []
    for group in _card_groups(profile, persona_deltas):
        lines.append([])
        lines.extend(_group_lines(group))
    return lines


def format_profile_card(
    profile: BuildProfile, persona_deltas: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    """The card as plain lines — what a pipe reads, and what the tests pin."""
    return [_plain_text(line) for line in _card_segment_lines(profile, persona_deltas)]


def print_profile_card(
    profile: BuildProfile, persona_deltas: Mapping[str, Mapping[str, Any]]
) -> None:
    """Print the card through the reporter; advisory, never raises.

    ``init`` has created the repo by the time this runs, and a card that
    cannot be derived — a config shape this reader never met — must not turn
    that into a failure.
    """
    from .phase_reporter import current_reporter

    try:
        lines = _card_segment_lines(profile, persona_deltas)
    except Exception as exc:  # noqa: BLE001 — see docstring: the card is advisory
        logger.debug("Profile card skipped: %s", exc)
        return
    reporter = current_reporter()
    for line in lines:
        if line:
            reporter.echo_segments(line)
        else:
            reporter.echo("")
