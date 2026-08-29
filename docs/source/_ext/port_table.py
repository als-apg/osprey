"""Sphinx extension that renders the frozen host-port layout as a table.

Usage in RST files::

    .. osprey-ports::

The rows come from :data:`osprey.port_layout.LAYOUT`, so the published table
is the layout itself rather than a hand-kept copy of it: a slot added, moved or
renamed in the module shows up in the docs on the next build, and nothing here
has to be edited to keep step.

The env-var column comes from the registry
(``FRAMEWORK_WEB_SERVERS[key].port_env_var``) for the per-user panel families.
Every other slot leaves it empty, which says only that the registry declares no
variable for it -- not that no environment variable can reach it; see
:func:`_env_var_text` for the web terminal, which is the one row where the
difference matters.

Both imports are deliberately at module scope and deliberately unguarded: if
either one fails, the extension fails to load and the docs build stops. A table
that quietly rendered empty — or without its ports — would publish a ports
reference with no ports in it, which is worse than a red build.
"""

from typing import Any

from docutils import nodes
from sphinx.application import Sphinx
from sphinx.util.docutils import SphinxDirective

from osprey.port_layout import DEFAULT_PORT_BASE, LAYOUT, PortSlot, index_bounds
from osprey.registry.web import FRAMEWORK_WEB_SERVERS

#: Column headers, in render order.
COLUMNS = ("Offset", "Port", "Service", "Tier", "Override key", "Environment variable")

#: Relative widths for the six columns, in the same order.
COLUMN_WIDTHS = (10, 14, 18, 12, 26, 20)

#: Port-family name → the registry definition that owns it. The family name is
#: exactly what the layout calls a per-user slot, which is what lets a row find
#: its env var; ``port_family`` covers the one server whose family is named
#: differently from its registry key (``lattice_dashboard`` → ``lattice``).
_DEFINITIONS_BY_FAMILY = {
    (definition.port_family or key): definition for key, definition in FRAMEWORK_WEB_SERVERS.items()
}

#: Placeholder for a cell with nothing in it — an em dash reads as "there is
#: none here", where a blank cell reads as "this was not filled in".
_EMPTY = "—"


def _band(entry: PortSlot) -> tuple[int, int]:
    """Return the inclusive ``(lowest, highest)`` offset the slot occupies.

    Args:
        entry: The slot to measure.

    Returns:
        ``(offset, offset)`` for a single-port slot, and the first and last
        offset of the band for an indexed one — the per-user families, the
        dispatch workers, the extra virtual-accelerator instances and the
        facility band.
    """
    low, high = index_bounds(entry)
    return (entry.offset + low, entry.offset + high)


def _offset_text(entry: PortSlot) -> str:
    """Return the Offset cell, e.g. ``+803`` or ``+100 to +199``."""
    first, last = _band(entry)
    if first == last:
        return f"+{first}"
    return f"+{first} to +{last}"


def _port_text(entry: PortSlot) -> str:
    """Return the Port cell at the default base, e.g. ``10803``."""
    first, last = _band(entry)
    if first == last:
        return str(DEFAULT_PORT_BASE + first)
    return f"{DEFAULT_PORT_BASE + first}-{DEFAULT_PORT_BASE + last}"


def _env_var_text(entry: PortSlot) -> str:
    """Return the env var that overrides this slot's port, or the placeholder.

    Args:
        entry: The slot being rendered.

    Returns:
        The registry's ``port_env_var`` for a per-user panel family, and
        :data:`_EMPTY` for every other slot. The web terminal itself is a
        per-user family but is not a companion server, so it has no registry
        entry and so no value for this column. That is not the same as having
        no env override: ``osprey web --port`` moves it, and inside a
        multi-user container the declared ``OSPREY_TERMINAL_WEB_PORT``
        (``cli/web_cmd.py``) outranks both that flag and the config.
    """
    definition = _DEFINITIONS_BY_FAMILY.get(entry.name)
    if entry.per_index and definition is not None:
        return definition.port_env_var
    return _EMPTY


def _cell(text: str, *, literal: bool) -> nodes.entry:
    """Return one table cell.

    Args:
        text: The cell's text.
        literal: Render it as inline code. Ports, slot names, config keys and
            env vars are all things a reader copies, so they are set in code;
            the tier name is prose.

    Returns:
        The populated ``entry`` node.
    """
    entry = nodes.entry()
    paragraph = nodes.paragraph()
    if literal:
        paragraph += nodes.literal(text=text)
    else:
        paragraph += nodes.Text(text)
    entry += paragraph
    return entry


def _row(entry: PortSlot) -> nodes.row:
    """Return the table row for one slot."""
    row = nodes.row()
    row += _cell(_offset_text(entry), literal=True)
    row += _cell(_port_text(entry), literal=True)
    row += _cell(entry.name, literal=True)
    row += _cell(entry.tier, literal=False)
    row += _cell(entry.config_key or _EMPTY, literal=entry.config_key is not None)
    env_var = _env_var_text(entry)
    row += _cell(env_var, literal=env_var != _EMPTY)
    return row


def _header_row() -> nodes.row:
    """Return the header row."""
    row = nodes.row()
    for heading in COLUMNS:
        cell = nodes.entry()
        cell += nodes.paragraph(text=heading)
        row += cell
    return row


class OspreyPortsDirective(SphinxDirective):
    """Render :data:`osprey.port_layout.LAYOUT` as a table.

    Usage::

        .. osprey-ports::

    Six columns: the offset from ``deployment.port_base``, the resulting port
    at the default base, the slot name, the tier, the config key that overrides
    the slot, and the environment variable that overrides it.
    """

    required_arguments = 0
    optional_arguments = 0
    option_spec: dict[str, Any] = {}
    has_content = False

    def run(self) -> list[nodes.Node]:
        """Build the table node."""
        table = nodes.table()
        table["classes"].append("osprey-ports")

        group = nodes.tgroup(cols=len(COLUMNS))
        table += group
        for width in COLUMN_WIDTHS:
            group += nodes.colspec(colwidth=width)

        head = nodes.thead()
        head += _header_row()
        group += head

        body = nodes.tbody()
        for entry in LAYOUT:
            body += _row(entry)
        group += body

        return [table]


def setup(app: Sphinx) -> dict[str, Any]:
    """Register the ``osprey-ports`` directive with Sphinx."""
    app.add_directive("osprey-ports", OspreyPortsDirective)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
