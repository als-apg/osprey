"""Re-map the GTLView/BTSView CANOPY panels from legacy names to the demo grammar.

The element table (``injector_elements.tsv``, read by ``_elements``) pairs each
legacy name a panel binds with the demo channel that replaces it. This script
reads that table and offers two commands::

    python scripts/facility_demo/panel_rename.py check --facility <facility.json>
    python scripts/facility_demo/panel_rename.py apply --library <dir>

``check`` exits 0 only when every element-table ``demo_name`` is a channel of
the facility file, and otherwise lists the missing names and exits 1.

``apply`` rewrites, in place, every screen under ``<library>/screens/gtb`` and
``<library>/screens/bts`` (each panel with its own copy of the embedded
sub-screens). It rewrites binding sites only:

* the widget props ``pv`` and ``readbackPV``, and an ``xy-plot`` curve's
  ``x_channel``/``y_channel``;
* a ``write_pv`` action's ``pvName``;
* a rule's ``pvs[].name``;
* the values of a screen formula's ``bindings``;
* the embed macro values in ``MACRO_SITES``.

Free text (``text``, the ``DESC``/``BPM``/``TV`` macros, ``meta.comment``) is
never touched, nor is a value naming a local (``loc://``) or formula
(``fox://``) channel, nor a value that still holds a macro reference.

A legacy name can carry several rows, one per role (a paddle's command and its
state share one legacy name). Each site therefore picks its row through the
site table: an embed macro by ``MACRO_SITES``, a ``write_pv`` target or the
``pv`` of a write-capable widget the setpoint, a ``readbackPV`` the readback.
Any other site takes the legacy name's only row, and fails when it has
several. A name already in the demo grammar's table is left as it is, so a
second ``apply`` changes nothing.

Before re-mapping, ``apply`` makes the structural edits of ``_panel_edits``
(the paddle split, the lamp LED, the knob-mechanic and Matlab cuts, the
cross-links, the source-bug fixes) and deletes ``screens/common/KnobPanel``.

``apply`` is all or nothing: when any site names a legacy PV the table does
not resolve, or a panel lacks a widget an edit needs, it lists every such
problem and writes or deletes no file.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _elements import TABLE, ElementRow, load_elements
from _panel_edits import KNOB_PANEL, PADDLE_RB, Edits

#: The screen directories ``apply`` rewrites, relative to the library.
PANEL_DIRS = ("screens/gtb", "screens/bts")

#: Prefixes of channel names that are not facility channels.
LOCAL_PREFIXES = ("loc://", "fox://")


@dataclass(frozen=True)
class Selector:
    """Which of a legacy name's rows a binding site resolves to.

    ``field`` is ``"site"`` (match the row template's panel site) or
    ``"role"`` (match the row's role key).
    """

    field: str
    value: str

    def matches(self, row: ElementRow) -> bool:
        """Whether ``row`` is the kind of row this site binds."""
        if self.field == "role":
            return row.role == self.value
        return row.template.site == self.value

    def __str__(self) -> str:
        return f"{self.field} {self.value}"


SETPOINT = Selector("site", "sp")
READBACK = Selector("site", "rb")

#: The embed macros that bind a channel, and the row each one resolves to.
MACRO_SITES: dict[str, Selector] = {
    "CNTL_PV": SETPOINT,
    "MON_PV": READBACK,
    "COLOR_PV": Selector("site", "status"),
    "BPM_X": Selector("site", "position"),
    "BPM_Y": Selector("site", "position"),
    "paddle": Selector("role", "tv_paddle_sp"),
    PADDLE_RB: Selector("role", "tv_paddle_rb"),
    "light": Selector("role", "tv_lamp_rb"),
}

#: Widget types whose ``pv`` writes to the machine (CANOPY's write-capable set).
WRITABLE_TYPES = frozenset(
    {
        "pv-button",
        "pv-toggle",
        "pv-slider",
        "pv-text-input",
        "pv-checkbox",
        "pv-enum-combobox",
        "pv-spinbox",
        "pv-radio-group",
    }
)


class RenameError(Exception):
    """A binding site the element table cannot resolve."""


@dataclass(frozen=True)
class Site:
    """One binding site of a screen: where it is, what it holds, its selector."""

    file: Path
    where: str
    value: str
    selector: Selector | None
    holder: Any
    key: Any


class RenameMap:
    """The element table, keyed by legacy name."""

    def __init__(self, rows: Sequence[ElementRow]) -> None:
        self.by_legacy: dict[str, list[ElementRow]] = {}
        for row in rows:
            if row.legacy_pv is not None:
                self.by_legacy.setdefault(row.legacy_pv, []).append(row)
        self.demo_names = frozenset(row.demo_name for row in rows)

    def resolve(self, legacy: str, selector: Selector | None) -> str:
        """The demo name replacing ``legacy`` at a site with ``selector``.

        Raises:
            RenameError: The name is not in the table; or the site has a
                selector and no row, or several rows, match it; or the site has
                none and the name has several rows.
        """
        rows = self.by_legacy.get(legacy)
        if not rows:
            raise RenameError(f"{legacy!r} is not in the element table")
        if selector is None:
            if len(rows) > 1:
                roles = ", ".join(row.role for row in rows)
                raise RenameError(
                    f"{legacy!r} has several rows ({roles}) and this site is not in the role table"
                )
            return rows[0].demo_name
        matching = [row for row in rows if selector.matches(row)]
        if len(matching) != 1:
            roles = ", ".join(row.role for row in rows)
            count = "no row" if not matching else "several rows"
            raise RenameError(f"{legacy!r} has {count} for {selector} (rows: {roles})")
        return matching[0].demo_name

    def paddle_readback(self, paddle: str) -> str:
        """The ``paddle_rb`` macro value of a ``_tv`` instance whose ``paddle`` is ``paddle``.

        A legacy name is kept, for the rename to resolve to the readback row;
        a demo name (a panel re-mapped before the split) gives its readback.
        """
        if paddle not in self.demo_names:
            return paddle
        legacy = next(
            (
                name
                for name, rows in self.by_legacy.items()
                if any(r.demo_name == paddle for r in rows)
            ),
            None,
        )
        if legacy is None:
            raise RenameError(f"{paddle!r} has no legacy name to pair a readback with")
        return self.resolve(legacy, MACRO_SITES[PADDLE_RB])

    def needs_rename(self, value: str) -> bool:
        """Whether a site value is a name ``apply`` maps (not local, macro or demo)."""
        stripped = value.strip()
        if not stripped or stripped.startswith(LOCAL_PREFIXES):
            return False
        if "${" in stripped or "$(" in stripped:
            return False
        return stripped not in self.demo_names


# --- finding the binding sites ---------------------------------------------------


def _node_sites(file: Path, node: dict[str, Any], path: str) -> Iterator[Site]:
    props = node.get("props")
    if isinstance(props, dict):
        node_type = node.get("type")
        if isinstance(props.get("pv"), str):
            selector = SETPOINT if node_type in WRITABLE_TYPES else None
            yield Site(file, f"{path}.props.pv", props["pv"], selector, props, "pv")
        if isinstance(props.get("readbackPV"), str):
            yield Site(
                file, f"{path}.props.readbackPV", props["readbackPV"], READBACK, props, "readbackPV"
            )
        curves = props.get("curves")
        if node_type == "xy-plot" and isinstance(curves, list):
            for index, curve in enumerate(curves):
                for key in ("x_channel", "y_channel"):
                    if isinstance(curve, dict) and isinstance(curve.get(key), str):
                        where = f"{path}.props.curves[{index}].{key}"
                        yield Site(file, where, curve[key], None, curve, key)
        actions = props.get("actions")
        for index, action in enumerate(actions if isinstance(actions, list) else []):
            if (
                isinstance(action, dict)
                and action.get("type") == "write_pv"
                and isinstance(action.get("pvName"), str)
            ):
                where = f"{path}.props.actions[{index}].pvName"
                yield Site(file, where, action["pvName"], SETPOINT, action, "pvName")
        macros = props.get("macros")
        if isinstance(macros, dict):
            for name, selector in MACRO_SITES.items():
                if isinstance(macros.get(name), str):
                    where = f"{path}.props.macros.{name}"
                    yield Site(file, where, macros[name], selector, macros, name)
    rules = node.get("rules")
    for index, rule in enumerate(rules if isinstance(rules, list) else []):
        pvs = rule.get("pvs") if isinstance(rule, dict) else None
        for pv_index, pv in enumerate(pvs if isinstance(pvs, list) else []):
            if isinstance(pv, dict) and isinstance(pv.get("name"), str):
                where = f"{path}.rules[{index}].pvs[{pv_index}].name"
                yield Site(file, where, pv["name"], None, pv, "name")


def _walk(file: Path, node: Any, path: str) -> Iterator[Site]:
    if isinstance(node, dict):
        if "type" in node and ("props" in node or "rules" in node):
            yield from _node_sites(file, node, path)
        for key, value in node.items():
            yield from _walk(file, value, f"{path}.{key}" if path else key)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _walk(file, value, f"{path}[{index}]")


def binding_sites(file: Path, document: dict[str, Any]) -> Iterator[Site]:
    """Every binding site of one screen document, in document order."""
    formulas = document.get("formulas")
    for index, formula in enumerate(formulas if isinstance(formulas, list) else []):
        bindings = formula.get("bindings") if isinstance(formula, dict) else None
        if isinstance(bindings, dict):
            for key, value in bindings.items():
                if isinstance(value, str):
                    where = f"formulas[{index}].bindings.{key}"
                    yield Site(file, where, value, None, bindings, key)
    yield from _walk(file, document.get("root"), "root")


# --- the commands ----------------------------------------------------------------


def panel_files(library: Path) -> list[Path]:
    """The screens ``apply`` rewrites, sorted."""
    files: list[Path] = []
    for relative in PANEL_DIRS:
        files += sorted((library / relative).rglob("*.screen.json"))
    return files


def _render(document: dict[str, Any]) -> str:
    return json.dumps(document, indent=2, ensure_ascii=False) + "\n"


def apply(library: Path, rows: Sequence[ElementRow]) -> list[Path]:
    """Edit and re-map the library's panels in place; return the files changed or deleted.

    Raises:
        RenameError: One or more sites cannot be resolved; the message lists
            every one, and no file is written.
    """
    mapping = RenameMap(rows)
    files = panel_files(library)
    if not files:
        raise RenameError(f"no screens under {', '.join(PANEL_DIRS)} of {library}")
    rendered: dict[Path, str] = {}
    errors: list[str] = []
    for file in files:
        text = file.read_text(encoding="utf-8")
        document = json.loads(text)
        relative = file.relative_to(library / "screens").as_posix()
        edits = Edits(mapping.paddle_readback)
        edits.edit(relative, document)
        errors += [f"{file.relative_to(library)}: {error}" for error in edits.errors]
        for site in list(binding_sites(file, document)):
            if not mapping.needs_rename(site.value):
                continue
            try:
                site.holder[site.key] = mapping.resolve(site.value.strip(), site.selector)
            except RenameError as error:
                errors.append(f"{file.relative_to(library)}: {site.where}: {error}")
        new_text = _render(document)
        if new_text != text:
            rendered[file] = new_text
    if errors:
        raise RenameError(
            f"{len(errors)} site(s) cannot be edited or re-mapped:\n" + "\n".join(errors)
        )
    for file, new_text in rendered.items():
        file.write_text(new_text, encoding="utf-8")
    changed = list(rendered)
    knob_panel = library / KNOB_PANEL
    if knob_panel.exists():
        knob_panel.unlink()
        changed.append(knob_panel)
    return sorted(changed)


def missing_names(facility: Path, rows: Sequence[ElementRow]) -> list[str]:
    """Element-table ``demo_name`` values that are not channels of ``facility``."""
    document = json.loads(facility.read_text(encoding="utf-8"))
    channels = document.get("channels") or []
    present = {channel.get("id") for channel in channels if isinstance(channel, dict)}
    wanted = dict.fromkeys(row.demo_name for row in rows)
    return [name for name in wanted if name not in present]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--table", type=Path, default=TABLE, help="the element table (TSV)")
    commands = parser.add_subparsers(dest="command", required=True)
    check_parser = commands.add_parser("check", help="every demo_name is in a facility.json")
    check_parser.add_argument("--facility", type=Path, required=True)
    apply_parser = commands.add_parser("apply", help="re-map a CANOPY library in place")
    apply_parser.add_argument("--library", type=Path, required=True)
    args = parser.parse_args(argv)

    rows = load_elements(args.table)
    if args.command == "check":
        missing = missing_names(args.facility, rows)
        if missing:
            print(
                f"{len(missing)} element-table name(s) missing from {args.facility}:",
                file=sys.stderr,
            )
            for name in missing:
                print(f"  {name}", file=sys.stderr)
            return 1
        print(f"all {len({row.demo_name for row in rows})} element-table names present")
        return 0
    try:
        changed = apply(args.library, rows)
    except RenameError as error:
        print(error, file=sys.stderr)
        return 1
    written = [file for file in changed if file.exists()]
    print(f"re-mapped {len(written)} screen(s)")
    for file in written:
        print(f"  {file.relative_to(args.library)}")
    for file in changed:
        if not file.exists():
            print(f"  deleted {file.relative_to(args.library)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
