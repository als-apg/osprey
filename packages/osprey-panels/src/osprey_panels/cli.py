"""``osprey-panels write|check|classes``: operator screens from a facility file.

Experimental: the commands, the rule tables and the screens may change in any
release.

Every input is named on the command line; nothing is looked up from a
checkout or a profile. Reports go to stdout, warnings and errors to stderr.
Exit codes: 0 when the run did what it was asked, 1 on failing problems or a
refused input, 2 on a usage error.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from osprey_panels import emit
from osprey_panels.classes import summary, summary_line
from osprey_panels.describe import (
    FRONT_CONTESTED,
    FRONT_NONE,
    FRONT_READING,
    FRONT_STATED,
    HOP_FOUND,
    STRIP_BY_PLACE,
    STRIP_NOWHERE,
    STRIP_PLACED,
    card_counts,
    card_rule_counts,
    chain_states,
    front_states,
    ladder_states,
    press_counts,
    strip_states,
    tile_less,
)
from osprey_panels.document import FacilityDocumentError
from osprey_panels.model import PanelGraph, load_graph
from osprey_panels.tables import PanelTableError
from osprey_panels.view import VIEW_KINDS

__all__ = ["main"]

#: The view kinds ``--kind`` takes: the engine's own list, so the two cannot drift.
VIEW_CHOICES: tuple[str, ...] = tuple(VIEW_KINDS)

PROG = "osprey-panels"


class _Refused(Exception):
    """An input the command will not proceed with; already printed."""


# -- printers -------------------------------------------------------------------------


def _report(line: str) -> None:
    """A line of the verb's report, on stdout."""
    sys.stdout.write(f"{line}\n")


def _warn(summary_text: str) -> None:
    """Something the verb worked around, on stderr."""
    sys.stderr.write(f"warning: {summary_text}\n")


def _fail(summary_text: str, cause: str | None = None) -> None:
    """A failure or a refusal, on stderr: what failed, then why, one step in."""
    lines = [f"error: {summary_text}", *(f"  {line}" for line in (cause or "").splitlines())]
    sys.stderr.write("".join(f"{line}\n" for line in lines))


# -- inputs ---------------------------------------------------------------------------


def _graph(facility: Path, tables: Path) -> PanelGraph:
    """The panel graph of the facility file and its rule tables."""
    if not tables.is_dir():
        _fail("The rule tables cannot be read.", f"no directory at {tables}")
        raise _Refused
    try:
        graph = load_graph(facility, tables)
    except (FacilityDocumentError, PanelTableError, OSError) as error:
        _fail("The facility cannot be read.", str(error))
        raise _Refused from error
    if graph.unread:
        _report(
            f"{graph.unread} values of the facility's rows cannot be read "
            "(a rack unit that is no integer, for example); drawn without them"
        )
    if graph.withheld:
        _report(f"{graph.withheld} things withheld as part of an excluded thing")
    return graph


def _theme_roles(theme: Path | None) -> Mapping[str, Mapping[str, str]]:
    """The theme's roles, or none without a theme."""
    if theme is None:
        return {}
    try:
        return emit.load_theme(theme)
    except emit.EmitError as error:
        _fail("The theme cannot be used.", str(error))
        raise _Refused from error


def _draw(tables: Path, draw_file: Path | None, theme: Path | None) -> emit.Draw:
    """The drawing settings: ``--draw``, else ``draw.json`` beside the rule tables, else none."""
    roles = _theme_roles(theme)
    path = draw_file if draw_file is not None else tables / emit.DRAW_FILE
    if draw_file is None and not path.is_file():
        _report(f"no {emit.DRAW_FILE} beside the rule tables: drawing with the empty settings")
        return emit.Draw()
    try:
        return emit.load_draw(path, theme_roles=roles)
    except emit.EmitError as error:
        _fail("The drawing settings cannot be used.", str(error))
        raise _Refused from error


def _index(out: Path) -> dict[str, Any] | None:
    """The index under *out*, or ``None`` when there is none to read."""
    try:
        index = json.loads((out / emit.INDEX).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return index if isinstance(index, dict) else None


def _index_classes(out: Path) -> dict[str, Any] | None:
    """The class table of the index under *out*, or ``None`` when there is none to read."""
    index = _index(out)
    classes = index.get("classes") if index is not None else None
    return classes if isinstance(classes, dict) else None


def _card_rule_terms(row: Any) -> str:
    """The terms a ``cards.csv`` row selects devices by, as one line."""
    terms = [f"class {row.cls}"]
    if row.under:
        terms.append(f"under {row.under}")
    if row.devices:
        terms.append(f"devices {row.devices}")
    if row.attribute:
        terms.append(f"attribute {row.attribute[0]}={row.attribute[1]}")
    if row.when:
        terms.append(f"when {';'.join(row.when)}")
    return ", ".join(terms)


def _most(states: Counter[str]) -> str:
    """`` (most: <reason>, <n>)`` for the commonest reason in *states*; empty when none."""
    if not states:
        return ""
    why, n = states.most_common(1)[0]
    return f" (most: {why}, {n})"


def _megabytes(size: int) -> str:
    return f"{size / 1_000_000:.1f} MB"


# -- write ----------------------------------------------------------------------------


def _write(args: argparse.Namespace) -> int:
    graph = _graph(args.facility, args.tables)
    draw = _draw(args.tables, args.draw, args.theme)
    out: Path = args.out
    keep: list[Callable[[Any], bool]] = []
    if args.kinds:
        kinds = set(args.kinds)
        keep.append(lambda target: target.view in kinds)
    if args.place is not None:
        try:
            keep.append(emit.under(graph, args.place))
        except KeyError:
            _fail("No such place.", f"the facility holds no place {args.place!r}")
            return 1
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            emit.write(
                graph,
                out,
                draw=draw,
                only=(lambda t: all(test(t) for test in keep)) if keep else None,
            )
        for warning in caught:
            _warn(str(warning.message))
    except emit.EmitError as error:
        _fail("Nothing was written.", str(error))
        return 1
    index = json.loads((out / emit.INDEX).read_text(encoding="utf-8"))
    screens = index["screens"]
    count = Counter(entry["kind"] for entry in screens)
    sizes = Counter[str]()
    for entry in screens:
        sizes[entry["kind"]] += entry["bytes"]
    _report(f"{len(screens)} screens, {_megabytes(index['total_bytes'])}, in {out}")
    _report(f"  classes: {summary_line(summary(index['classes']))}")
    for kind in VIEW_CHOICES:
        if count[kind]:
            _report(f"  {kind}: {count[kind]} screens, {_megabytes(sizes[kind])}")
    devices = [e["thing"] for e in screens if e["kind"] == "card" and e["thing_kind"] == "device"]
    if devices:
        _report_device_cards(graph, devices)
    if index["outside"]:
        _report(f"  {len(index['outside'])} screens not written (outside the selection)")
    largest = sorted(screens, key=lambda entry: (-entry["bytes"], entry["file"]))[: args.top]
    if largest:
        _report("largest:")
        for entry in largest:
            _report(f"  {entry['bytes']:>12,}  {entry['file']}")
    return 0


def _report_device_cards(graph: PanelGraph, devices: list[str]) -> None:
    """The recipes of the written device cards and how whole each recipe's parts are drawn."""
    recipes = Counter(
        getattr(graph.recipe(device), "recipe", None) or "default" for device in devices
    )
    named = ", ".join(f"{n} {name}" for name, n in sorted(recipes.items()) if name != "default")
    _report(
        f"  device cards: {recipes['default']} with the default recipe"
        + (f", {named}" if named else "")
    )
    for recipe_name, where in sorted(strip_states(graph, devices).items()):
        total = sum(where.values())
        _report(
            f"  strip {recipe_name}: {where[STRIP_PLACED]} / {total} placed, "
            f"{where[STRIP_BY_PLACE]} by their place, {where[STRIP_NOWHERE]} not stated"
        )
    for chain, states in sorted(chain_states(graph, devices).items()):
        drawn = states.pop(HOP_FOUND, 0)
        short = sum(states.values())
        _report(
            f"  chain {chain}: {drawn} drawn whole, {short} with a hop not drawn{_most(states)}"
        )
    for recipe_name, states in sorted(front_states(graph, devices).items()):
        total = states[FRONT_STATED] + states[FRONT_NONE] + states[FRONT_CONTESTED]
        _report(
            f"  front {recipe_name}: {total} connectors, {states[FRONT_STATED]} with "
            f"something stated on them, {states[FRONT_CONTESTED]} contested, "
            f"{states[FRONT_NONE]} with nothing stated, "
            f"and {states[FRONT_READING]} readings; devices: "
            f"{states['devices: all']} all stated, {states['devices: some']} some, "
            f"{states['devices: none']} none, "
            f"{states['devices: readings only']} with readings only"
        )
    for recipe_name, states in sorted(ladder_states(graph, devices).items()):
        whole = states.pop(HOP_FOUND, 0)
        _report(
            f"  ladder {recipe_name}: {whole} drawn whole, "
            f"{sum(states.values())} with a part not served{_most(states)}"
        )


# -- check ----------------------------------------------------------------------------


def _check(args: argparse.Namespace) -> int:
    graph = _graph(args.facility, args.tables)
    draw = _draw(args.tables, args.draw, args.theme)
    out: Path = args.out
    problems = emit.check(graph, out, max_bytes=args.max_bytes, draw=draw)
    for standing, classes in emit.picture_classes(graph).items():
        _report(f"pictures, {standing}: {emit.named_classes(classes)}")
    loose = emit.unplaced_parts(graph)
    counted = ", ".join(f"{cls} {n}" for cls, n in loose.items()) or "none"
    _report(f"pictures, parts not placed: {counted}")
    devices = [t.id for t in graph.things() if t.kind == "device"]
    presses = press_counts(graph, devices)
    _report(
        f"device cards, presses: {presses.stated} with a stated value, {presses.inferred} "
        f"with an inferred value (said so on the card), {presses.left_out} left out "
        f"(no value stated) on {presses.left_out_cards} cards, {presses.withheld} "
        f"withheld, join inferred"
    )
    bare_classes, bare = tile_less(graph, devices)
    _report(f"default card without tiles: {bare_classes} classes, {bare} devices")
    cardless, ownerless, by_default = card_counts(graph)
    _report(
        f"no card: {cardless} devices, {cardless - ownerless} drawn on their owner's card, "
        f"{ownerless} with no owner stated (no index entry)"
    )
    _report(f"  by default (no channel, no drawing): {by_default} devices")
    for number, (row, n) in enumerate(card_rule_counts(graph), start=1):
        _report(f"  cards.csv row {number}, {row.action} ({_card_rule_terms(row)}): {n} devices")
    for table, n in sorted(graph.redundant_rows().items()):
        _report(f"redundant: {table}: {n} rows restate the facility file or the default")
    for column, n in sorted(graph.redundant_columns.items()):
        _report(f"redundant: {column}: {n} rows state the default; the value may be left empty")
    count = Counter(problem.kind for problem in problems)
    for kind in sorted(count):
        _report(f"{kind}: {count[kind]}")
        shown = [problem for problem in problems if problem.kind == kind][: args.limit]
        for problem in shown:
            _report(f"  {problem.file}: {problem.detail}")
        if count[kind] > len(shown):
            _report(f"  ... and {count[kind] - len(shown)} more")
    class_table = _index_classes(out)
    if class_table is not None:
        _report(f"classes: {summary_line(summary(class_table))}")
    failing = [p for p in problems if p.failing or args.strict]
    if failing:
        _fail(f"{len(failing)} problems in the written screens.")
        return 1
    _report("The written screens match the model.")
    return 0


# -- classes --------------------------------------------------------------------------


def _classes(args: argparse.Namespace) -> int:
    out: Path = args.out
    classes = _index_classes(out)
    if classes is None:
        _fail("No class table.", f"{out / emit.INDEX} is missing or holds none; write again.")
        return 1
    rows = sorted(classes.items(), key=lambda kv: (-kv[1].get("members", 0), kv[0]))
    shown = rows if args.limit <= 0 else rows[: args.limit]
    _report(f"{'members':>8} {'off':>5} {'cut':>5}  class  (standard)")
    for name, row in shown:
        _report(
            f"{row.get('members', 0):>8} {row.get('off_standard', 0):>5} {row.get('cut', 0):>5}"
            f"  {name}  ({row.get('standard', '')})"
        )
    if len(rows) > len(shown):
        _report(f"  ... and {len(rows) - len(shown)} more")
    per_view: dict[str, Counter[str]] = {}
    for row in classes.values():
        view = str(row.get("key", {}).get("view", ""))
        tally = per_view.setdefault(view, Counter())
        tally["screens"] += row.get("members", 0)
        tally["classes"] += 1
        tally["single"] += row.get("members", 0) == 1
        tally["off"] += row.get("off_standard", 0)
        tally["cut"] += row.get("cut", 0)
    _report(f"{'view':<12} {'screens':>8} {'classes':>8} {'single':>7} {'off':>6} {'cut':>6}")
    for view in sorted(per_view, key=lambda v: -per_view[v]["screens"]):
        t = per_view[view]
        _report(
            f"{view:<12} {t['screens']:>8} {t['classes']:>8} {t['single']:>7}"
            f" {t['off']:>6} {t['cut']:>6}"
        )
    if args.show_differs:
        index = _index(out) or {}
        for entry in index.get("screens", ()):
            if entry.get("differs"):
                _report(f"{entry['file']} ({entry.get('class')}):")
                for words in entry["differs"]:
                    _report(f"  {words}")
    _report(f"classes: {summary_line(summary(classes))}")
    return 0


# -- parser ---------------------------------------------------------------------------


def _positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"{text} is not at least 1")
    return value


def _inputs(parser: argparse.ArgumentParser) -> None:
    """The inputs ``write`` and ``check`` read: facility, tables, draw and theme."""
    parser.add_argument(
        "--facility", type=Path, required=True, metavar="FILE", help="The facility file."
    )
    parser.add_argument(
        "--tables", type=Path, required=True, metavar="DIR", help="The rule tables directory."
    )
    parser.add_argument(
        "--draw",
        type=Path,
        default=None,
        metavar="FILE",
        help="The drawing settings (JSON); default: draw.json beside the rule tables.",
    )
    parser.add_argument(
        "--theme",
        type=Path,
        default=None,
        metavar="FILE",
        help="A theme file whose state colours the drawing colours are held against; "
        "default: the engine's fallback state colours only.",
    )
    parser.add_argument(
        "--out", type=Path, required=True, metavar="DIR", help="The output directory."
    )


def build_parser() -> argparse.ArgumentParser:
    """The command's argument parser."""
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Write operator screens from a facility file, and check a written set. "
        "Experimental: the commands, the rule tables and the screens may change in any release.",
    )
    verbs = parser.add_subparsers(dest="verb", metavar="{write,check,classes}", required=True)

    write = verbs.add_parser(
        "write",
        help="Write the screens of the facility under --out, with an index.",
        description="Write the screens of the facility under --out, with an index. Files a "
        "former run wrote and that are no screen any more are deleted; nothing else in --out "
        "is touched, and a non-empty --out without an index.json is refused. With --kind and "
        "--under, only the screens that match both are written; links to the others are "
        "still written, and check notes them. Prints the screen count and bytes per view kind "
        "and the largest screens.",
    )
    _inputs(write)
    write.add_argument(
        "--kind",
        dest="kinds",
        action="append",
        choices=VIEW_CHOICES,
        default=[],
        metavar="KIND",
        help=f"Write only this view kind; repeatable. One of {', '.join(VIEW_CHOICES)}.",
    )
    write.add_argument(
        "--under",
        dest="place",
        default=None,
        metavar="PLACE_ID",
        help="Write only this place and all within it.",
    )
    write.add_argument(
        "--top", type=int, default=10, metavar="N", help="Largest screens to list (default 10)."
    )
    write.set_defaults(run=_write)

    check = verbs.add_parser(
        "check",
        help="Compare the screens under --out with what the model generates now.",
        description="Compare the screens under --out with what the model generates now; "
        "writes nothing. Exits 1 on any failing problem, a screen the Screen-IR schema refuses "
        "among them; notes (outside_set, a card or checkout over its budget, unbound_probe, "
        "unbound_picture, ambiguous_picture) fail only under --strict. Prints the device "
        "classes by how pictures.csv draws them, the card counts, the redundant table rows, "
        "then the count per problem kind and the first --limit of each.",
    )
    _inputs(check)
    check.add_argument(
        "--strict", action="store_true", help="Links to screens outside the written set fail too."
    )
    check.add_argument(
        "--max-bytes", type=_positive, default=None, metavar="N", help="Screens over N bytes fail."
    )
    check.add_argument(
        "--limit", type=int, default=20, metavar="N", help="Problems printed per kind (default 20)."
    )
    check.set_defaults(run=_check)

    classes = verbs.add_parser(
        "classes",
        help="Print the screen classes the index under --out records.",
        description="Print the screen classes the index under --out records; reads nothing "
        "else. One row per class, by member count: members, screens off the standard, screens "
        "with a label cut, the class and its standard's file. Then per view kind its screens, "
        "classes, classes of one screen, screens off their standard and screens with a label "
        "cut. The last line is the summary a gate reads.",
    )
    classes.add_argument(
        "--out",
        type=Path,
        required=True,
        metavar="DIR",
        help="The output directory a write filled.",
    )
    classes.add_argument(
        "--limit",
        type=int,
        default=30,
        metavar="N",
        help="Classes listed; 0 lists all (default 30).",
    )
    classes.add_argument(
        "--differs",
        dest="show_differs",
        action="store_true",
        help="List each screen off its standard and how.",
    )
    classes.set_defaults(run=_classes)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command.

    Args:
        argv: The arguments after the program name; ``None`` reads ``sys.argv``.

    Returns:
        The exit code: 0 ok, 1 failing problems or a refused input, 2 a usage error.
    """
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exit_:
        return exit_.code if isinstance(exit_.code, int) else 2
    try:
        return int(args.run(args))
    except _Refused:
        return 1
