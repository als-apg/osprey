"""The structural guard on how OSPREY's CLI reaches a terminal.

The output migration moved every printed line onto one renderer and every table
and panel onto one pair of factories. That work is done; this file is what keeps
it done. It is a drift tripwire, not a test of behaviour -- it reads the source
of ``src/osprey/cli``, ``src/osprey/deployment`` and ``src/osprey/health/render.py``
with :mod:`ast` and asks four structural questions:

1. Is a :class:`~rich.console.Console` built anywhere but ``styles.py``? The two
   consoles there are the process's only consoles, and they are identity-stable
   for its lifetime, so a third one built elsewhere writes past the theme, past
   ``set_theme``'s re-theming, and past any live region that owns the cursor.
2. Is ``click.echo`` called outside the handful of seams that genuinely need
   raw bytes? ``click.echo`` writes at the stream. It does not know about the
   renderer's altitudes, it does not know a live region is mounted, and it does
   not know which stream a machine reader is parsing.
3. Is the builtin ``print`` called at all? It has no legal use here.
4. Is a :class:`~rich.table.Table` or :class:`~rich.panel.Panel` constructed
   outside ``styles.py``? The box, padding and border tokens are spelled in the
   factories and nowhere else, so a hand-built surface is a look that retheming
   cannot reach.

**How to answer a failure.** Almost always: route the line through
:mod:`osprey.cli.output` (``report``/``note``/``section``/``warn``/``fail``), or
build the surface with :func:`osprey.cli.styles.data_table`,
:func:`~osprey.cli.styles.layout_table` or :func:`~osprey.cli.styles.panel`. Only
when a caller's contract is the *bytes* -- a ``--json`` document, ``--help``
text, a diff piped to a file -- does an allowlist entry below become the right
answer, and then it needs a sentence saying which caller reads those bytes.

**Two dispositions, on purpose.** Rules 1, 3 and 4 are zero-tolerance: their
allowlists are *exact*, so the console inventory and the set of hand-built
surfaces cannot drift in either direction without somebody looking at it. Rule 2
is a *ratchet*: its entries are upper bounds, so retiring an echo seam never reds
this file, while adding one to an already-listed file trips the guard exactly
like adding one to a clean file. The asymmetry is deliberate -- a migration that
removes a raw echo must not have to edit this guard in the same breath, whereas
a fifth Console appearing anywhere is news either way.

**On matching.** Everything here matches on the shape of the call, never on a
line number, so the allowlist survives the files moving underneath it. Import
aliases are read per module, so renaming an import on the way in
(``from rich.table import Table as RT``) does not buy an exemption.

Rich's two classmethod constructors are decided one at a time rather than
inheriting whatever their spelling implies: ``Table.grid()`` is allowed, because
a layout table draws nothing and so carries none of the tokens ``styles.py``
owns (``live_render.py`` relies on this); ``Panel.fit()`` is forbidden and
matched explicitly, because it spells border and padding exactly like the
constructor does.

*Standing rule for whoever adds rule 5.* A guard that matches on the called name
inherits a family of exemptions it never chose -- every classmethod constructor,
every import alias. So when you add a rule, enumerate the constructor's alternate
spellings (``.fit``, ``.grid``, ``.from_*``) and decide each one here in writing.
The spellings you do not name are the ones a well-meaning developer will reach
for.

**Scope, and what sits outside it.** The trees below are this feature's scope.
Two live CLI output surfaces are outside them and are NOT guarded:
``services/channel_finder/tools/validate_database.py`` and ``preview_database.py``,
which render everything ``osprey channel-finder validate`` and ``osprey
channel-finder preview`` print. Both hand-build their tables and panels and hold
a module-level console. The CLI injects the shared console when it calls them, so
the theme does reach that output, but their box and padding tokens are spelled
outside ``styles.py``. A green run here is therefore not a claim about the CLI's
entire visual surface -- only about the trees named below.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "osprey"

#: Whole trees walked by every rule.
_GUARDED_TREES = (SRC / "cli", SRC / "deployment")

#: Single files walked by every rule. ``health/render.py`` is here rather than
#: the whole of ``health`` because it is the only module in that package that
#: prints; its ``render_json`` is the repo's reference machine seam.
_GUARDED_FILES = (SRC / "health" / "render.py",)


# ---------------------------------------------------------------------------
# Call matchers
#
# Each takes one ast.Call and the names that module bound from click and rich,
# and answers whether the call is the thing its rule forbids. They match the
# CALLED NAME, so an attribute call (``console.print``, ``Table.grid``) is a
# different call from the bare one, which is exactly the distinction the rules
# need -- with the named exceptions below, each of which was decided rather than
# inherited.
# ---------------------------------------------------------------------------


def _called_name(call: ast.Call) -> str | None:
    """The final name in ``call``'s callee: ``a.b.C()`` gives ``"C"``.

    Returns ``None`` for a callee that is not a name at all -- a subscript, a
    call of a call -- because such a callee names nothing this file can rule on.
    """
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


@dataclass(frozen=True)
class _Bindings:
    """What one module called the printing machinery it imported.

    Matching on the called name alone reads only the *unaliased* spelling, which
    hands anyone an accidental exemption by renaming an import. A module that
    already has its own ``Table`` or ``Console`` would alias on import for
    ordinary reasons, so the aliased spelling is not an exotic case; collecting
    the bindings per module is what keeps the rules reading the same code a
    person does.
    """

    #: Names bound to click's raw stream writers (``echo``, ``secho``).
    click_writers: frozenset[str]
    #: Names bound to :class:`rich.console.Console`.
    consoles: frozenset[str]
    #: Names bound to :class:`rich.table.Table` or :class:`rich.panel.Panel`.
    surfaces: frozenset[str]
    #: Names bound to rich's module-level ``print``.
    rich_prints: frozenset[str]
    #: Names bound to the ``rich`` package itself, for the dotted ``rich.print``.
    rich_modules: frozenset[str]


#: Where each guarded name lives, as ``module -> {imported name: binding field}``.
_IMPORT_SOURCES: dict[str, dict[str, str]] = {
    "click": {"echo": "click_writers", "secho": "click_writers"},
    "rich": {"print": "rich_prints"},
    "rich.console": {"Console": "consoles"},
    "rich.table": {"Table": "surfaces"},
    "rich.panel": {"Panel": "surfaces"},
}

#: The unaliased spellings, always in scope. A module that imports nothing still
#: gets these, so ``rich.table.Table(...)`` is matched on its trailing name.
_DEFAULT_BINDINGS: dict[str, frozenset[str]] = {
    "click_writers": frozenset(),
    "consoles": frozenset({"Console"}),
    "surfaces": frozenset({"Table", "Panel"}),
    "rich_prints": frozenset({"print"}),
    "rich_modules": frozenset(),
}


def _bindings(tree: ast.Module) -> _Bindings:
    """Read one module's import statements into a :class:`_Bindings`.

    Handles ``from X import Y``, ``from X import Y as Z`` and ``import rich [as
    r]``. Deliberately does NOT chase re-exports or module-attribute rebinding
    (``T = Table``): those are not idioms anybody reaches for by accident, and
    chasing them would mean interpreting the module rather than reading it.
    """
    collected: dict[str, set[str]] = {
        field: set(names) for field, names in _DEFAULT_BINDINGS.items()
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for imported, field in _IMPORT_SOURCES.get(node.module or "", {}).items():
                for alias in node.names:
                    if alias.name == imported:
                        collected[field].add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "rich":
                    collected["rich_modules"].add(alias.asname or "rich")
    return _Bindings(**{field: frozenset(names) for field, names in collected.items()})


def _is_console_construction(call: ast.Call, bindings: _Bindings) -> bool:
    """A ``Console(...)`` construction, however the name was imported.

    Matches the bare ``Console(...)``, the dotted ``console.Console(...)`` (whose
    trailing name is the same), and any alias the module bound, such as
    ``from rich.console import Console as C``.
    """
    return _called_name(call) in bindings.consoles


def _is_click_echo(call: ast.Call, bindings: _Bindings) -> bool:
    """A raw ``click.echo``/``click.secho``, dotted or imported bare.

    Deliberately NOT every ``.echo``: the phase reporter's ``echo`` is a
    renderer method that routes through the console, and
    ``echo = current_reporter().echo`` is a legal alias several verbs take. Only
    click's own writer is raw, so only click's own writer is matched -- via the
    dotted ``click.`` receiver, or via a name the module imported from click.

    Known gap: ``import click as c; c.echo(...)`` is missed, because the dotted
    branch pins the receiver to the literal name ``click``. Left as-is on the
    reviewer's measurement that nobody writes that import, and recorded here so
    the asymmetry with the rich aliases is a decision rather than an oversight.
    """
    func = call.func
    if isinstance(func, ast.Attribute) and func.attr in ("echo", "secho"):
        return isinstance(func.value, ast.Name) and func.value.id == "click"
    return isinstance(func, ast.Name) and func.id in bindings.click_writers


def _is_bare_print(call: ast.Call, bindings: _Bindings) -> bool:
    """A call of the builtin ``print``, and nothing that merely ends in it.

    ``current_reporter().out().print(table)`` and ``_target_console().print(...)``
    are Rich console prints and are how the CLI is supposed to draw. Matching on
    :class:`ast.Name` rather than on the trailing attribute is what separates the
    two, and getting that wrong would make this rule fire on the renderer itself.

    Also matches rich's own ``print``, under both spellings: ``from rich import
    print`` binds the same bare name, and the dotted ``rich.print(...)`` is
    caught through the module binding. Rich's ``print`` draws on rich's global
    console, past ``styles.py`` and past any mounted live region, which is the
    failure rule 1 exists to prevent.

    **Known blind spot: the builtin bound as a DEFAULT PARAMETER VALUE.** This
    rule matches calls, so ``def f(..., emit: Callable[[str], None] = print)``
    is invisible to it. No such site is left in the trees this guard walks --
    the two that existed (``auth_credentials.ensure_auth_credentials``'s
    ``echo`` and ``reset.reset_deployment``'s ``emit``) now require the sink --
    but nothing here would catch a new one, so a green run is not proof that
    the trees hold no reachable builtin print.
    """
    func = call.func
    if isinstance(func, ast.Attribute) and func.attr == "print":
        return isinstance(func.value, ast.Name) and func.value.id in bindings.rich_modules
    return isinstance(func, ast.Name) and func.id in bindings.rich_prints


def _is_surface_construction(call: ast.Call, bindings: _Bindings) -> bool:
    """A ``Table(...)`` or ``Panel(...)`` construction, however it is spelled.

    Aliases count: ``from rich.table import Table as RT`` then ``RT(box=None)``
    is the same construction wearing a different name.

    Rich offers two classmethod constructors here, and they are decided
    separately rather than sharing the fate their spelling gives them:

    - ``Table.grid()`` is **allowed**. It returns a layout table whose whole
      point is that it draws nothing, so it carries no box, padding or border
      tokens for ``styles.py`` to own. ``live_render.py`` uses it.
    - ``Panel.fit()`` is **forbidden**, and matched here explicitly. It is a full
      panel construction -- ``Panel.fit(text, border_style=..., padding=...)``
      spells exactly the tokens the factory exists to hold. Reading it as exempt
      would be an accident of it being a classmethod, not a decision.

    Matching ``.fit`` on any bound surface name is exact enough in practice:
    ``Table`` has no ``fit`` and ``Panel`` has no ``grid``, so neither name can
    borrow the other's disposition.
    """
    func = call.func
    if isinstance(func, ast.Attribute) and func.attr == "fit":
        return isinstance(func.value, ast.Name) and func.value.id in bindings.surfaces
    return _called_name(call) in bindings.surfaces


# ---------------------------------------------------------------------------
# Allowlists
#
# Keyed by path relative to ``src/osprey``, POSIX-spelled. The value is a count
# and the reason for it. Under a zero-tolerance rule the count is exact; under
# the ratchet it is an upper bound. Read the reason before adding a line to one
# of these files: it says what makes the existing calls legal, and a new call is
# only covered if the same sentence is true of it.
# ---------------------------------------------------------------------------

#: Rule 1, zero-tolerance. The whole console inventory of the process, exactly.
_CONSOLE_ALLOWLIST: dict[str, tuple[int, str]] = {
    "cli/styles.py": (
        2,
        "The stdout and stderr consoles, both built by _make_console. These are "
        "the CLI's only consoles: set_theme() re-themes them in place so that "
        "`from .styles import console` aliases taken at import time stay correct, "
        "which only works while nothing else builds one.",
    ),
}

#: Rule 2, the ratchet. Seams whose contract is the bytes. Upper bounds, so a
#: seam that is later migrated away only ever lowers a number here.
_ECHO_ALLOWLIST: dict[str, tuple[int, str]] = {
    # --- Reviewed seams. Each of these is deliberate and was ruled on during
    # --- the output migration.
    "cli/main.py": (
        1,
        "The bare-group `ctx.get_help()` echo. Help text is click's own rendered "
        "bytes and belongs on stdout unwrapped.",
    ),
    "cli/audit_cmd.py": (
        1,
        "_display_report's `click.echo(report.model_dump_json(indent=2))`. The "
        "verb's body runs inside output.machine_mode(), so every human line is "
        "on stderr and this is the one document on stdout. It does not route "
        "through the renderer precisely so the document lands on real stdout.",
    ),
    "cli/users_cmd.py": (
        1,
        "The users-env byte render. A caller redirects this to a file, so the "
        "bytes are the contract.",
    ),
    "cli/config_cmd.py": (
        1,
        "config_cmd._emit, the module's own single byte seam.",
    ),
    "cli/query_cmd.py": (
        2,
        "The two machine seams: the --json document, and the agent-text "
        "passthrough that a caller pipes.",
    ),
    "cli/scaffold_cmd.py": (
        4,
        "Two classes, both already blessed elsewhere. Two are help passthrough "
        "seams -- `click.echo(ctx.get_help())` on a bare group, the same class as "
        "main.py. The other two are byte-parity diff seams, the same class as "
        "users_cmd's byte render: each echoes a joined difflib.unified_diff with "
        "framework:/yours: headers and nothing else, for a reader who is piping "
        "or applying it. The matching no-differences branches print through the "
        "console with markup, so the theming lives on the human path and the raw "
        "path carries diff bytes only.",
    ),
}

#: Rule 3, zero-tolerance. Nothing, anywhere. The builtin has no legal use in
#: these trees, and as of this guard's first run there was not one call of it to
#: grandfather, so this allowlist starts empty and should stay that way.
_PRINT_ALLOWLIST: dict[str, tuple[int, str]] = {}

#: Rule 4, zero-tolerance. The factory bodies, exactly.
_SURFACE_ALLOWLIST: dict[str, tuple[int, str]] = {
    "cli/styles.py": (
        3,
        "The three factory bodies: data_table, layout_table and panel. This is "
        "the module the rule exists to funnel everything into, so the "
        "constructions here are the rule being satisfied rather than broken.",
    ),
}


@dataclass(frozen=True)
class _Rule:
    """One structural question, its matcher, its budget and its remedy."""

    #: What a failure calls the thing, in the words a reader would search for.
    label: str
    #: Takes a call and its module's :class:`_Bindings`; True if it violates.
    matches: object
    #: Path relative to ``src/osprey`` mapped to (count, reason).
    allowlist: dict[str, tuple[int, str]]
    #: The sentence a failure ends with, telling the reader what to do.
    remedy: str
    #: True for a zero-tolerance rule, whose counts are exact in both
    #: directions. False for the ratchet, whose counts are upper bounds.
    exact: bool


_RULES: dict[str, _Rule] = {
    "console": _Rule(
        label="Console(...) construction",
        matches=_is_console_construction,
        allowlist=_CONSOLE_ALLOWLIST,
        remedy=(
            "Import `console` or `err_console` from osprey.cli.styles instead of "
            "building one, or print through osprey.cli.output."
        ),
        exact=True,
    ),
    "click_echo": _Rule(
        label="raw click.echo/click.secho call",
        matches=_is_click_echo,
        allowlist=_ECHO_ALLOWLIST,
        remedy=(
            "Print through osprey.cli.output (report/note/section/warn/fail). If "
            "the bytes really are the contract -- a --json document, --help text "
            "-- add an entry to _ECHO_ALLOWLIST saying which caller reads them."
        ),
        exact=False,
    ),
    "bare_print": _Rule(
        label="builtin print(...) call",
        matches=_is_bare_print,
        allowlist=_PRINT_ALLOWLIST,
        remedy=(
            "Print through osprey.cli.output. A raw write lands inside a mounted "
            "live region instead of above it, and takes no altitude."
        ),
        exact=True,
    ),
    "surface": _Rule(
        label="direct rich Table(...)/Panel(...) construction",
        matches=_is_surface_construction,
        allowlist=_SURFACE_ALLOWLIST,
        remedy=(
            "Build it with osprey.cli.styles.data_table(), layout_table() or "
            "panel(). Pass content-shaped options (title, show_lines) as "
            "keywords; do not respell box, padding or border."
        ),
        exact=True,
    ),
}


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------


def _guarded_sources() -> list[tuple[str, str]]:
    """Every guarded module, as ``(path relative to src/osprey, source)``."""
    paths: list[Path] = []
    for tree in _GUARDED_TREES:
        paths.extend(sorted(tree.rglob("*.py")))
    paths.extend(path for path in _GUARDED_FILES if path.exists())
    return [(path.relative_to(SRC).as_posix(), path.read_text(encoding="utf-8")) for path in paths]


def _find(rule: _Rule, source: str) -> list[int]:
    """Line numbers in ``source`` where ``rule`` matches, in order."""
    tree = ast.parse(source)
    bindings = _bindings(tree)
    return sorted(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and rule.matches(node, bindings)
    )


def _over_budget(rule: _Rule, sources: Iterable[tuple[str, str]]) -> list[str]:
    """Failure lines for every guarded file that does not match its allowance.

    An empty list means the tree is clean under ``rule``. This is the whole
    decision, so the standing test and every self-test below drive it: a guard
    that cannot be shown to go red is not a guard.

    Under a zero-tolerance rule a file holding FEWER calls than its entry is
    reported too. That is not an obstacle to cleanup -- the message says the
    news is good and to lower the number -- it is what keeps the console and
    surface inventories honest, because an entry nobody ever has to revisit
    stops describing the tree.
    """
    problems: list[str] = []
    for relpath, source in sources:
        found = _find(rule, source)
        allowed, reason = rule.allowlist.get(relpath, (0, ""))
        if len(found) == allowed:
            continue
        if len(found) < allowed:
            if not rule.exact:
                continue
            problems.append(
                f"{relpath} holds {len(found)} {rule.label}(s) but its entry "
                f"claims {allowed}. This is good news: lower the number in the "
                f"allowlist to {len(found)}, and reword the reason if it no "
                f"longer describes what is left.\n"
                f"    what the entry says today: {reason}"
            )
            continue
        locations = ", ".join(f"{relpath}:{line}" for line in found)
        if allowed:
            limit = "exactly" if rule.exact else "at most"
            problems.append(
                f"{relpath} holds {len(found)} of the {rule.label}s but is "
                f"allowed {limit} {allowed}.\n"
                f"    at: {locations}\n"
                f"    why the existing ones are allowed: {reason}\n"
                f"    {rule.remedy}"
            )
        else:
            problems.append(
                f"{relpath} holds {len(found)} {rule.label}(s) and is on no "
                f"allowlist.\n"
                f"    at: {locations}\n"
                f"    {rule.remedy}"
            )
    return problems


@pytest.mark.parametrize("rule_name", sorted(_RULES))
def test_output_stays_inside_the_hierarchy(rule_name: str) -> None:
    """No guarded module reaches a terminal outside the renderer and factories."""
    problems = _over_budget(_RULES[rule_name], _guarded_sources())
    assert not problems, (
        f"\n\n{len(problems)} file(s) drifted out of the CLI output hierarchy:\n\n"
        + "\n\n".join(problems)
        + "\n"
    )


def test_the_pending_review_block_stays_separable() -> None:
    """Pending entries stay labelled and stay out of the permanent allowlist.

    Written for whoever retires the block: it catches a half-deletion that left
    a ``PENDING REVIEW.`` entry behind among the blessed seams, and it catches
    the reverse mistake of promoting an entry by moving it without dropping the
    label. It passes once the block is gone, because the pending dict is read
    through :func:`globals` rather than referenced directly.
    """
    pending: dict[str, tuple[int, str]] = globals().get("_ECHO_PENDING_REVIEW", {})
    for relpath, (_allowed, reason) in _ECHO_ALLOWLIST.items():
        labelled = reason.startswith("PENDING REVIEW.")
        if relpath in pending:
            assert labelled, f"{relpath} is in the pending block but is not labelled as pending"
        else:
            assert not labelled, (
                f"{relpath} is labelled PENDING REVIEW but sits outside the pending block. "
                "Either move it back inside the block, or drop the label now that it is blessed."
            )


def test_the_guarded_trees_are_actually_being_read() -> None:
    """The scan reaches real modules, so a green run means something.

    A file list that silently came back empty -- a moved package, a typo in a
    path constant -- would make all four rules pass by reading nothing.
    """
    sources = _guarded_sources()
    relpaths = {relpath for relpath, _ in sources}
    assert len(sources) > 50, f"only {len(sources)} guarded modules found under {SRC}"
    for expected in ("cli/styles.py", "cli/output.py", "health/render.py"):
        assert expected in relpaths, f"{expected} is not being scanned"


@pytest.mark.parametrize("rule_name", sorted(_RULES))
def test_no_allowlist_entry_outlives_its_file(rule_name: str) -> None:
    """Every allowlisted path still exists in the scanned set.

    ``_over_budget`` iterates sources and looks each one up in the allowlist, so
    an entry whose file was deleted or renamed is simply never visited and would
    survive forever with nothing to trip it. That is the staleness direction the
    exact rules claim to own -- their shrink-check only fires while the file is
    still there -- so it is pinned here rather than assumed.
    """
    relpaths = {relpath for relpath, _ in _guarded_sources()}
    stale = sorted(set(_RULES[rule_name].allowlist) - relpaths)
    assert not stale, (
        f"{rule_name} allows {stale}, which the scan does not reach. Either the "
        "file moved (update the key) or it is gone (drop the entry)."
    )


# ---------------------------------------------------------------------------
# Proof the guard can go red
#
# Every rule is driven against source that violates it, through the same
# _over_budget the standing test uses. If a matcher is ever loosened into
# uselessness -- or a budget is raised to infinity -- these fail rather than the
# guard quietly passing forever.
# ---------------------------------------------------------------------------

#: One violating module per rule, plus the near-miss each matcher must NOT
#: flag. The near-misses are the real content here: they are the calls that
#: actually appear in the guarded trees, and a matcher that flags them would
#: have to be defanged to get the suite green again.
_VIOLATION_SAMPLES: dict[str, tuple[str, str]] = {
    "console": (
        "from rich.console import Console\nextra = Console(stderr=True)\n",
        "from .styles import console\nconsole.print('hello')\n",
    ),
    "click_echo": (
        "import click\nclick.echo('raw bytes')\n",
        "from .phase_reporter import current_reporter\n"
        "echo = current_reporter().echo\n"
        "echo('through the reporter')\n",
    ),
    "bare_print": (
        "print('straight at stdout')\n",
        "from .phase_reporter import current_reporter\n"
        "current_reporter().out().print('through the console')\n",
    ),
    "surface": (
        "from rich.table import Table\ntable = Table(box=None)\n",
        "from rich.table import Table\nfrom .styles import data_table\n"
        "grid = Table.grid(padding=(0, 1))\ntable = data_table(title='Services')\n",
    ),
}


@pytest.mark.parametrize("rule_name", sorted(_RULES))
def test_each_rule_fires_on_a_violation(rule_name: str) -> None:
    """A new offending module in an unlisted file is reported by name."""
    violating, _ = _VIOLATION_SAMPLES[rule_name]
    problems = _over_budget(_RULES[rule_name], [("cli/newly_added_cmd.py", violating)])
    assert len(problems) == 1, f"{rule_name} did not fire on its violation sample"
    assert "cli/newly_added_cmd.py" in problems[0]
    assert _RULES[rule_name].remedy.split(".")[0] in problems[0]


@pytest.mark.parametrize("rule_name", sorted(_RULES))
def test_each_rule_ignores_its_near_miss(rule_name: str) -> None:
    """The legal spelling next door stays legal, so the guard is not a blunt grep."""
    _, near_miss = _VIOLATION_SAMPLES[rule_name]
    assert not _over_budget(_RULES[rule_name], [("cli/newly_added_cmd.py", near_miss)])


#: The alternate spellings of a guarded call, and whether each is a violation.
#: Every entry is a decision recorded in the docstrings above; this table is
#: what stops a decision from quietly reverting. ``id`` doubles as the reason.
_ALTERNATE_SPELLINGS: list[tuple[str, str, bool, str]] = [
    (
        "surface",
        "from rich.panel import Panel\np = Panel.fit('boxed')\n",
        True,
        "Panel.fit is a full construction: it spells border and padding itself",
    ),
    (
        "surface",
        "from rich.table import Table\ng = Table.grid(padding=(0, 1))\n",
        False,
        "Table.grid draws nothing, so it holds none of the tokens styles.py owns",
    ),
    (
        "surface",
        "from rich.table import Table as RT\nt = RT(box=None)\n",
        True,
        "an aliased Table is the same construction under another name",
    ),
    (
        "surface",
        "from rich.panel import Panel as P\np = P.fit('boxed')\n",
        True,
        "Panel.fit through an alias is still Panel.fit",
    ),
    (
        "console",
        "from rich.console import Console as C\nc = C(stderr=True)\n",
        True,
        "an aliased Console is still a third console",
    ),
    (
        "console",
        "from rich import console\nc = console.Console()\n",
        True,
        "the dotted spelling resolves to the same trailing name",
    ),
    (
        "bare_print",
        "import rich\nrich.print('x')\n",
        True,
        "rich.print draws on rich's global console, past styles.py",
    ),
    (
        "bare_print",
        "from rich import print\nprint('x')\n",
        True,
        "rich's print rebinds the same bare name the builtin uses",
    ),
    (
        "click_echo",
        "from click import echo as e\ne('x')\n",
        True,
        "an aliased click.echo is still a raw write",
    ),
    (
        "click_echo",
        "import click as c\nc.echo('x')\n",
        False,
        "KNOWN GAP, documented in _is_click_echo: the dotted branch pins the "
        "receiver to the literal name click",
    ),
]


@pytest.mark.parametrize(
    ("rule_name", "source", "should_fire", "why"),
    _ALTERNATE_SPELLINGS,
    ids=[f"{rule}-{why[:40]}" for rule, _src, _fire, why in _ALTERNATE_SPELLINGS],
)
def test_alternate_spellings_are_decided_not_inherited(
    rule_name: str, source: str, should_fire: bool, why: str
) -> None:
    """Each classmethod and alias spelling keeps the disposition it was given.

    Matching on the called name hands out exemptions nobody chose. Every one of
    those spellings is pinned here, including the two that are deliberately NOT
    violations, so a later edit cannot silently flip one.
    """
    problems = _over_budget(_RULES[rule_name], [("cli/newly_added_cmd.py", source)])
    assert bool(problems) is should_fire, why


@pytest.mark.parametrize("rule_name", sorted(_RULES))
def test_a_budgeted_file_still_trips_when_it_grows(rule_name: str) -> None:
    """An allowlisted file gets a budget, not an exemption.

    Adding one more raw call to a file that already has entries is the likeliest
    way this guard would be defeated by accident, so it is the case pinned here:
    the offending file's own budget is reused and exceeded by exactly one.
    """
    rule = _RULES[rule_name]
    if not rule.allowlist:
        pytest.skip(f"{rule_name} allows nothing anywhere, so there is no budget to exceed")
    relpath, (budget, _reason) = next(iter(rule.allowlist.items()))
    violating, _ = _VIOLATION_SAMPLES[rule_name]
    # The sample's single violating statement, repeated one past the budget.
    body = violating.rsplit("\n", 2)[-2]
    grown = violating + "\n".join([body] * budget) + "\n"
    problems = _over_budget(rule, [(relpath, grown)])
    assert len(problems) == 1, f"{relpath} exceeded its budget without tripping {rule_name}"
    assert f"allowed {'exactly' if rule.exact else 'at most'} {budget}" in problems[0]
    assert "why the existing ones are allowed" in problems[0]


@pytest.mark.parametrize("rule_name", sorted(_RULES))
def test_a_zero_tolerance_rule_trips_when_its_inventory_shrinks(rule_name: str) -> None:
    """An exact entry is checked in both directions, so it cannot go stale.

    The ratchet is exempt by design: retiring an echo seam must not red this
    file. For the other three, an entry claiming more than the tree holds is a
    description that has stopped being true, and the message says so in those
    words rather than reading as a failure the reader has to undo.
    """
    rule = _RULES[rule_name]
    if not rule.exact:
        pytest.skip(f"{rule_name} is the ratchet: fewer calls is always fine")
    if not rule.allowlist:
        pytest.skip(f"{rule_name} allows nothing anywhere, so nothing can shrink")
    relpath, (allowed, _reason) = next(iter(rule.allowlist.items()))
    problems = _over_budget(rule, [(relpath, "x = 1\n")])
    assert len(problems) == 1, f"{relpath} lost all {allowed} of its calls without tripping"
    assert "This is good news" in problems[0]
    assert "lower the number in the allowlist to 0" in problems[0]
