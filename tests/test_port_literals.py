"""Release gate: no retired framework port literal creeps back into the tree.

Every framework host port now derives from ``deployment.port_base`` (default
10000) through :mod:`osprey.port_layout`.  The numbers that scheme replaced --
the 80xx/90xx block that used to be scattered across defaults, schemas,
templates, presets and prose -- are *retired*: none of them may appear again as
a literal in shipped source, templates or documentation.

This is the shipped form of the throwaway scan that produced
``.claude/plans/port-block-layout/research/inventory.md`` section F.  Two
findings from that scan shape the rule:

* **One pass, no port-context predicate.**  The throwaway scan required a
  port-context token (``port``, ``localhost``, ``listen``, ...) on the line.
  Section F.4 re-ran it with the predicate inverted and found that every
  remaining context-free hit was real -- a docstring, a preset comment, a
  commented example.  A context predicate would therefore only have hidden
  work, so the shipped lint drops it and runs a single guarded pass.

* **The digit-boundary guard replaces the predicate's precision.**  The number
  pattern is ``(?<![\\d.#])(\\d{4,5})(?![\\d.])``, so a retired run sitting
  inside a longer number is not a hit: a magnet-strength tuple
  (``-57.9465``), a version, a timestamp.  See :func:`_retired_on_line`.

The single suppression idiom is the ``# osprey:not-a-port`` marker.  A line
carrying that text anywhere is skipped, which lets ``.j2`` / ``.rst`` / ``.html``
sources use their own comment syntax.  Use it only when the number *is* the
point -- history a comment is recording, or a stdlib-only contract that cannot
import :mod:`osprey.port_layout`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Shipped sources only.  ``tests/`` is deliberately out of scope: test files
# pin old behaviour on purpose (overrides, fixtures, this file's own retired-set
# definition), and a lint that flagged them would have to allowlist itself.
SCAN_ROOTS = ("src/osprey", "packages", "docs/source")

# Text formats that carry port literals.  Two extensions are deliberately
# absent:
#   * ``.json`` -- ``src/osprey/interfaces/vendor_manifest.json`` records
#     SHA-256 digests, and a hex digest contains 4-5 digit runs that land in the
#     retired ranges by chance (``...fa9071cf...``).  No OSPREY port default is
#     spelled in JSON, so excluding the format costs nothing.
#   * ``.js`` -- the only JavaScript carrying such runs is vendored/minified
#     bundle payload, which the path exclusions below already cover; OSPREY's
#     own scripts read ports from the DOM or an API, never from a literal.
SCAN_SUFFIXES = frozenset({".py", ".j2", ".yml", ".yaml", ".sh", ".conf", ".rst", ".md", ".html"})

# Path exclusions.  Third-party bundles are not ours to reword, and a minified
# payload is one enormous line in which any 4-5 digit run is coincidence.
# ``_version.py`` is written by setuptools-scm and carries the abbreviated git
# hash of the checkout, which can spell any 4-5 digit run by chance.
EXCLUDED_PATH_PARTS = ("static/vendor/",)
EXCLUDED_NAME_GLOBS = ("*.min.*", "_version.py")

MARKER = "osprey:not-a-port"

# A 4-5 digit run that is not part of a longer number.  The lookbehind rejects a
# run preceded by a digit, a decimal point, or ``#``; the lookahead rejects one
# followed by a digit or a decimal point.
_NUMBER_RE = re.compile(r"(?<![\d.#])(\d{4,5})(?![\d.])")

# CSS colours.  The ``#`` lookbehind above only skips a run that *immediately*
# follows the hash, so ``#0d8090`` would still expose ``8090``.  Matches falling
# inside a colour-shaped token are dropped instead.  (No such collision exists
# in the tree today -- this guards the 8085-8097 / 9070-9100 ranges, where a
# plausible colour literal could land.)
_HEX_COLOUR_RE = re.compile(r"#(?:[0-9a-fA-F]{8}|[0-9a-fA-F]{6}|[0-9a-fA-F]{3,4})\b")


def _retired_ports() -> frozenset[int]:
    """The numbers the port-block layout replaced.

    Ranges come from ``inventory.md`` section F: the contiguous 80xx and 90xx
    bands that held framework defaults, plus the per-family 9x00 anchors of the
    old "one hundred per family" scheme and the handful of loose literals
    (bluesky lane ports, the ariel sidecar, the VA stand-in port).
    """
    numbers: set[int] = set()
    numbers.update(range(8085, 8098))  # web/artifacts/ariel/panels/lattice band
    numbers.update(range(9070, 9101))  # auth, nginx, dispatcher-adjacent band
    numbers.update(n for n in range(9200, 9701) if n % 100 == 0)  # family anchors
    numbers.update(
        {
            9800,  # qmd sidecar
            9900,  # event dispatcher
            9901,  # dispatch worker index 1
            8090,  # bluesky bridge
            8091,  # bluesky tiled
            8095,  # bluesky web sidecar
            8190,  # second-lane bluesky bridge
            5074,  # VA stand-in (5064 is the CA protocol port and is NOT retired)
        }
    )
    return frozenset(numbers)


RETIRED_PORTS = _retired_ports()


# Disposition codes.  Every hit is one of these; the failure message prints the
# list so whoever trips the lint can classify a new hit without reading this
# file.  Only the last two are suppressions -- the first four describe numbers
# that were never retired in the first place, and exist here because a reader
# staring at a 4-digit literal needs the vocabulary to tell them apart.
DISPOSITION_CODES: tuple[tuple[str, str], ...] = (
    (
        "container-internal",
        "a container-side port on that line -- fixed by the image, not by the "
        "host layout; keep the literal and comment it",
    ),
    (
        "protocol",
        "a protocol default (5064/5065/5075/5076 Channel Access, 443, a bare-URI "
        "DSN default); it is the wire contract, not an OSPREY slot",
    ),
    (
        "third-party",
        "a port owned by something OSPREY only dials -- a model server, an "
        "external facility store, a JVM bridge",
    ),
    (
        "named-exception",
        "a stdlib-only module that cannot import osprey.port_layout and must "
        "still carry a default; marker sits on the module-constant line",
    ),
    (
        f"marked ({MARKER})",
        "prose, a docstring or a comment where the retired number IS the point "
        "-- recorded history or a worked example; the line carries the marker",
    ),
    (
        "layout lookup",
        "not a suppression: the honest disposition for almost every hit -- "
        "replace the literal with osprey.port_layout.default_port(...) at the "
        "base the caller can resolve",
    ),
)


@dataclass(frozen=True)
class Allowed:
    """One allowlist entry: a retired number that may stand at a known path."""

    path_glob: str
    number: int
    code: str
    reason: str


# Deliberately empty.  Task 7.8's sweep and Task 7.10's docs sweep resolved every
# real hit either by deriving the number from the layout or by marking the one
# line where the number is the explanation
# (``src/osprey/registry/mcp.py`` -- the removed BLUESKY_BRIDGE_URL default).
# The marker handles that case, so no path/number pair needs standing
# permission.  Add an entry here only when a number genuinely cannot carry a
# marker (a strict format, a generated file); keep it to one file and one
# number with a one-line reason -- never a broad directory.
ALLOWLIST: tuple[Allowed, ...] = ()


def _is_allowed(rel_path: str, number: int, allowlist: tuple[Allowed, ...] = ALLOWLIST) -> bool:
    return any(entry.number == number and fnmatch(rel_path, entry.path_glob) for entry in allowlist)


def _retired_on_line(line: str) -> list[int]:
    """Retired numbers on one line, after the guard, the marker and colours."""
    if MARKER in line:
        return []
    colour_spans = [m.span() for m in _HEX_COLOUR_RE.finditer(line)]
    found: list[int] = []
    for match in _NUMBER_RE.finditer(line):
        if any(start <= match.start() and match.end() <= end for start, end in colour_spans):
            continue
        number = int(match.group(1))
        if number in RETIRED_PORTS:
            found.append(number)
    return found


def _is_excluded(rel_path: str) -> bool:
    if any(part in rel_path for part in EXCLUDED_PATH_PARTS):
        return True
    name = rel_path.rsplit("/", 1)[-1]
    return any(fnmatch(name, glob) for glob in EXCLUDED_NAME_GLOBS)


def _scan_files() -> list[Path]:
    files: list[Path] = []
    for root in SCAN_ROOTS:
        base = _REPO_ROOT / root
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.suffix not in SCAN_SUFFIXES:
                continue
            if "__pycache__" in path.parts or path.name.endswith(".bak"):
                continue
            rel_path = path.relative_to(_REPO_ROOT).as_posix()
            if _is_excluded(rel_path):
                continue
            files.append(path)
    return files


@dataclass(frozen=True)
class Hit:
    rel_path: str
    lineno: int
    number: int
    text: str

    def __str__(self) -> str:
        return f"{self.rel_path}:{self.lineno}: {self.number} — {self.text}"


def _scan_repo() -> list[Hit]:
    hits: list[Hit] = []
    for path in _scan_files():
        rel_path = path.relative_to(_REPO_ROOT).as_posix()
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for lineno, line in enumerate(content.splitlines(), start=1):
            for number in _retired_on_line(line):
                if _is_allowed(rel_path, number):
                    continue
                hits.append(Hit(rel_path, lineno, number, line.strip()))
    return hits


def _failure_message(hits: list[Hit]) -> str:
    codes = "\n".join(f"  - {code}: {why}" for code, why in DISPOSITION_CODES)
    return (
        f"{len(hits)} retired port literal(s) found. Every framework host port "
        f"derives from deployment.port_base via osprey.port_layout; these "
        f"numbers were replaced and must not come back:\n"
        + "\n".join(f"  {hit}" for hit in hits)
        + "\n\nClassify each hit as one of:\n"
        + codes
        + f"\n\nIf the number really must stand, put `{MARKER}` on that line "
        f"with a reason (that is the only suppression idiom), or -- when the "
        f"line cannot carry a comment -- add one Allowed(...) entry to "
        f"ALLOWLIST in {Path(__file__).name}."
    )


def test_no_retired_port_literals() -> None:
    """No retired framework port number survives in src/, packages/ or docs/."""
    files = _scan_files()
    assert files, f"scan matched no files under {SCAN_ROOTS} — globs are wrong"
    hits = _scan_repo()
    assert not hits, _failure_message(hits)


# --- self-tests: pin the scanner's own logic, not the tree ------------------


def test_guard_drops_a_retired_run_inside_a_longer_number() -> None:
    """A magnet strength is not a port: the decimal tail defeats the guard.

    Shaped after the real line the throwaway scan tripped on --
    ``src/osprey/simulation/lattice/ring.py`` ``_SD = (0.20300000, -57.9465)``
    -- but with a genuinely retired run in the tail, so the assertion tests the
    guard rather than the retired set.
    """
    assert _retired_on_line("_SD = (0.20300000, -57.9800)") == []
    assert _retired_on_line("version = 19900") == []
    assert _retired_on_line("value = 9900.5") == []


def test_guard_drops_a_retired_run_inside_a_hex_colour() -> None:
    """``#0d8090`` exposes 8090 to the number regex; the colour rule drops it."""
    assert _retired_on_line("  --teal: #0d8090;") == []
    # ...and the immediate-hash form is handled by the lookbehind alone.
    assert _retired_on_line("  --brand: #8090ab;") == []


def test_marker_suppresses_the_line_it_sits_on() -> None:
    line = "DISPATCH = 'http://127.0.0.1:9900'"
    assert _retired_on_line(line) == [9900]
    assert _retired_on_line(f"{line}  # {MARKER} — recorded history") == []


def test_a_bare_retired_number_is_a_hit() -> None:
    """No port-context token required — a bare number in prose still hits."""
    assert _retired_on_line("the old scheme put the gallery on 8086") == [8086]
    assert _retired_on_line("port: 9800") == [9800]
    assert sorted(_retired_on_line("8091 and 9901")) == [8091, 9901]
    # A number outside the retired set is not a hit.
    assert _retired_on_line("port: 10100") == []
    assert _retired_on_line("ca_port: 5064") == []


def test_allowlist_entry_suppresses_one_path_and_number() -> None:
    entries = (Allowed("src/osprey/thing.py", 9800, "third-party", "someone else's port"),)
    assert _is_allowed("src/osprey/thing.py", 9800, entries)
    assert not _is_allowed("src/osprey/thing.py", 9900, entries)
    assert not _is_allowed("src/osprey/other.py", 9800, entries)


def test_excluded_paths_are_skipped() -> None:
    assert _is_excluded("src/osprey/interfaces/static/vendor/plotly.js")
    assert _is_excluded("src/osprey/interfaces/static/app.min.js")
    assert not _is_excluded("src/osprey/registry/web.py")
