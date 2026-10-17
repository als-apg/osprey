"""The injector's beam order, read from the MML lattice files.

MML describes the injector in two AT lattice scripts of its machine folder:
``GTB/gtblattice.m`` (gun, linac and the booster transfer line) and
``BTS/btslattice.m`` (booster to storage ring). This module reads them without
MATLAB and returns, per demo section, every element of that section's lattice
arrays in beam order:

``GTL``
    ``gtblattice.m`` GUN.
``LN``
    ``gtblattice.m`` LINAC.
``LTB``
    ``gtblattice.m`` LTB1 followed by LTB2.
``BTS``
    ``btslattice.m`` BTS.

Each element is one occurrence of an array token: the MML variable name (not
the element's ``FamName``) and its occurrence number ``n`` within the section,
so ``GTL:TV#1`` is the first ``TV`` in GUN and ``LTB:BPM#2`` the second ``BPM``
in LTB1 + LTB2. That ``<SEC>:<token>#<n>`` key is the element table's
``deck_element``. Each element carries the AT constructor that defines it, its
family name, length, normalized quadrupole strength ``K`` (1/m^2, as MML hands
it to AT), bend angle and edge angles (rad), and the source line of its token.

The reader executes the files' assignments in order, the way MATLAB does: a
name defined twice (the files redefine ``HCM``/``VCM`` per section with a
different length) means its latest definition at the point an array is built.
It understands numeric expressions over earlier scalars, ``pi`` and
``getbrho``; the AT constructors ``marker``, ``drift``, ``corrector``,
``quadrupole``, ``rbend``, ``sbend`` and ``aperture``; and arrays of element and
array names. Any other statement (control flow, ``buildlat``, the magnet
measurement tables) is skipped, and a name it leaves unbound is an error
wherever it is used. An array token with no definition, an unknown section and
a missing array are errors too, raised as ``ValueError``.

``Section.energy`` is the file's default beam energy in eV, the energy its
normalized strengths were computed at. The demo runs at its own energies, so
a consumer keeps ``K`` and the angles (the optics) and scales fields by the
energy ratio.
"""

from __future__ import annotations

import ast
import math
import os
import re
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

#: The environment variable naming the machine folder of an MML checkout, the one
#: holding ``GTB/`` and ``BTS/``.
MML_ROOT_ENV = "OSPREY_MML_MACHINE_ROOT"
#: That machine folder, when the environment names one.
MML_ROOT: Path | None = Path(os.environ[MML_ROOT_ENV]) if os.environ.get(MML_ROOT_ENV) else None

#: Each section: its lattice file (relative to ``MML_ROOT``) and its arrays in order.
LATTICES: dict[str, tuple[str, tuple[str, ...]]] = {
    "GTL": ("GTB/gtblattice.m", ("GUN",)),
    "LN": ("GTB/gtblattice.m", ("LINAC",)),
    "LTB": ("GTB/gtblattice.m", ("LTB1", "LTB2")),
    "BTS": ("BTS/btslattice.m", ("BTS",)),
}

#: The counted element kinds, by AT constructor; a ``BPM`` marker is a ``monitor``.
KINDS = {
    "corrector": "corrector",
    "quadrupole": "quad",
    "rbend": "bend",
    "sbend": "bend",
}

#: Electron rest energy in GeV, as MML's ``getbrho`` takes it.
ELECTRON_REST_GEV = 0.51099906e-3


def getbrho(energy_gev: float) -> float:
    """Magnetic rigidity in T*m of an electron of kinetic energy ``energy_gev``."""
    e0 = ELECTRON_REST_GEV
    return (10 / 2.99792458) * math.sqrt((energy_gev + e0) ** 2 - e0**2)


#: The functions an expression may call.
_FUNCTIONS: dict[str, Callable[..., float]] = {"getbrho": getbrho}

#: The AT constructors and the position of each argument they carry.
_CONSTRUCTORS: dict[str, dict[str, int]] = {
    "marker": {},
    "aperture": {},
    "drift": {"length": 1},
    "corrector": {"length": 1},
    "quadrupole": {"length": 1, "k": 2},
    "rbend": {"length": 1, "angle": 2, "entrance_angle": 3, "exit_angle": 4, "k": 5},
    "sbend": {"length": 1, "angle": 2, "entrance_angle": 3, "exit_angle": 4, "k": 5},
}

_ASSIGNMENT = re.compile(r"^\s*([A-Za-z_]\w*)\s*=(?!=)\s*(.*?)\s*$", re.DOTALL)
_CALL = re.compile(r"^([A-Za-z_]\w*)\s*\((.*)\)$", re.DOTALL)
_NAME = re.compile(r"[A-Za-z_]\w*")
_SEPARATORS = re.compile(r"[\s,;]+")


@dataclass(frozen=True)
class Definition:
    """One AT element as a constructor call defines it."""

    type: str
    family: str
    length: float = 0.0
    k: float = 0.0
    angle: float = 0.0
    entrance_angle: float = 0.0
    exit_angle: float = 0.0

    @property
    def kind(self) -> str | None:
        """``corrector``, ``quad``, ``bend``, ``monitor`` (a BPM marker) or ``None``."""
        if self.type == "marker":
            return "monitor" if self.family == "BPM" else None
        return KINDS.get(self.type)


@dataclass(frozen=True)
class Element:
    """One occurrence of a token in a section's lattice arrays."""

    section: str
    name: str
    occurrence: int
    definition: Definition
    line: int

    @property
    def deck_element(self) -> str:
        """``<SEC>:<token>#<n>``, the element table's key for this occurrence."""
        return f"{self.section}:{self.name}#{self.occurrence}"

    @property
    def type(self) -> str:
        return self.definition.type

    @property
    def family(self) -> str:
        return self.definition.family

    @property
    def length(self) -> float:
        return self.definition.length

    @property
    def k(self) -> float:
        return self.definition.k

    @property
    def angle(self) -> float:
        return self.definition.angle

    @property
    def kind(self) -> str | None:
        return self.definition.kind


@dataclass(frozen=True)
class Section:
    """A section's elements in beam order, and the energy its strengths are for."""

    name: str
    elements: tuple[Element, ...]
    energy: float


@dataclass(frozen=True)
class _Token:
    name: str
    definition: Definition
    line: int


@dataclass
class _Program:
    """What executing a lattice file binds: scalars, elements and arrays."""

    scalars: dict[str, float]
    elements: dict[str, Definition]
    arrays: dict[str, list[_Token]]


# --- statements ------------------------------------------------------------------


def _statements(text: str) -> list[list[tuple[int, str]]]:
    """The file's statements, each a list of ``(line number, code)`` segments.

    Comments are dropped, ``...`` continues a line, an open bracket or
    parenthesis continues the statement, and ``;`` at depth zero ends one.
    """
    statements: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    depth = 0
    for number, raw in enumerate(text.splitlines(), start=1):
        code = raw.split("%", 1)[0]
        continued = code.rstrip().endswith("...")
        if continued:
            code = code.rstrip()[: -len("...")]
        segment = ""
        for char in code:
            if char in "[(":
                depth += 1
            elif char in "])":
                depth = max(depth - 1, 0)
            if char == ";" and depth == 0:
                current.append((number, segment))
                statements.append(current)
                current, segment = [], ""
                continue
            segment += char
        current.append((number, segment))
        if not continued and depth == 0:
            statements.append(current)
            current = []
    if current:
        statements.append(current)
    return [statement for statement in statements if "".join(c for _, c in statement).strip()]


def _split_arguments(text: str) -> list[str]:
    arguments, depth, current = [], 0, ""
    for char in text:
        if char in "[(":
            depth += 1
        elif char in "])":
            depth -= 1
        if char == "," and depth == 0:
            arguments.append(current.strip())
            current = ""
        else:
            current += char
    arguments.append(current.strip())
    return arguments


# --- expressions -----------------------------------------------------------------


class _Unevaluable(Exception):
    """An expression outside the subset this reader evaluates."""


def _evaluate(expression: str, scalars: Mapping[str, float]) -> float:
    try:
        tree = ast.parse(expression.strip().replace("^", "**"), mode="eval")
    except SyntaxError as error:
        raise _Unevaluable(expression) from error
    return _value(tree.body, scalars)


def _value(node: ast.expr, scalars: Mapping[str, float]) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id == "pi":
            return math.pi
        if node.id in scalars:
            return scalars[node.id]
        raise _Unevaluable(node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
        operand = _value(node.operand, scalars)
        return -operand if isinstance(node.op, ast.USub) else operand
    if isinstance(node, ast.BinOp):
        left, right = _value(node.left, scalars), _value(node.right, scalars)
        operations: dict[type, Callable[[float, float], float]] = {
            ast.Add: lambda a, b: a + b,
            ast.Sub: lambda a, b: a - b,
            ast.Mult: lambda a, b: a * b,
            ast.Div: lambda a, b: a / b,
            ast.Pow: lambda a, b: a**b,
        }
        if type(node.op) in operations:
            return operations[type(node.op)](left, right)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCTIONS
        and not node.keywords
    ):
        return _FUNCTIONS[node.func.id](*(_value(arg, scalars) for arg in node.args))
    raise _Unevaluable(ast.dump(node))


# --- execution -------------------------------------------------------------------


def _definition(
    constructor: str, arguments: list[str], scalars: Mapping[str, float], line: int
) -> Definition:
    positions = _CONSTRUCTORS[constructor]
    family = arguments[0].strip()
    if not (len(family) >= 2 and family[0] == family[-1] == "'"):
        raise ValueError(f"line {line}: {constructor} needs a quoted family name")
    values: dict[str, float] = {}
    for field, position in positions.items():
        if position >= len(arguments):
            raise ValueError(f"line {line}: {constructor} has no argument {position + 1}")
        try:
            values[field] = _evaluate(arguments[position], scalars)
        except _Unevaluable as error:
            raise ValueError(
                f"line {line}: cannot evaluate {arguments[position]!r} ({error})"
            ) from error
    return Definition(type=constructor, family=family[1:-1], **values)


def _array(segments: list[tuple[int, str]], program: _Program, line: int) -> list[_Token] | None:
    """An array of element and array names, or ``None`` for any other bracket."""
    tokens: list[_Token] = []
    for number, code in segments:
        if _SEPARATORS.sub("", _NAME.sub("", code)):
            return None
        for name in _NAME.findall(code):
            if name in program.arrays:
                tokens.extend(program.arrays[name])
            elif name in program.elements:
                tokens.append(_Token(name, program.elements[name], number))
            else:
                raise ValueError(f"line {number}: array token {name} is not defined")
    if not tokens:
        raise ValueError(f"line {line}: empty array")
    return tokens


def _bracket_body(segments: list[tuple[int, str]]) -> list[tuple[int, str]] | None:
    """The segments between an assignment's ``[`` and its closing ``]``."""
    stripped = "".join(code for _, code in segments).strip()
    if stripped.endswith("'"):
        stripped = stripped[:-1].rstrip()
    if not (stripped.startswith("[") and stripped.endswith("]")):
        return None
    body: list[tuple[int, str]] = []
    opened = closed = False
    for number, code in segments:
        if not opened:
            if "[" not in code:
                continue
            code = code.split("[", 1)[1]
            opened = True
        if "]" in code:
            code = code.rsplit("]", 1)[0]
            closed = True
        body.append((number, code))
        if closed:
            break
    return body


def execute(text: str) -> _Program:
    """Run a lattice file's assignments in order and return what they bind."""
    program = _Program(scalars={}, elements={}, arrays={})
    for statement in _statements(text):
        line = statement[0][0]
        joined = "".join(code for _, code in statement).strip()
        assignment = _ASSIGNMENT.match(joined)
        if assignment is None:
            continue
        name, rhs = assignment.groups()
        first_offset = statement[0][1].index("=") + 1 if "=" in statement[0][1] else 0
        rhs_segments = [(statement[0][0], statement[0][1][first_offset:]), *statement[1:]]
        call = _CALL.match(rhs)
        body = _bracket_body(rhs_segments)
        if call and call.group(1) in _CONSTRUCTORS:
            constructor, arguments = call.group(1), _split_arguments(call.group(2))
            program.elements[name] = _definition(constructor, arguments, program.scalars, line)
            program.arrays.pop(name, None)
            program.scalars.pop(name, None)
        elif body is not None:
            tokens = _array(body, program, line)
            if tokens is not None:
                program.arrays[name] = tokens
                program.elements.pop(name, None)
            program.scalars.pop(name, None)
        else:
            try:
                program.scalars[name] = _evaluate(rhs, program.scalars)
            except (_Unevaluable, ZeroDivisionError, TypeError, ValueError):
                program.scalars.pop(name, None)
                continue
            program.elements.pop(name, None)
            program.arrays.pop(name, None)
    return program


def parse(text: str, section: str) -> Section:
    """``section``'s elements in beam order, read from its lattice file's ``text``."""
    if section not in LATTICES:
        raise ValueError(f"unknown section {section!r}; expected one of {sorted(LATTICES)}")
    arrays = LATTICES[section][1]
    program = execute(text)
    missing = [name for name in arrays if name not in program.arrays]
    if missing:
        raise ValueError(f"{section}: lattice arrays not found: {missing}")
    if "Energy" not in program.scalars:
        raise ValueError(f"{section}: the lattice file sets no default Energy")
    seen: dict[str, int] = defaultdict(int)
    elements = []
    for array in arrays:
        for token in program.arrays[array]:
            seen[token.name] += 1
            elements.append(
                Element(section, token.name, seen[token.name], token.definition, token.line)
            )
    return Section(section, tuple(elements), program.scalars["Energy"])


def load(section: str, root: Path | None = MML_ROOT) -> Section:
    """``section`` read from the MML checkout at ``root``."""
    if section not in LATTICES:
        raise ValueError(f"unknown section {section!r}; expected one of {sorted(LATTICES)}")
    if root is None:
        raise ValueError(f"no MML checkout: set {MML_ROOT_ENV} to its machine folder")
    path = root / LATTICES[section][0]
    return parse(path.read_text(encoding="latin-1"), section)


def load_all(root: Path | None = MML_ROOT) -> dict[str, Section]:
    """Every section, in beam order, read from the MML checkout at ``root``."""
    return {section: load(section, root) for section in LATTICES}
