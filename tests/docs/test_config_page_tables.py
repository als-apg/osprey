"""The configuration reference's hand-kept tables state what the code does.

``docs/source/reference/configuration/config.rst`` keeps three kinds of fact
by hand that the code also holds, and nothing else compares the two:

* **The probe table** under "Probe checks". Its ``type`` cells are the probe
  registry, :data:`osprey.health.probes.PROBE_TYPES`, in both directions, and
  "Six probe types ship" is its size. A row's parameters are the inline
  literals in its cell that name a key (status words, example values and other
  sections excluded), and each one is *read* by that type's module under
  ``src/osprey/health/probes/``: the probe's ``spec`` mapping is subscripted
  with it, passed it as the first argument of ``spec.get``, or tested with
  ``in spec``. The reverse holds too: every key a module reads is in its row,
  apart from the keys the runner hands every probe. A "(default ``x``)" beside
  a parameter is the default its ``spec.get`` falls back to.
* **Stated defaults.** Every ``Key / Default`` table row, and every default a
  sentence or a ``What it does`` cell states outside a code block ("Default
  ``7979``", "(default 500)", "``read`` (default)", "unset by default" and
  the like), equals the ``default:`` column of
  ``src/osprey/profiles/config_key_manifest.yml`` for its key. A sentence's key
  is the nearest inline literal before it that the manifest declares, else the
  nearest key-like literal, else its section (see :data:`_SECTION_KEYS`).
  ``""`` and the manifest's ``'""'`` are one value; a stated "unset" matches a
  manifest ``null``, ``""`` or ``no-fallback``; a manifest ``derived`` default
  is skipped, since the page states what the derivation gives. A stated
  default whose key is not a manifest key with a ``default:`` is listed in
  :data:`_NOT_MANIFEST_DEFAULTS` with its reason, not guessed.
* **The image axes.** The page's ``images:`` example and its environment
  example both render every OSPREY-built image as
  ``registry.example.org/accelerator/<project><suffix>:2026.08.1`` through
  :func:`~osprey.deployment.compose_generator.resolve_image_defaults`; with no
  axis set the tag is the table's ``local`` and there is no prefix; and a
  registry's trailing slash is added once and never doubled. The sentence
  that points back at the service-image table spells no row count, for the
  reason ``test_service_image_table`` gives for image counts.

Each check is a function of the page text, so the negative controls feed it a
page with one fact planted wrong.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from functools import cache
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.deployment.compose_generator import (
    _OSPREY_IMAGE_SUFFIXES,
    DEFAULT_IMAGE_TAG,
    resolve_image_axes,
    resolve_image_defaults,
)
from osprey.health.config import _RESERVED_CHECK_KEYS
from osprey.health.models import Status
from osprey.health.probes import PROBE_TYPES
from tests.docs.test_service_image_table import _table_rows

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PAGE = _REPO_ROOT / "docs" / "source" / "reference" / "configuration" / "config.rst"
_PROBES = _REPO_ROOT / "src" / "osprey" / "health" / "probes"
_MANIFEST = _REPO_ROOT / "src" / "osprey" / "profiles" / "config_key_manifest.yml"

_NUMBER_WORDS = {
    word: n
    for n, word in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen "
        "fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()
    )
}

#: Keys the runner puts in every probe's ``spec`` (``osprey.health.runner``):
#: the reserved check keys, and the category the check sits in.
_RUNNER_KEYS = _RESERVED_CHECK_KEYS | {"category"}

#: Literals in a probe row that are not parameter names, with the reason.
_NOT_PARAMS = {
    "cborg": "an example provider name",
    "archiver_freshness": "the archiver_freshness row naming its own type",
}

#: Sections whose prose states a default without naming its key.
_SECTION_KEYS = {
    "The onboarding tour invite": "web.tour",
    "The control-target picker": "web.control_target_picker",
    "How often the file watchers re-read the tree": "web.file_watch_reconcile_interval_s",
}

#: Stated defaults whose subject is not a manifest key with a ``default:``.
_NOT_MANIFEST_DEFAULTS = {
    "expect_status": "an http probe parameter; held by the probe-table test",
    "max_age_s": "an archiver_freshness probe parameter; held by the probe-table test",
    "timeout_status": "a key of one health check, not of config.yml",
    "Timeouts": "the per-probe timeout table's column header",
    "suite_timeout_s": "health.suite_timeout_s has no manifest entry",
}

_LITERAL = re.compile(r"``([^`]+)``")
_STATED_DEFAULT = re.compile(
    "|".join(
        [
            r"\(default (?:``)?(?P<paren>[^`)]+?)(?:``)?\)",
            r"\b[Dd]efault (?:is )?``(?P<lead>[^`]+)``",
            r"\bdefaults to ``(?P<to>[^`]+)``",
            r"``(?P<trail>[^`]+)``(?: \(default\)|, the default| - The default| \(the default\))",
            r"\bAbsent means ``(?P<absent>[^`]+)``",
            r"(?P<unset>\bunset by default)",
        ]
    )
)
_KEY_LIKE = re.compile(r"[a-z_][a-z0-9_-]*(?:\.[a-z0-9_<>-]+)*")
_UNSET = object()
_PROSE = object()


@cache
def _page() -> str:
    return _PAGE.read_text(encoding="utf-8")


@cache
def _manifest() -> dict[str, Any]:
    return yaml.safe_load(_MANIFEST.read_text(encoding="utf-8"))["keys"]


def _list_table(text: str, after: str) -> list[list[str]]:
    """The rows of the first ``list-table`` after *after*, each cell on one line."""
    return _list_table_at(text, text.index(".. list-table::", text.index(after)))


def _list_table_at(text: str, start: int) -> list[list[str]]:
    """The rows of the ``list-table`` whose directive starts at *start*."""
    rows: list[list[str]] = []
    for line in text[start:].splitlines()[1:]:
        if line and not line.startswith(" "):
            break
        if line.startswith("   * - "):
            rows.append([line[7:].strip()])
        elif line.startswith("     - ") and rows:
            rows[-1].append(line[7:].strip())
        elif line.strip() and rows:
            rows[-1][-1] += " " + line.strip()
    return rows


def _prose(text: str) -> str:
    """*text* with every code block's and literal block's body blanked."""
    out: list[str] = []
    skip_from: int | None = None
    for line in text.splitlines():
        indent = len(line) - len(line.lstrip())
        if skip_from is not None:
            if not line.strip() or indent > skip_from:
                out.append("")
                continue
            skip_from = None
        out.append(line)
        stripped = line.strip()
        if stripped.startswith(".. code-block::") or (
            stripped.endswith("::") and not stripped.startswith("..")
        ):
            skip_from = indent
    return "\n".join(out)


# --- the probe table -------------------------------------------------------


def _probe_rows(text: str) -> dict[str, str]:
    """``type`` -> purpose cell, from the probe table."""
    rows = _list_table(text, "probe types ship")[1:]
    return {_LITERAL.fullmatch(row[0])[1]: row[1] for row in rows}  # type: ignore[index]


def _row_params(cell: str) -> set[str]:
    """The parameter names a probe row's cell spells."""
    statuses = {status.value for status in Status}
    names = set()
    for literal in _LITERAL.findall(cell):
        name = literal.split(": ", 1)[0]
        if re.fullmatch(r"[a-z_]+", name) and name not in statuses:
            names.add(name)
    return names - set(_NOT_PARAMS)


def _literal(node: ast.expr) -> Any:
    """*node*'s value when it is a literal, else ``None`` (a named constant, a call)."""
    try:
        return ast.literal_eval(node)
    except ValueError:
        return None


@cache
def _probe_reads(probe: str) -> dict[str, Any]:
    """Every key the probe module reads from ``spec``, with its ``get`` default."""
    tree = ast.parse((_PROBES / f"{probe}.py").read_text(encoding="utf-8"))
    reads: dict[str, Any] = {}

    def is_spec(node: ast.AST) -> bool:
        return isinstance(node, ast.Name) and node.id == "spec"

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and is_spec(node.func.value)
            and isinstance(node.args[0], ast.Constant)
        ):
            reads[node.args[0].value] = _literal(node.args[1]) if len(node.args) > 1 else None
        elif (
            isinstance(node, ast.Subscript)
            and is_spec(node.value)
            and isinstance(node.slice, ast.Constant)
        ):
            reads.setdefault(node.slice.value, None)
        elif (
            isinstance(node, ast.Compare)
            and isinstance(node.ops[0], ast.In)
            and is_spec(node.comparators[0])
            and isinstance(node.left, ast.Constant)
        ):
            reads.setdefault(node.left.value, None)
    return reads


def _probe_row_problems(text: str, probe: str) -> list[str]:
    """What the probe table says about *probe* that its module does not do."""
    cell = _probe_rows(text)[probe]
    documented = _row_params(cell)
    reads = _probe_reads(probe)
    read = set(reads) - _RUNNER_KEYS
    problems = [f"{probe}: ``{name}`` is not read by the probe" for name in documented - read]
    problems += [f"{probe}: reads ``{name}``, the row omits it" for name in read - documented]
    for name, stated in re.findall(r"``(\w+)`` \(default (?:``)?([^`)]+?)(?:``)?\)", cell):
        if name in reads and yaml.safe_load(stated) != reads[name]:
            problems.append(f"{probe}: {name} defaults to {reads[name]!r}, the row says {stated}")
    return problems


def test_the_probe_table_rows_are_the_probe_registry() -> None:
    text = _page()
    assert set(_probe_rows(text)) == PROBE_TYPES
    spelled = re.search(r"(\w+) probe types ship", text)
    assert spelled is not None
    assert _NUMBER_WORDS[spelled[1].lower()] == len(PROBE_TYPES)


@pytest.mark.parametrize("probe", sorted(PROBE_TYPES))
def test_each_probe_row_names_what_its_probe_reads(probe: str) -> None:
    assert _probe_row_problems(_page(), probe) == []


def test_a_parameter_the_probe_does_not_read_is_caught() -> None:
    planted = _page().replace("``expect_status``", "``expect_code``", 1)
    assert _probe_row_problems(planted, "http") == [
        "http: ``expect_code`` is not read by the probe",
        "http: reads ``expect_status``, the row omits it",
    ]


def test_a_wrong_probe_default_is_caught() -> None:
    planted = _page().replace("``max_age_s`` (default 600)", "``max_age_s`` (default 60)", 1)
    assert _probe_row_problems(planted, "archiver_freshness") == [
        "archiver_freshness: max_age_s defaults to 600.0, the row says 60"
    ]


def test_every_not_param_entry_is_on_the_page() -> None:
    cells = " ".join(_probe_rows(_page()).values())
    assert [name for name in _NOT_PARAMS if f"``{name}``" not in cells] == []


# --- stated defaults -------------------------------------------------------


def _table_defaults(text: str) -> Iterator[tuple[str, str]]:
    """``(key, default cell)`` for every row of every ``Key / Default`` table."""
    for match in re.finditer(r"\.\. list-table::", text):
        rows = _list_table_at(text, match.start())
        if rows and rows[0][:2] == ["Key", "Default"]:
            for row in rows[1:]:
                yield _LITERAL.fullmatch(row[0])[1], row[1]  # type: ignore[index]


def _subject(before: str, heading: str) -> str:
    """The key a stated default is about: see the module docstring."""
    names = [literal.split(": ", 1)[0] for literal in _LITERAL.findall(before)]
    declared = [name for name in names if name in _manifest()]
    if declared:
        return declared[-1]
    keys = [name for name in names if _KEY_LIKE.fullmatch(name)]
    if keys:
        return keys[-1]
    return _SECTION_KEYS.get(heading, heading)


def _prose_defaults(text: str) -> Iterator[tuple[str, str]]:
    """``(subject, stated value)`` for every default stated outside a code block."""
    heading = ""
    for paragraph in re.split(r"\n\s*\n", _prose(text)):
        lines = paragraph.strip().splitlines()
        if len(lines) == 2 and lines[1] and set(lines[1]) <= set("=-~^"):
            heading = lines[0].strip()
            continue
        flat = " ".join(paragraph.split())
        for match in _STATED_DEFAULT.finditer(flat):
            groups = {name: value for name, value in match.groupdict().items() if value}
            ((kind, value),) = groups.items()
            stated = "unset" if kind == "unset" else f"``{value}``"
            yield _subject(flat[: match.start()], heading), stated


def _stated(cell: str) -> object:
    """A stated default as a value: parsed literal, :data:`_UNSET` or :data:`_PROSE`."""
    if cell.startswith("unset"):
        return _UNSET
    literal = _LITERAL.match(cell)
    if literal:
        return yaml.safe_load(literal[1])
    return _PROSE


def _same(stated: object, default: object) -> bool:
    """Whether a stated default is the manifest's: see the module docstring."""
    if default == '""':
        default = ""
    if stated is _UNSET:
        return default in (None, "", "no-fallback")
    if isinstance(stated, (int, float)) and isinstance(default, (int, float)):
        if isinstance(stated, bool) or isinstance(default, bool):
            return stated is default
        return float(stated) == float(default)
    return stated == default


def _default_problems(text: str) -> tuple[list[str], set[str]]:
    """Stated defaults that disagree with the manifest, and subjects it has no default for."""
    problems: list[str] = []
    unlisted: set[str] = set()
    for subject, cell in [*_table_defaults(text), *_prose_defaults(text)]:
        entry = _manifest().get(subject)
        if not isinstance(entry, dict) or "default" not in entry:
            unlisted.add(subject)
            continue
        default = entry["default"]
        if default == "derived":
            continue
        stated = _stated(cell)
        if stated is _PROSE or not _same(stated, default):
            problems.append(f"{subject}: the page says {cell}, the manifest {default!r}")
    return problems, unlisted


def test_every_stated_default_is_the_manifest_default() -> None:
    problems, _ = _default_problems(_page())
    assert problems == []


def test_stated_defaults_without_a_manifest_default_are_the_listed_ones() -> None:
    _, unlisted = _default_problems(_page())
    assert unlisted == set(_NOT_MANIFEST_DEFAULTS)


def test_the_tables_and_the_prose_both_reach_the_check() -> None:
    """Each source of stated defaults yields rows, so a parser gone blind fails here."""
    table_keys = {key for key, _ in _table_defaults(_page())}
    prose_keys = {key for key, _ in _prose_defaults(_page())}
    assert {"web.chat_max_sessions", "web.panels.canopy.vole_bytes_max"} <= table_keys
    assert {
        "phoebus.port",
        "phoebus.agent_access",
        "python_executor.execution_timeout_seconds",
        "ariel.entry_text.read_chars",
        "ariel.entry_url_template",
        "claude_code.thinking.display",
        "web.tour",
        "web.terminal",
        "health.disk.max_used_percent",
    } <= prose_keys


@pytest.mark.parametrize(
    ("right", "wrong"),
    [
        ("     - ``1800``", "     - ``1700``"),
        ("Default ``7979``", "Default ``7980``"),
        ("``read`` (default)", "``read_write`` (default)"),
    ],
)
def test_a_wrong_default_is_caught(right: str, wrong: str) -> None:
    assert _page().count(right) == 1
    problems, _ = _default_problems(_page().replace(right, wrong))
    assert len(problems) == 1


# --- the image axes --------------------------------------------------------

_EXPECTED_IMAGE = r"registry\.example\.org/accelerator/{project}[^:]*:2026\.08\.1"


def _code_block(text: str, marker: str) -> str:
    """The dedented body of the code block that holds *marker*."""
    at = text.index(marker)
    lines = text[text.rindex(".. code-block::", 0, at) :].splitlines()[2:]
    end = next(i for i, line in enumerate(lines) if line and not line.startswith("   "))
    return "\n".join(line[3:] for line in lines[:end])


def _clear_axes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OSPREY_IMAGE_REGISTRY", raising=False)
    monkeypatch.delenv("OSPREY_IMAGE_TAG", raising=False)


def test_the_images_example_moves_every_built_image(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_axes(monkeypatch)
    config = yaml.safe_load(_code_block(_page(), "images:\n     registry:"))
    config["project_name"] = "demo"
    images = resolve_image_defaults(config)
    assert set(images) == set(_OSPREY_IMAGE_SUFFIXES)
    pattern = re.compile(_EXPECTED_IMAGE.format(project="demo"))
    assert [ref for ref in images.values() if not pattern.fullmatch(ref)] == []


def test_the_environment_example_moves_every_built_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_axes(monkeypatch)
    block = _code_block(_page(), "OSPREY_IMAGE_TAG=")
    for name, value in re.findall(r"(OSPREY_IMAGE_\w+)=(\S+)", block):
        monkeypatch.setenv(name, value)
    images = resolve_image_defaults({"project_name": "demo"})
    pattern = re.compile(_EXPECTED_IMAGE.format(project="demo"))
    assert [ref for ref in images.values() if not pattern.fullmatch(ref)] == []


def test_with_no_axis_set_the_tag_is_the_tables_and_there_is_no_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_axes(monkeypatch)
    rows = {row[0]: row[3] for row in _list_table(_page(), ".. _deployment-image-axes:")[1:]}
    assert rows["Registry"] == "no prefix at all"
    assert rows["Tag"] == f"``{DEFAULT_IMAGE_TAG}``"
    assert resolve_image_axes({}) == ("", DEFAULT_IMAGE_TAG)


@pytest.mark.parametrize(
    "registry", ["registry.example.org/accelerator", "registry.example.org/accelerator/"]
)
def test_a_registry_trailing_slash_is_added_once_never_doubled(
    monkeypatch: pytest.MonkeyPatch, registry: str
) -> None:
    _clear_axes(monkeypatch)
    prefix, _ = resolve_image_axes({"images": {"registry": registry}})
    assert prefix == "registry.example.org/accelerator/"


def test_the_axes_sentence_spells_no_row_count() -> None:
    """The service-image table states its own length; see ``test_service_image_table``."""
    assert _table_rows()
    words = "|".join(_NUMBER_WORDS)
    assert re.findall(rf"\b(?:{words}|\d+)\s+rows above\b", _page(), re.IGNORECASE) == []
