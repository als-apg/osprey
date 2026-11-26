"""No moved deployment-root path is spelt by hand outside the root's declaration.

:mod:`osprey.repo_layout` is the one place the deployment root's paths are
spelt; every other module in ``src/`` and ``packages/*/src`` reads them from
it. This guard reads the shipped Python for the spellings of the root paths a
later layout moves that are not ordinary words, and for the display paths of
the declared children, and fails on one it finds outside the declaration
unless the site is listed in :data:`ALLOWED` with its count and the reason it
may stay. :data:`ALLOWED` is therefore the inventory of the Python sites that
change by hand when those paths move. What counts as a spelling is
:mod:`tests._layout_spelling`'s rule; a declared key passed to a lookup, or
keying a table, is an identity and not a spelling.

The copies the declaration cannot reach (a package or shipped file that cannot
import it) are each held equal to it here or by the test named beside them.
"""

from __future__ import annotations

import ast
import functools
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

import pytest

from osprey import repo_layout
from tests._layout_spelling import (
    CATEGORIES,
    EQUAL_TO_SOURCE,
    SOURCE_ROOTS,
    TEMPLATE_TREE,
    Site,
    _division_operands,
    _is_path_call,
    compare,
    scan,
    scan_source,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The declaration, the one file that spells the deployment root's paths.
DECLARATION = "src/osprey/repo_layout.py"

#: The root paths a later layout moves that are not ordinary words, the display
#: paths of the declared children, and the moved-from paths no reader falls back to,
#: each beside the path it moved to.
GUARDED = (
    "web-terminal-context",
    "agent/context",
    "agent/output-styles",
    "ci-extra.yml",
    "ops/ci-extra.yml",
    "osprey.service",
    "ops/osprey.service",
    "data/facility",
    "data/code_review",
    "agent/code_review",
    "data/canopy",
    "data/.runtime",
    "var/runtime",
    "ops/scripts/verify.sh",
    "ops/scripts/osprey-boot-hook.sh",
)


def _keys() -> frozenset[str]:
    """The declared keys, which are identities wherever a lookup or a table takes them."""
    return frozenset(
        {row.key for row in repo_layout.ROOT_ENTRIES}
        | {item.key for item in repo_layout.ROOT_CHILDREN}
    )


_CATALOG = "src/osprey/agent_runner/build_artifacts/catalog.py"
_CONVENTIONS = "src/osprey/cli/profile_conventions.py"
_GALLERY = "src/osprey/interfaces/web_terminal/scaffold_gallery_service.py"
_CLAUDE_CODE_FILES = "src/osprey/interfaces/web_terminal/claude_code_files.py"
_TEMPLATE_MANAGER = "src/osprey/cli/templates/manager.py"

#: ``(file, constant text) -> (count, category)`` for each site still allowed
#: to spell a guarded path.
ALLOWED: Mapping[tuple[str, str], tuple[int, str]] = {
    # The render's per-user context tree. It is named by the convention key a
    # profile spells (``exclude: {web-terminal-context: ...}``), which is not a
    # path and did not move; the source tree it renders from is agent/context.
    (_CONVENTIONS, "docker/web-terminal-context"): (1, EQUAL_TO_SOURCE),
    (_CONVENTIONS, "docker/web-terminal-context/base.md"): (1, EQUAL_TO_SOURCE),
    (_TEMPLATE_MANAGER, "docker/web-terminal-context"): (1, EQUAL_TO_SOURCE),
    ("src/osprey/deployment/web_terminals/seeding.py", "docker/web-terminal-context"): (
        1,
        EQUAL_TO_SOURCE,
    ),
    # The framework's own template trees, not laid out as a deployment.
    (_TEMPLATE_MANAGER, "claude_code/web-terminal-context"): (1, TEMPLATE_TREE),
    ("src/osprey/cli/profile_cmd.py", "web-terminal-context"): (1, TEMPLATE_TREE),
}


@functools.cache
def _sites() -> tuple[Site, ...]:
    """Every guarded spelling in the shipped Python, read once per process."""
    return tuple(scan(REPO_ROOT, GUARDED, exempt={DECLARATION}, keys=_keys()))


def test_no_root_path_is_spelt_by_hand() -> None:
    problems = compare(_sites(), ALLOWED)
    assert problems == [], (
        f"read these paths from {DECLARATION} instead of spelling them; a site that "
        "may keep its spelling goes in ALLOWED with its count and category:\n" + "\n".join(problems)
    )


def test_every_allowed_site_has_a_category() -> None:
    assert {category for _count, category in ALLOWED.values()} <= CATEGORIES


def test_every_guarded_spelling_is_a_declared_path() -> None:
    """Each spelling is a row's path, a child's display path or a moved-from path."""
    declared = (
        {row.path for row in repo_layout.ROOT_ENTRIES}
        | {
            repo_layout.child_display_path(item.parent, item.key)
            for item in repo_layout.ROOT_CHILDREN
        }
        | {move.old for move in repo_layout.retired_moves()}
    )
    for spelling in GUARDED:
        assert spelling in declared or any(path.startswith(f"{spelling}/") for path in declared), (
            spelling
        )


#: A plain word: a successor spelt so is an ordinary word the guard cannot read.
_PLAIN_WORD = re.compile(r"[a-z]+")


def test_every_guarded_moved_from_path_has_its_successor_guarded() -> None:
    """A moved-from path is guarded beside the path it moved to, so neither is spelt.

    A successor that is a plain word (``facility``) is no spelling the guard
    can read; its moved-from path stays guarded alone.
    """
    missing = [
        move.new
        for move in repo_layout.retired_moves()
        if move.old in GUARDED and move.new not in GUARDED and not _PLAIN_WORD.fullmatch(move.new)
    ]
    assert missing == []


# --- no join rebuilds a dissolved directory's old child -----------------------------------


def _dissolved_children() -> dict[str, frozenset[str]]:
    """Each dissolved directory's name beside the names its moved-out children had."""
    return {
        name: frozenset(
            move.old.split("/")[1]
            for move in repo_layout.retired_moves()
            if move.old.startswith(f"{name}/")
        )
        for name in repo_layout.dissolved_names()
    }


def _operand_text(node: ast.expr) -> str | None:
    """A join operand's text, an f-string's fields as ``{}``; ``None`` when not constant."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(
            value.value if isinstance(value, ast.Constant) else "{}" for value in node.values
        )
    return None


def _segment_is(segment: str, names: frozenset[str]) -> bool:
    """Whether *segment* spells one of *names*, an f-string field standing for any text."""
    pattern = "[^/]+".join(re.escape(part) for part in segment.split("{}"))
    return any(re.fullmatch(pattern, name) for name in names)


def dissolved_joins(source: str, file: str) -> list[str]:
    """Each join in *source* of a dissolved directory followed by one of its old children.

    A join is a ``/`` chain, ``Path(...)``, ``.joinpath(...)`` or
    ``os.path.join(...)``; its constant operands are split into segments, and
    an operand that is not constant breaks the run. ``Path("data") /
    "landing"`` rebuilds a path that no longer exists in a deployment; a
    render-side path reads its row's ``render`` instead.
    """
    dissolved = _dissolved_children()
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            operands = list(_division_operands(node))
        elif isinstance(node, ast.Call) and _is_path_call(node.func):
            operands = list(node.args)
        else:
            continue
        segments: list[str | None] = []
        for operand in operands:
            text = _operand_text(operand)
            segments.extend(text.strip("/").split("/") if text else [None])
        for parent, child in zip(segments, segments[1:], strict=False):
            if parent in dissolved and child is not None and _segment_is(child, dissolved[parent]):
                found.append(f"{file}:{node.lineno}: {parent}/{child}")
    return sorted(set(found))


def test_no_join_spells_a_dissolved_directorys_old_child() -> None:
    found = []
    for pattern in SOURCE_ROOTS:
        for root in sorted(REPO_ROOT.glob(pattern)):
            for path in sorted(root.rglob("*.py")):
                file = path.relative_to(REPO_ROOT).as_posix()
                found.extend(dissolved_joins(path.read_text(encoding="utf-8"), file))
    assert found == [], (
        "a dissolved directory's children moved; read the row's path, or its render "
        "for a render-side path, from the declaration:\n" + "\n".join(found)
    )


def test_the_join_check_knows_the_dissolved_children() -> None:
    children = _dissolved_children()
    assert set(children) == set(repo_layout.dissolved_names()) == {"data"}
    assert {"facility", "canopy", "landing", ".runtime", "code_review"} <= children["data"]
    assert "bluesky_curve" in children["data"]


_PLANTED_JOINS = """
A = Path("data") / "landing"
B = root / "data" / "facility" / "records"
C = os.path.join(root, "data", "canopy", "library")
D = root.joinpath("data/.runtime")
E = root / "data" / f"{lane}_curve"
F = render / entry("facility").render
G = render / "data" / "graph"
H = root / "data" / name / "facility"
I = root / "facility" / "data"
"""


def test_the_join_check_finds_each_planted_join() -> None:
    assert dissolved_joins(_PLANTED_JOINS, "planted.py") == [
        "planted.py:2: data/landing",
        "planted.py:3: data/facility",
        "planted.py:4: data/canopy",
        "planted.py:5: data/.runtime",
        "planted.py:6: data/{}_curve",
    ]


# --- the scanner finds every form a spelling takes ---------------------------------------

_PLANTED = '''
"""A docstring naming ci-extra.yml is prose, not code."""
LITERAL = "osprey.service"
JOINED = Path("data") / "facility"
NESTED = root / "ops" / "scripts" / "verify.sh"
SENTENCE = "Edit the reviewer under data/code_review and build again."
FORMATTED = f"data/.runtime/{lane}_curve"
STYLES = "build/agent/output-styles"
'''

#: The six forms, each with the line it is planted on.
_FORMS = {
    "a literal": (3, "osprey.service"),
    "a joined path": (4, "data/facility"),
    "a join through a variable": (5, "ops/scripts/verify.sh"),
    "a prose sentence": (6, "data/code_review"),
    "an f-string": (7, "data/.runtime"),
    "a spelling inside a longer path": (8, "agent/output-styles"),
}


@pytest.mark.parametrize("form", list(_FORMS))
def test_the_scanner_finds_a_planted_spelling(form: str) -> None:
    found = {
        (site.line, site.spelling) for site in scan_source(_PLANTED, "planted.py", GUARDED, _keys())
    }
    assert _FORMS[form] in found


def test_the_scanner_finds_the_six_forms_and_nothing_else() -> None:
    found = [
        (site.line, site.spelling) for site in scan_source(_PLANTED, "planted.py", GUARDED, _keys())
    ]
    assert found == sorted(_FORMS.values())


_IDENTITIES = """
UNIT = entry("osprey.service").path
CHILD = child_display_path("data", "facility")
RENDER = {"web-terminal-context": render, "output-styles": render}
NAME = repo_layout.display_path("ci-extra.yml")
PATTERN = "my-output-styles"
"""


def test_a_declared_key_is_no_spelling() -> None:
    assert scan_source(_IDENTITIES, "identities.py", GUARDED, _keys()) == []


def test_a_key_outside_a_lookup_is_still_a_spelling() -> None:
    sites = scan_source('open("osprey.service")\nX = ["ci-extra.yml"]\n', "k.py", GUARDED, _keys())
    assert [site.spelling for site in sites] == ["osprey.service", "ci-extra.yml"]


# --- the private copies the declaration cannot reach are held equal ----------------------


def test_the_facility_packages_display_prefix_equals_the_declared_child() -> None:
    from osprey_facility import source_tree

    prefix = repo_layout.entry("facility").path
    assert source_tree._DISPLAY_PREFIX == prefix
    assert source_tree.display_path("limits.yaml") == f"{prefix}/limits.yaml"


def test_the_bluesky_template_mounts_and_ignores_the_derived_curve_directory() -> None:
    """Compose mounts, and git ignores, the directory the certificates are minted in."""
    from osprey.cli.init_cmd import _repo_gitignore
    from osprey.deployment.container_lifecycle import _bluesky_curve_dir
    from osprey.utils.workspace import STATE_DIR_NAME

    runtime = repo_layout.child_display_path("var", "runtime")
    assert _bluesky_curve_dir("lane").as_posix() == f"{runtime}/lane_curve"

    compose = (REPO_ROOT / "src/osprey/templates/services/bluesky/docker-compose.yml.j2").read_text(
        encoding="utf-8"
    )
    mounts = re.findall(r"- \./(\S+)/\{\{ lane_key \}\}_curve/(\w+):/app/curve:ro", compose)
    assert sorted(side for _parent, side in mounts) == ["bridge", "queueserver"]
    assert {parent for parent, _side in mounts} == {runtime}

    # The certificates lie in the state zone, which the repo .gitignore ignores
    # whole and anchored at the root.
    assert PurePosixPath(runtime).parts[0] == STATE_DIR_NAME
    rules = [line for line in _repo_gitignore().splitlines() if line and not line.startswith("#")]
    assert f"/{STATE_DIR_NAME}/" in rules


def test_the_code_review_prompt_default_is_where_init_puts_the_starter() -> None:
    from osprey.cli.templates.preset_data import STARTER_PROMPT_PATH
    from osprey.services.code_review.settings import DEFAULT_PROMPT

    assert (
        DEFAULT_PROMPT
        == STARTER_PROMPT_PATH
        == repo_layout.display_path("code_review", "reviewer.md")
    )
