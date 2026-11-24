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

import functools
import re
from collections.abc import Mapping
from pathlib import Path

import pytest

from osprey import repo_layout
from tests._layout_spelling import (
    CATEGORIES,
    EQUAL_TO_SOURCE,
    GENERATED_SCHEMA,
    PINNED_COPY,
    TEMPLATE_TREE,
    Site,
    compare,
    scan,
    scan_source,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The declaration, the one file that spells the deployment root's paths.
DECLARATION = "src/osprey/repo_layout.py"

#: The root paths a later layout moves that are not ordinary words, and the
#: display paths of the declared children.
GUARDED = (
    "web-terminal-context",
    "output-styles",
    "ci-extra.yml",
    "osprey.service",
    "data/facility",
    "data/code_review",
    "data/canopy",
    "data/.runtime",
    "scripts/verify.sh",
    "scripts/osprey-boot-hook.sh",
)


def _keys() -> frozenset[str]:
    """The declared keys, which are identities wherever a lookup or a table takes them."""
    return frozenset(
        {row.key for row in repo_layout.ROOT_ENTRIES}
        | {item.key for item in repo_layout.ROOT_CHILDREN}
    )


_CATALOG = "src/osprey/agent_runner/build_artifacts/catalog.py"
_CONVENTIONS = "src/osprey/cli/profile_conventions.py"
_GENERATED = "packages/osprey-facility/src/osprey_facility/schema/_generated/core.py"
_GALLERY = "src/osprey/interfaces/web_terminal/scaffold_gallery_service.py"
_CLAUDE_CODE_FILES = "src/osprey/interfaces/web_terminal/claude_code_files.py"
_CANOPY = "src/osprey/integrations/canopy/__init__.py"
_TEMPLATE_MANAGER = "src/osprey/cli/templates/manager.py"

#: ``(file, constant text) -> (count, category)`` for each site still allowed
#: to spell a guarded path.
ALLOWED: Mapping[tuple[str, str], tuple[int, str]] = {
    # The facility package names its tree's position in messages; it cannot
    # import the framework (held equal below).
    (
        "packages/osprey-facility/src/osprey_facility/source_tree.py",
        "data/facility",
    ): (1, PINNED_COPY),
    # Canopy's standard-library-only package: the screen library and the
    # facility collections below the project root (tests/integrations/canopy/
    # test_layout_copies.py).
    (_CANOPY, "data/canopy/library"): (1, PINNED_COPY),
    (_CANOPY, "data/canopy/collections"): (1, PINNED_COPY),
    # The render's facility tree, where the runtime reads it whatever the
    # source layout.
    ("src/osprey/facility/settings.py", "data/facility"): (1, EQUAL_TO_SOURCE),
    (_CONVENTIONS, "data/facility"): (1, EQUAL_TO_SOURCE),
    # Claude Code's own output-styles directory under the render's .claude/,
    # and the artifact names keyed by the convention.
    (_CATALOG, "output-styles/control-operator"): (1, EQUAL_TO_SOURCE),
    (_CATALOG, ".claude/output-styles/control-operator.md"): (1, EQUAL_TO_SOURCE),
    (_CONVENTIONS, ".claude/output-styles"): (1, EQUAL_TO_SOURCE),
    (
        "src/osprey/cli/build_profile_emit.py",
        "Output styles installed into .claude/output-styles/.",
    ): (1, EQUAL_TO_SOURCE),
    ("src/osprey/cli/templates/manifest.py", "output-styles/"): (1, EQUAL_TO_SOURCE),
    (_CLAUDE_CODE_FILES, "output-styles"): (1, EQUAL_TO_SOURCE),
    (_CLAUDE_CODE_FILES, "output-styles/"): (1, EQUAL_TO_SOURCE),
    (_GALLERY, "output-styles"): (2, EQUAL_TO_SOURCE),
    # The render's per-user context tree.
    (_CONVENTIONS, "docker/web-terminal-context"): (1, EQUAL_TO_SOURCE),
    (_CONVENTIONS, "docker/web-terminal-context/base.md"): (1, EQUAL_TO_SOURCE),
    (_TEMPLATE_MANAGER, "docker/web-terminal-context"): (1, EQUAL_TO_SOURCE),
    ("src/osprey/deployment/web_terminals/seeding.py", "docker/web-terminal-context"): (
        1,
        EQUAL_TO_SOURCE,
    ),
    # The framework's own template trees, not laid out as a deployment.
    (_CATALOG, "claude/output-styles/control-operator.md.j2"): (1, TEMPLATE_TREE),
    ("src/osprey/cli/templates/artifact_library.py", "output-styles"): (1, TEMPLATE_TREE),
    (_TEMPLATE_MANAGER, "claude_code/web-terminal-context"): (1, TEMPLATE_TREE),
    ("src/osprey/cli/profile_cmd.py", "web-terminal-context"): (1, TEMPLATE_TREE),
    (
        "src/osprey/integrations/canopy/repo_checks.py",
        "src/osprey/templates/apps/demo/data/canopy/**",
    ): (1, TEMPLATE_TREE),
    # Field descriptions generated from the facility schema.
    (
        _GENERATED,
        "One facility, described once: its places, devices, channels and groups, the models "
        "that simulate it and how each model is wired to channels, the write limits and the "
        "scenarios. `osprey build` combines the sources under data/facility/ into one file of "
        "this shape. Every list whose order carries meaning sets `list_elements_ordered`; a "
        "multivalued slot without it is a set, compared and written sorted by its string "
        "form, or by its key when its entries are keyed (`scheme_names` by name and scheme, "
        "`citations` by document, version and locator).",
    ): (1, GENERATED_SCHEMA),
    (_GENERATED, "Relative to data/facility/."): (1, GENERATED_SCHEMA),
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
    """Each spelling is a row's path or a child's display path."""
    declared = {row.path for row in repo_layout.ROOT_ENTRIES} | {
        repo_layout.child_display_path(item.parent, item.key) for item in repo_layout.ROOT_CHILDREN
    }
    for spelling in GUARDED:
        assert spelling in declared or any(path.startswith(f"{spelling}/") for path in declared), (
            spelling
        )


# --- the scanner finds every form a spelling takes ---------------------------------------

_PLANTED = '''
"""A docstring naming ci-extra.yml is prose, not code."""
LITERAL = "osprey.service"
JOINED = Path("data") / "facility"
NESTED = root / "scripts" / "verify.sh"
SENTENCE = "Edit the reviewer under data/code_review and build again."
FORMATTED = f"data/.runtime/{lane}_curve"
STYLES = ".claude/output-styles"
'''

#: The six forms, each with the line it is planted on.
_FORMS = {
    "a literal": (3, "osprey.service"),
    "a joined path": (4, "data/facility"),
    "a join through a variable": (5, "scripts/verify.sh"),
    "a prose sentence": (6, "data/code_review"),
    "an f-string": (7, "data/.runtime"),
    "a spelling inside a render path": (8, "output-styles"),
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

    prefix = repo_layout.child_display_path("data", "facility")
    assert source_tree._DISPLAY_PREFIX == prefix
    assert source_tree.display_path("limits.yaml") == f"{prefix}/limits.yaml"


def test_the_bluesky_template_mounts_and_ignores_the_derived_curve_directory() -> None:
    """Compose mounts, and git ignores, the directory the certificates are minted in."""
    from osprey.deployment.container_lifecycle import _bluesky_curve_dir

    runtime = repo_layout.child_display_path("data", "runtime")
    assert _bluesky_curve_dir("lane").as_posix() == f"{runtime}/lane_curve"

    compose = (REPO_ROOT / "src/osprey/templates/services/bluesky/docker-compose.yml.j2").read_text(
        encoding="utf-8"
    )
    mounts = re.findall(r"- \./(\S+)/\{\{ lane_key \}\}_curve/(\w+):/app/curve:ro", compose)
    assert sorted(side for _parent, side in mounts) == ["bridge", "queueserver"]
    assert {parent for parent, _side in mounts} == {runtime}

    gitignore = (REPO_ROOT / "src/osprey/templates/project/gitignore").read_text(encoding="utf-8")
    rules = [line for line in gitignore.splitlines() if line and not line.startswith("#")]
    assert f"/{runtime}/" in rules


def test_the_code_review_prompt_default_is_where_init_puts_the_starter() -> None:
    from osprey.cli.templates.preset_data import STARTER_PROMPT_PATH
    from osprey.services.code_review.settings import DEFAULT_PROMPT

    assert DEFAULT_PROMPT == f"{repo_layout.entry('data').path}/{STARTER_PROMPT_PATH}"
