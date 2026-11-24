"""Three listings the docs retype from the deployment-root tables state what the code holds.

- The ``--force`` entry of ``reference/cli.rst`` names exactly the entries
  ``osprey init --force`` replaces (discarded, merged, rewritten) and exactly the
  entries it never touches, in the order the computed ``--force`` help names them.
- The "is already an OSPREY deployment repo" entry of ``how-to/build-profiles.rst``
  names exactly the entries ``--force`` replaces, in the same order.
- The profile-sources list of ``architecture/python-executor.rst`` holds exactly
  :data:`~osprey.mcp_server.python_executor.executor.PROFILE_SOURCE_ENTRIES` without
  the retired paths, which stay protected but are not listed for operators. The list
  is grouped by kind, so it compares as a set.

Each comparison is a function taking the code-side table and returning the
differences it finds, so the negative controls below can drop an entry from, or add
one to, the code side and watch the page reported.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from osprey import repo_layout
from osprey.cli import init_cmd
from osprey.mcp_server.python_executor.executor import PROFILE_SOURCE_ENTRIES

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOCS = _REPO_ROOT / "docs" / "source"
_CLI_PAGE = _DOCS / "reference" / "cli.rst"
_BUILD_PROFILES_PAGE = _DOCS / "how-to" / "build-profiles.rst"
_EXECUTOR_PAGE = _DOCS / "architecture" / "python-executor.rst"

#: One RST inline literal.
_LITERAL = re.compile(r"``([^`]+)``")
#: How the cli page names the one CI file init scaffolds, in place of its path.
_CI_FILE_PHRASE = "the CI file"
#: A token of the cli page's never-touches clause: a literal, or the CI-file phrase.
_PRESERVED_TOKEN = re.compile(rf"``(?P<literal>[^`]+)``|(?P<ci>{re.escape(_CI_FILE_PHRASE)})")

#: A name no table holds, planted on the code side to see the page reported as lacking it.
_PLANTED = "planted-entry/"


# --- page passages -----------------------------------------------------------


def _paragraph(text: str, lead: str) -> str:
    """The paragraph of *text* opening with *lead*, its lines joined by single spaces."""
    lines = text.splitlines()
    starts = [n for n, line in enumerate(lines) if line.startswith(lead)]
    assert len(starts) == 1, f"expected one paragraph opening with {lead!r}, found {len(starts)}"
    body: list[str] = []
    for line in lines[starts[0] :]:
        if not line.strip():
            break
        body.append(line.strip())
    return " ".join(body)


def _between(paragraph: str, start: str, end: str, page: str) -> str:
    """The text of *paragraph* after *start* and before the next *end*."""
    match = re.search(re.escape(start) + r"(.*?)" + re.escape(end), paragraph)
    assert match is not None, f"{page}: no clause between {start!r} and {end!r}"
    return match[1]


def _literals(clause: str) -> list[str]:
    return _LITERAL.findall(clause)


def _cli_force_listing(text: str) -> dict[str, list[str]]:
    """The cli page's ``--force`` entry: discarded, rewritten, merged and preserved names."""
    paragraph = _paragraph(text, "``--force`` — ")
    replaced = _between(paragraph, "discarding edits to", ". In ``", "cli.rst")
    discarded, rewritten = (
        replaced.split(", and rewriting", 1) if ", and rewriting" in replaced else (replaced, "")
    )
    assert paragraph.count("Never touches") == 1, "cli.rst: no single 'Never touches' clause"
    preserved = paragraph.split("Never touches", 1)[1]
    return {
        "discarded": _literals(discarded),
        "rewritten": _literals(rewritten),
        "merged": _literals(_between(paragraph, "In ", " the entries OSPREY ships", "cli.rst")),
        "preserved": [
            m["literal"] if m["literal"] is not None else m["ci"]
            for m in _PRESERVED_TOKEN.finditer(preserved)
        ],
    }


def _build_profiles_force_listing(text: str) -> dict[str, list[str]]:
    """The build-profiles page's ``--force`` sentence: replaced (in order) and merged names."""
    page = "build-profiles.rst"
    paragraph = _paragraph(text, '**"is already an OSPREY deployment repo"**')
    return {
        "replaced": _literals(_between(paragraph, "which rewrites", ", losing any edit", page)),
        "merged": _literals(
            _between(paragraph, "In ", " it resets the entries OSPREY ships", page)
        ),
    }


def _protected_listing(text: str) -> list[str]:
    """The names the executor page's profile-sources bullet lists, nested bullets included."""
    lines = text.splitlines()
    starts = [n for n, line in enumerate(lines) if line.startswith("- **The profile sources**")]
    assert len(starts) == 1, f"expected one profile-sources bullet, found {len(starts)}"
    body: list[str] = []
    for line in lines[starts[0] + 1 :]:
        if line.startswith("- "):
            break
        body.append(line)
    return _literals("\n".join(body))


# --- code-side tables --------------------------------------------------------


def _replaced_by_wording(
    rows: Sequence[repo_layout.RootEntry], wording: Mapping[str, str]
) -> dict[str | None, list[str]]:
    """The replaced *rows*' listing names grouped by how init words them, table order.

    ``None`` holds the entries plainly replaced, losing any edit.
    """
    grouped: dict[str | None, list[str]] = {None: [], "merged": [], "rewritten": []}
    for row in rows:
        grouped[wording.get(row.key)].append(repo_layout.listing_name(row))
    return grouped


def _replaced_rows() -> tuple[repo_layout.RootEntry, ...]:
    return repo_layout.rows_with_init(repo_layout.InitRole.REPLACED)


def _scaffolded_names() -> list[str]:
    return [
        repo_layout.listing_name(row)
        for row in repo_layout.rows_with_init(repo_layout.InitRole.SCAFFOLDED)
    ]


def _live_source_entries(entries: Sequence[str]) -> list[str]:
    """*entries* without the retired paths of the source zone."""
    retired = set(repo_layout.retired_paths(repo_layout.rows_in_zone(repo_layout.Zone.SOURCE)))
    return [entry for entry in entries if entry not in retired]


# --- comparisons -------------------------------------------------------------


def _sequence_drift(page: Sequence[str], code: Sequence[str], where: str, what: str) -> list[str]:
    """Names missing from or extra on the page, then an order mismatch once the sets agree."""
    drift = [
        f"{where}: {name} is missing from the {what} list" for name in code if name not in page
    ]
    drift += [
        f"{where}: the {what} list names {name}, which the code does not"
        for name in page
        if name not in code
    ]
    if not drift and list(page) != list(code):
        drift.append(f"{where}: the {what} list reads {list(page)}, the code orders {list(code)}")
    return drift


def _cli_drift(
    page: Mapping[str, list[str]],
    replaced: Mapping[str | None, list[str]],
    preserved: Sequence[str],
    scaffolded: Sequence[str],
) -> list[str]:
    where = "reference/cli.rst --force"
    if _CI_FILE_PHRASE in page["preserved"] and len(scaffolded) != 1:
        return [f"{where}: the page says {_CI_FILE_PHRASE!r}, init scaffolds {list(scaffolded)}"]
    page_preserved = [
        scaffolded[0] if name == _CI_FILE_PHRASE else name for name in page["preserved"]
    ]
    return [
        *_sequence_drift(page["discarded"], replaced[None], where, "discarded"),
        *_sequence_drift(page["rewritten"], replaced["rewritten"], where, "rewritten"),
        *_sequence_drift(page["merged"], replaced["merged"], where, "merged"),
        *_sequence_drift(page_preserved, preserved, where, "never-touches"),
    ]


def _build_profiles_drift(
    page: Mapping[str, list[str]], replaced: Mapping[str | None, list[str]]
) -> list[str]:
    where = "how-to/build-profiles.rst --force"
    return [
        *_sequence_drift(
            page["replaced"], [*replaced[None], *replaced["rewritten"]], where, "replaced"
        ),
        *_sequence_drift(page["merged"], replaced["merged"], where, "merged"),
    ]


def _protected_drift(page: Sequence[str], entries: Sequence[str]) -> list[str]:
    where = "architecture/python-executor.rst profile sources"
    listed = {name.rstrip("/") for name in page}
    code = set(entries)
    drift = [f"{where}: {name} is missing from the list" for name in sorted(code - listed)]
    drift += [
        f"{where}: the list names {name}, which PROFILE_SOURCE_ENTRIES does not hold"
        for name in sorted(listed - code)
    ]
    return drift


# --- the pages as written ----------------------------------------------------


def _cli_page() -> dict[str, list[str]]:
    return _cli_force_listing(_CLI_PAGE.read_text(encoding="utf-8"))


def _build_profiles_page() -> dict[str, list[str]]:
    return _build_profiles_force_listing(_BUILD_PROFILES_PAGE.read_text(encoding="utf-8"))


def _executor_page() -> list[str]:
    return _protected_listing(_EXECUTOR_PAGE.read_text(encoding="utf-8"))


def test_the_cli_force_entry_names_what_init_replaces_and_preserves() -> None:
    drift = _cli_drift(
        _cli_page(),
        _replaced_by_wording(_replaced_rows(), init_cmd.REPLACED_WORDING),
        init_cmd.PRESERVED_BY_FORCE,
        _scaffolded_names(),
    )

    assert drift == []


def test_the_build_profiles_force_sentence_names_what_init_replaces() -> None:
    drift = _build_profiles_drift(
        _build_profiles_page(),
        _replaced_by_wording(_replaced_rows(), init_cmd.REPLACED_WORDING),
    )

    assert drift == []


def test_the_protected_paths_list_is_the_live_profile_source_set() -> None:
    assert _protected_drift(_executor_page(), _live_source_entries(PROFILE_SOURCE_ENTRIES)) == []


def test_the_retired_source_paths_are_protected_but_not_listed() -> None:
    retired = repo_layout.retired_paths(repo_layout.rows_in_zone(repo_layout.Zone.SOURCE))
    listed = {name.rstrip("/") for name in _executor_page()}

    assert retired
    assert set(retired) <= set(PROFILE_SOURCE_ENTRIES)
    assert not set(retired) & listed


# --- negative controls -------------------------------------------------------


def _one_convention_and_one_zone_row(
    rows: Sequence[repo_layout.RootEntry],
) -> tuple[repo_layout.RootEntry, repo_layout.RootEntry]:
    """The first convention-directory row and the first other row of *rows*."""
    conventions = {row.key for row in repo_layout.convention_rows()}
    convention = next(row for row in rows if row.key in conventions)
    zone = next(row for row in rows if row.key not in conventions)
    return convention, zone


@pytest.mark.parametrize("pick", [0, 1], ids=["convention source", "zone entry"])
def test_a_replaced_entry_dropped_from_the_code_is_reported_on_both_pages(pick: int) -> None:
    rows = _replaced_rows()
    dropped = _one_convention_and_one_zone_row(rows)[pick]
    grouped = _replaced_by_wording(
        [row for row in rows if row.key != dropped.key], init_cmd.REPLACED_WORDING
    )
    name = repo_layout.listing_name(dropped)

    cli = _cli_drift(_cli_page(), grouped, init_cmd.PRESERVED_BY_FORCE, _scaffolded_names())
    build = _build_profiles_drift(_build_profiles_page(), grouped)

    assert any("cli.rst" in line and name in line and "does not" in line for line in cli), cli
    assert any("build-profiles.rst" in line and name in line for line in build), build


@pytest.mark.parametrize("pick", [0, 1], ids=["convention source", "zone entry"])
def test_a_preserved_entry_dropped_from_the_code_is_reported(pick: int) -> None:
    preserved = list(init_cmd.PRESERVED_BY_FORCE)
    conventions = {repo_layout.listing_name(row) for row in repo_layout.convention_rows()}
    dropped = [name for name in preserved if (name in conventions) == (pick == 0)][0]

    drift = _cli_drift(
        _cli_page(),
        _replaced_by_wording(_replaced_rows(), init_cmd.REPLACED_WORDING),
        [name for name in preserved if name != dropped],
        _scaffolded_names(),
    )

    assert any(dropped in line and "never-touches" in line for line in drift), drift


def test_an_entry_the_code_gains_is_reported_missing_from_each_force_listing() -> None:
    grouped = _replaced_by_wording(_replaced_rows(), init_cmd.REPLACED_WORDING)
    grouped[None] = [*grouped[None], _PLANTED]

    cli = _cli_drift(
        _cli_page(), grouped, [*init_cmd.PRESERVED_BY_FORCE, _PLANTED], _scaffolded_names()
    )
    build = _build_profiles_drift(_build_profiles_page(), grouped)

    assert f"reference/cli.rst --force: {_PLANTED} is missing from the discarded list" in cli
    assert f"reference/cli.rst --force: {_PLANTED} is missing from the never-touches list" in cli
    assert (
        f"how-to/build-profiles.rst --force: {_PLANTED} is missing from the replaced list" in build
    )


def test_a_reordered_force_table_is_reported() -> None:
    grouped = _replaced_by_wording(_replaced_rows(), init_cmd.REPLACED_WORDING)
    grouped[None] = list(reversed(grouped[None]))

    assert any("orders" in line for line in _build_profiles_drift(_build_profiles_page(), grouped))


@pytest.mark.parametrize("pick", [0, 1], ids=["convention source", "zone entry"])
def test_a_source_entry_dropped_from_the_code_is_reported_on_the_executor_page(pick: int) -> None:
    live = _live_source_entries(PROFILE_SOURCE_ENTRIES)
    source_rows = list(repo_layout.rows_in_zone(repo_layout.Zone.SOURCE))
    dropped = _one_convention_and_one_zone_row(source_rows)[pick].path

    drift = _protected_drift(_executor_page(), [entry for entry in live if entry != dropped])

    assert drift == [
        f"architecture/python-executor.rst profile sources: the list names {dropped}, "
        "which PROFILE_SOURCE_ENTRIES does not hold"
    ]


def test_a_source_entry_the_code_gains_is_reported_missing_from_the_executor_page() -> None:
    live = _live_source_entries(PROFILE_SOURCE_ENTRIES)
    planted = _PLANTED.rstrip("/")

    assert _protected_drift(_executor_page(), [*live, planted]) == [
        f"architecture/python-executor.rst profile sources: {planted} is missing from the list"
    ]
