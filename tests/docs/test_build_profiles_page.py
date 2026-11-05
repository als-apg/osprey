"""The build-profiles how-to states facts a reader acts on; each is checked against the code.

* the tree under "What ``osprey init`` writes" lists exactly the top-level
  entries a fresh ``osprey init`` writes for the two bundled presets, and the
  paragraph under it names exactly the entries only one preset writes;
* the "Valid events" list is :data:`CLAUDE_CODE_HOOK_EVENTS`, in its order;
* the ``osprey build`` options table names every option the command takes;
* every message quoted under "Troubleshooting" is copied from a string literal
  in ``src/osprey``.
"""

from __future__ import annotations

import ast
import os
import re
import textwrap
from functools import cache
from pathlib import Path

import pytest
from click.testing import CliRunner

from osprey.cli.build_cmd import build
from osprey.cli.init_cmd import init
from osprey.cli.templates.claude_code import CLAUDE_CODE_HOOK_EVENTS

REPO_ROOT = Path(__file__).resolve().parents[2]
PAGE = REPO_ROOT / "docs/source/how-to/build-profiles.rst"
SRC = REPO_ROOT / "src/osprey"

PRESETS = ("control-assistant", "hello-world")

#: Tree entries a fresh ``osprey init`` need not write, each with the reason.
PAGE_ONLY_ENTRIES = {
    "build/": "absent until the first `osprey build`",
}

#: Entries whose own tree line states when init writes them, so the
#: preset-specific paragraph does not name them. With no provider key in the
#: environment, ``.env`` (and its lock) comes only from a profile's
#: ``env.defaults``, which ``control-assistant`` declares and ``hello-world`` not.
SELF_DESCRIBED_ENTRIES = {".env", ".env.lock"}

#: Each message quoted under "Troubleshooting", whitespace-normalized, mapped
#: to the string-literal text it is copied from. A literal's ``{...}`` fields
#: are written ``{}``; the page shows what they print, or ``…``.
QUOTED_MESSAGES = {
    "No OSPREY deployment repo found: no profile.yml in … or any parent directory.": [
        "No OSPREY deployment repo found: no {} in {} or any parent directory.",
    ],
    "profile.yml was materialized from preset … at …; the preset bundled with OSPREY … "
    "is … — it has moved on since": [
        "{} was materialized from preset {} at {}{}; the preset bundled with OSPREY {} "
        "is {} — it has moved on since",
    ],
    "Profile convention directories are invalid": [
        "Profile convention directories are invalid ({} problem(s)):",
    ],
    "web-terminal-context/<name> is a file": ["{}/{} is a file: {}/ holds one "],
    "project/ mirror writes N build-owned path(s)": ["{}/ mirror writes {} build-owned path(s)"],
    "Repo root has N unrecognized top-level entry/entries": [
        "Repo root has %d unrecognized top-level entry/entries: %s",
    ],
    "Unknown profile key(s): 'overlay'": ["Unknown {} key(s): {}. "],
    "is already an OSPREY deployment repo": ["{target} is already an OSPREY deployment repo"],
    "already exists, is not empty, and is not an OSPREY deployment repo": [
        "{target} already exists, is not empty, and is not an OSPREY deployment repo",
    ],
    "OSPREY X does not satisfy requires_osprey_version": [
        "OSPREY %s does not satisfy requires_osprey_version: %s",
    ],
}


def _page() -> str:
    return PAGE.read_text(encoding="utf-8")


def _section(title: str) -> str:
    """The text of the section headed *title*, up to the next heading of any level."""
    lines = _page().splitlines()
    start = lines.index(title) + 2
    for index in range(start, len(lines) - 1):
        underline = lines[index + 1]
        if (
            lines[index].strip()
            and underline
            and set(underline) <= set("=-~^")
            and (len(underline) >= len(lines[index]) > 0)
        ):
            return "\n".join(lines[start:index])
    return "\n".join(lines[start:])


def _code_block(text: str, language: str) -> list[str]:
    """The dedented body lines of the first ``code-block:: <language>`` in *text*."""
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.strip() != f".. code-block:: {language}":
            continue
        body: list[str] = []
        for following in lines[index + 1 :]:
            if following.strip() and not following.startswith(" "):
                break
            body.append(following)
        return textwrap.dedent("\n".join(body)).strip("\n").splitlines()
    raise AssertionError(f"no {language} block in the section")


def _literals(text: str) -> list[str]:
    return re.findall(r"``([^`]+)``", text)


def _paragraph_containing(text: str, needle: str) -> str:
    for paragraph in re.split(r"\n\s*\n", text):
        if needle in paragraph:
            return " ".join(paragraph.split())
    raise AssertionError(f"no paragraph containing {needle!r}")


@pytest.fixture(scope="module")
def init_listings(tmp_path_factory: pytest.TempPathFactory) -> dict[str, set[str]]:
    """The top-level entries a fresh ``osprey init`` writes, per preset."""
    listings: dict[str, set[str]] = {}
    for preset in PRESETS:
        repo = tmp_path_factory.mktemp(preset) / "my-facility"
        result = CliRunner().invoke(
            init,
            [str(repo), "--preset", preset, "--no-git"],
            # No provider key from the shell: `.env` must come from the preset.
            env={name: None for name in os.environ if name.endswith("_API_KEY")},
        )
        assert result.exit_code == 0, result.output
        # `--no-git`: the page states the git repo and first commit in prose.
        listings[preset] = {
            f"{entry.name}/" if entry.is_dir() else entry.name for entry in repo.iterdir()
        }
    return listings


def _tree_entries() -> set[str]:
    block = _code_block(_section("What ``osprey init`` writes"), "text")
    assert block[0] == "my-facility/"
    return {match.group(1) for line in block[1:] if (match := re.match(r"^  (\S+)", line))}


@pytest.mark.slow
def test_init_tree_lists_what_init_writes(init_listings: dict[str, set[str]]) -> None:
    written = set().union(*init_listings.values())
    listed = _tree_entries()
    assert written - listed == set(), "osprey init writes entries the tree does not show"
    assert listed - written - set(PAGE_ONLY_ENTRIES) == set(), (
        "the tree shows entries osprey init does not write"
    )


@pytest.mark.slow
def test_conditional_entries_paragraph_names_the_preset_specific_ones(
    init_listings: dict[str, set[str]],
) -> None:
    written = set().union(*init_listings.values())
    common = set.intersection(*init_listings.values())
    conditional = written - common - set(PAGE_ONLY_ENTRIES) - SELF_DESCRIBED_ENTRIES
    paragraph = _paragraph_containing(
        _section("What ``osprey init`` writes"), "appear only when the preset calls for them"
    )
    named = paragraph.split("appear only when")[0]
    assert set(_literals(named)) == conditional

    every_but = re.search(
        r"``control-assistant`` deployment has all of them except (.+?);", paragraph
    )
    only = re.search(r"``hello-world`` deployment has only (.+?)\.$", paragraph)
    assert every_but and only, paragraph
    assert conditional - set(_literals(every_but.group(1))) == (
        init_listings["control-assistant"] & conditional
    )
    assert set(_literals(only.group(1))) == init_listings["hello-world"] & conditional


def test_valid_hook_events_are_the_events_a_profile_may_declare() -> None:
    paragraph = _paragraph_containing(_section("Custom hooks"), "Valid events:")
    assert _literals(paragraph) == list(CLAUDE_CODE_HOOK_EVENTS)


def test_build_options_table_names_every_build_option() -> None:
    building = _section("Building")
    table = building.split(".. list-table::", 1)[1].split("\n\n.. ", 1)[0]
    rows = re.findall(r"^   \* - ``([^`]+)``", table, flags=re.MULTILINE)
    listed = {opt for row in rows for opt in re.findall(r"--[\w-]+", row)}
    declared = {
        opt
        for param in build.params
        for opt in param.opts
        if opt.startswith("--") and opt != "--help" and not getattr(param, "hidden", False)
    }
    assert listed == declared


@cache
def _source_templates() -> frozenset[str]:
    """Every string literal under ``src/osprey``, with an f-string's fields as ``{}``.

    Implicitly concatenated literals are one node, so a message split over
    source lines is one template here.
    """
    templates: set[str] = set()
    for path in SRC.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                templates.add(node.value)
            elif isinstance(node, ast.JoinedStr):
                templates.add(
                    "".join(
                        part.value if isinstance(part, ast.Constant) else "{}"
                        for part in node.values
                    )
                )
    return frozenset(templates)


def test_troubleshooting_quotes_are_the_messages_osprey_prints() -> None:
    section = _section("Troubleshooting")
    quoted = {" ".join(q.split()) for q in re.findall(r'\*\*"(.+?)"\*\*', section, re.DOTALL)}
    assert quoted == set(QUOTED_MESSAGES), "update QUOTED_MESSAGES with the page's quotes"

    templates = _source_templates()
    missing = [
        fragment
        for fragments in QUOTED_MESSAGES.values()
        for fragment in fragments
        if not any(fragment in template for template in templates)
    ]
    assert missing == [], "quoted messages no longer in src/osprey"
