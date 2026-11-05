"""The hello-world tutorial quotes the profile ``osprey init`` writes, and counts it right.

Step 2 of the tutorial shows excerpts of the emitted ``profile.yml`` and says
how many hooks it declares and how many commented blocks it carries; Step 1
shows the summary ``osprey init`` prints. A reader compares those with their
own copy, so each is checked against the real thing:

* every line of the ``hooks:`` and ``config:`` excerpts is a line of the
  preset, in the preset's order (``...`` and ``# ...`` lines mark a gap);
* the hook count, the helper-script count and the commented-block count the
  page names equal the ones a fresh ``osprey init --preset hello-world``
  produces;
* the file rows of the init summary on the page are rows ``osprey init`` prints.
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey.agent_runner.build_artifacts.catalog import BuildArtifactCatalog
from osprey.cli.init_cmd import init

REPO_ROOT = Path(__file__).resolve().parents[2]
TUTORIAL = REPO_ROOT / "docs/source/getting-started/hello-world-tutorial.rst"
PRESET = REPO_ROOT / "src/osprey/profiles/presets/hello-world.yml"

#: Lines in an excerpt that stand for omitted preset lines.
ELISIONS = {"...", "# ..."}

#: Number words the page spells counts with.
WORDS = {
    word: number
    for number, word in enumerate(
        "one two three four five six seven eight nine ten eleven twelve thirteen "
        "fourteen fifteen sixteen seventeen eighteen nineteen twenty".split(),
        start=1,
    )
}

#: An emitted commented block opens with a ``# --- Title ---`` banner.
BLOCK_BANNER = re.compile(r"^# --- ", re.MULTILINE)


def _tutorial() -> str:
    return TUTORIAL.read_text(encoding="utf-8")


def _code_blocks(text: str, language: str) -> list[list[str]]:
    """The dedented body lines of every ``code-block:: <language>`` on the page."""
    blocks: list[list[str]] = []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.strip() != f".. code-block:: {language}":
            continue
        body: list[str] = []
        for following in lines[index + 1 :]:
            if following.strip() and not following.startswith(" "):
                break
            body.append(following)
        blocks.append(textwrap.dedent("\n".join(body)).strip("\n").splitlines())
    return blocks


def _first_block_starting(language: str, first_line: str) -> list[str]:
    for block in _code_blocks(_tutorial(), language):
        if block and block[0] == first_line:
            return block
    raise AssertionError(f"the tutorial has no {language} block starting {first_line!r}")


@pytest.fixture(scope="module")
def initialized(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    """A fresh hello-world repo, and what ``osprey init`` printed making it."""
    repo = tmp_path_factory.mktemp("tutorial") / "my-first-agent"
    result = CliRunner().invoke(
        init,
        [str(repo), "--preset", "hello-world", "--no-git"],
        env={"ANTHROPIC_API_KEY": "sk-ant-test"},
    )
    assert result.exit_code == 0, result.output
    return repo, result.output


@pytest.mark.parametrize("first_line", ["hooks:", "config:"])
def test_profile_excerpt_lines_are_preset_lines_in_order(first_line: str) -> None:
    excerpt = _first_block_starting("yaml", first_line)
    preset = PRESET.read_text(encoding="utf-8").splitlines()
    position = 0
    for line in excerpt:
        if line.strip() in ELISIONS:
            continue
        try:
            position = preset.index(line, position) + 1
        except ValueError:
            pytest.fail(f"excerpt line is not a preset line after line {position}: {line!r}")


def test_the_hook_counts_the_page_names_are_the_profiles(
    initialized: tuple[Path, str],
) -> None:
    repo, _ = initialized
    hooks = yaml.safe_load((repo / "profile.yml").read_text(encoding="utf-8"))["hooks"]
    names = {f"hooks/{hook}" for hook in hooks}
    catalog = BuildArtifactCatalog.default()
    helpers = [
        name
        for name in catalog.closure(names) - names
        if (artifact := catalog.get(name)) is not None and artifact.output_path.endswith(".py")
    ]
    text = _tutorial()
    # The opening paragraph and Step 2 each say how many hooks the profile declares.
    named_hooks = re.findall(
        r"\b(\w+)\s+safety\s+and\s+bookkeeping\s+hooks\b|library —\s+(\w+)\s+hooks\b", text
    )
    assert len(named_hooks) == 2, named_hooks
    stated = {WORDS.get(word, -1) for pair in named_hooks for word in pair if word}
    assert stated == {len(hooks)}
    helper_counts = {
        WORDS.get(word, -1) for word in re.findall(r"\b(\w+)\s+helper\s+scripts\b", text)
    }
    assert helper_counts == {len(helpers)}


def test_the_commented_block_count_is_the_profiles(initialized: tuple[Path, str]) -> None:
    repo, _ = initialized
    emitted = len(BLOCK_BANNER.findall((repo / "profile.yml").read_text(encoding="utf-8")))
    text = _tutorial()
    named = re.findall(
        r"exactly\s+\*\*(\d+)\*\*\s+such\s+blocks|those\s+(\d+)\s+commented\s+blocks", text
    )
    # Step 2 and Next Steps each give the count.
    assert len(named) == 2, named
    stated = {int(number) for pair in named for number in pair if number}
    assert stated == {emitted}


def test_the_init_summary_rows_are_what_init_prints(initialized: tuple[Path, str]) -> None:
    _, output = initialized
    printed = {line.strip() for line in output.splitlines()}
    summary = _first_block_starting("text", "✓ Created my-first-agent")
    rows = [line.strip() for line in summary if line.strip()]
    missing = [row for row in rows if row not in printed]
    assert missing == []
