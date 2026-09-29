"""Retired symbols, paths and environment names never come back.

``RETIRED`` maps each retired token to the stage that deleted its LAST producer
or reader; a token joins the map in that stage's close. Once its stage is at or
before ``CURRENT_BATCH``, the token must not appear in any tracked text file
under ``SCAN_ROOTS``. A token whose stage is still ahead is not enforced yet.

The map holds symbols, dotted module paths, file paths and environment names
only. Retired CLI verb spellings live in ``_RETIRED_SPELLINGS``
(``tests/cli/test_lifecycle_invariants.py``) and retired configuration keys in
the ``deleted:`` section of ``src/osprey/profiles/config_key_manifest.yml``.

History keeps the names on purpose: ``changelog.d/`` and ``CHANGELOG.md`` are
never swept. ``SELF_EXEMPT`` names the files that spell retired tokens as data
(this guard and its sibling guards, and the key manifest); every entry must be a
tracked file, so a stale exemption fails the guard.

A token matches as a whole name: a letter, digit or underscore on either side
means a different, longer name.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path

import pytest

from tests.facility._batches import BATCHES, CURRENT_BATCH

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Tracked trees the sweep reads: tests (``tests/e2e`` included), scripts, docs,
#: the rendered templates and the deployment code and images.
SCAN_ROOTS: tuple[str, ...] = (
    "docker",
    "docs",
    "scripts",
    "src/osprey/deployment",
    "src/osprey/templates",
    "tests",
)

#: History records that the retired names existed and when they went.
EXCLUDED: tuple[str, ...] = ("changelog.d/", "CHANGELOG.md")

#: Files that spell retired tokens as data.
SELF_EXEMPT: tuple[str, ...] = (
    "src/osprey/profiles/config_key_manifest.yml",
    "tests/docs/test_environment_variable_page.py",
    "tests/docs/test_mml_converter_retired.py",
    "tests/facility/test_no_retired_shapes.py",
)

#: Retired token -> the stage that deleted its last producer or reader.
RETIRED: dict[str, str] = {}

_NAME_CHAR = "A-Za-z0-9_"


def token_pattern(tokens: Iterable[str]) -> str | None:
    """Return one pattern matching any of ``tokens`` as a whole name, or None."""
    ordered = sorted(set(tokens), key=lambda token: (-len(token), token))
    if not ordered:
        return None
    alternation = "|".join(re.escape(token) for token in ordered)
    return rf"(?<![{_NAME_CHAR}])(?:{alternation})(?![{_NAME_CHAR}])"


def enforced(retired: Mapping[str, str], current: int = CURRENT_BATCH) -> set[str]:
    """Return the tokens whose stage is at or before ``current``."""
    return {
        token
        for token, stage in retired.items()
        if stage in BATCHES and BATCHES.index(stage) <= current
    }


def scan(root: Path, tokens: Iterable[str], exempt: Iterable[str] = SELF_EXEMPT) -> list[str]:
    """Return ``path:line:token`` for every enforced token in a swept file."""
    pattern = token_pattern(tokens)
    if pattern is None:
        return []
    pathspec = [
        *SCAN_ROOTS,
        *(f":(exclude){path}" for path in EXCLUDED),
        *(f":(exclude){path}" for path in exempt),
    ]
    result = subprocess.run(
        ["git", "grep", "-I", "-n", "-z", "-P", "-e", pattern, "--", *pathspec],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode in (0, 1), result.stderr
    compiled = re.compile(pattern)
    offenders: list[str] = []
    for line in result.stdout.split("\n"):
        if not line:
            continue
        path, lineno, content = line.split("\0", 2)
        offenders.extend(f"{path}:{lineno}:{hit}" for hit in compiled.findall(content))
    return sorted(offenders)


def stage_problems(retired: Mapping[str, str]) -> list[str]:
    """Return one line per entry whose tag names no stage."""
    return sorted(
        f"{token}: tag {stage!r} names no stage"
        for token, stage in retired.items()
        if stage not in BATCHES
    )


def stale_exemptions(root: Path, exempt: Iterable[str] = SELF_EXEMPT) -> list[str]:
    """Return the exempt paths that are not tracked files under ``root``."""
    tracked = set(
        subprocess.run(
            ["git", "ls-files", "--", *exempt],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
    )
    return sorted(path for path in exempt if path not in tracked)


# --- the pattern ------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "from .pairing import assign_readbacks",
        "osprey.channel_roster.database",
        "src/osprey/channel_roster/pairing.py",
        "os.environ['OLD_ENV_NAME']",
    ],
)
def test_pattern_finds_a_whole_name(text: str) -> None:
    pattern = token_pattern(
        ["assign_readbacks", "osprey.channel_roster.database", "pairing.py", "OLD_ENV_NAME"]
    )
    assert pattern is not None
    assert re.search(pattern, text) is not None


@pytest.mark.parametrize(
    "text",
    ["assign_readbacks_v2", "_assign_readbacks", "OLD_ENV_NAME_SUFFIX", "repairing.py"],
)
def test_pattern_leaves_longer_names(text: str) -> None:
    pattern = token_pattern(["assign_readbacks", "OLD_ENV_NAME", "pairing.py"])
    assert pattern is not None
    assert re.search(pattern, text) is None


def test_an_empty_map_sweeps_nothing() -> None:
    assert token_pattern([]) is None
    assert scan(REPO_ROOT, []) == []


# --- the map and the exemptions ---------------------------------------------------


def test_every_tag_names_a_stage() -> None:
    assert stage_problems(RETIRED) == []


def test_map_is_sorted() -> None:
    assert list(RETIRED) == sorted(RETIRED)


def test_exemptions_are_sorted() -> None:
    assert list(SELF_EXEMPT) == sorted(SELF_EXEMPT)


def test_every_exemption_is_a_tracked_file() -> None:
    assert stale_exemptions(REPO_ROOT) == []


def test_only_reached_stages_are_enforced() -> None:
    retired = {
        "due_name": BATCHES[CURRENT_BATCH],
        "old_name": BATCHES[0],
        "later_name": BATCHES[-1],
        "unknown_name": "0z",
    }
    assert enforced(retired) == {"due_name", "old_name"}
    assert stage_problems(retired) == ["unknown_name: tag '0z' names no stage"]


# --- the sweep --------------------------------------------------------------------


def _planted_repo(root: Path, files: Mapping[str, str | bytes]) -> Path:
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    return root


def test_planted_token_is_flagged_where_swept(tmp_path: Path) -> None:
    root = _planted_repo(
        tmp_path,
        {
            "tests/e2e/test_planted.py": "from x import retired_name\n",
            "docs/source/page.rst": "Set ``RETIRED_ENV``.\n",
            "src/osprey/templates/app.j2": "{{ retired_name }}\n",
            "src/osprey/cli/main.py": "retired_name = 1\n",
            "changelog.d/1.removed.md": "Removed retired_name.\n",
            "CHANGELOG.md": "retired_name\n",
            "tests/facility/test_no_retired_shapes.py": "RETIRED = {'retired_name': '1b'}\n",
            "tests/data/blob.bin": b"\x00retired_name\x00",
            "scripts/tool.py": "retired_name_v2 = 1\n",
        },
    )
    assert scan(root, ["retired_name", "RETIRED_ENV"]) == [
        "docs/source/page.rst:1:RETIRED_ENV",
        "src/osprey/templates/app.j2:1:retired_name",
        "tests/e2e/test_planted.py:1:retired_name",
    ]


def test_a_stale_exemption_is_refused(tmp_path: Path) -> None:
    root = _planted_repo(tmp_path, {"tests/kept.py": "\n"})
    assert stale_exemptions(root, ("tests/gone.py", "tests/kept.py")) == ["tests/gone.py"]


# --- the tree ---------------------------------------------------------------------


def test_no_retired_token_survives() -> None:
    offenders = scan(REPO_ROOT, enforced(RETIRED))
    assert offenders == [], "retired tokens still present:\n" + "\n".join(offenders)
