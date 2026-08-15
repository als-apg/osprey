"""Removal gate: nothing shipped still calls the web tier's env file ``.env.production``.

The env file every per-user web-terminal container runs with is ``.env.users``,
and the verb that renders it is ``osprey users env``. Both names changed at once,
so both old spellings are swept by one pattern: ``.env.production`` (the
artifact) and ``env-production`` (the verb).

The sweep covers ``src/osprey/`` — code, comments, shipped templates and the
emitted CI pipeline, all of which reach an operator — and ``docs/source/``.
``CHANGELOG.md`` is out of scope by design: it records the rename, and rewriting
released history to satisfy a grep would be worse than the grep failing.

Underscore spellings (``env_production.py``, ``ensure_env_production``) are
Python identifiers, not names an operator ever types or a deploy ever writes.
They are deliberately not swept, and the pattern's ``[.-]`` is what keeps them
out of it.

The counterpart in ``tests/cli/test_lifecycle_invariants.py`` sweeps the verb
alone with the retired-spelling machinery. Neither subsumes the other: that one
covers ``src/`` only, this one is the half that reaches the docs tree.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

#: The artifact and the verb, in one pattern. ``[.-]`` and not ``.``: an
#: unescaped dot would condemn ``env_production.py`` and every function named
#: after it, which are live Python names rather than retired operator-facing
#: ones.
RETIRED_NAME = re.compile(r"env[.-]production")

ROOTS = ("src/osprey", "docs/source")
SCAN_SUFFIXES = (
    ".py",
    ".md",
    ".rst",
    ".txt",
    ".j2",
    ".yml",
    ".yaml",
    ".json",
    ".toml",
    ".sh",
)

#: Shipped files that carry no extension at all, admitted by name instead.
#: An extension-only rule cannot see them — and the two files this rename was
#: most likely to be reverted in are both in that class: the auth sidecar's
#: ``.dockerignore``, which listed the old name outright, and the project
#: ``gitignore`` template, which explains in a comment which env files carry
#: secrets. ``Dockerfile`` joins them because a ``COPY`` of the old name would
#: bake a stale artifact into an image.
#:
#: Matched with any leading dot stripped, because the same kind of file ships
#: both ways: ``templates/project/gitignore`` has no dot (it is renamed on
#: emit) while ``auth_sidecar/.dockerignore`` keeps one.
SCAN_STEMS = frozenset({"gitignore", "dockerignore", "Dockerfile"})

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: The only occurrences allowed to remain, keyed by ``(repo-relative path,
#: exact stripped line)`` and not by file alone — both files below also hold
#: lines that WERE renamed, so a file-level exemption would stop the sweep
#: applying to them at all.
#:
#: Every entry is the one-way migration that carries a pre-rename deployment's
#: secrets onto the current name. It has to name the file it is moving; a
#: migration that could not spell the old name would have nothing to find.
_ALLOWLIST: dict[tuple[str, str], str] = {
    (
        "src/osprey/deployment/web_terminals/env_production.py",
        'LEGACY_USERS_ENV_FILENAME = ".env.production"',
    ): "the legacy name the migration moves FROM",
    (
        "src/osprey/deployment/web_terminals/env_production.py",
        '"""Move a leftover ``.env.production`` onto :data:`USERS_ENV_FILENAME`.',
    ): "names the file that migration moves",
    (
        "src/osprey/deployment/container_lifecycle.py",
        "# Carry a pre-rename .env.production onto .env.users before anything reads",
    ): "names the file the `up` path migrates",
    (
        "docs/source/how-to/deploy-project.rst",
        "mv .env.production .env.users",
    ): "the manual rename an upgrading operator runs when `up` has not done it yet",
}


def _shipped_sources() -> list[Path]:
    files: list[Path] = []
    for root in ROOTS:
        base = _REPO_ROOT / root
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix not in SCAN_SUFFIXES and path.name.lstrip(".") not in SCAN_STEMS:
                continue
            if "__pycache__" in path.parts:
                continue
            files.append(path)
    return files


def _hits() -> list[tuple[str, int, str]]:
    """Every ``(repo-relative path, line number, stripped line)`` still naming it."""
    found: list[tuple[str, int, str]] = []
    for path in _shipped_sources():
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:  # pragma: no cover - defensive, no such file today
            continue
        if not RETIRED_NAME.search(content):
            continue
        relative = str(path.relative_to(_REPO_ROOT))
        for number, line in enumerate(content.splitlines(), start=1):
            if RETIRED_NAME.search(line):
                found.append((relative, number, line.strip()))
    return found


def test_no_shipped_text_still_names_the_old_env_file_or_verb() -> None:
    offenders = [hit for hit in _hits() if (hit[0], hit[2]) not in _ALLOWLIST]
    assert offenders == [], (
        "The web tier's env file is `.env.users` and the verb that renders it is "
        "`osprey users env`. Old spellings remain:\n"
        + "\n".join(f"{path}:{number}: {line}" for path, number, line in offenders)
    )


def test_every_exemption_still_has_something_to_explain() -> None:
    """A migration line that stops naming the old file should leave the list.

    Exact-line keys make this sharp: an allowlisted line that was edited — even
    to fix a typo — stops matching and is reported, so the exemption is
    re-argued rather than inherited.
    """
    seen = {(path, line) for path, _, line in _hits()}
    stale = sorted(set(_ALLOWLIST) - seen)
    assert stale == [], f"allowlist entries with no remaining occurrence: {stale}"


def test_the_extensionless_shipped_files_are_swept() -> None:
    """The guard has to cover its own deliverable.

    Both files below named the old spelling before this rename, and neither
    carries an extension. An admission rule written on suffixes alone sweeps
    every ``.py`` and ``.rst`` in the tree while leaving out precisely the
    files a reintroduction is most likely to land in — and reports a clean
    result either way.
    """
    scanned = {str(path.relative_to(_REPO_ROOT)) for path in _shipped_sources()}
    for required in (
        "src/osprey/templates/modules/web_terminals/auth_sidecar/.dockerignore",
        "src/osprey/templates/project/gitignore",
    ):
        assert required in scanned, f"{required} is shipped but the sweep cannot see it"


@pytest.mark.parametrize(
    "name,text",
    (
        ("regress.rst", "   osprey users env-production --output .env.users\n"),
        ("regress.rst", "   the file lands at ``.env.production`` beside ``.env``\n"),
        (".dockerignore", ".env.production\n"),
        ("gitignore", "# secrets, including the generated .env.production\n"),
    ),
)
def test_the_sweep_would_catch_a_regression(
    name: str, text: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sweep that finds nothing looks identical to one whose pattern is broken.

    Both spellings, because the pattern carries an alternation and an
    alternation can lose a branch — and both extensionless spellings, because
    the admission rule has two halves and the stem half is the newer one.
    """
    fake_root = tmp_path / "repo"
    (fake_root / "docs" / "source").mkdir(parents=True)
    (fake_root / "docs" / "source" / name).write_text(text, encoding="utf-8")
    monkeypatch.setattr(f"{__name__}._REPO_ROOT", fake_root, raising=True)

    with pytest.raises(AssertionError, match="Old spellings remain"):
        test_no_shipped_text_still_names_the_old_env_file_or_verb()


def test_the_underscore_spelling_is_not_swept() -> None:
    """The Python module and its functions keep their names; the sweep must let them.

    A pattern written with a bare ``.`` would match ``env_production.py`` and
    report the module every reader of this criterion is meant to keep.
    """
    assert not RETIRED_NAME.search("from ...env_production import ensure_env_production")
    assert RETIRED_NAME.search("osprey users env-production")
    assert RETIRED_NAME.search(".env.production")
