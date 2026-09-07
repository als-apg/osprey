"""Guard: nothing OSPREY ships names a real person, site or institution.

An operator reads the framework's own prose — an agent description, a log line,
a `--help` example, a comment in a rendered compose file — as a statement about
*their* deployment. When that prose carries the name of whoever happened to
write it, it is wrong everywhere else, and it is wrong in the one place a
deployer cannot edit: files that `osprey init --force` and `profile expand`
regenerate.

The rule this pins is therefore not "fewer names" but "no identities": shipped
text under ``src/osprey/`` and ``docs/source/`` names no real person, mailbox,
institution or site. Magnitudes stay — "~135,000 documents", "~41 minutes" —
because they are facts about the software's behaviour rather than about whose
machine produced them. Examples use the placeholder cast the roster
documentation already uses: ``alice`` and ``carol`` at ``example.org``, a
facility called *Example Research Facility*, a ``simulation`` gateway, a
project at ``~/my-assistant``.

Removing a literal is a one-line edit; keeping it removed is what needs a
guard, because every new pull request is an opportunity to add one back and
nothing else would notice.

**The list grows.** It starts at what has actually been swept out of the tree,
because a pattern that fires on the current tree is not a guard, it is a
failing test. As each remaining literal is removed, its pattern joins
:data:`DENIED` in the same change that removes it: ``lbl.gov``,
``\\bALS\\b`` and ``ALS-U`` outside the simulation and virtual-accelerator
packages, ``BELLA``, ``GEECS``.

Three kinds of surface legitimately name an institution and will need an
``allow`` entry rather than an edit when their patterns land: the shipped
provider adapters for named LLM gateways, the named ingestion adapter, and the
packaging and escalation metadata that has to spell the upstream project's own
``owner/repo``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Everything OSPREY ships that a deployer or operator can read. Code and
#: comments as much as documentation: a rendered template's comment reaches a
#: deployment, and a docstring reaches the API reference.
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
    ".js",
    ".html",
    ".css",
)


@dataclass(frozen=True)
class Denied:
    """One identity the shipped tree may not name."""

    name: str
    """How the guard's failure message calls it."""

    pattern: re.Pattern[str]
    """What a reintroduction looks like."""

    why: str
    """What is wrong with shipping it, in one clause."""

    sample: str
    """A line this pattern must match, so a broken pattern cannot read as a
    clean tree."""

    allow: frozenset[str] = frozenset()
    """Repo-relative paths that may keep it, each for a stated reason."""


#: The identities already swept out of ``src/osprey`` and ``docs/source``.
#: Each entry is here because the tree is clean of it *now*; see the module
#: docstring for the ones still to come.
DENIED: tuple[Denied, ...] = (
    Denied(
        name="maintainer account",
        pattern=re.compile(r"thellert", re.IGNORECASE),
        why="a maintainer's own login and mailbox is not an example anyone can copy",
        sample='  # bare username, e.g. "thellert"',
    ),
    Denied(
        name="site EPICS gateway host",
        pattern=re.compile(r"cagw-alsdmz|pvgatemain1", re.IGNORECASE),
        why=(
            "a gateway address is one site's own infrastructure — unreachable "
            "everywhere else, and read as this deployment's own machine when it "
            "ships in an example"
        ),
        sample="  epics_gateway: cagw-alsdmz.example-site.org:5064",
    ),
)


def _shipped_sources() -> list[Path]:
    files: list[Path] = []
    for root in ROOTS:
        base = _REPO_ROOT / root
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix not in SCAN_SUFFIXES:
                continue
            if "__pycache__" in path.parts:
                continue
            files.append(path)
    return files


def _hits(denied: Denied) -> list[tuple[str, int, str]]:
    """Every ``(repo-relative path, line number, stripped line)`` naming it."""
    found: list[tuple[str, int, str]] = []
    for path in _shipped_sources():
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:  # pragma: no cover - defensive
            continue
        if not denied.pattern.search(content):
            continue
        relative = str(path.relative_to(_REPO_ROOT))
        for number, line in enumerate(content.splitlines(), start=1):
            if denied.pattern.search(line):
                found.append((relative, number, line.strip()))
    return found


@pytest.mark.parametrize("denied", DENIED, ids=lambda d: d.name)
def test_no_shipped_text_names_a_real_identity(denied: Denied) -> None:
    offenders = [hit for hit in _hits(denied) if hit[0] not in denied.allow]
    assert offenders == [], (
        f"shipped text names {denied.name} — {denied.why}. "
        f"Use the placeholder cast (alice/carol@example.org, Example Research "
        f"Facility, ~/my-assistant, simulation) instead:\n"
        + "\n".join(f"{path}:{number}: {line}" for path, number, line in offenders)
    )


@pytest.mark.parametrize("denied", DENIED, ids=lambda d: d.name)
def test_every_exemption_still_has_something_to_explain(denied: Denied) -> None:
    """An exemption whose file stopped carrying the literal should be deleted.

    Otherwise the allowlist quietly becomes a list of files the guard does not
    cover, which is the failure mode a guard is written to prevent.
    """
    seen = {path for path, _, _ in _hits(denied)}
    stale = sorted(denied.allow - seen)
    assert stale == [], f"{denied.name}: exemptions with no remaining occurrence: {stale}"


@pytest.mark.parametrize("denied", DENIED, ids=lambda d: d.name)
def test_the_sweep_would_catch_a_regression(
    denied: Denied, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sweep that finds nothing reads exactly like one whose pattern is dead.

    Each entry carries the line it must trip on, so a pattern that is edited
    into uselessness fails here rather than reporting a clean tree forever.
    """
    fake_root = tmp_path / "repo"
    (fake_root / "src" / "osprey").mkdir(parents=True)
    (fake_root / "src" / "osprey" / "regress.py").write_text(
        f"# {denied.sample}\n", encoding="utf-8"
    )
    monkeypatch.setattr(f"{__name__}._REPO_ROOT", fake_root, raising=True)

    with pytest.raises(AssertionError, match="shipped text names"):
        test_no_shipped_text_names_a_real_identity(denied)


def test_the_placeholder_cast_is_not_swept() -> None:
    """The names the rule tells an author to use must survive the rule.

    A guard that condemned its own replacement text would push the next author
    back to a real one.
    """
    placeholders = (
        "alice@example.org",
        "carol@example.org",
        "Example Research Facility",
        "~/my-assistant",
        "epics_gateway=simulation",
    )
    for denied in DENIED:
        for placeholder in placeholders:
            assert not denied.pattern.search(placeholder), (
                f"{denied.name} condemns the placeholder {placeholder!r}"
            )


def test_the_sweep_reaches_shipped_templates_and_docs() -> None:
    """The guard has to cover the surfaces the literals actually lived in.

    A rendered template's comment and a how-to page are exactly where an
    identity survives review, and both would be missed by a rule written for
    ``.py`` alone.
    """
    scanned = {str(path.relative_to(_REPO_ROOT)) for path in _shipped_sources()}
    for required in (
        "src/osprey/templates/modules/web_terminals/docker-compose.web.yml.j2",
        "src/osprey/profiles/presets/control-assistant.yml",
        "docs/source/how-to/web-terminal/multi-user/login.rst",
    ):
        assert required in scanned, f"{required} is shipped but the sweep cannot see it"
