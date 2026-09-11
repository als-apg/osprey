"""Theme-lab prose names themes, families and token documents that exist.

The export is written to be read on its own — a proposer implements a theme from
it without opening the lab — so a family or document it names has to be real.
Nothing checks that by running the lab: the markdown is a string, every name in
it is prose, and a family that was renamed or never shipped reads exactly like
one that did. This scans the source for the names it hands out and holds each
against the token tree.
"""

import json
import re
from pathlib import Path

import osprey.interfaces.design_system as design_system_pkg

_DESIGN_SYSTEM = Path(design_system_pkg.__file__).parent
_THEME_LAB_JS = _DESIGN_SYSTEM / "static" / "js" / "theme-lab.js"
_TOKENS = _DESIGN_SYSTEM / "tokens"
_TOKENS_CSS = _DESIGN_SYSTEM / "static" / "css" / "tokens.css"

#: A backticked span with no whitespace — the shape a name is written in.
_BACKTICKED_RE = re.compile(r"`([^`\s]+)`")

#: A bare lowercase identifier: no dot, so `accent.base` and `tokens.css` are
#: not candidates, and no slash.
_BARE_NAME_RE = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")

#: Backticked bare words in theme-lab.js that name neither a theme nor a family:
#: JS globals, CSS function names, and the vocabulary the prose uses for its own
#: concepts. Checked in BOTH directions — a word that becomes a family name has
#: to leave this set, and a word that leaves the file has to leave it too — so
#: the set cannot quietly grow into a blanket exemption.
_NON_THEME_WORDS = frozenset(
    {
        "document",
        "hover",
        "hsl",
        "lightness",
        "null",
        "on",
        "reason",
        "rgba",
        "value",
        "window",
    }
)


def _shipped() -> tuple[set[str], set[str]]:
    """``(theme ids, family ids)`` as ``tokens/themes/*.json`` declares them."""
    ids: set[str] = set()
    families: set[str] = set()
    for path in sorted((_TOKENS / "themes").glob("*.json")):
        extensions = json.loads(path.read_text(encoding="utf-8"))["$extensions"]
        ids.add(extensions["id"])
        families.add(extensions["family"])
    return ids, families


def _emitted_property_names() -> set[str]:
    """Custom-property names the generator writes, without the ``--`` prefix.

    The prose quotes a few of these (``wt-accent-system-tint-04``), and they are
    hyphenated bare words like a family id is.
    """
    return set(re.findall(r"--([\w-]+)\s*:", _TOKENS_CSS.read_text(encoding="utf-8")))


def _backticked() -> list[str]:
    return _BACKTICKED_RE.findall(_THEME_LAB_JS.read_text(encoding="utf-8"))


def _resolves_under_tokens(token: str) -> bool:
    """Whether a backticked `*.json` reference names something in ``tokens/``.

    Three spellings all appear in the prose and all have to work: a bare file
    name (``core.json``), a path with or without the ``tokens/`` prefix
    (``themes/dark.json``), and a glob (``tokens/interfaces/*.json``).
    """
    relative = token[len("tokens/") :] if token.startswith("tokens/") else token
    if "*" in relative:
        return any(_TOKENS.glob(relative))
    if "/" in relative:
        return (_TOKENS / relative).is_file()
    return any(candidate.name == relative for candidate in _TOKENS.rglob("*.json"))


def test_every_token_document_named_exists() -> None:
    """A `*.json` the export tells a proposer to edit is a real file."""
    documents = {token for token in _backticked() if token.endswith(".json")}
    assert documents, "theme-lab.js names no token document at all"
    missing = sorted(token for token in documents if not _resolves_under_tokens(token))
    assert not missing, f"theme-lab.js names token documents that do not exist: {missing}"


def test_every_theme_or_family_named_is_shipped() -> None:
    """No prose names a theme family the token tree does not have.

    The failure this catches is silent and durable: a renamed or never-shipped
    family reads as authoritative in an exported proposal, and the proposer
    follows it into a change set that cannot land.
    """
    ids, families = _shipped()
    exempt = _NON_THEME_WORDS | _emitted_property_names()

    unknown = sorted(
        {
            token
            for token in _backticked()
            if _BARE_NAME_RE.match(token) and token not in exempt and token not in ids | families
        }
    )
    assert not unknown, (
        "theme-lab.js names themes/families that are not in tokens/themes/: "
        f"{unknown} — shipped ids {sorted(ids)}, families {sorted(families)}"
    )


def test_the_non_theme_word_list_has_no_stale_entries() -> None:
    """Every exemption still earns its place in the file."""
    present = set(_backticked())
    stale = sorted(_NON_THEME_WORDS - present)
    assert not stale, f"words exempted but no longer backticked in theme-lab.js: {stale}"
