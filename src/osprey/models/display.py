"""Model names as a person reads them, and the Claude family an id belongs to.

A wire id (``claude-haiku-4-5-20251001``) is what a gateway serves and what
configuration names. A person reads a shorter name: the family and version with
no vendor in front (``Haiku 4.5``), the way nobody calls GPT-6 "OpenAI-GPT-6".
Any id outside the four Claude families is shown as itself, minus a
``vendor/`` routing prefix. Display only: nothing here reaches a wire.

The same parser answers which Claude family and version an id is, which is how
the build fills Claude Code's own ``haiku``/``sonnet``/``opus`` alias names from
the ids a gateway serves (:func:`claude_code_alias_candidates`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Claude Code's own alias names. Claude Code reads one model env var per name
#: and asks for its ``haiku`` for background work; the names are its contract.
CLAUDE_CODE_ALIASES: tuple[str, ...] = ("haiku", "sonnet", "opus")

_FAMILY_RE = re.compile(r"^claude-?(?P<family>fable|opus|sonnet|haiku)(?P<rest>.*)$")
_REGION_PREFIX_RE = re.compile(r"^(?:[a-z]{2}\.)?anthropic\.")
_DATE_RE = re.compile(r"^\d{8}$")
_REVISION_RE = re.compile(r"^v\d+$")


@dataclass(frozen=True)
class _ClaudeId:
    family: str
    version: tuple[int, ...]
    qualifiers: tuple[str, ...]


def _strip_vendor(model_id: str) -> str:
    """Drop one leading ``vendor/`` segment and a Bedrock ``us.anthropic.`` prefix."""
    bare = model_id.split("/", 1)[1] if "/" in model_id else model_id
    return _REGION_PREFIX_RE.sub("", bare)


def _parse(model_id: str) -> _ClaudeId | None:
    """Family, version and trailing qualifiers of a Claude id; ``None`` otherwise.

    The version is the run of numbers after the family, stopping at an 8-digit
    release date or a ``-v1`` revision suffix. Argo's compact spelling carries
    its version as one digit run (``claudeopus41``), read one digit per part.
    """
    match = _FAMILY_RE.match(_strip_vendor(model_id))
    if match is None:
        return None
    rest = match.group("rest")
    version: list[int] = []
    compact = re.match(r"\d+", rest)
    if compact:
        version = [int(digit) for digit in compact.group()]
        rest = rest[compact.end() :]
    qualifiers: list[str] = []
    for token in (t for t in re.split(r"[-.]", rest) if t):
        if _DATE_RE.match(token) or _REVISION_RE.match(token):
            continue
        if token.isdigit() and not qualifiers:
            version.append(int(token))
        else:
            qualifiers.append(token)
    return _ClaudeId(match.group("family"), tuple(version), tuple(qualifiers))


def display_model_name(model_id: str) -> str:
    """The name a person reads for ``model_id``.

    ``claude-sonnet-5`` → ``Sonnet 5``, ``claude-haiku-4-5-20251001`` →
    ``Haiku 4.5``, ``claudeopus41`` → ``Opus 4.1``; ``gpt-6-sol`` stays
    ``gpt-6-sol`` and ``ollama/gpt-oss:20b`` reads ``gpt-oss:20b``.
    """
    parsed = _parse(model_id)
    if parsed is None:
        return _strip_vendor(model_id)
    parts = [parsed.family.capitalize()]
    if parsed.version:
        parts.append(".".join(str(n) for n in parsed.version))
    parts.extend(parsed.qualifiers)
    return " ".join(parts)


def claude_code_alias_candidates(model_ids: list[str] | tuple[str, ...]) -> dict[str, str]:
    """The served id each Claude Code alias name would pick, by family.

    For every alias name, the ids of that family compete: the highest version
    wins, an id with no trailing qualifier beats one with, and a remaining tie
    goes to the id that sorts first. Aliases with no candidate are absent, so a
    gateway serving no Claude model yields ``{}``.
    """
    best: dict[str, tuple[tuple[tuple[int, ...], bool], str]] = {}
    for model_id in sorted(model_ids):
        parsed = _parse(model_id)
        if parsed is None or parsed.family not in CLAUDE_CODE_ALIASES:
            continue
        rank = (parsed.version, not parsed.qualifiers)
        current = best.get(parsed.family)
        if current is None or rank > current[0]:
            best[parsed.family] = (rank, model_id)
    return {alias: best[alias][1] for alias in CLAUDE_CODE_ALIASES if alias in best}
