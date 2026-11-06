"""The agent sessions a scenario narrates: what an operator asked, and what was done.

A facility scenario's ``sessions`` block states past conversations with the
deployment's agent — the prompts, the tool calls and their results, the answers
and the artifacts each turn left behind — so a fresh deployment opens with a
history to look back on. This module owns that block's shape:

* :func:`parse_sessions` turns the YAML list into frozen
  :class:`ScenarioSession` records and refuses a malformed block with a
  :class:`ValueError` naming the offending path.
* :func:`resolve` places every session, turn and artifact at an aware instant
  against an anchor, through :func:`~osprey_connectors.relative_time.resolve_relative_timestamp`
  (the logbook's ``{days_ago, time}`` grammar).
* :func:`to_dict` / :func:`from_dict` carry a :class:`ResolvedSession` through
  the JSON seed file losslessly.

A session is always in the past: its ``when`` is at least one day back, and its
turns (``when`` plus each turn's ``at_offset_s``) never run past that day's
midnight, because the resolver pins a time of day and does not clamp.

Tools are named by ``{server, name}`` — the OSPREY MCP server and its tool —
never by a client's spelling of them (``mcp__<server>__<name>``). A turn lists
every channel it names in ``channels:`` and every logbook id it cites in
``logbook_ids:``; for the ``controls`` server's channel-taking tools the channels
in a step's input are cross-checked against that list here, so the list is the
one place the facility check (the scenario validator) has to look.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from datetime import time as dtime
from typing import Any

from osprey_connectors.relative_time import RelativeTimestamp, resolve_relative_timestamp

__all__ = [
    "ARTIFACT_FORMS",
    "AnswerStep",
    "ArtifactSpec",
    "ResolvedArtifact",
    "ResolvedPart",
    "ResolvedSession",
    "ResolvedTurn",
    "ScenarioSession",
    "SessionPart",
    "SessionTurn",
    "ToolStep",
    "from_dict",
    "parse_sessions",
    "resolve",
    "to_dict",
]

# Closed key sets, one per level of the block.
_SESSION_KEYS = ("slug", "title", "when", "parts")
_WHEN_KEYS = ("days_ago", "time")
_PART_KEYS = ("title", "turns")
_TURN_KEYS = ("prompt", "at_offset_s", "channels", "logbook_ids", "steps", "artifacts")
_TOOL_STEP_KEYS = ("tool", "input", "result", "approved")
_TOOL_REF_KEYS = ("server", "name")
_ANSWER_STEP_KEYS = ("answer",)
_ARTIFACT_KEYS = ("slug", "form", "title", "source")

ARTIFACT_FORMS = ("plot", "note", "data")

# Slugs feed stable ids (``<identity>/<scenario>/<slug>/...``), so they stay to
# one lower-case, path-safe spelling.
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")

# The ``controls`` server's channel-taking tools: input key holding a list of
# channel names, or (for ``channel_write``) a list of operations each naming one.
_CONTROLS_SERVER = "controls"
_CHANNEL_LIST_TOOLS = ("channel_read", "archiver_read", "channel_limits")
_CHANNEL_WRITE_TOOL = "channel_write"

_SECONDS_PER_DAY = 86400


@dataclass(frozen=True)
class ToolStep:
    """One tool call of a turn and the result it returned.

    Attributes:
        server: The OSPREY MCP server the tool belongs to (``controls``).
        name: The tool's name on that server (``channel_write``).
        input: The call's arguments.
        result: What the tool returned: text, or a JSON value.
        approved: Whether the operator approved the call (a write).
    """

    server: str
    name: str
    input: dict[str, Any]
    result: Any
    approved: bool = False


@dataclass(frozen=True)
class AnswerStep:
    """The agent's prose reply within a turn."""

    answer: str


@dataclass(frozen=True)
class ArtifactSpec:
    """An artifact a turn produced.

    Attributes:
        slug: Stable name of the artifact within its session.
        form: ``plot``, ``note`` or ``data``.
        title: Title the artifact is stored under.
        source: Where its content comes from, relative to the scenario.
    """

    slug: str
    form: str
    title: str
    source: str


@dataclass(frozen=True)
class SessionTurn:
    """One operator prompt and everything the agent did in answer.

    Attributes:
        prompt: What the operator typed.
        at_offset_s: Seconds after the session's ``when`` the prompt was sent.
        channels: Every channel the turn names.
        logbook_ids: Every logbook entry id the turn cites.
        steps: Tool calls and answers, in order.
        artifacts: Artifacts the turn left behind.
    """

    prompt: str
    at_offset_s: float
    channels: tuple[str, ...]
    logbook_ids: tuple[str, ...]
    steps: tuple[ToolStep | AnswerStep, ...]
    artifacts: tuple[ArtifactSpec, ...] = ()


@dataclass(frozen=True)
class SessionPart:
    """One transcript of a session; a second part is the half after a ``/clear``."""

    title: str
    turns: tuple[SessionTurn, ...]


@dataclass(frozen=True)
class ScenarioSession:
    """A past agent session a scenario states.

    Attributes:
        slug: Stable name of the session within its scenario.
        title: Sidebar title.
        when: When the session started, relative to the anchor.
        parts: The session's transcripts, in order.
    """

    slug: str
    title: str
    when: RelativeTimestamp
    parts: tuple[SessionPart, ...]


@dataclass(frozen=True)
class ResolvedArtifact:
    """An artifact placed at the instant its turn produced it."""

    spec: ArtifactSpec
    at: datetime


@dataclass(frozen=True)
class ResolvedTurn:
    """A turn placed at the instant its prompt was sent."""

    turn: SessionTurn
    at: datetime
    artifacts: tuple[ResolvedArtifact, ...] = field(default=())


@dataclass(frozen=True)
class ResolvedPart:
    """A part whose turns are placed."""

    part: SessionPart
    turns: tuple[ResolvedTurn, ...]


@dataclass(frozen=True)
class ResolvedSession:
    """A session placed against an anchor: every instant is timezone-aware."""

    session: ScenarioSession
    at: datetime
    parts: tuple[ResolvedPart, ...]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_sessions(raw: Any, prefix: str = "sessions") -> tuple[ScenarioSession, ...]:
    """Parse a scenario's ``sessions`` block into the session model.

    Args:
        raw: The decoded YAML list (``None`` reads as no sessions).
        prefix: Location prefix every error message starts with, so a caller
            can name the scenario the block came from.

    Returns:
        The sessions, in block order.

    Raises:
        ValueError: The block is malformed; the message names the path of the
            offending value (``sessions[0].parts[1].turns[2]: ...``).
    """
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError(f"{prefix}: must be a list of sessions, got {raw!r}")
    sessions: list[ScenarioSession] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        where = f"{prefix}[{index}]"
        session = _parse_session(where, item)
        if session.slug in seen:
            raise ValueError(f"{where}.slug: two sessions are both named {session.slug!r}")
        seen.add(session.slug)
        sessions.append(session)
    return tuple(sessions)


def _parse_session(where: str, raw: Any) -> ScenarioSession:
    _closed(where, raw, _SESSION_KEYS, required=_SESSION_KEYS)
    slug = _slug(f"{where}.slug", raw["slug"])
    title = _text(f"{where}.title", raw["title"])
    when = _when(f"{where}.when", raw["when"])
    parts_raw = _nonempty_list(f"{where}.parts", raw["parts"])
    parts = tuple(_parse_part(f"{where}.parts[{i}]", p) for i, p in enumerate(parts_raw))
    _check_offsets(where, when, parts)
    _check_artifact_slugs(where, parts)
    return ScenarioSession(slug=slug, title=title, when=when, parts=parts)


def _parse_part(where: str, raw: Any) -> SessionPart:
    _closed(where, raw, _PART_KEYS, required=_PART_KEYS)
    title = _text(f"{where}.title", raw["title"])
    turns_raw = _nonempty_list(f"{where}.turns", raw["turns"])
    turns = tuple(_parse_turn(f"{where}.turns[{i}]", t) for i, t in enumerate(turns_raw))
    return SessionPart(title=title, turns=turns)


def _parse_turn(where: str, raw: Any) -> SessionTurn:
    _closed(where, raw, _TURN_KEYS, required=("prompt", "steps"))
    prompt = _text(f"{where}.prompt", raw["prompt"])
    offset = _offset(f"{where}.at_offset_s", raw.get("at_offset_s", 0))
    channels = _names(f"{where}.channels", raw.get("channels", []))
    logbook_ids = _names(f"{where}.logbook_ids", raw.get("logbook_ids", []))
    steps_raw = _nonempty_list(f"{where}.steps", raw["steps"])
    steps = tuple(_parse_step(f"{where}.steps[{i}]", s) for i, s in enumerate(steps_raw))
    artifacts_raw = raw.get("artifacts", [])
    if not isinstance(artifacts_raw, list):
        raise ValueError(f"{where}.artifacts: must be a list, got {artifacts_raw!r}")
    artifacts = tuple(
        _parse_artifact(f"{where}.artifacts[{i}]", a) for i, a in enumerate(artifacts_raw)
    )
    turn = SessionTurn(
        prompt=prompt,
        at_offset_s=offset,
        channels=channels,
        logbook_ids=logbook_ids,
        steps=steps,
        artifacts=artifacts,
    )
    _check_step_channels(where, turn)
    return turn


def _parse_step(where: str, raw: Any) -> ToolStep | AnswerStep:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{where}: must be a mapping, got {raw!r}")
    if "answer" in raw:
        _closed(where, raw, _ANSWER_STEP_KEYS, required=_ANSWER_STEP_KEYS)
        return AnswerStep(answer=_text(f"{where}.answer", raw["answer"]))
    if "tool" not in raw:
        raise ValueError(f"{where}: a step is either a tool call ('tool') or an 'answer'")
    _closed(where, raw, _TOOL_STEP_KEYS, required=("tool", "input", "result"))
    server, name = _tool_ref(f"{where}.tool", raw["tool"])
    tool_input = raw["input"]
    if not isinstance(tool_input, Mapping):
        raise ValueError(f"{where}.input: must be a mapping, got {tool_input!r}")
    result = raw["result"]
    if not isinstance(result, (str, Mapping, list)):
        raise ValueError(f"{where}.result: must be text, a mapping or a list, got {result!r}")
    approved = raw.get("approved", False)
    if not isinstance(approved, bool):
        raise ValueError(f"{where}.approved: must be true or false, got {approved!r}")
    return ToolStep(
        server=server, name=name, input=dict(tool_input), result=result, approved=approved
    )


def _tool_ref(where: str, raw: Any) -> tuple[str, str]:
    if not isinstance(raw, Mapping):
        raise ValueError(
            f"{where}: must be a mapping {{server, name}} naming an OSPREY server's tool, "
            f"got {raw!r}"
        )
    _closed(where, raw, _TOOL_REF_KEYS, required=_TOOL_REF_KEYS)
    server = _text(f"{where}.server", raw["server"])
    name = _text(f"{where}.name", raw["name"])
    for key, value in (("server", server), ("name", name)):
        if value.startswith("mcp__") or "__" in value:
            raise ValueError(
                f"{where}.{key}: names the OSPREY {key} itself, never a client's "
                f"'mcp__<server>__<tool>' spelling, got {value!r}"
            )
    return server, name


def _parse_artifact(where: str, raw: Any) -> ArtifactSpec:
    _closed(where, raw, _ARTIFACT_KEYS, required=_ARTIFACT_KEYS)
    slug = _slug(f"{where}.slug", raw["slug"])
    form = raw["form"]
    if form not in ARTIFACT_FORMS:
        raise ValueError(f"{where}.form: must be one of {list(ARTIFACT_FORMS)}, got {form!r}")
    return ArtifactSpec(
        slug=slug,
        form=form,
        title=_text(f"{where}.title", raw["title"]),
        source=_text(f"{where}.source", raw["source"]),
    )


# ---------------------------------------------------------------------------
# Cross-checks
# ---------------------------------------------------------------------------


def _check_step_channels(where: str, turn: SessionTurn) -> None:
    """Refuse a controls tool step naming a channel its turn's ``channels:`` lacks."""
    listed = set(turn.channels)
    for index, step in enumerate(turn.steps):
        if not isinstance(step, ToolStep) or step.server != _CONTROLS_SERVER:
            continue
        at = f"{where}.steps[{index}].input"
        for path, channel in _input_channels(at, step):
            if channel not in listed:
                raise ValueError(
                    f"{path}: channel {channel!r} is not in the turn's channels: list "
                    f"{sorted(listed)}"
                )


def _input_channels(at: str, step: ToolStep) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    if step.name in _CHANNEL_LIST_TOOLS:
        if "channels" not in step.input:
            return found
        channels = step.input["channels"]
        if not isinstance(channels, list):
            raise ValueError(f"{at}.channels: must be a list of channel names, got {channels!r}")
        for i, channel in enumerate(channels):
            if not isinstance(channel, str):
                raise ValueError(f"{at}.channels[{i}]: must be a channel name, got {channel!r}")
            found.append((f"{at}.channels[{i}]", channel))
    elif step.name == _CHANNEL_WRITE_TOOL:
        if "operations" not in step.input:
            return found
        operations = step.input["operations"]
        if not isinstance(operations, list):
            raise ValueError(f"{at}.operations: must be a list, got {operations!r}")
        for i, op in enumerate(operations):
            op_at = f"{at}.operations[{i}]"
            if not isinstance(op, Mapping) or not isinstance(op.get("channel"), str):
                raise ValueError(f"{op_at}: must be a mapping naming a 'channel', got {op!r}")
            found.append((f"{op_at}.channel", op["channel"]))
    return found


def _check_offsets(where: str, when: RelativeTimestamp, parts: tuple[SessionPart, ...]) -> None:
    """Turns run forward in time, across parts, and end before the day's midnight."""
    previous = 0.0
    start = _seconds_of_day(when.time)
    for p, part in enumerate(parts):
        for t, turn in enumerate(part.turns):
            at = f"{where}.parts[{p}].turns[{t}].at_offset_s"
            if turn.at_offset_s < previous:
                raise ValueError(
                    f"{at}: turns run forward in time; {turn.at_offset_s!r} comes before the "
                    f"previous turn's {previous!r}"
                )
            previous = turn.at_offset_s
            if start + turn.at_offset_s >= _SECONDS_PER_DAY:
                raise ValueError(
                    f"{at}: the session starts at {when.time.isoformat()} and this turn "
                    f"{turn.at_offset_s!r} s later runs past midnight; a session stays within "
                    "its day"
                )


def _check_artifact_slugs(where: str, parts: tuple[SessionPart, ...]) -> None:
    seen: set[str] = set()
    for p, part in enumerate(parts):
        for t, turn in enumerate(part.turns):
            for a, artifact in enumerate(turn.artifacts):
                if artifact.slug in seen:
                    raise ValueError(
                        f"{where}.parts[{p}].turns[{t}].artifacts[{a}].slug: two artifacts of "
                        f"the session are both named {artifact.slug!r}"
                    )
                seen.add(artifact.slug)


# ---------------------------------------------------------------------------
# Scalars
# ---------------------------------------------------------------------------


def _closed(where: str, raw: Any, keys: Sequence[str], *, required: Sequence[str]) -> None:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{where}: must be a mapping, got {raw!r}")
    unknown = sorted(set(raw) - set(keys))
    if unknown:
        raise ValueError(f"{where}: unknown keys {unknown}")
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"{where}: missing keys {missing}")


def _text(where: str, raw: Any) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{where}: must be a non-empty string, got {raw!r}")
    return raw


def _nonempty_list(where: str, raw: Any) -> list[Any]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{where}: must be a non-empty list, got {raw!r}")
    return raw


def _slug(where: str, raw: Any) -> str:
    if not isinstance(raw, str) or not _SLUG_RE.match(raw):
        raise ValueError(f"{where}: must be a lower-case slug (letters, digits, '-'), got {raw!r}")
    return raw


def _names(where: str, raw: Any) -> tuple[str, ...]:
    if not isinstance(raw, list):
        raise ValueError(f"{where}: must be a list of names, got {raw!r}")
    return tuple(_text(f"{where}[{i}]", item) for i, item in enumerate(raw))


def _offset(where: str, raw: Any) -> float:
    if (
        isinstance(raw, bool)
        or not isinstance(raw, (int, float))
        or not math.isfinite(raw)
        or raw < 0
    ):
        raise ValueError(f"{where}: must be a non-negative number of seconds, got {raw!r}")
    return float(raw) if isinstance(raw, float) else int(raw)


def _when(where: str, raw: Any) -> RelativeTimestamp:
    """The logbook's ``{days_ago, time}`` grammar, at least one day back."""
    if not isinstance(raw, Mapping):
        raise ValueError(f"{where}: must be a mapping with 'days_ago' and 'time', got {raw!r}")
    _closed(where, raw, _WHEN_KEYS, required=_WHEN_KEYS)
    days_ago = raw["days_ago"]
    if isinstance(days_ago, bool) or not isinstance(days_ago, int) or days_ago < 1:
        raise ValueError(
            f"{where}.days_ago: must be an integer of at least 1 (a session is always in "
            f"the past), got {days_ago!r}"
        )
    raw_time = raw["time"]
    if not isinstance(raw_time, str):
        raise ValueError(f"{where}.time: must be an 'HH:MM:SS' string, got {raw_time!r}")
    try:
        parsed = dtime.fromisoformat(raw_time)
    except ValueError:
        raise ValueError(
            f"{where}.time: must be a valid 'HH:MM:SS' time of day, got {raw_time!r}"
        ) from None
    if parsed.tzinfo is not None:
        raise ValueError(
            f"{where}.time: is local time and must not carry a timezone offset, got {raw_time!r}"
        )
    return RelativeTimestamp(days_ago=days_ago, time=parsed)


def _seconds_of_day(value: dtime) -> float:
    return value.hour * 3600 + value.minute * 60 + value.second + value.microsecond / 1e6


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def resolve(sessions: Sequence[ScenarioSession], anchor: datetime) -> tuple[ResolvedSession, ...]:
    """Place every session, turn and artifact at an aware instant.

    A session starts at its ``when`` resolved against ``anchor`` (calendar days
    back, wall-clock ``time`` in the anchor's timezone); a turn is placed
    ``at_offset_s`` seconds after that, and a turn's artifacts at the turn's
    instant.

    Raises:
        ValueError: ``anchor`` is naive.
    """
    if anchor.tzinfo is None or anchor.utcoffset() is None:
        raise ValueError(f"resolve: the anchor must be timezone-aware, got {anchor!r}")
    resolved: list[ResolvedSession] = []
    for session in sessions:
        start = resolve_relative_timestamp(session.when, anchor)
        parts = tuple(
            ResolvedPart(part=part, turns=tuple(_resolve_turn(t, start) for t in part.turns))
            for part in session.parts
        )
        resolved.append(ResolvedSession(session=session, at=start, parts=parts))
    return tuple(resolved)


def _resolve_turn(turn: SessionTurn, start: datetime) -> ResolvedTurn:
    at = start + timedelta(seconds=turn.at_offset_s)
    return ResolvedTurn(
        turn=turn,
        at=at,
        artifacts=tuple(ResolvedArtifact(spec=a, at=at) for a in turn.artifacts),
    )


# ---------------------------------------------------------------------------
# Seed-file round trip
# ---------------------------------------------------------------------------


def to_dict(resolved: ResolvedSession) -> dict[str, Any]:
    """A resolved session as JSON-ready data: the block's own shape plus ``at`` instants.

    The session, every turn and every artifact carry ``at`` as an ISO-8601
    string with its UTC offset; :func:`from_dict` reads it back.
    """
    session = resolved.session
    return {
        "slug": session.slug,
        "title": session.title,
        "when": {"days_ago": session.when.days_ago, "time": session.when.time.isoformat()},
        "at": resolved.at.isoformat(),
        "parts": [
            {"title": rp.part.title, "turns": [_turn_to_dict(rt) for rt in rp.turns]}
            for rp in resolved.parts
        ],
    }


def _turn_to_dict(resolved: ResolvedTurn) -> dict[str, Any]:
    turn = resolved.turn
    return {
        "prompt": turn.prompt,
        "at_offset_s": turn.at_offset_s,
        "at": resolved.at.isoformat(),
        "channels": list(turn.channels),
        "logbook_ids": list(turn.logbook_ids),
        "steps": [_step_to_dict(step) for step in turn.steps],
        "artifacts": [
            {
                "slug": ra.spec.slug,
                "form": ra.spec.form,
                "title": ra.spec.title,
                "source": ra.spec.source,
                "at": ra.at.isoformat(),
            }
            for ra in resolved.artifacts
        ],
    }


def _step_to_dict(step: ToolStep | AnswerStep) -> dict[str, Any]:
    if isinstance(step, AnswerStep):
        return {"answer": step.answer}
    return {
        "tool": {"server": step.server, "name": step.name},
        "input": step.input,
        "result": step.result,
        "approved": step.approved,
    }


def from_dict(raw: Mapping[str, Any], prefix: str = "session") -> ResolvedSession:
    """Read back what :func:`to_dict` wrote, validating it as :func:`parse_sessions` does.

    Raises:
        ValueError: The data is not a resolved session; the message names the path.
    """
    if not isinstance(raw, Mapping):
        raise ValueError(f"{prefix}: must be a mapping, got {raw!r}")
    if "at" not in raw:
        raise ValueError(f"{prefix}: missing keys ['at']")
    session_at = _instant(f"{prefix}.at", raw["at"])
    plain = {key: value for key, value in raw.items() if key not in ("at", "parts")}
    parts_raw = raw.get("parts")
    if not isinstance(parts_raw, list):
        raise ValueError(f"{prefix}.parts: must be a list, got {parts_raw!r}")
    turn_ats: list[list[tuple[datetime, list[datetime]]]] = []
    plain_parts: list[Any] = []
    for p, part in enumerate(parts_raw):
        if not isinstance(part, Mapping) or not isinstance(part.get("turns"), list):
            raise ValueError(f"{prefix}.parts[{p}]: must be a mapping with a 'turns' list")
        ats: list[tuple[datetime, list[datetime]]] = []
        plain_turns: list[Any] = []
        for t, turn in enumerate(part["turns"]):
            plain_turn, turn_at, artifact_ats = _split_turn(f"{prefix}.parts[{p}].turns[{t}]", turn)
            ats.append((turn_at, artifact_ats))
            plain_turns.append(plain_turn)
        turn_ats.append(ats)
        plain_parts.append({**part, "turns": plain_turns})
    plain["parts"] = plain_parts
    session = _parse_session(prefix, plain)
    parts = tuple(
        ResolvedPart(
            part=part,
            turns=tuple(
                ResolvedTurn(
                    turn=turn,
                    at=turn_at,
                    artifacts=tuple(
                        ResolvedArtifact(spec=spec, at=artifact_at)
                        for spec, artifact_at in zip(turn.artifacts, artifact_ats, strict=True)
                    ),
                )
                for turn, (turn_at, artifact_ats) in zip(part.turns, ats, strict=True)
            ),
        )
        for part, ats in zip(session.parts, turn_ats, strict=True)
    )
    return ResolvedSession(session=session, at=session_at, parts=parts)


def _split_turn(where: str, turn: Any) -> tuple[dict[str, Any], datetime, list[datetime]]:
    """A seed-file turn without its instants, its ``at``, and its artifacts' ``at``s."""
    if not isinstance(turn, Mapping) or "at" not in turn:
        raise ValueError(f"{where}: must be a mapping carrying 'at'")
    artifacts = turn.get("artifacts", [])
    if not isinstance(artifacts, list):
        raise ValueError(f"{where}.artifacts: must be a list, got {artifacts!r}")
    artifact_ats: list[datetime] = []
    plain_artifacts: list[Any] = []
    for a, artifact in enumerate(artifacts):
        if not isinstance(artifact, Mapping) or "at" not in artifact:
            raise ValueError(f"{where}.artifacts[{a}]: must be a mapping carrying 'at'")
        artifact_ats.append(_instant(f"{where}.artifacts[{a}].at", artifact["at"]))
        plain_artifacts.append({k: v for k, v in artifact.items() if k != "at"})
    plain_turn = {k: v for k, v in turn.items() if k not in ("at", "artifacts")}
    plain_turn["artifacts"] = plain_artifacts
    return plain_turn, _instant(f"{where}.at", turn["at"]), artifact_ats


def _instant(where: str, raw: Any) -> datetime:
    if not isinstance(raw, str):
        raise ValueError(f"{where}: must be an ISO-8601 instant, got {raw!r}")
    try:
        value = datetime.fromisoformat(raw)
    except ValueError:
        raise ValueError(f"{where}: must be an ISO-8601 instant, got {raw!r}") from None
    if value.tzinfo is None:
        raise ValueError(f"{where}: must carry a UTC offset, got {raw!r}")
    return value
