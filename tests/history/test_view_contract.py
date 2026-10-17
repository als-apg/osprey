"""The history page's view contract: every query file renders and parses.

"Parses" is checked by a tokenizer that knows the store's lexical rules
(comments, quoted strings and identifiers, typed parameters): one statement,
balanced brackets, every parameter filled, and every table read is a
``telemetry`` table the store has, never a ``system.*`` one.
"""

from __future__ import annotations

import datetime
import re

import pytest

from osprey.deployment import telemetry_config as tc
from osprey.deployment.telemetry_store import ddl
from osprey.history import queries

_NOW = datetime.datetime(2026, 3, 4, 12, 30, 15, 123456, tzinfo=datetime.UTC)
_SAMPLES = {
    "String": "jean-luc's \\ session",
    "UInt64": 2**64 - 1,
    "DateTime64(9)": _NOW,
    "Array(String)": ["queueserver", "bluesky-va-queueserver"],
}
_TOKEN = re.compile(
    r"""
    (?P<comment>--[^\n]*)
  | (?P<space>\s+)
  | (?P<string>'(?:[^'\\]|\\.)*')
  | (?P<quoted>`(?:[^`\\]|\\.)*`)
  | (?P<param>\{[A-Za-z_][A-Za-z0-9_]*:[^{}]+\})
  | (?P<number>\d+(?:\.\d+)?)
  | (?P<word>[A-Za-z_][A-Za-z0-9_]*)
  | (?P<op><=|>=|!=|<>|\|\||[-+*/%<>=,.()\[\]])
  | (?P<bad>.)
    """,
    re.VERBOSE | re.DOTALL,
)


def _values(name: str) -> dict[str, object]:
    return {param: _SAMPLES[kind] for param, kind in queries.parameters(name).items()}


def _tokens(sql: str) -> list[str]:
    out: list[str] = []
    for match in _TOKEN.finditer(sql):
        kind = match.lastgroup
        if kind == "bad":
            raise AssertionError(f"unexpected character {match.group()!r} at {match.start()}")
        if kind not in ("comment", "space"):
            out.append(match.group())
    return out


def _parse(sql: str) -> list[str]:
    tokens = _tokens(sql)
    assert tokens[0].upper() in ("SELECT", "WITH"), tokens[:3]
    pairs = {"(": ")", "[": "]"}
    stack: list[str] = []
    for token in tokens:
        if token in pairs:
            stack.append(pairs[token])
        elif token in pairs.values():
            assert stack and stack.pop() == token, f"unbalanced {token}"
    assert not stack, f"unclosed {stack}"
    return tokens


def _read_tables(tokens: list[str]) -> list[tuple[str, str]]:
    reads = []
    for i, token in enumerate(tokens):
        if token.upper() in ("FROM", "JOIN") and tokens[i + 1] != "(":
            assert tokens[i + 2] == ".", f"unqualified table after {token}: {tokens[i + 1]}"
            reads.append((tokens[i + 1], tokens[i + 3]))
    return reads


@pytest.mark.parametrize("name", queries.QUERIES)
def test_every_query_renders_and_parses(name: str) -> None:
    rendered = queries.render(name, _values(name))
    assert "{" not in re.sub(r"--[^\n]*", "", rendered), "a parameter was left unfilled"
    tokens = _parse(rendered)
    assert not any(token.startswith("{") for token in tokens), "a parameter was left unfilled"
    assert ";" not in tokens
    reads = _read_tables(tokens)
    assert reads, "reads no table"
    for database, table in reads:
        assert database == ddl.DATABASE, f"{name} reads {database}.{table}"
        assert table in ddl.TABLES_BY_NAME, f"{name} reads unknown table {table}"


@pytest.mark.parametrize("name", queries.QUERIES)
def test_no_query_reads_a_system_table(name: str) -> None:
    text = re.sub(r"--[^\n]*", "", queries.load(name)).lower()
    assert "system." not in text
    assert "system" not in [token.lower() for token in _tokens(queries.load(name))]


@pytest.mark.parametrize("name", queries.QUERIES)
def test_every_query_is_bounded(name: str) -> None:
    tokens = [t.upper() for t in _tokens(queries.load(name))]
    limits = [
        int(tokens[i + 1]) for i, t in enumerate(tokens) if t == "LIMIT" and tokens[i + 1].isdigit()
    ]
    assert limits, f"{name} has no LIMIT"
    # A page never asks for more rows than a view user or reader may receive.
    assert max(limits) <= tc.DEFAULT_LIMITS["result_rows"]


@pytest.mark.parametrize(
    "name", ["conversation", "tool_log", "approvals", "preflight", queries.RECIPE]
)
def test_row_queries_deduplicate_on_their_natural_key(name: str) -> None:
    assert "LIMIT 1 BY" in queries.load(name)


# Body names an agent event's kind, not the record: two events of one kind in
# one millisecond differ only in their attributes.
_LOG_ROW_IDENTITY = "cityHash64(Body, toJSONString(LogAttributes))"


def _hash_calls(tokens: list[str]) -> list[str]:
    calls = []
    for i, token in enumerate(tokens):
        if token == "cityHash64":
            depth, j = 0, i + 1
            while True:
                depth += {"(": 1, ")": -1}.get(tokens[j], 0)
                j += 1
                if depth == 0:
                    break
            calls.append(tokens[i:j])
    return calls


def _arms(tokens: list[str]) -> list[list[str]]:
    arms: list[list[str]] = [[]]
    for token in tokens:
        if token.upper() == "UNION":
            arms.append([])
        else:
            arms[-1].append(token)
    return arms


def test_every_log_row_hash_includes_its_attributes() -> None:
    identity = _tokens(_LOG_ROW_IDENTITY)
    hashed = set()
    for name in queries.QUERIES:
        for arm in _arms(_tokens(queries.load(name))):
            if (ddl.DATABASE, "otel_logs") not in _read_tables(arm):
                continue
            for call in _hash_calls(arm):
                assert call == identity, f"{name} keys an otel_logs row by {' '.join(call)}"
                hashed.add(name)
    assert hashed >= {"conversation", "load_full_log", queries.RECIPE}, hashed


@pytest.mark.parametrize(
    ("name", "cursor"),
    [
        ("session_list", ("cursor_ts", "cursor_user", "cursor_session")),
        ("conversation", ("after_ts", "after_hash")),
        ("tool_log", ("after_ts", "after_hash")),
        (queries.RECIPE, ("after_at", "after_source", "after_hash")),
    ],
)
def test_paged_queries_page_by_keyset(name: str, cursor: tuple[str, ...]) -> None:
    text = queries.load(name)
    assert set(cursor) <= set(queries.parameters(name))
    assert "OFFSET" not in text.upper()
    tuple_compare = re.search(r"\)\s*[<>]\s*\(\{" + cursor[0], text)
    assert tuple_compare, f"{name} does not compare its keyset tuple to the cursor"


def test_the_recipe_pages_within_a_readers_result_cap() -> None:
    text = queries.load(queries.RECIPE)
    assert text.rstrip().endswith(f"LIMIT {queries.PAGE_LIMIT}")
    assert queries.PAGE_LIMIT < tc.DEFAULT_LIMITS["result_rows"]
    assert "ORDER BY at, source, row_hash" in text
    # Tool calls are placed where they started.
    assert "osprey.audit.duration_ms" in text


def test_the_conversation_reads_prompts_and_responses_only() -> None:
    text = queries.load("conversation")
    assert f"Event IN ('{queries.USER_PROMPT_EVENT}', '{queries.ASSISTANT_RESPONSE_EVENT}')" in text
    for event, attribute in queries.CONTENT_ATTRIBUTES.items():
        assert f"'{event}'" in text and f"'{attribute}'" in text
    assert f"'{queries.REDACTED}'" in text
    assert "api_request" not in text
    assert f"substring(LogAttributes[content_key], 1, {queries.PREVIEW_CHARS})" in text


def test_the_session_list_reads_prompts_only() -> None:
    text = queries.load("session_list")
    assert f"Event = '{queries.USER_PROMPT_EVENT}'" in text
    assert f"LIMIT {queries.SESSION_LIST_LIMIT}" in text
    assert queries.SESSION_LIST_WINDOW_DAYS == 3


def test_the_preflight_band_never_matches_on_behalf_of_alone() -> None:
    text = queries.load("preflight")
    assert "User IN {queueservers:Array(String)}" in text
    assert "Surface = 'preflight'" in text


def test_parameters_are_sent_typed_never_spliced() -> None:
    params = queries.query_parameters(
        "preflight",
        {"queueservers": ["queueserver"], "user": "a'b", "from": _NOW, "to": _NOW},
    )
    assert params == {
        "param_queueservers": "['queueserver']",
        "param_user": "a'b",
        "param_from": "2026-03-04 12:30:15.123456000",
        "param_to": "2026-03-04 12:30:15.123456000",
    }
    rendered = queries.render(
        "load_full_ledger", {"user": "a'b", "tool_use_id": "t", "surface": "s"}
    )
    assert "User = 'a\\'b'" in rendered


def test_a_missing_or_stray_parameter_is_refused() -> None:
    with pytest.raises(queries.QueryError, match="missing session"):
        queries.query_parameters("agents", {"user": "a"})
    with pytest.raises(queries.QueryError, match="not taken extra"):
        queries.query_parameters("agents", {"user": "a", "session": "s", "extra": 1})
    with pytest.raises(queries.QueryError, match="no history query"):
        queries.load("nope")


def test_the_first_cursor_is_older_than_any_kept_row() -> None:
    assert queries.FIRST_CURSOR.year == 2000
    params = queries.query_parameters(
        "conversation",
        {"user": "u", "session": "s", "after_ts": queries.FIRST_CURSOR, "after_hash": 0},
    )
    assert params["param_after_ts"] == "2000-01-01 00:00:00.000000000"
