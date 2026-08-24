"""Client-side Cypher gate for the graph MCP server.

Write enforcement is server-side (``Session.execute_read`` rejects write clauses with
``Neo.ClientError.Statement.AccessMode``), so this module does not vet write keywords.
It only closes the gaps read mode leaves open: extension procedures and functions that
execute strings or reach the network (``apoc.*``, ``n10s.*`` ...) and ``LOAD CSV``.

The gate is a comment-, string- and backtick-aware token scanner. Comments and string
literals are dropped before classification, so ``RETURN 'CALL apoc.load.json(x)'``
passes while ``/* it's */ CALL apoc.load.json(...)`` is refused. Everything after a
``CALL`` that is not a subquery is treated as a procedure invocation and must be
allowlisted (default deny); dotted names in a denied namespace followed by ``(`` are
refused wherever they appear, so string-executing APOC *functions* in ``RETURN`` or
``WHERE`` are caught too. Pure functions, no I/O, no neo4j import.
"""

from __future__ import annotations

READ_PROCEDURE_ALLOWLIST: frozenset[str] = frozenset({"db.labels", "db.relationshiptypes"})
"""Lowercased dotted procedure/function names that may run through ``read_cypher``."""

DENIED_NAMESPACES: frozenset[str] = frozenset({"apoc", "n10s", "db", "dbms", "gds"})
"""Leading identifier segments whose invocations are refused unless allowlisted."""

_NOT_AVAILABLE = (
    "Extension procedures/functions and CSV import are not available through this tool."
)

_ID = "id"
_PUNCT = "punct"


class GateRefusal(ValueError):
    """Raised when a query is refused before dialing; str(exc) is the operator-facing reason."""


def _tokenize(query: str) -> list[tuple[str, str]]:
    """Return ``(kind, text)`` tokens with comments and string literals removed.

    Identifier tokens are uppercased; backtick-escaped identifiers become a single
    identifier token with the backticks stripped (a doubled backtick is a literal one).
    Every other non-blank character is a one-character punctuation token.
    """
    tokens: list[tuple[str, str]] = []
    i, n = 0, len(query)
    while i < n:
        ch = query[i]
        pair = query[i : i + 2]
        if pair == "//":
            end = query.find("\n", i)
            i = n if end < 0 else end + 1
        elif pair == "/*":
            end = query.find("*/", i + 2)
            i = n if end < 0 else end + 2
        elif ch in "'\"":
            i += 1
            while i < n and query[i] != ch:
                i += 2 if query[i] == "\\" else 1
            i += 1
        elif ch == "`":
            name: list[str] = []
            i += 1
            while i < n:
                if query[i] == "`":
                    if query[i + 1 : i + 2] != "`":
                        break
                    i += 1
                name.append(query[i])
                i += 1
            i += 1
            tokens.append((_ID, "".join(name).upper()))
        elif ch.isalnum() or ch == "_":
            start = i
            while i < n and (query[i].isalnum() or query[i] == "_"):
                i += 1
            tokens.append((_ID, query[start:i].upper()))
        elif ch.isspace():
            i += 1
        else:
            tokens.append((_PUNCT, ch))
            i += 1
    return tokens


def _dotted_name(tokens: list[tuple[str, str]], start: int) -> tuple[str, int]:
    """Join the identifier chain ``a . b . c`` starting at ``start``.

    Returns the lowercased dotted name and the index of the first token after it.
    """
    parts = [tokens[start][1]]
    i = start + 1
    while i + 1 < len(tokens) and tokens[i] == (_PUNCT, ".") and tokens[i + 1][0] == _ID:
        parts.append(tokens[i + 1][1])
        i += 2
    return ".".join(parts).lower(), i


def _classify_call(tokens: list[tuple[str, str]], start: int) -> int:
    """Classify the tokens after a ``CALL`` and return where scanning resumes.

    A ``{`` (optionally preceded by a variable-scope clause ``( ... )``) opens a
    subquery, which is allowed and scanned from the inside. Anything else is a
    procedure invocation and must be allowlisted.
    """
    i = start
    if i < len(tokens) and tokens[i] == (_PUNCT, "("):
        depth = 0
        while i < len(tokens):
            if tokens[i] == (_PUNCT, "("):
                depth += 1
            elif tokens[i] == (_PUNCT, ")"):
                depth -= 1
                if depth == 0:
                    break
            i += 1
        i += 1
    if i < len(tokens) and tokens[i] == (_PUNCT, "{"):
        return i + 1
    if i != start or i >= len(tokens) or tokens[i][0] != _ID:
        raise GateRefusal(
            "Refused: CALL is not followed by a recognizable subquery or procedure name. "
            + _NOT_AVAILABLE
        )
    name, after = _dotted_name(tokens, i)
    if name not in READ_PROCEDURE_ALLOWLIST:
        raise GateRefusal(f"Refused: procedure '{name}' is not allowlisted. " + _NOT_AVAILABLE)
    return after


def vet_query(query: str) -> None:
    """Return silently when ``query`` may be sent to the store; raise GateRefusal otherwise."""
    tokens = _tokenize(query)
    if not tokens:
        raise GateRefusal("Refused: empty query (nothing left after removing comments).")
    i = 0
    while i < len(tokens):
        kind, text = tokens[i]
        if kind != _ID:
            i += 1
            continue
        previous = tokens[i - 1] if i else None
        # ``n.call`` / ``n:Call`` are property and label positions, never a clause.
        clause_position = previous not in ((_PUNCT, "."), (_PUNCT, ":"))
        if text == "CALL" and clause_position:
            i = _classify_call(tokens, i + 1)
            continue
        if text == "LOAD" and clause_position and tokens[i + 1 : i + 2] == [(_ID, "CSV")]:
            raise GateRefusal("Refused: LOAD CSV is not permitted. " + _NOT_AVAILABLE)
        name, after = _dotted_name(tokens, i)
        if previous != (_PUNCT, ".") and tokens[after : after + 1] == [(_PUNCT, "(")]:
            if name.split(".")[0] in DENIED_NAMESPACES and name not in READ_PROCEDURE_ALLOWLIST:
                raise GateRefusal(
                    f"Refused: invocation of '{name}' is not allowlisted. " + _NOT_AVAILABLE
                )
        i = after
    return None
