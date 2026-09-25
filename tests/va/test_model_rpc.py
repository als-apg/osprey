"""Tests for the model RPC wire contract.

In-process only: requests and replies are p4p Values built and read back
here, with no PVAccess server or client context. The contract module is the
one definition the Virtual Accelerator server and its clients share, so the
round trips below are what keeps the two ends from drifting.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import math
from pathlib import Path

import pytest

pytest.importorskip("p4p")  # p4p arrives with the virtual-accelerator extra

from p4p.nt import NTURI, NTScalar

from osprey.services.virtual_accelerator.serving import model_rpc
from osprey.services.virtual_accelerator.serving.model_rpc import (
    ERR_NOT_READY,
    ERR_TIMEOUT,
    REQUEST_TYPE,
    RPC_PV,
    RPC_TIMEOUT_S,
    VERBS,
    ModelRpcError,
    RpcRequest,
    build_request,
    error_reply,
    ok_reply,
    parse_reply,
    parse_request,
)

MODULE_PATH = Path(model_rpc.__file__)


def _raw_request(**query: object):
    """An NTURI request carrying exactly ``query``, bypassing build_request's checks."""
    fields = []
    for name, value in query.items():
        fields.append((name, "as" if isinstance(value, list) else "s"))
    return NTURI(fields).wrap(RPC_PV, kws=query)


def _full_raw_request(values: str, *, verb: str = "set"):
    return REQUEST_TYPE.wrap(RPC_PV, kws={"verb": verb, "names": [], "values": values, "token": ""})


class TestConstants:
    def test_channel_and_verbs(self):
        assert RPC_PV == "model_rpc"
        assert VERBS == ("info", "get", "diff", "status", "set", "reset")

    def test_timeout(self):
        assert RPC_TIMEOUT_S == 30.0

    def test_error_texts_are_distinct_sentences(self):
        assert isinstance(ERR_TIMEOUT, str) and ERR_TIMEOUT
        assert ERR_NOT_READY != ERR_TIMEOUT

    def test_not_ready_is_the_runners_string(self):
        # runner.py cannot be imported without the CA server library, so its
        # assignment is read from the source rather than from the module. It
        # must bind this name, not restate the sentence: two literals could
        # drift apart, one name cannot.
        tree = ast.parse(MODULE_PATH.with_name("runner.py").read_text())
        bound = [
            ast.unparse(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "NOT_READY" for t in node.targets)
        ]
        assert bound == ["ERR_NOT_READY"]


class TestImports:
    def test_imports_only_p4p_and_the_standard_library(self):
        # The module is imported by the server AND by a client that has
        # neither the server library nor the model installed.
        allowed = {"__future__", "collections", "dataclasses", "json", "math", "numbers"}
        allowed |= {"typing", "p4p"}
        roots = set()
        for node in ast.walk(ast.parse(MODULE_PATH.read_text())):
            if isinstance(node, ast.Import):
                roots |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                assert node.level == 0, "no relative imports"
                roots.add((node.module or "").split(".")[0])
        assert roots <= allowed, sorted(roots - allowed)


class TestRequestShape:
    def test_is_an_nturi_on_the_rpc_channel(self):
        request = build_request("status")
        assert request.getID() == "epics:nt/NTURI:1.0"
        assert request.path == RPC_PV

    def test_carries_every_query_field_even_when_empty(self):
        query = build_request("status").query
        assert query.verb == "status"
        assert list(query.names) == []
        assert query.values == ""
        assert query.token == ""

    def test_values_travel_as_a_json_object(self):
        query = build_request("set", values={"A": 1.5}).query
        assert json.loads(query.values) == {"A": 1.5}


class TestRequestRoundTrip:
    @pytest.mark.parametrize("verb", VERBS)
    def test_every_verb_with_all_fields(self, verb):
        request = build_request(verb, names=("A", "B"), values={"A": 1.5, "B": -2}, token="tok")
        parsed = parse_request(request)
        assert parsed == RpcRequest(
            verb=verb, names=("A", "B"), values={"A": 1.5, "B": -2.0}, token="tok"
        )

    @pytest.mark.parametrize("verb", VERBS)
    def test_every_verb_with_defaults(self, verb):
        assert parse_request(build_request(verb)) == RpcRequest(verb=verb)

    def test_defaults_are_empty(self):
        request = RpcRequest(verb="get")
        assert request.names == ()
        assert request.values == {}
        assert request.token == ""

    def test_integers_arrive_as_floats(self):
        values = parse_request(build_request("set", values={"A": 3})).values
        assert values == {"A": 3.0}
        assert type(values["A"]) is float

    def test_empty_values_object_is_no_values(self):
        assert parse_request(build_request("set", values={})).values == {}

    def test_names_accept_any_iterable(self):
        request = build_request("get", names=iter(["X", "Y"]))
        assert parse_request(request).names == ("X", "Y")

    def test_request_is_frozen(self):
        parsed = parse_request(build_request("get"))
        with pytest.raises(dataclasses.FrozenInstanceError):
            parsed.verb = "set"  # type: ignore[misc]


class TestRequestRefusedOnTheWire:
    """What a server refuses from a client that did not use build_request."""

    @pytest.mark.parametrize(
        "values",
        [
            '{"A": NaN}',
            '{"A": Infinity}',
            '{"A": -Infinity}',
            '{"A": 1e999}',
            '{"A": "1.0"}',
            '{"A": {"x": 1.0}}',
            '{"A": [1.0]}',
            '{"A": true}',
            '{"A": null}',
        ],
    )
    def test_non_finite_or_non_numeric_entry(self, values):
        with pytest.raises(ModelRpcError, match="'A'"):
            parse_request(_full_raw_request(values))

    @pytest.mark.parametrize("values", ["not json", '{"A": 1.0', "{'A': 1.0}"])
    def test_non_json_values(self, values):
        with pytest.raises(ModelRpcError, match="JSON"):
            parse_request(_full_raw_request(values))

    @pytest.mark.parametrize("values", ["[1.0, 2.0]", "1.0", '"A"', "NaN", "null"])
    def test_values_that_are_not_an_object(self, values):
        with pytest.raises(ModelRpcError, match="object"):
            parse_request(_full_raw_request(values))

    @pytest.mark.parametrize("verb", ["frobnicate", "", "GET", "set "])
    def test_unknown_verb(self, verb):
        with pytest.raises(ModelRpcError, match="verb"):
            parse_request(_full_raw_request("", verb=verb))

    def test_missing_verb(self):
        with pytest.raises(ModelRpcError, match="verb"):
            parse_request(_raw_request(names=["A"]))

    def test_no_query_at_all(self):
        with pytest.raises(ModelRpcError, match="query"):
            parse_request(NTScalar("s").wrap("status"))

    def test_names_sent_as_one_string(self):
        with pytest.raises(ModelRpcError, match="names"):
            parse_request(_raw_request(verb="get", names="A"))

    def test_omitted_optional_fields_read_as_empty(self):
        # A generic PVA client sends only the fields it is given.
        assert parse_request(_raw_request(verb="status")) == RpcRequest(verb="status")


class TestRequestRefusedAtTheClient:
    @pytest.mark.parametrize(
        "bad",
        [math.nan, math.inf, -math.inf, "1.0", {"x": 1.0}, [1.0], True, None],
        ids=["nan", "inf", "-inf", "string", "nested", "list", "bool", "none"],
    )
    def test_non_finite_or_non_numeric_entry(self, bad):
        with pytest.raises(ModelRpcError, match="'A'"):
            build_request("set", values={"A": bad})

    def test_unknown_verb(self):
        with pytest.raises(ModelRpcError, match="verb"):
            build_request("frobnicate")

    def test_names_as_one_string(self):
        with pytest.raises(ModelRpcError, match="names"):
            build_request("get", names="AB")

    def test_non_string_name(self):
        with pytest.raises(ModelRpcError, match="names"):
            build_request("get", names=["A", 1])  # type: ignore[list-item]

    def test_values_not_a_mapping(self):
        with pytest.raises(ModelRpcError, match="object"):
            build_request("set", values=[("A", 1.0)])  # type: ignore[arg-type]


class TestReply:
    @pytest.mark.parametrize(
        "result",
        [
            {"A": 1.5, "B": [1.0, 2.0]},
            [1, "two", None],
            None,
            "text",
            0.25,
            {"nested": {"deep": {"x": True}}},
        ],
    )
    def test_ok_round_trip(self, result):
        reply = ok_reply(result)
        assert reply.getID() == "epics:nt/NTScalar:1.0"
        assert json.loads(reply.value) == {"ok": True, "result": result}
        assert parse_reply(reply) == result

    @pytest.mark.parametrize("message", ["no such variable 'Q'", ERR_NOT_READY, ERR_TIMEOUT])
    def test_error_round_trip(self, message):
        reply = error_reply(message)
        assert json.loads(reply.value) == {"ok": False, "error": message}
        with pytest.raises(ModelRpcError) as caught:
            parse_reply(reply)
        assert str(caught.value) == message

    def test_array_like_results_become_lists(self):
        class ArrayLike:
            def tolist(self):
                return [1.0, 2.0]

        assert parse_reply(ok_reply({"orbit": ArrayLike()})) == {"orbit": [1.0, 2.0]}

    def test_non_finite_result_survives(self):
        # Requests refuse non-finite numbers; results report what the model holds.
        result = parse_reply(ok_reply({"x": math.nan}))
        assert math.isnan(result["x"])

    def test_accepts_the_unwrapped_string_a_client_context_returns(self):
        # A p4p client Context unwraps an NTScalar reply to its (str) value.
        assert parse_reply(ok_reply([1, 2]).value) == [1, 2]
        with pytest.raises(ModelRpcError, match="refused"):
            parse_reply(error_reply("refused").value)

    @pytest.mark.parametrize(
        "text", ["not json", "[1]", '{"result": 1}', '{"ok": "yes"}', '{"ok": false}']
    )
    def test_malformed_reply(self, text):
        with pytest.raises(ModelRpcError, match="reply"):
            parse_reply(NTScalar("s").wrap(text))
