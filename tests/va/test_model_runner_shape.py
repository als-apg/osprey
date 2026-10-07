"""What the model runner does, asserted against its syntax tree.

:mod:`osprey.services.virtual_accelerator.serving.runner` imports the
serving package's runner, which imports the compiled Channel Access server
extension, so it is not importable on a host without a working build of it.
Its behaviour is proven against the deployed container. What is checked here,
with no import at all, is the handful of properties whose violation would be
silent in that container -- a leaked write token, a model touched off the run
loop's thread, a call left unanswered -- so that they fail in a unit run
instead.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

RUNNER = (
    Path(__file__).resolve().parents[2]
    / "src/osprey/services/virtual_accelerator/serving/runner.py"
)
RUNNER_CLASS = "ModelRunner"

#: A reference to the model write token, by either of the names it has in
#: the runner: the constructor argument and the attribute it is kept on.
_TOKEN_NAMES = frozenset({"model_write_token", "_model_write_token"})

#: Calls whose arguments reach a log, a terminal or a warning.
_OUTPUT_METHODS = frozenset(
    {"debug", "info", "warning", "warn", "error", "exception", "critical", "log"}
)


def _mentions_token(node: ast.AST) -> bool:
    return any(
        (isinstance(each, ast.Name) and each.id in _TOKEN_NAMES)
        or (isinstance(each, ast.Attribute) and each.attr in _TOKEN_NAMES)
        for each in ast.walk(node)
    )


def _token_leaks(tree: ast.AST) -> list[str]:
    """Every place in ``tree`` that would put the token into text.

    That is a logging, ``print`` or ``warnings.warn`` call, a ``format``
    call, an f-string, a ``%`` interpolation or a raised exception, any of
    which mentions the token. Passing the token on as an argument to another
    call is none of these, which is how the model surface receives it.
    """
    leaks = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            emits = (isinstance(func, ast.Name) and func.id == "print") or (
                isinstance(func, ast.Attribute) and func.attr in _OUTPUT_METHODS | {"format"}
            )
        else:
            emits = isinstance(node, (ast.JoinedStr, ast.Raise)) or (
                isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod)
            )
        if emits and _mentions_token(node):
            leaks.append(ast.unparse(node))
    return leaks


@pytest.fixture(scope="module")
def tree() -> ast.Module:
    return ast.parse(RUNNER.read_text(encoding="utf-8"))


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name!r} not found")


def _method(tree: ast.Module, cls: str, name: str) -> ast.FunctionDef:
    for node in _class(tree, cls).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"method {cls}.{name} not found")


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name!r} not found")


def _try(function: ast.FunctionDef) -> ast.Try:
    tries = [node for node in function.body if isinstance(node, ast.Try)]
    assert len(tries) == 1, f"{function.name} has exactly one try block"
    return tries[0]


def _statements(function: ast.FunctionDef) -> list[str]:
    """The function's statements other than its docstring, unparsed."""
    return [
        ast.unparse(statement)
        for statement in function.body
        if not (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant))
    ]


def test_the_model_write_token_is_never_logged(tree: ast.Module) -> None:
    """Nor printed, warned, formatted into text or raised: the token is the
    one thing that gates a model write, and a log is readable by far more
    people than the write is allowed to."""
    assert _token_leaks(tree) == []


@pytest.mark.parametrize(
    "leak",
    [
        "LOG.info('armed with %s', self._model_write_token)",
        "print(model_write_token)",
        "warnings.warn(f'token {model_write_token}')",
        "text = 'token %s' % self._model_write_token",
        "raise ValueError(model_write_token)",
        "text = '{}'.format(self._model_write_token)",
    ],
)
def test_the_leak_check_catches_a_leak(leak: str) -> None:
    """The check above is only as good as its detector: each of these would
    put the token into text, and each is caught."""
    assert _token_leaks(ast.parse(leak))


def test_the_leak_check_lets_the_token_be_passed_on() -> None:
    """Handing the token to the object that checks it is not a leak."""
    passed_on = ast.parse(
        "surface = ModelSurface.for_view(model_write_token=self._model_write_token)"
    )
    assert _token_leaks(passed_on) == []


def test_the_rpc_handler_never_touches_the_model(tree: ast.Module) -> None:
    """It runs on a p4p worker thread, which is no more allowed to touch the
    model than the Channel Access server thread is. The verb is dispatched by
    a job instead, and the job runs on the run loop."""
    rpc = ast.unparse(_method(tree, RUNNER_CLASS, "_rpc"))

    assert "self.model" not in rpc
    assert "self._surface" not in rpc
    assert "_surface_reply(" not in rpc
    # Empty values, so the cycle carrying the job runs no model pass of its
    # own: the job is the whole of what a read costs the loop.
    assert "self._enqueue({}, jobs=[" in rpc


def test_a_call_arriving_before_the_driver_exists_is_refused(tree: ast.Module) -> None:
    """The PVA server is listening from the moment it is created, which is
    before the driver a served value is read through exists."""
    rpc = ast.unparse(_method(tree, RUNNER_CLASS, "_rpc"))

    assert "self.ca_driver" in rpc
    assert "op.done(error=ERR_NOT_READY)" in rpc


def test_a_request_that_is_not_one_is_refused_without_the_run_loop(tree: ast.Module) -> None:
    """A malformed request never reaches the model, so it never needs the
    loop's thread -- and would otherwise occupy a cycle to be told so."""
    rpc = _method(tree, RUNNER_CLASS, "_rpc")
    parsing = _try(rpc)
    enqueues = [
        node
        for node in ast.walk(rpc)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "self._enqueue"
    ]

    assert "parse_request" in ast.unparse(parsing.body)
    refusal = parsing.handlers[0]
    assert "op.done(error=" in ast.unparse(refusal)
    assert enqueues
    assert refusal.end_lineno is not None
    assert all(refusal.end_lineno < enqueue.lineno for enqueue in enqueues)


def test_a_call_the_run_loop_never_reaches_is_answered_anyway(tree: ast.Module) -> None:
    """A client told nothing waits out its own timeout knowing only that the
    server did not answer; the contract's own message says why."""
    rpc = ast.unparse(_method(tree, RUNNER_CLASS, "_rpc"))

    assert "threading.Timer(RPC_TIMEOUT_S, call.complete, (error_reply(ERR_TIMEOUT),))" in rpc
    # Daemon, so a call still being waited on never holds up a shutdown.
    assert "timeout.daemon = True" in rpc
    assert "timeout.start()" in rpc


def test_a_call_is_answered_exactly_once(tree: ast.Module) -> None:
    """The job and the timer race to answer, and p4p completes an operation
    once: a second completion is an error the client it was meant for never
    sees."""
    complete = ast.unparse(_method(tree, "_RpcCall", "complete"))

    assert "with self._lock" in complete
    assert "self._answered" in complete
    assert "return False" in complete


def test_the_job_answers_its_call_on_every_path(tree: ast.Module) -> None:
    """The run loop logs a job that raises and moves on to the next item. So
    a job that returned without answering would cost its client the whole
    timeout, and tell it nothing when the timeout expired."""
    dispatched = _try(_function(tree, "_surface_reply"))

    assert "ok_reply(" in ast.unparse(dispatched.body)
    assert dispatched.handlers
    assert all("error_reply(" in ast.unparse(handler) for handler in dispatched.handlers)
    assert "call.complete(_surface_reply(" in ast.unparse(_method(tree, RUNNER_CLASS, "_reply_now"))
    assert "kept.append(_surface_reply(" in ast.unparse(_method(tree, RUNNER_CLASS, "_keep_reply"))


@pytest.mark.parametrize(
    ("method", "reply"),
    [
        (
            "_reply_now",
            "call.complete(_surface_reply(self._surface, request, driver, self.queue.qsize()))",
        ),
        ("_send_kept", "call.complete(reply)"),
    ],
)
def test_the_reply_goes_out_before_the_timer_is_cancelled(
    tree: ast.Module, method: str, reply: str
) -> None:
    """Cancelling first would leave a client with nothing at all if the reply
    itself could not be delivered; this way the timer still answers, late,
    with the reason."""
    assert _statements(_method(tree, RUNNER_CLASS, method))[-2:] == [reply, "timeout.cancel()"]


def test_the_loop_records_what_only_it_can_see(tree: ast.Module) -> None:
    """``status`` reports the run loop's own queue, which the surface is told
    from the one thread that can read it."""
    rpc = ast.unparse(_method(tree, RUNNER_CLASS, "_rpc"))
    reply = ast.unparse(_function(tree, "_surface_reply"))

    assert "qsize" not in rpc
    for job in ("_reply_now", "_keep_reply"):
        assert "self.queue.qsize()" in ast.unparse(_method(tree, RUNNER_CLASS, job))
    assert "surface.record_queue_depth(queue_depth)" in reply
    assert "surface.record_cycle(" in reply


def test_a_write_the_surface_did_not_refuse_itself_is_still_recorded(tree: ast.Module) -> None:
    """The surface records the refusals it raises. A write that failed some
    other way is a refused write too, and is the one such failure this
    handler has to record for ``status`` itself -- while a refused *read* is
    recorded by neither, being no write at all."""
    reply = ast.unparse(_function(tree, "_surface_reply"))

    assert "MODEL_WRITE_VERBS" in reply
    assert "surface.record_refusal(" in reply


def test_the_diff_verb_reads_what_the_control_system_serves(tree: ast.Module) -> None:
    """Its whole answer is the served value beside the model's truth, and
    only the driver knows the first of the two."""
    assert "surface.diff(driver.getParam)" in ast.unparse(_function(tree, "_dispatch"))


def test_every_verb_the_contract_admits_is_dispatched(tree: ast.Module) -> None:
    """A verb the contract admits and this dispatch does not would be parsed,
    enqueued, and answered with whatever the fall-through verb happens to
    be."""
    pytest.importorskip("p4p")
    from osprey.services.virtual_accelerator.serving.model_rpc import VERBS

    dispatch = _function(tree, "_dispatch")
    answered = {
        node.func.attr
        for node in ast.walk(dispatch)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and ast.unparse(node.func.value) == "surface"
    }

    assert answered == set(VERBS)


def test_the_reported_endpoint_is_the_port_the_server_binds(tree: ast.Module) -> None:
    """Read from the variable the pvAccess server itself binds from, so the
    address a client is told to use cannot drift from the one in use."""
    endpoint = ast.unparse(_function(tree, "_pva_endpoint"))

    assert "EPICS_PVAS_SERVER_PORT" in endpoint
    assert "DEFAULT_PVA_PORT" in endpoint
