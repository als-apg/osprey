"""Tests for the armed raw-put block the execution wrapper emits in readwrite runs.

A readwrite run may write, but only through a connector: the connector checks
limits and records the write. The emitted block arms
:mod:`osprey.runtime.raw_put_block` so that a raw client put the connector did
not let through is refused with :data:`RAW_CLIENT_WRITE_MARKER`.

Emission is asserted on the literals the install call receives, parsed from the
generated source with :mod:`ast` — the tables they must match live in
``write_surface``, so a drift between the two fails here. Behaviour is asserted
by running the emitted block in a **real subprocess**: the block patches client
modules process-wide, and running it in-process would leak those patches into
every test after it.
"""

import ast
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from osprey.runtime.raw_put_block import _refusal_text
from osprey.services.python_executor.execution import wrapper as wrapper_module
from osprey.services.python_executor.execution.wrapper import (
    ARMED_RPC_REFUSALS,
    READONLY_REFUSAL_MARKER,
    ExecutionWrapper,
    armed_contract,
)
from osprey.services.python_executor.write_surface import _ARMED_BLOCKED, _ARMED_RPC
from osprey_connectors.control_system.limits_validator import (
    ChannelLimitsConfig,
    LimitsValidator,
)
from osprey_connectors.errors import RAW_CLIENT_WRITE_MARKER

P4P_RPC_TEXT = "rpc is not mediated and cannot be approved — use the supervised write path"
TANGO_COMMAND_TEXT = (
    "Tango command refused in a limits-checked run: a command carries no value to bound"
)

_SRC_ROOT = str(Path(__file__).resolve().parents[4] / "src")


def _validator() -> LimitsValidator:
    return LimitsValidator(
        {
            "TEST:PV": ChannelLimitsConfig(
                channel_address="TEST:PV", min_value=0, max_value=100, writable=True
            )
        },
        {"mode": "exclusive", "on_violation": "error"},
    )


def _install_call(source: str) -> ast.Call:
    """The top-level ``_ns["install"](...)`` call of an emitted block."""
    calls = [
        node.value
        for node in ast.parse(source).body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Subscript)
        and ast.literal_eval(node.value.func.slice) == "install"
    ]
    assert len(calls) == 1, source[-2000:]
    return calls[0]


def _install_literals(source: str) -> tuple[list, dict]:
    call = _install_call(source)
    args = [ast.literal_eval(arg) for arg in call.args]
    kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
    return args, kwargs


def _grouped(table: dict[tuple[str, str], str]) -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {}
    for dotted, attr in table:
        rows.setdefault(dotted, []).append(attr)
    return rows


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [_SRC_ROOT, env.get("PYTHONPATH")]))
    return env


# --------------------------------------------------------------------------
# Which runs get the block
# --------------------------------------------------------------------------


def test_readonly_run_emits_no_armed_block():
    assert ExecutionWrapper(execution_mode="readonly")._get_armed_block() == ""


def test_readwrite_run_emits_the_armed_block_and_no_readonly_guard():
    wrapper = ExecutionWrapper(execution_mode="readwrite")
    assert wrapper._get_readonly_guard() == ""
    args, _kwargs = _install_literals(wrapper._get_armed_block())
    assert args == ["armed"]


@pytest.mark.parametrize("validator", [None, "real"])
def test_block_is_emitted_with_and_without_a_limits_validator(validator):
    limits = _validator() if validator else None
    block = ExecutionWrapper(limits_validator=limits, execution_mode="readwrite")._get_armed_block()
    assert _install_literals(block)[0] == ["armed"]


# --------------------------------------------------------------------------
# The literals the install receives
# --------------------------------------------------------------------------


def test_blocked_targets_are_the_write_surface_table_grouped_in_order():
    _args, kwargs = _install_literals(
        ExecutionWrapper(execution_mode="readwrite")._get_armed_block()
    )
    rows = kwargs["blocked_targets"]
    assert [dotted for dotted, _attrs in rows] == list(_grouped(_ARMED_BLOCKED))
    assert {dotted: list(attrs) for dotted, attrs in rows} == _grouped(_ARMED_BLOCKED)
    assert {(d, a) for d, attrs in rows for a in attrs} == set(_ARMED_BLOCKED)


def test_rpc_targets_are_the_write_surface_table_grouped_in_order():
    _args, kwargs = _install_literals(
        ExecutionWrapper(execution_mode="readwrite")._get_armed_block()
    )
    rows = kwargs["rpc_targets"]
    assert [dotted for dotted, _attrs in rows] == list(_grouped(_ARMED_RPC))
    assert {dotted: list(attrs) for dotted, attrs in rows} == _grouped(_ARMED_RPC)


def test_passed_rows_are_neither_blocked_nor_refused():
    _args, kwargs = _install_literals(
        ExecutionWrapper(execution_mode="readwrite")._get_armed_block()
    )
    emitted = {
        (d, a) for d, attrs in kwargs["blocked_targets"] + kwargs["rpc_targets"] for a in attrs
    }
    from osprey.services.python_executor.write_surface import _ARMED_PASSED

    assert emitted.isdisjoint(_ARMED_PASSED)


def test_pvaccess_puts_are_left_to_the_limits_guard():
    """The armed block does not refuse a PVAccess put: the connector cannot
    carry one yet, so the run's limits guard checks it instead."""
    from osprey.services.python_executor.write_surface import _ARMED_CHECKED

    _args, kwargs = _install_literals(
        ExecutionWrapper(execution_mode="readwrite")._get_armed_block()
    )
    emitted = {
        (d, a) for d, attrs in kwargs["blocked_targets"] + kwargs["rpc_targets"] for a in attrs
    }
    assert _ARMED_CHECKED
    assert emitted.isdisjoint(_ARMED_CHECKED)


@pytest.mark.parametrize(("validator", "expected"), [(None, False), ("real", True)])
def test_refuse_rpc_follows_the_limits_validator(validator, expected):
    limits = _validator() if validator else None
    block = ExecutionWrapper(limits_validator=limits, execution_mode="readwrite")._get_armed_block()
    assert _install_literals(block)[1]["refuse_rpc"] is expected


def test_marker_is_the_raw_client_write_marker_and_not_the_readonly_one():
    _args, kwargs = _install_literals(
        ExecutionWrapper(execution_mode="readwrite")._get_armed_block()
    )
    assert kwargs["marker"] == RAW_CLIENT_WRITE_MARKER
    assert READONLY_REFUSAL_MARKER not in kwargs["marker"]


def test_every_rpc_row_resolves_to_its_refusal_text():
    """The install raises on an rpc row without text; every row must have one."""
    _args, kwargs = _install_literals(
        ExecutionWrapper(execution_mode="readwrite")._get_armed_block()
    )
    texts = kwargs["rpc_refusals"]
    assert texts == ARMED_RPC_REFUSALS
    for dotted, attr in _ARMED_RPC:
        expected = P4P_RPC_TEXT if dotted.startswith("p4p.") else TANGO_COMMAND_TEXT
        assert _refusal_text(texts, dotted, attr) == expected, (dotted, attr)


def test_emitted_contract_is_the_shared_contract():
    for limits, refuse in ((None, False), (_validator(), True)):
        block = ExecutionWrapper(
            limits_validator=limits, execution_mode="readwrite"
        )._get_armed_block()
        assert _install_literals(block)[1] == armed_contract(refuse_rpc=refuse)


def test_the_install_is_a_bare_top_level_call():
    """No ``try`` around it: a failed install must abort the script."""
    block = ExecutionWrapper(execution_mode="readwrite")._get_armed_block()
    tree = ast.parse(block)
    assert not any(isinstance(node, ast.Try) for node in ast.walk(tree))
    _install_call(block)  # found among the module's top-level statements


# --------------------------------------------------------------------------
# Where the block sits in the whole wrapper
# --------------------------------------------------------------------------


@pytest.mark.parametrize("validator", [None, "real"])
def test_block_sits_after_limits_checking_and_before_user_code(tmp_path, validator):
    limits = _validator() if validator else None
    script = ExecutionWrapper(limits_validator=limits, execution_mode="readwrite").create_wrapper(
        "USER_CODE_SENTINEL = 1", tmp_path
    )
    install = script.index('    "armed",')
    assert install < script.index("USER_CODE_SENTINEL")
    assert install < script.index("OSPREY filesystem guard")
    if limits is not None:
        injection = script.index("_runtime_module._limits_validator = _limits_validator")
        assert injection < install
    else:
        assert "_limits_validator = _limits_validator" not in script


def test_readonly_wrapper_carries_no_armed_install(tmp_path):
    # Matched on the emitted call's argument line: the embedded engine source
    # names both modes in its own docstring.
    script = ExecutionWrapper(execution_mode="readonly").create_wrapper("x = 1", tmp_path)
    assert '\n    "armed",\n' not in script
    assert '\n    "readonly",\n' in script


# --------------------------------------------------------------------------
# Behaviour, in a real subprocess
# --------------------------------------------------------------------------

_FAKE_CLIENTS = textwrap.dedent(
    """
    import sys
    import types

    epics = types.ModuleType("epics")
    epics.__file__ = "<fake epics>"
    exec("def caput(pvname, value, **kw):\\n    return 'wrote'\\n", epics.__dict__)
    sys.modules["epics"] = epics

    tango = types.ModuleType("tango")
    tango.__file__ = "<fake tango>"
    exec(
        "class DeviceProxy:\\n"
        "    def __init__(self, name):\\n"
        "        self.name = name\\n"
        "    def write_attribute(self, attr, value):\\n"
        "        return 'wrote'\\n"
        "    def command_inout(self, cmd, *args):\\n"
        "        return 'ran'\\n",
        tango.__dict__,
    )
    sys.modules["tango"] = tango
    """
)

_PROBE = textwrap.dedent(
    """
    import json
    import sys

    from osprey_connectors.control_system.write_door import open_door

    def attempt(call):
        try:
            return {"ok": call()}
        except Exception as error:
            return {
                "type": type(error).__name__,
                "reason": getattr(error, "reason", None),
                "text": str(error),
            }

    proxy = tango.DeviceProxy("sys/dev/1")
    out = {
        "caput": attempt(lambda: epics.caput("SR:PV", 1.0)),
        "tango_write": attempt(lambda: proxy.write_attribute("current", 1.0)),
        "tango_command": attempt(lambda: proxy.command_inout("On")),
    }
    with open_door():
        out["caput_in_door"] = attempt(lambda: epics.caput("SR:PV", 1.0))
    print("RESULT " + json.dumps(out))
    """
)


def _run_block(block: str) -> dict:
    script = "\n".join((_FAKE_CLIENTS, block, _PROBE))
    proc = subprocess.run(  # fixed argv, generated script
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=_env(),
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    line = next(line for line in proc.stdout.splitlines() if line.startswith("RESULT "))
    return json.loads(line.removeprefix("RESULT "))


def test_emitted_block_refuses_raw_puts_and_passes_door_puts():
    out = _run_block(ExecutionWrapper(execution_mode="readwrite")._get_armed_block())

    for name in ("caput", "tango_write"):
        assert out[name]["type"] == "ChannelWriteBlockedError", out[name]
        assert out[name]["reason"] == "RAW_CLIENT_WRITE"
        assert RAW_CLIENT_WRITE_MARKER in out[name]["text"]
    assert out["caput_in_door"] == {"ok": "wrote"}
    # Without a limits validator the rpc rows are left alone.
    assert out["tango_command"] == {"ok": "ran"}


def test_emitted_block_refuses_rpc_when_limits_checked():
    block = ExecutionWrapper(
        limits_validator=_validator(), execution_mode="readwrite"
    )._get_armed_block()
    out = _run_block(block)

    assert out["tango_command"] == {
        "type": "RuntimeError",
        "reason": None,
        "text": TANGO_COMMAND_TEXT,
    }
    assert out["caput"]["reason"] == "RAW_CLIENT_WRITE"


def test_failed_install_aborts_the_whole_wrapper_before_user_code(tmp_path, monkeypatch):
    """A block that cannot install stops the run; the user code never executes."""
    monkeypatch.setattr(wrapper_module, "RAW_CLIENT_WRITE_MARKER", "")
    sentinel = tmp_path / "user_code_ran.txt"
    script = tmp_path / "wrapped_script.py"
    script.write_text(
        ExecutionWrapper(execution_mode="readwrite").create_wrapper(
            f"open({str(sentinel)!r}, 'w').write('ran')", tmp_path
        ),
        encoding="utf-8",
    )

    proc = subprocess.run(  # fixed argv, generated script
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        env=_env(),
        timeout=300,
    )

    assert proc.returncode != 0
    assert "armed mode needs a non-empty refusal marker" in proc.stderr
    assert not sentinel.exists()


def test_whole_readwrite_wrapper_runs_clean_with_the_block(tmp_path):
    script = tmp_path / "wrapped_script.py"
    script.write_text(
        ExecutionWrapper(execution_mode="readwrite").create_wrapper(
            "results = {'value': 41}", tmp_path
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(  # fixed argv, generated script
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        env=_env(),
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads((tmp_path / "results.json").read_text()) == {"value": 41}
