"""Tests for the p4p/PVAccess guards in the generated execution-wrapper monkeypatch.

The wrapper emits *source text* that runs inside the executor subprocess, so
these tests execute that generated source against fake ``p4p`` modules injected
into ``sys.modules``. A fake ``epics`` module is injected too, so exec'ing the
block never patches a real pyepics install in the test process.
"""

import asyncio
import contextlib
import gc
import io
import sys
from types import ModuleType

import pytest

from osprey.connectors.control_system.limits_validator import (
    ChannelLimitsConfig,
    LimitsValidator,
)
from osprey.errors import ChannelLimitsViolationError
from osprey.services.python_executor.execution.wrapper import ExecutionWrapper

pytestmark = pytest.mark.unit

RPC_REFUSAL = "rpc is not mediated and cannot be approved"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class RecordingContext:
    """Stand-in for ``p4p.client.<flavor>.Context``.

    ``puts``/``rpcs`` record what actually reached the (fake) network, so a test
    can prove validation happened *before* any I/O.
    """

    def __init__(self):
        self.puts = []
        self.rpcs = []

    def put(self, name, values, request=None, timeout=5.0, **kwargs):
        self.puts.append((name, values))
        return "put-done"

    def rpc(self, name, value=None, request=None, timeout=5.0):
        self.rpcs.append((name, value))
        return "rpc-done"


class AsyncRecordingContext(RecordingContext):
    """Stand-in for the asyncio flavor, whose real ``put`` is a coroutine."""

    async def put(self, name, values, request=None, timeout=5.0, **kwargs):
        self.puts.append((name, values))
        return "put-done"


def _make_fake_epics():
    mod = ModuleType("epics")

    def caput(pvname, value, wait=False, timeout=60, **kwargs):
        return 1

    class PV:
        def __init__(self, pvname):
            self.pvname = pvname

        def put(self, value, wait=False, timeout=60, **kwargs):
            return 1

    mod.caput = caput
    mod.PV = PV
    return mod


def _install_fake_p4p(monkeypatch, flavors=("thread", "asyncio"), bases=None, broken=()):
    """Inject a fake ``p4p`` package exposing a fresh Context per flavor.

    Returns ``{flavor: context_class}``. Flavors not listed are left
    unimportable, so the "not available" branch is exercised for them.
    ``bases`` overrides the base class for a flavor; flavors named in
    ``broken`` raise ``RuntimeError`` when their ``Context`` is imported.
    """
    bases = bases or {}
    p4p_mod = ModuleType("p4p")
    client_mod = ModuleType("p4p.client")
    p4p_mod.client = client_mod
    monkeypatch.setitem(sys.modules, "p4p", p4p_mod)
    monkeypatch.setitem(sys.modules, "p4p.client", client_mod)

    classes = {}
    for flavor in flavors:
        flavor_mod = ModuleType(f"p4p.client.{flavor}")
        if flavor in broken:

            def _boom(attr, _flavor=flavor):
                raise RuntimeError(f"{_flavor} client is broken")

            flavor_mod.__getattr__ = _boom
        else:
            ctx_cls = type(
                f"{flavor.capitalize()}Context", (bases.get(flavor, RecordingContext),), {}
            )
            flavor_mod.Context = ctx_cls
            classes[flavor] = ctx_cls
        setattr(client_mod, flavor, flavor_mod)
        monkeypatch.setitem(sys.modules, f"p4p.client.{flavor}", flavor_mod)

    # Flavors we did not create must fail to import even if p4p is installed
    # on this host (Linux VA images ship it; macOS dev machines do not).
    for flavor in ("thread", "asyncio", "cothread"):
        if flavor not in flavors:
            monkeypatch.setitem(sys.modules, f"p4p.client.{flavor}", None)
    return classes


def _block_p4p(monkeypatch):
    """Make every p4p import fail, whatever the host actually has installed."""
    for name in (
        "p4p",
        "p4p.client",
        "p4p.client.thread",
        "p4p.client.asyncio",
        "p4p.client.cothread",
    ):
        monkeypatch.setitem(sys.modules, name, None)


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def _make_validator():
    limits = {
        "TEST:MAG:SP": ChannelLimitsConfig(
            channel_address="TEST:MAG:SP", min_value=0.0, max_value=10.0, writable=True
        ),
        "TEST:OTHER:SP": ChannelLimitsConfig(
            channel_address="TEST:OTHER:SP", min_value=0.0, max_value=10.0, writable=True
        ),
        "TEST:RO": ChannelLimitsConfig(channel_address="TEST:RO", writable=False),
    }
    return LimitsValidator(limits, {"allow_unlisted_channels": False})


def _make_permissive_validator():
    """Same limits, but unlisted channels are allowed.

    This is the configuration in which mis-classifying a batch as a scalar is
    a real bypass: the "name" (a list/array/generator object) is unlisted, so
    a scalar validation of it would simply be waved through while p4p went on
    to write every channel in the batch.
    """
    strict = _make_validator()
    return LimitsValidator(strict.limits, {"allow_unlisted_channels": True})


def _run_monkeypatch(monkeypatch, validator=None):
    """Execute the generated monkeypatch block; return ``(namespace, stdout)``."""
    validator = validator or _make_validator()
    monkeypatch.setitem(sys.modules, "epics", _make_fake_epics())

    # The block injects the validator into osprey.runtime as a side effect.
    import osprey.runtime as runtime_module

    monkeypatch.setattr(runtime_module, "_limits_validator", None, raising=False)

    source = ExecutionWrapper(limits_validator=validator)._get_limits_checking_monkeypatch()
    namespace: dict = {}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(source, "<generated-wrapper>", "exec"), namespace)
    out = buf.getvalue()
    # The whole block is wrapped in a broad try/except that prints and swallows.
    assert "Limits checking setup failed" not in out, out
    return namespace, out


# ---------------------------------------------------------------------------
# Generated-source shape
# ---------------------------------------------------------------------------


def test_generated_source_patches_every_p4p_flavor():
    wrapper = ExecutionWrapper(limits_validator=_make_validator())
    source = wrapper._get_limits_checking_monkeypatch()
    assert "from p4p.client.thread import Context" in source
    assert "from p4p.client.asyncio import Context" in source
    assert "from p4p.client.cothread import Context" in source
    # Fail-closed batch iteration, not a truncating plain zip.
    assert "strict=True" in source
    # Scalar-vs-batch keys on str, exactly as p4p's own put() does.
    assert "isinstance(_name, str)" in source
    assert RPC_REFUSAL in source


def test_no_p4p_code_when_limits_checking_disabled():
    assert ExecutionWrapper(limits_validator=None)._get_limits_checking_monkeypatch() == ""


# ---------------------------------------------------------------------------
# Scalar puts
# ---------------------------------------------------------------------------


def test_scalar_put_within_limits_reaches_the_client(monkeypatch):
    classes = _install_fake_p4p(monkeypatch)
    _run_monkeypatch(monkeypatch)

    ctxt = classes["thread"]()
    assert ctxt.put("TEST:MAG:SP", 5.0) == "put-done"
    assert ctxt.puts == [("TEST:MAG:SP", 5.0)]


def test_scalar_put_out_of_bounds_raises_before_any_network_call(monkeypatch):
    classes = _install_fake_p4p(monkeypatch)
    _run_monkeypatch(monkeypatch)

    ctxt = classes["thread"]()
    with pytest.raises(ChannelLimitsViolationError):
        ctxt.put("TEST:MAG:SP", 99.0)
    assert ctxt.puts == []


def test_scalar_put_to_unlisted_channel_is_blocked(monkeypatch):
    classes = _install_fake_p4p(monkeypatch)
    _run_monkeypatch(monkeypatch)

    ctxt = classes["thread"]()
    with pytest.raises(ChannelLimitsViolationError):
        ctxt.put("TEST:NOT:IN:DB", 1.0)
    assert ctxt.puts == []


def test_scalar_put_to_read_only_channel_is_blocked(monkeypatch):
    classes = _install_fake_p4p(monkeypatch)
    _run_monkeypatch(monkeypatch)

    ctxt = classes["thread"]()
    with pytest.raises(ChannelLimitsViolationError):
        ctxt.put("TEST:RO", 1.0)
    assert ctxt.puts == []


def test_put_forwards_extra_arguments(monkeypatch):
    classes = _install_fake_p4p(monkeypatch)
    _run_monkeypatch(monkeypatch)

    ctxt = classes["thread"]()
    assert ctxt.put("TEST:MAG:SP", 1.0, timeout=1.5, wait=True) == "put-done"
    assert ctxt.puts == [("TEST:MAG:SP", 1.0)]


# ---------------------------------------------------------------------------
# Batch puts
# ---------------------------------------------------------------------------


def test_batch_put_validates_every_pair(monkeypatch):
    classes = _install_fake_p4p(monkeypatch)
    _run_monkeypatch(monkeypatch)

    ctxt = classes["thread"]()
    names = ["TEST:MAG:SP", "TEST:OTHER:SP"]
    values = [1.0, 2.0]
    assert ctxt.put(names, values) == "put-done"
    assert ctxt.puts == [(names, values)]


def test_batch_put_rejects_the_batch_when_one_pair_is_invalid(monkeypatch):
    classes = _install_fake_p4p(monkeypatch)
    _run_monkeypatch(monkeypatch)

    ctxt = classes["thread"]()
    with pytest.raises(ChannelLimitsViolationError):
        ctxt.put(["TEST:MAG:SP", "TEST:OTHER:SP"], [1.0, 999.0])
    assert ctxt.puts == []


def test_batch_put_pairs_names_with_values_positionally(monkeypatch):
    """A value legal for one channel must not launder a write to another."""
    classes = _install_fake_p4p(monkeypatch)
    _run_monkeypatch(monkeypatch)

    ctxt = classes["thread"]()
    with pytest.raises(ChannelLimitsViolationError):
        ctxt.put(["TEST:MAG:SP", "TEST:RO"], [1.0, 2.0])
    assert ctxt.puts == []


@pytest.mark.parametrize(
    ("names", "values"),
    [
        (["TEST:MAG:SP", "TEST:OTHER:SP"], [1.0]),
        (["TEST:MAG:SP"], [1.0, 2.0]),
    ],
)
def test_batch_put_length_mismatch_raises_before_any_network_call(monkeypatch, names, values):
    classes = _install_fake_p4p(monkeypatch)
    _run_monkeypatch(monkeypatch)

    ctxt = classes["thread"]()
    with pytest.raises(ValueError):
        ctxt.put(names, values)
    assert ctxt.puts == []


def test_batch_put_with_non_sequence_values_raises(monkeypatch):
    classes = _install_fake_p4p(monkeypatch)
    _run_monkeypatch(monkeypatch)

    ctxt = classes["thread"]()
    with pytest.raises(ValueError):
        ctxt.put(["TEST:MAG:SP", "TEST:OTHER:SP"], 1.0)
    assert ctxt.puts == []


# ---------------------------------------------------------------------------
# rpc
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flavor", ["thread", "asyncio"])
def test_rpc_is_refused_unconditionally(monkeypatch, flavor):
    classes = _install_fake_p4p(monkeypatch)
    _run_monkeypatch(monkeypatch)

    ctxt = classes[flavor]()
    with pytest.raises(RuntimeError, match=RPC_REFUSAL):
        ctxt.rpc("TEST:MAG:SP", 1.0)
    assert ctxt.rpcs == []


def test_rpc_is_refused_even_for_an_in_limits_channel(monkeypatch):
    classes = _install_fake_p4p(monkeypatch)
    _run_monkeypatch(monkeypatch)

    ctxt = classes["thread"]()
    with pytest.raises(RuntimeError, match="supervised write path"):
        ctxt.rpc("TEST:MAG:SP")
    assert ctxt.rpcs == []


# ---------------------------------------------------------------------------
# asyncio flavor
# ---------------------------------------------------------------------------


def test_asyncio_flavor_put_is_validated(monkeypatch):
    classes = _install_fake_p4p(monkeypatch)
    _run_monkeypatch(monkeypatch)

    ctxt = classes["asyncio"]()
    with pytest.raises(ChannelLimitsViolationError):
        ctxt.put("TEST:MAG:SP", 99.0)
    assert ctxt.puts == []
    assert ctxt.put("TEST:MAG:SP", 2.0) == "put-done"


def test_each_flavor_is_patched_independently(monkeypatch):
    """The thread and asyncio Context classes are distinct objects."""
    classes = _install_fake_p4p(monkeypatch)
    _run_monkeypatch(monkeypatch)

    assert classes["thread"].put is not RecordingContext.put
    assert classes["asyncio"].put is not RecordingContext.put
    assert classes["thread"].rpc is not RecordingContext.rpc
    assert classes["asyncio"].rpc is not RecordingContext.rpc


def test_one_missing_flavor_does_not_stop_the_others(monkeypatch):
    """cothread is absent here, yet thread/asyncio are still patched."""
    classes = _install_fake_p4p(monkeypatch, flavors=("thread", "asyncio"))
    _, out = _run_monkeypatch(monkeypatch)

    assert "p4p.client.cothread not available" in out
    assert "Monkeypatched p4p.client.thread" in out
    assert "Monkeypatched p4p.client.asyncio" in out
    ctxt = classes["asyncio"]()
    with pytest.raises(ChannelLimitsViolationError):
        ctxt.put("TEST:MAG:SP", 99.0)


def test_cothread_flavor_is_patched_when_importable(monkeypatch):
    classes = _install_fake_p4p(monkeypatch, flavors=("thread", "asyncio", "cothread"))
    _, out = _run_monkeypatch(monkeypatch)

    assert "Monkeypatched p4p.client.cothread" in out
    ctxt = classes["cothread"]()
    with pytest.raises(ChannelLimitsViolationError):
        ctxt.put("TEST:MAG:SP", 99.0)
    assert ctxt.puts == []


# ---------------------------------------------------------------------------
# p4p absent
# ---------------------------------------------------------------------------


def test_absent_p4p_prints_disabled_and_does_not_crash(monkeypatch):
    _block_p4p(monkeypatch)
    namespace, out = _run_monkeypatch(monkeypatch)

    for flavor in ("thread", "asyncio", "cothread"):
        assert f"p4p.client.{flavor} not available - PVA limits checking disabled" in out
    # The rest of the block still completed.
    assert "Runtime channel limits checking ENABLED" in out
    assert namespace["_limits_validator"] is not None


# ---------------------------------------------------------------------------
# Batch discrimination: anything that is not a str is a batch, as in p4p itself
# ---------------------------------------------------------------------------


def test_generator_of_names_is_batch_validated(monkeypatch):
    """A non-list iterable of names must not slip through the scalar path."""
    classes = _install_fake_p4p(monkeypatch)
    _run_monkeypatch(monkeypatch, _make_permissive_validator())

    ctxt = classes["thread"]()
    names = (n for n in ["TEST:MAG:SP", "TEST:RO"])
    with pytest.raises(ChannelLimitsViolationError):
        ctxt.put(names, [1.0, 2.0])
    assert ctxt.puts == []


def test_generator_of_names_forwards_a_reusable_sequence(monkeypatch):
    """Validating a one-shot iterable must not leave the client an empty batch."""
    classes = _install_fake_p4p(monkeypatch)
    _run_monkeypatch(monkeypatch)

    ctxt = classes["thread"]()
    names = ["TEST:MAG:SP", "TEST:OTHER:SP"]
    values = [1.0, 2.0]
    assert ctxt.put((n for n in names), values) == "put-done"
    assert ctxt.puts == [(tuple(names), values)]


def test_generator_of_names_length_mismatch_raises(monkeypatch):
    classes = _install_fake_p4p(monkeypatch)
    _run_monkeypatch(monkeypatch, _make_permissive_validator())

    ctxt = classes["thread"]()
    names = (n for n in ["TEST:MAG:SP", "TEST:OTHER:SP"])
    with pytest.raises(ValueError):
        ctxt.put(names, [1.0])
    assert ctxt.puts == []


def test_numpy_array_of_names_is_batch_validated(monkeypatch):
    np = pytest.importorskip("numpy")
    classes = _install_fake_p4p(monkeypatch)
    _run_monkeypatch(monkeypatch, _make_permissive_validator())

    ctxt = classes["thread"]()
    names = np.array(["TEST:MAG:SP", "TEST:OTHER:SP"])
    with pytest.raises(ChannelLimitsViolationError):
        ctxt.put(names, [1.0, 999.0])
    assert ctxt.puts == []

    assert ctxt.put(names, [1.0, 2.0]) == "put-done"
    assert ctxt.puts == [(tuple(names), [1.0, 2.0])]


def test_non_iterable_name_fails_closed(monkeypatch):
    classes = _install_fake_p4p(monkeypatch)
    _run_monkeypatch(monkeypatch, _make_permissive_validator())

    ctxt = classes["thread"]()
    with pytest.raises(ValueError):
        ctxt.put(42, [1.0])
    assert ctxt.puts == []


def test_unlisted_channel_in_an_iterable_batch_is_blocked(monkeypatch):
    classes = _install_fake_p4p(monkeypatch)
    _run_monkeypatch(monkeypatch)

    ctxt = classes["thread"]()
    with pytest.raises(ChannelLimitsViolationError):
        ctxt.put((n for n in ["TEST:MAG:SP", "TEST:NOT:IN:DB"]), [1.0, 2.0])
    assert ctxt.puts == []


# ---------------------------------------------------------------------------
# One broken flavor must not disarm the others
# ---------------------------------------------------------------------------


def test_a_flavor_that_raises_does_not_skip_the_remaining_flavors(monkeypatch):
    """A non-ImportError failure in one flavor is contained to that flavor."""
    classes = _install_fake_p4p(
        monkeypatch, flavors=("thread", "asyncio", "cothread"), broken=("thread",)
    )
    _, out = _run_monkeypatch(monkeypatch)

    # Contained: the outer swallow-all handler never saw it.
    assert "p4p.client.thread guard failed" in out
    assert "thread client is broken" in out
    assert "Monkeypatched p4p.client.asyncio" in out
    assert "Monkeypatched p4p.client.cothread" in out

    for flavor in ("asyncio", "cothread"):
        ctxt = classes[flavor]()
        with pytest.raises(ChannelLimitsViolationError):
            ctxt.put("TEST:MAG:SP", 99.0)
        assert ctxt.puts == []


# ---------------------------------------------------------------------------
# asyncio flavor with a genuinely async put()
# ---------------------------------------------------------------------------


def test_async_put_within_limits_is_awaitable(monkeypatch):
    classes = _install_fake_p4p(monkeypatch, bases={"asyncio": AsyncRecordingContext})
    _run_monkeypatch(monkeypatch)

    ctxt = classes["asyncio"]()
    assert asyncio.run(ctxt.put("TEST:MAG:SP", 2.0)) == "put-done"
    assert ctxt.puts == [("TEST:MAG:SP", 2.0)]


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_async_put_out_of_bounds_raises_synchronously(monkeypatch):
    """The violation surfaces at call time, so no unawaited coroutine is made."""
    classes = _install_fake_p4p(monkeypatch, bases={"asyncio": AsyncRecordingContext})
    _run_monkeypatch(monkeypatch)

    ctxt = classes["asyncio"]()
    with pytest.raises(ChannelLimitsViolationError):
        ctxt.put("TEST:MAG:SP", 99.0)  # no await
    gc.collect()  # a stray coroutine would warn "never awaited" here
    assert ctxt.puts == []
