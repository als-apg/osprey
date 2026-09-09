"""Tests for the p4p/PVAccess guards in the generated execution-wrapper monkeypatch.

The wrapper emits *source text* that runs inside the executor subprocess, so
these tests execute that generated source against fake ``p4p`` modules injected
into ``sys.modules``. Every other client the block imports is faked or blocked
too, so exec'ing it never patches a real install — pyepics, aioca and p4p are
all present in this environment, and a patch that escaped here would outlive
the test. The autouse fixture below is the second net under that.
"""

import asyncio
import contextlib
import gc
import importlib
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
from osprey.services.python_executor.write_surface import _CLIENT_WRITE_TARGETS

pytestmark = pytest.mark.unit

RPC_REFUSAL = "rpc is not mediated and cannot be approved"


# ---------------------------------------------------------------------------
# Restoration
# ---------------------------------------------------------------------------


def _resolve_target(dotted):
    """Resolve a write-surface row to the object that carries its attributes.

    A copy of the resolver the emitted guard uses, kept for the same reason
    ``test_readonly_guard.py`` keeps one: the guard has to be self-contained —
    it runs before user code in a subprocess that may not be able to import
    OSPREY at all — so there is no function to share.
    """
    parts = dotted.split(".")
    for cut in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:cut]))
        except ImportError:
            continue
        for attr in parts[cut:]:
            try:
                obj = getattr(obj, attr)
            except AttributeError:
                return None
        return obj
    return None


@pytest.fixture(autouse=True)
def _restore_patched_targets():
    """Undo any client patch that escaped the fakes, after every test.

    The block these tests exec is written to patch installed control-system
    clients, and it is exec'd in the test process. The fakes below are what
    normally absorbs that, but a client the fakes miss — a row added to the
    write surface, a spelling the harness does not yet stand in for — would
    otherwise leave the installed library patched for the remainder of the
    session, and unrelated tests would fail on a limits refusal far from here.

    Snapshotting happens before a test injects its fakes, so it captures the
    real objects; restoring writes back to those same objects and is therefore
    unaffected by whatever ``sys.modules`` held in between.
    """
    saved = []
    for dotted, attrs in _CLIENT_WRITE_TARGETS:
        target = _resolve_target(dotted)
        if target is None:
            continue
        for attr in attrs:
            if hasattr(target, attr):
                saved.append((target, attr, getattr(target, attr)))
    yield
    for target, attr, value in saved:
        setattr(target, attr, value)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class RawRecordingContext:
    """Stand-in for ``p4p.client.raw.Context``, the base every flavor subclasses.

    ``raw_puts`` records what reached the (fake) network through the raw layer.
    A flavor put lands here through ``super().put`` exactly as p4p's own do, so
    a test can tell a forwarded write apart from a re-validated one. The
    signature is the real one: a raw put names ONE channel and spells its value
    ``builder``.
    """

    def __init__(self):
        self.puts = []
        self.rpcs = []
        self.raw_puts = []

    def put(self, name, handler, builder=None, request=None, get=True):
        self.raw_puts.append((name, builder))
        return "raw-put-done"

    def rpc(self, name, handler, value=None, request=None):
        self.rpcs.append((name, value))
        return "raw-rpc-done"


class HandlerGetRawContext(RawRecordingContext):
    """A raw fake whose ``get`` carries the real signature: it answers a HANDLER.

    ``p4p.client.raw.Context.get(self, name, handler, request=None)`` hands the
    value to a callback instead of returning it, so the one-argument read the
    step check makes raises ``TypeError`` against it and the write fails closed
    on the reader-error branch. ``RawRecordingContext`` has no ``get`` at all,
    which fails closed one branch EARLIER — on "no reader supplied" — so
    without this fake the branch the installed library actually takes goes
    untested.
    """

    def get(self, name, handler, request=None):
        handler(1.0)
        return "raw-get-started"


class RecordingContext:
    """Stand-in for the flavor half of ``p4p.client.<flavor>.Context``.

    ``puts``/``rpcs`` record what actually reached the (fake) network, so a test
    can prove validation happened *before* any I/O. It is mixed with a fresh
    raw base per test rather than subclassing one here, so the guard's patch of
    the raw class cannot outlive the test that installed it.
    """

    def put(self, name, values, request=None, timeout=5.0, **kwargs):
        self.puts.append((name, values))
        super().put(name, None, builder=values)
        return "put-done"

    def rpc(self, name, value=None, request=None, timeout=5.0):
        self.rpcs.append((name, value))
        return "rpc-done"


class AsyncRecordingContext(RecordingContext):
    """Stand-in for the asyncio flavor, whose real ``put`` is a coroutine."""

    async def put(self, name, values, request=None, timeout=5.0, **kwargs):
        self.puts.append((name, values))
        super(RecordingContext, self).put(name, None, builder=values)
        return "put-done"


class _FakeChid:
    """What ``epics.ca`` addresses a channel by: an opaque id, not a name."""

    def __init__(self, pvname):
        self.pvname = pvname


def _make_fake_epics():
    """An ``epics`` whose writes funnel through ``epics.ca.put``, as pyepics' do.

    ``caput`` and ``PV.put`` reach the network through ``ca.put`` on a real
    install, and the fake keeps that chain so the block's pyepics branch finds
    the same shape here. ``ca`` is looked up on the module at call time, so a
    ``PV.put`` reaches whatever wrapper the block installed.
    """
    mod = ModuleType("epics")
    ca = ModuleType("epics.ca")

    def _ca_name(chid):
        return chid.pvname

    def _ca_create_channel(pvname, **kwargs):
        return _FakeChid(pvname)

    def _ca_put(chid, value, wait=False, timeout=60, **kwargs):
        return 1

    def _ca_get(chid, timeout=60, **kwargs):
        return 1.0

    ca.name = _ca_name
    ca.create_channel = _ca_create_channel
    ca.put = _ca_put
    ca.get = _ca_get

    class PV:
        def __init__(self, pvname):
            self.pvname = pvname
            self.chid = ca.create_channel(pvname)

        def put(self, value, wait=False, timeout=60, **kwargs):
            return ca.put(self.chid, value, wait=wait, timeout=timeout, **kwargs)

        def get(self, **kwargs):
            return ca.get(self.chid)

    def caput(pvname, value, wait=False, timeout=60, **kwargs):
        return PV(pvname).put(value, wait=wait, timeout=timeout, **kwargs)

    def caget(pvname, timeout=60, **kwargs):
        return ca.get(ca.create_channel(pvname), timeout=timeout)

    mod.ca = ca
    mod.PV = PV
    mod.caput = caput
    mod.caget = caget
    return mod


def _install_fake_p4p(
    monkeypatch, flavors=("thread", "asyncio"), bases=None, broken=(), raw_base=None
):
    """Inject a fake ``p4p`` package exposing a fresh Context per flavor.

    Returns ``{flavor: context_class}`` plus ``"raw"`` for the base class, which
    is created fresh per call: the guard patches it, and a class shared between
    tests would carry that patch into the next one. Each flavor class is built
    on that base, as p4p's own are, so a flavor put re-enters the raw guard the
    way the real client does. Flavors not listed are left unimportable, so the
    "not available" branch is exercised for them. ``bases`` overrides the flavor
    half of a class; ``raw_base`` overrides the base the raw Context is built
    on; flavors named in ``broken`` raise ``RuntimeError`` when their
    ``Context`` is imported.
    """
    bases = bases or {}
    p4p_mod = ModuleType("p4p")
    client_mod = ModuleType("p4p.client")
    p4p_mod.client = client_mod
    monkeypatch.setitem(sys.modules, "p4p", p4p_mod)
    monkeypatch.setitem(sys.modules, "p4p.client", client_mod)

    raw_mod = ModuleType("p4p.client.raw")
    raw_cls = type("RawContext", (raw_base or RawRecordingContext,), {})
    raw_mod.Context = raw_cls
    client_mod.raw = raw_mod
    monkeypatch.setitem(sys.modules, "p4p.client.raw", raw_mod)

    classes = {"raw": raw_cls}
    for flavor in flavors:
        flavor_mod = ModuleType(f"p4p.client.{flavor}")
        if flavor in broken:

            def _boom(attr, _flavor=flavor):
                raise RuntimeError(f"{_flavor} client is broken")

            flavor_mod.__getattr__ = _boom
        else:
            ctx_cls = type(
                f"{flavor.capitalize()}Context",
                (bases.get(flavor, RecordingContext), raw_cls),
                {},
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
    # The C extension is installed here and nothing fakes it, so a test that
    # reached it would patch the real library.
    monkeypatch.setitem(sys.modules, "p4p._p4p", None)
    return classes


def _block_p4p(monkeypatch):
    """Make every p4p import fail, whatever the host actually has installed."""
    for name in (
        "p4p",
        "p4p.client",
        "p4p.client.thread",
        "p4p.client.asyncio",
        "p4p.client.cothread",
        "p4p.client.raw",
        "p4p._p4p",
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
        "TEST:STEP:SP": ChannelLimitsConfig(
            channel_address="TEST:STEP:SP",
            min_value=0.0,
            max_value=10.0,
            max_step=1.0,
            writable=True,
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


def _count_validations(namespace):
    """Record every ``validate`` the installed guards make; return the log.

    The guards close over the validator OBJECT, so shadowing the bound method
    on that instance counts calls wherever they come from. This is what tells a
    put forwarded through the raw base apart from one validated twice on its
    way there.
    """
    validator = namespace["_limits_validator"]
    original = validator.validate
    calls = []

    def _counting(channel_address, value, **kwargs):
        calls.append((channel_address, value))
        return original(channel_address, value, **kwargs)

    validator.validate = _counting
    return calls


def _run_monkeypatch(monkeypatch, validator=None):
    """Execute the generated monkeypatch block; return ``(namespace, stdout)``."""
    validator = validator or _make_validator()
    fake_epics = _make_fake_epics()
    monkeypatch.setitem(sys.modules, "epics", fake_epics)
    monkeypatch.setitem(sys.modules, "epics.ca", fake_epics.ca)

    # The other clients the block imports are made absent rather than faked:
    # this module is about p4p, and aioca is really installed here — exec'ing
    # the block against it would rebind the installed library for the rest of
    # the session. Their behaviour is covered in test_client_limits_monkeypatch.
    for name in ("aioca", "aioca._catools", "pvaccess"):
        monkeypatch.setitem(sys.modules, name, None)

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
    assert "from p4p.client.raw import Context" in source
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

    for flavor in ("thread", "asyncio", "cothread", "raw"):
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


# ---------------------------------------------------------------------------
# Payloads that carry no number to check
# ---------------------------------------------------------------------------


def test_builder_callable_raises_before_any_call(monkeypatch):
    """p4p invokes a builder later, so there is nothing to check now."""
    classes = _install_fake_p4p(monkeypatch)
    _run_monkeypatch(monkeypatch)

    ctxt = classes["thread"]()
    with pytest.raises(ValueError, match="builder callable"):
        ctxt.put("TEST:MAG:SP", lambda value: value)
    assert ctxt.puts == []
    assert ctxt.raw_puts == []


def test_dict_payload_out_of_range_raises(monkeypatch):
    """A structure must be checked on the number it carries, not waved through."""
    classes = _install_fake_p4p(monkeypatch)
    _run_monkeypatch(monkeypatch)

    ctxt = classes["thread"]()
    with pytest.raises(ChannelLimitsViolationError):
        ctxt.put("TEST:MAG:SP", {"value": 99.0})
    assert ctxt.puts == []


def test_dict_payload_within_limits_forwards_unreduced(monkeypatch):
    """The client is handed the structure it was given, not the number."""
    classes = _install_fake_p4p(monkeypatch)
    _run_monkeypatch(monkeypatch)

    ctxt = classes["thread"]()
    assert ctxt.put("TEST:MAG:SP", {"value": 5.0}) == "put-done"
    assert ctxt.puts == [("TEST:MAG:SP", {"value": 5.0})]


def test_dict_payload_without_a_value_field_fails_closed(monkeypatch):
    classes = _install_fake_p4p(monkeypatch)
    _run_monkeypatch(monkeypatch)

    ctxt = classes["thread"]()
    with pytest.raises(ValueError, match="no 'value' field"):
        ctxt.put("TEST:MAG:SP", {"severity": 0})
    assert ctxt.puts == []


def test_value_object_payload_out_of_range_raises(monkeypatch):
    """A p4p ``Value`` carries the number in a ``value`` field."""

    class FakeValue:
        value = 99.0

    classes = _install_fake_p4p(monkeypatch)
    _run_monkeypatch(monkeypatch)

    ctxt = classes["thread"]()
    with pytest.raises(ChannelLimitsViolationError):
        ctxt.put("TEST:MAG:SP", FakeValue())
    assert ctxt.puts == []


def test_json_string_payload_out_of_range_raises(monkeypatch):
    """p4p decodes a '{'-leading string itself, AFTER the guard has run.

    ``Context.put`` does ``json.loads`` on this payload inside its own loop, so
    a guard that stopped at scalar/dict/``Value`` would hand the validator a
    str, fail ``float()``, skip every check, and let p4p write the number it
    decoded.
    """
    classes = _install_fake_p4p(monkeypatch)
    _run_monkeypatch(monkeypatch)

    ctxt = classes["thread"]()
    with pytest.raises(ChannelLimitsViolationError):
        ctxt.put("TEST:MAG:SP", '{"value": 999.0}')
    assert ctxt.puts == []
    assert ctxt.raw_puts == []


def test_json_string_payload_within_limits_forwards_exactly_once(monkeypatch):
    """In range, the client is handed the STRING it was given, checked once."""
    classes = _install_fake_p4p(monkeypatch)
    namespace, _ = _run_monkeypatch(monkeypatch)
    validations = _count_validations(namespace)

    ctxt = classes["thread"]()
    assert ctxt.put("TEST:MAG:SP", '{"value": 5.0}') == "put-done"
    assert validations == [("TEST:MAG:SP", 5.0)]
    assert ctxt.puts == [("TEST:MAG:SP", '{"value": 5.0}')]
    assert ctxt.raw_puts == [("TEST:MAG:SP", '{"value": 5.0}')]


def test_json_string_payload_without_a_value_field_fails_closed(monkeypatch):
    classes = _install_fake_p4p(monkeypatch)
    _run_monkeypatch(monkeypatch)

    ctxt = classes["thread"]()
    with pytest.raises(ValueError, match="no 'value' field"):
        ctxt.put("TEST:MAG:SP", '{"severity": 0}')
    assert ctxt.puts == []


def test_undecodable_json_string_payload_fails_closed(monkeypatch):
    """p4p refuses this payload too — the guard must not wave it through."""
    classes = _install_fake_p4p(monkeypatch)
    _run_monkeypatch(monkeypatch)

    ctxt = classes["thread"]()
    with pytest.raises(ValueError, match="does not decode"):
        ctxt.put("TEST:MAG:SP", "{value: 999.0")
    assert ctxt.puts == []


def test_json_bytes_payload_out_of_range_raises(monkeypatch):
    """Decoded one spelling stricter than p4p, which errs closed.

    p4p's own test is ``value[:1] == '{'``, which a bytes payload never
    matches, so p4p forwards it undecoded. Refusing an out-of-range one here
    costs a string write that begins with a brace and buys immunity to a p4p
    that later decodes bytes too.
    """
    classes = _install_fake_p4p(monkeypatch)
    _run_monkeypatch(monkeypatch)

    ctxt = classes["thread"]()
    with pytest.raises(ChannelLimitsViolationError):
        ctxt.put("TEST:MAG:SP", b'{"value": 999.0}')
    assert ctxt.puts == []


def test_plain_string_payload_is_not_treated_as_json(monkeypatch):
    """Only a '{'-leading string is a structure; the rest is a string write."""
    classes = _install_fake_p4p(monkeypatch)
    _run_monkeypatch(monkeypatch)

    ctxt = classes["thread"]()
    assert ctxt.put("TEST:MAG:SP", "Off") == "put-done"
    assert ctxt.puts == [("TEST:MAG:SP", "Off")]


def test_value_object_without_a_value_field_fails_closed(monkeypatch):
    """A ``Value`` carrying other fields fails closed like a dict does.

    p4p spells "is this field present" as ``Value.has(name)``, and a real
    ``Value`` with no ``value`` field answers the ``getattr`` fallback with
    ITSELF — which the validator then skips as non-numeric. Dicts already fail
    closed here; this is the same trade for the other structure spelling.
    """

    class FakeValueWithoutValue:
        def has(self, name):
            return name == "severity"

        severity = 0

    classes = _install_fake_p4p(monkeypatch)
    _run_monkeypatch(monkeypatch)

    ctxt = classes["thread"]()
    with pytest.raises(ValueError, match="no 'value' field"):
        ctxt.put("TEST:MAG:SP", FakeValueWithoutValue())
    assert ctxt.puts == []


def test_value_object_with_a_value_field_is_checked_on_its_number(monkeypatch):
    """The ``has()`` probe must not get in the way of a well-formed ``Value``."""

    class FakeValueWithValue:
        value = 99.0

        def has(self, name):
            return name == "value"

    classes = _install_fake_p4p(monkeypatch)
    _run_monkeypatch(monkeypatch)

    ctxt = classes["thread"]()
    with pytest.raises(ChannelLimitsViolationError):
        ctxt.put("TEST:MAG:SP", FakeValueWithValue())
    assert ctxt.puts == []


def test_batch_put_reduces_every_pair(monkeypatch):
    classes = _install_fake_p4p(monkeypatch)
    _run_monkeypatch(monkeypatch)

    ctxt = classes["thread"]()
    with pytest.raises(ChannelLimitsViolationError):
        ctxt.put(["TEST:MAG:SP", "TEST:OTHER:SP"], [{"value": 1.0}, {"value": 999.0}])
    assert ctxt.puts == []


# ---------------------------------------------------------------------------
# The raw base: the direct route, and the flavours' way through it
# ---------------------------------------------------------------------------


def test_direct_raw_put_out_of_bounds_raises(monkeypatch):
    """A script driving raw.Context itself is checked like any other client."""
    classes = _install_fake_p4p(monkeypatch)
    _run_monkeypatch(monkeypatch)

    ctxt = classes["raw"]()
    with pytest.raises(ChannelLimitsViolationError):
        ctxt.put("TEST:MAG:SP", None, builder=99.0)
    assert ctxt.raw_puts == []


def test_direct_raw_put_within_limits_reaches_the_client(monkeypatch):
    classes = _install_fake_p4p(monkeypatch)
    _run_monkeypatch(monkeypatch)

    ctxt = classes["raw"]()
    assert ctxt.put("TEST:MAG:SP", None, builder=5.0) == "raw-put-done"
    assert ctxt.raw_puts == [("TEST:MAG:SP", 5.0)]


def test_direct_raw_put_with_a_builder_callable_raises(monkeypatch):
    classes = _install_fake_p4p(monkeypatch)
    _run_monkeypatch(monkeypatch)

    ctxt = classes["raw"]()
    with pytest.raises(ValueError, match="builder callable"):
        ctxt.put("TEST:MAG:SP", None, builder=lambda value: value)
    assert ctxt.raw_puts == []


def test_direct_raw_put_to_an_unlisted_channel_is_blocked(monkeypatch):
    classes = _install_fake_p4p(monkeypatch)
    _run_monkeypatch(monkeypatch)

    ctxt = classes["raw"]()
    with pytest.raises(ChannelLimitsViolationError):
        ctxt.put("TEST:NOT:IN:DB", None, builder=1.0)
    assert ctxt.raw_puts == []


def test_raw_put_to_a_max_step_channel_fails_closed(monkeypatch):
    """Raw ``get`` answers a handler, so the step read cannot be made.

    The wrapper claims a max_step channel written straight through raw fails
    CLOSED. With a real-signature ``get`` installed on the raw fake, the
    guard's one-argument read raises ``TypeError`` and the validator turns any
    reader error into a refusal — which is the branch the installed library
    takes, rather than the "no reader supplied" one a ``get``-less fake hits.
    """
    classes = _install_fake_p4p(monkeypatch, raw_base=HandlerGetRawContext)
    _run_monkeypatch(monkeypatch)

    ctxt = classes["raw"]()
    with pytest.raises(ChannelLimitsViolationError) as excinfo:
        ctxt.put("TEST:STEP:SP", None, builder=5.0)
    assert excinfo.value.violation_type == "STEP_CHECK_FAILED"
    assert "Channel read failed" in excinfo.value.violation_reason
    assert ctxt.raw_puts == []


def test_raw_put_to_a_max_step_channel_fails_closed_without_a_reader(monkeypatch):
    """The other closed branch: a raw context exposing no ``get`` at all."""
    classes = _install_fake_p4p(monkeypatch)
    _run_monkeypatch(monkeypatch)

    ctxt = classes["raw"]()
    with pytest.raises(ChannelLimitsViolationError) as excinfo:
        ctxt.put("TEST:STEP:SP", None, builder=5.0)
    assert excinfo.value.violation_type == "STEP_CHECK_FAILED"
    assert "No way to read" in excinfo.value.violation_reason
    assert ctxt.raw_puts == []


def test_raw_rpc_is_refused(monkeypatch):
    classes = _install_fake_p4p(monkeypatch)
    _run_monkeypatch(monkeypatch)

    ctxt = classes["raw"]()
    with pytest.raises(RuntimeError, match=RPC_REFUSAL):
        ctxt.rpc("TEST:MAG:SP", None, 1.0)
    assert ctxt.rpcs == []


def test_flavor_put_forwards_through_raw_exactly_once(monkeypatch):
    """The re-entry through ``super().put`` must not be validated a second time."""
    classes = _install_fake_p4p(monkeypatch)
    namespace, _ = _run_monkeypatch(monkeypatch)
    validations = _count_validations(namespace)

    ctxt = classes["thread"]()
    assert ctxt.put("TEST:MAG:SP", 5.0) == "put-done"
    assert validations == [("TEST:MAG:SP", 5.0)]
    assert ctxt.raw_puts == [("TEST:MAG:SP", 5.0)]


def test_flavor_put_out_of_bounds_never_reaches_raw(monkeypatch):
    classes = _install_fake_p4p(monkeypatch)
    _run_monkeypatch(monkeypatch)

    ctxt = classes["thread"]()
    with pytest.raises(ChannelLimitsViolationError):
        ctxt.put("TEST:MAG:SP", 99.0)
    assert ctxt.raw_puts == []


def test_raw_guard_reports_itself_installed(monkeypatch):
    _install_fake_p4p(monkeypatch)
    _, out = _run_monkeypatch(monkeypatch)

    assert "Monkeypatched p4p.client.raw" in out
