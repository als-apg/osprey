"""Tests for the PVAccess limits guard a readwrite executor run installs.

The connector reads PVAccess but does not write it yet, so the armed raw-put
block lets raw ``p4p`` and ``pvaccess`` puts through
(``write_surface._ARMED_CHECKED``) and this guard limits-checks them instead,
as it did before that block existed. ``rpc`` is not the guard's: the armed
block refuses it (``test_armed_block.py``).

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


#: Marks a module as one of this suite's stand-ins, so ``_run_monkeypatch``
#: can tell a fake a test already installed from a real installed client.
_FAKE_CLIENT_MARKER = "_osprey_fake_client"


def _fake_module(name):
    """A module object marked as one of this suite's client stand-ins."""
    mod = ModuleType(name)
    setattr(mod, _FAKE_CLIENT_MARKER, True)
    return mod


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

    def put(self, name, handler, builder=None, request=None, get=True):  # noqa: ARG002 - p4p raw Context signature
        self.raw_puts.append((name, builder))
        return "raw-put-done"

    def rpc(self, name, handler, value=None, request=None):  # noqa: ARG002 - p4p raw Context signature
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

    def get(self, name, handler, request=None):  # noqa: ARG002 - p4p raw Context signature
        handler(1.0)
        return "raw-get-started"


class RecordingContext:
    """Stand-in for the flavor half of ``p4p.client.<flavor>.Context``.

    ``puts``/``rpcs`` record what actually reached the (fake) network, so a test
    can prove validation happened *before* any I/O. It is mixed with a fresh
    raw base per test rather than subclassing one here, so the guard's patch of
    the raw class cannot outlive the test that installed it.
    """

    def put(self, name, values, request=None, timeout=5.0, **kwargs):  # noqa: ARG002 - p4p Context signature
        self.puts.append((name, values))
        super().put(name, None, builder=values)
        return "put-done"

    def rpc(self, name, value=None, request=None, timeout=5.0):  # noqa: ARG002 - p4p Context signature
        self.rpcs.append((name, value))
        return "rpc-done"


class AsyncRecordingContext(RecordingContext):
    """Stand-in for the asyncio flavor, whose real ``put`` is a coroutine."""

    async def put(self, name, values, request=None, timeout=5.0, **kwargs):  # noqa: ARG002 - p4p Context signature
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

    def _ca_put(chid, value, wait=False, timeout=60, **kwargs):  # noqa: ARG001 - pyepics ca.put signature
        return 1

    def _ca_get(chid, timeout=60, **kwargs):  # noqa: ARG001 - pyepics ca.get signature
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

            def _boom(_attr, _flavor=flavor):
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


def _guard_source(validator):
    """The limits block and the PVAccess guard a readwrite run emits, in order."""
    wrapper = ExecutionWrapper(limits_validator=validator, execution_mode="readwrite")
    return "\n".join((wrapper._get_limits_checking_monkeypatch(), wrapper._get_pva_limits_guard()))


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
    for name in ("aioca", "aioca._catools"):
        monkeypatch.setitem(sys.modules, name, None)
    # pvaPy is faked by the tests that exercise it and absent for the rest.
    if getattr(sys.modules.get("pvaccess"), _FAKE_CLIENT_MARKER, False) is not True:
        monkeypatch.setitem(sys.modules, "pvaccess", None)

    # The block injects the validator into osprey.runtime as a side effect.
    import osprey.runtime as runtime_module

    monkeypatch.setattr(runtime_module, "_limits_validator", None, raising=False)

    source = _guard_source(validator)
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
    source = _guard_source(_make_validator())
    assert "from p4p.client.thread import Context" in source
    assert "from p4p.client.asyncio import Context" in source
    assert "from p4p.client.cothread import Context" in source
    assert "from p4p.client.raw import Context" in source
    # Fail-closed batch iteration, not a truncating plain zip.
    assert "strict=True" in source
    # Scalar-vs-batch keys on str, exactly as p4p's own put() does.
    assert "isinstance(_name, str)" in source
    # rpc belongs to the armed raw-put block, not to this guard.
    assert ".rpc = " not in source


def test_no_pva_guard_when_limits_checking_disabled():
    wrapper = ExecutionWrapper(limits_validator=None, execution_mode="readwrite")
    assert wrapper._get_pva_limits_guard() == ""


def test_no_pva_guard_in_a_readonly_run():
    """A readonly run refuses these puts outright; there is nothing to check."""
    wrapper = ExecutionWrapper(limits_validator=_make_validator(), execution_mode="readonly")
    assert wrapper._get_pva_limits_guard() == ""


def test_the_guard_says_so_when_limits_setup_failed():
    """Without a validator in scope the guard installs nothing and says why."""
    wrapper = ExecutionWrapper(limits_validator=_make_validator(), execution_mode="readwrite")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(wrapper._get_pva_limits_guard(), "<generated-wrapper>", "exec"), {})
    assert "PVAccess limits guard not installed" in buf.getvalue()


def test_the_guard_follows_the_limits_block_in_the_script():
    wrapper = ExecutionWrapper(limits_validator=_make_validator(), execution_mode="readwrite")
    script = wrapper.create_wrapper("pass")
    guard = wrapper._get_pva_limits_guard()
    assert guard and guard in script
    assert script.index(wrapper._get_limits_checking_monkeypatch()) < script.index(guard)
    assert script.index(guard) < script.index(wrapper._get_armed_block())


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
    # rpc is left for the armed raw-put block.
    assert classes["thread"].rpc is RecordingContext.rpc
    assert classes["asyncio"].rpc is RecordingContext.rpc


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


class _FakePvObject:
    """The structure a pvaccess ``Channel`` hands back and takes in.

    Shaped after the real binding, because pvaccess is not installed here and
    this fake is therefore the only evidence the guard binds correctly on it.
    pvaPy's no-arg ``getPyObject()`` answers the VALUE of the structure's
    ``value`` field and raises ``pvaccess.InvalidRequest`` when there is none;
    ``toDict()`` is the spelling that answers the whole structure as a dict.
    (The exception type is not part of the guard's contract — it lets whatever
    is raised propagate, and the write fails closed either way.)
    """

    def __init__(self, data):
        self._data = dict(data)

    def getPyObject(self):  # pvaccess spells it this way
        if "value" not in self._data:
            raise RuntimeError("PvObject has no value field")
        return self._data["value"]

    def toDict(self):  # pvaccess spells it this way
        return dict(self._data)


def _install_fake_pvaccess(monkeypatch, writes=None, reads=None, read=None):
    """Inject a ``pvaccess`` with the typed put family a Channel really carries.

    pvaPy spells a write as ``put`` plus a typed setter per scalar kind
    (``putDouble``, ``putInt``, …), which is why the block sweeps every
    ``put``-prefixed attribute rather than naming one. ``asyncPut``,
    ``parsePut`` and ``parsePutGet`` are the three writes whose names fall
    outside that prefix, and they are here so the sweep is tested against them.

    ``get`` answers a structure, as pvaPy's does, so a guard that reduces the
    payload but not the read is caught here. ``read`` replaces what it
    answers, and may raise: a channel that cannot be read is the case a step
    check has to fail closed on.
    """
    writes = [] if writes is None else writes
    reads = [] if reads is None else reads
    mod = _fake_module("pvaccess")

    class Channel:
        def __init__(self, name, provider=None):  # noqa: ARG002 - pvaPy Channel signature
            self._name = name
            self.current = 1.0

        def getName(self):  # pvaccess spells it this way
            return self._name

        def get(self, request=""):  # noqa: ARG002 - pvaPy Channel signature
            reads.append(self._name)
            if read is not None:
                return read(self._name)
            return _FakePvObject({"value": self.current})

        def put(self, value, request=""):  # noqa: ARG002 - pvaPy Channel signature
            writes.append((self._name, value))
            return "put-done"

        def putDouble(self, value, request=""):  # noqa: ARG002 - pvaPy Channel signature
            writes.append((self._name, value))
            return "put-done"

        def putGet(self, value, request=""):  # noqa: ARG002 - pvaPy Channel signature
            writes.append((self._name, value))
            return _FakePvObject({"value": value})

        def asyncPut(self, value, callback=None, request=""):  # noqa: ARG002 - pvaPy Channel signature
            # pvaPy's asynchronous write: a PvObject first, the completion
            # callback second. A write whose name does not start with "put".
            writes.append((self._name, value))
            return "async-put-done"

        def parsePut(self, args, request=""):  # noqa: ARG002 - pvaPy Channel signature
            # Takes a LIST OF JSON STRINGS, not a value object.
            writes.append((self._name, args))
            return "parse-put-done"

        def parsePutGet(self, args, request=""):  # noqa: ARG002 - pvaPy Channel signature
            writes.append((self._name, args))
            return _FakePvObject({"value": args})

    mod.Channel = Channel
    mod.PvObject = _FakePvObject
    monkeypatch.setitem(sys.modules, "pvaccess", mod)
    return mod


def _make_pva_validator():
    """The pvaPy tests' channels: one plain, one with ``max_step``."""
    limits = {
        "TEST:MAG:SP": ChannelLimitsConfig(
            channel_address="TEST:MAG:SP", min_value=0.0, max_value=10.0
        ),
        "TEST:MAG:STEP": ChannelLimitsConfig(
            channel_address="TEST:MAG:STEP",
            min_value=0.0,
            max_value=100.0,
            max_step=2.0,
        ),
    }
    return LimitsValidator(limits, {"allow_unlisted_channels": False})


# ---------------------------------------------------------------------------
# pvaPy (pvaccess)
# ---------------------------------------------------------------------------


def test_pvaccess_put_within_limits_reaches_the_channel(monkeypatch):
    writes: list = []
    reads: list = []
    mod = _install_fake_pvaccess(monkeypatch, writes, reads)
    _run_monkeypatch(monkeypatch, _make_pva_validator())

    assert mod.Channel("TEST:MAG:SP").put(5.0) == "put-done"
    assert writes == [("TEST:MAG:SP", 5.0)]
    # No max_step on this channel, so the guard buys no read for it.
    assert reads == []


def test_pvaccess_put_out_of_bounds_raises_before_the_channel(monkeypatch):
    """pvaPy is a PVAccess client of its own, not a p4p spelling.

    A ``pvaccess.Channel`` put never passes through a p4p ``Context``, so the
    p4p guards leave it unchecked; this is the guard that closes it.
    """
    writes: list = []
    mod = _install_fake_pvaccess(monkeypatch, writes)
    _run_monkeypatch(monkeypatch, _make_pva_validator())

    with pytest.raises(ChannelLimitsViolationError):
        mod.Channel("TEST:MAG:SP").put(99.0)
    assert writes == []


def test_pvaccess_typed_put_is_refused_out_of_range(monkeypatch):
    """The typed setters are the same write under another name.

    pvaPy spells one setter per scalar and array kind — ``putDouble``,
    ``putInt``, ``putScalarArray`` — so the guard sweeps every ``put``-prefixed
    attribute rather than naming ``put``. Naming them would go stale against
    the binding, and every name missed would be an unchecked write.
    """
    writes: list = []
    mod = _install_fake_pvaccess(monkeypatch, writes)
    _run_monkeypatch(monkeypatch, _make_pva_validator())

    with pytest.raises(ChannelLimitsViolationError):
        mod.Channel("TEST:MAG:SP").putDouble(99.0)
    assert writes == []


def test_pvaccess_structure_payload_is_reduced_before_validation(monkeypatch):
    """A ``PvObject`` payload is unwrapped to the number it carries.

    Handed the structure itself, the validator finds nothing numeric to check
    and lets the write past — so an out-of-range value dressed as a structure
    would reach the machine.
    """
    writes: list = []
    mod = _install_fake_pvaccess(monkeypatch, writes)
    _run_monkeypatch(monkeypatch, _make_pva_validator())

    with pytest.raises(ChannelLimitsViolationError):
        mod.Channel("TEST:MAG:SP").put(_FakePvObject({"value": 99.0}))
    assert writes == []


def test_pvaccess_dict_payload_is_reduced_before_validation(monkeypatch):
    """The plain dict a structure converts to is unwrapped the same way."""
    writes: list = []
    mod = _install_fake_pvaccess(monkeypatch, writes)
    _run_monkeypatch(monkeypatch, _make_pva_validator())

    with pytest.raises(ChannelLimitsViolationError):
        mod.Channel("TEST:MAG:SP").put({"value": 99.0})
    assert writes == []


def test_pvaccess_dict_without_a_value_field_fails_closed(monkeypatch):
    """A shape with no value under it is refused, not written unchecked."""
    writes: list = []
    mod = _install_fake_pvaccess(monkeypatch, writes)
    _run_monkeypatch(monkeypatch, _make_pva_validator())

    with pytest.raises(ValueError):
        mod.Channel("TEST:MAG:SP").put({"unit": "A"})
    assert writes == []


def test_pvaccess_callable_payload_fails_closed(monkeypatch):
    """A callable is not a value a limits database can be asked about."""
    writes: list = []
    mod = _install_fake_pvaccess(monkeypatch, writes)
    _run_monkeypatch(monkeypatch, _make_pva_validator())

    with pytest.raises(ValueError):
        mod.Channel("TEST:MAG:SP").put(lambda: 5.0)
    assert writes == []


def test_pvaccess_step_check_reads_through_the_channels_own_get(monkeypatch):
    """The step is measured over the same Channel the put goes through.

    pvaPy answers a get with a structure, so the reader reduces it exactly as
    it reduces a payload. Left unreduced, the structure is non-numeric, the
    validator SKIPS the step check, and a 49-unit step past a max of 2 reaches
    the machine — which is why this asserts the refusal is ``MAX_STEP_EXCEEDED``
    and not merely that something raised.
    """
    writes: list = []
    reads: list = []
    mod = _install_fake_pvaccess(monkeypatch, writes, reads)
    _run_monkeypatch(monkeypatch, _make_pva_validator())

    with pytest.raises(ChannelLimitsViolationError) as exc:
        mod.Channel("TEST:MAG:STEP").put(50.0)

    assert exc.value.violation_type == "MAX_STEP_EXCEEDED"
    assert reads == ["TEST:MAG:STEP"]
    assert writes == []


def test_pvaccess_put_within_max_step_reaches_the_channel(monkeypatch):
    """The reader is a real read, not a blanket refusal on max_step channels."""
    writes: list = []
    reads: list = []
    mod = _install_fake_pvaccess(monkeypatch, writes, reads)
    _run_monkeypatch(monkeypatch, _make_pva_validator())

    # The fake reads back 1.0, so this is a step of 1.0 against a max of 2.0.
    assert mod.Channel("TEST:MAG:STEP").put(2.0) == "put-done"
    assert reads == ["TEST:MAG:STEP"]
    assert writes == [("TEST:MAG:STEP", 2.0)]


def test_pvaccess_step_check_fails_closed_when_the_read_fails(monkeypatch):
    """A Channel that cannot be read refuses the write rather than skipping it."""
    writes: list = []
    reads: list = []

    def _raise(_name):
        raise OSError("channel unreachable")

    mod = _install_fake_pvaccess(monkeypatch, writes, reads, read=_raise)
    _run_monkeypatch(monkeypatch, _make_pva_validator())

    with pytest.raises(ChannelLimitsViolationError) as exc:
        mod.Channel("TEST:MAG:STEP").put(2.0)

    assert exc.value.violation_type == "STEP_CHECK_FAILED"
    assert writes == []


def test_pvaccess_async_put_out_of_range_is_refused(monkeypatch):
    """``asyncPut`` is a write whose name does not start with ``put``.

    pvaPy spells the asynchronous write ``asyncPut(pvObject, callback)`` — the
    same put with a completion callback bolted on, and the same value going to
    the machine. A sweep keyed on the ``put`` prefix alone leaves it out, so an
    approved run could send an out-of-range value with no check at all.
    """
    writes: list = []
    mod = _install_fake_pvaccess(monkeypatch, writes)
    _run_monkeypatch(monkeypatch, _make_pva_validator())

    with pytest.raises(ChannelLimitsViolationError):
        mod.Channel("TEST:MAG:SP").asyncPut(_FakePvObject({"value": 99.0}), lambda _r: None)
    assert writes == []


def test_pvaccess_async_put_within_limits_reaches_the_channel(monkeypatch):
    """The first positional is a PvObject, so the ordinary reduction covers it."""
    writes: list = []
    mod = _install_fake_pvaccess(monkeypatch, writes)
    _run_monkeypatch(monkeypatch, _make_pva_validator())

    channel = mod.Channel("TEST:MAG:SP")
    assert channel.asyncPut(_FakePvObject({"value": 5.0}), lambda _r: None) == "async-put-done"
    assert len(writes) == 1
    assert writes[0][0] == "TEST:MAG:SP"


def test_pvaccess_parse_put_cannot_be_limits_checked(monkeypatch):
    """``parsePut``/``parsePutGet`` carry JSON strings, so they are refused.

    Their payload is a list of ``field=value`` strings parsed against the
    channel's introspected structure — there is no value object to reduce, and
    the wrapper cannot know which string names the field the limits are about.
    Guessing would be a check that silently means nothing, so the write is
    refused with the same posture the p4p guard takes for a callable payload:
    an in-range value is refused too, and the caller is pointed at ``put``.
    """
    writes: list = []
    mod = _install_fake_pvaccess(monkeypatch, writes)
    _run_monkeypatch(monkeypatch, _make_pva_validator())

    with pytest.raises(ValueError, match="cannot be limits-checked"):
        mod.Channel("TEST:MAG:SP").parsePut(["value=5.0"])
    with pytest.raises(ValueError, match="cannot be limits-checked"):
        mod.Channel("TEST:MAG:SP").parsePutGet(["value=5.0"])
    assert writes == []


def test_pvaccess_structure_payload_over_max_step_is_refused(monkeypatch):
    """A structure payload is stepped against a structure read, both reduced.

    This is the shape a real pvaPy run has on both sides — ``PvObject`` in,
    ``PvObject`` back from ``get`` — and neither is a number until the guard
    reduces it. Unreduced, the validator finds nothing numeric and SKIPS the
    step check, so the assertion is on ``MAX_STEP_EXCEEDED`` specifically.
    """
    writes: list = []
    reads: list = []
    mod = _install_fake_pvaccess(monkeypatch, writes, reads)
    _run_monkeypatch(monkeypatch, _make_pva_validator())

    with pytest.raises(ChannelLimitsViolationError) as exc:
        mod.Channel("TEST:MAG:STEP").put(_FakePvObject({"value": 50.0}))

    assert exc.value.violation_type == "MAX_STEP_EXCEEDED"
    assert reads == ["TEST:MAG:STEP"]
    assert writes == []


def test_pvaccess_valueless_structure_payload_fails_closed(monkeypatch):
    """A structure with no value under it is refused by pvaPy's own accessor.

    ``PvObject.getPyObject()`` raises when the structure has no ``value``
    field, which happens before the guard's own dict branch is reached. The
    exception propagates and the write never leaves — a different exception
    type from the plain-dict refusal, the same fail-closed outcome.
    """
    writes: list = []
    mod = _install_fake_pvaccess(monkeypatch, writes)
    _run_monkeypatch(monkeypatch, _make_pva_validator())

    with pytest.raises(RuntimeError):
        mod.Channel("TEST:MAG:SP").put(_FakePvObject({"unit": "A"}))
    assert writes == []


def test_fake_pv_object_answers_the_way_pvapy_does():
    """The fake is the only surface standing in for binding on the real library.

    pvaPy's no-arg ``PvObject.getPyObject()`` answers the VALUE of the
    ``value`` field and raises ``pvaccess.InvalidRequest`` when there is none;
    ``toDict()`` is the spelling that answers the whole structure as a dict. A
    fake that answered a dict from ``getPyObject()`` would test the guard only
    against a shape a real PvObject never produces. (The exception type is not
    part of the contract — the guard lets whatever is raised propagate.)
    """
    payload = _FakePvObject({"value": 99.0, "unit": "A"})

    assert payload.getPyObject() == 99.0
    assert payload.toDict() == {"value": 99.0, "unit": "A"}
    with pytest.raises(RuntimeError):
        _FakePvObject({"unit": "A"}).getPyObject()


def test_pvaccess_without_a_channel_class_reports_the_gap(monkeypatch):
    """A pvaccess module with no ``Channel`` must not report itself guarded.

    The success line is the operator's only evidence the guard is on. A stub or
    broken install that carries no ``Channel`` wraps nothing, and saying so is
    the difference between a known gap and an unchecked write believed checked.
    """
    monkeypatch.setitem(sys.modules, "pvaccess", _fake_module("pvaccess"))

    _, out = _run_monkeypatch(monkeypatch, _make_pva_validator())
    assert "no Channel class" in out
    assert "✅ Monkeypatched pvaccess" not in out


def test_pvaccess_channel_without_a_put_reports_the_gap(monkeypatch):
    """A ``Channel`` the sweep found nothing on is the same gap, and says so."""
    mod = _fake_module("pvaccess")

    class Channel:
        def get(self):
            return 1.0

    mod.Channel = Channel
    monkeypatch.setitem(sys.modules, "pvaccess", mod)

    _, out = _run_monkeypatch(monkeypatch, _make_pva_validator())
    assert "⚠️  pvaccess guard failed: Channel has no put method" in out
    assert "✅ Monkeypatched pvaccess" not in out


def test_absent_pvaccess_leaves_the_p4p_guards_in_place(monkeypatch):
    _install_fake_p4p(monkeypatch)
    monkeypatch.setitem(sys.modules, "pvaccess", None)

    _, out = _run_monkeypatch(monkeypatch, _make_pva_validator())
    assert "pvaccess not available" in out
    assert "✅ Monkeypatched p4p.client.thread Context.put()" in out
