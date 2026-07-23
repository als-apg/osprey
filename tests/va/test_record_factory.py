"""Tests for the IOC record-construction factory.

Two process-boundary gotchas shape this file (both empirically confirmed
while writing it, not theoretical):

* softioc's ``builder``/``softioc`` modules are process-global: every record
  ever built in a process shares one namespace, and ``iocInit`` may only be
  called once per process. Tests that don't need a live CA round trip
  (contract-violation errors) build small, never-repeated synthetic address
  sets; the one test class that needs real counts against the full manifest
  shares a single module-scoped build.

* merely constructing a softioc record via ``builder.aIn``/``aOut`` in a
  process -- even without ever calling ``iocInit`` -- leaves that process
  unable to act as a pyepics Channel Access *client* afterwards
  (``create_channel returned 'Not supported by attached service'``).
  Since this test module needs ``ioc.records`` (hence ``softioc.builder``)
  for the non-live tests, the pytest process itself can never be the CA
  client for the live round-trip check. Both the IOC *and* the CA client
  therefore run in their own dedicated subprocesses; the CA-client
  subprocess entry point below is deliberately dispatched before this
  module imports ``ioc.records``, so it never inherits that poisoned state.
  This mirrors the real deployment anyway: the agent (CA client) and the VA
  container (CA server) are always separate processes.

The boot-value tests (``TestBootValues``) are the one exception to "build
real records": they monkeypatch the factory's ``_IN_BUILDERS``/
``_OUT_BUILDERS`` tables to a duck-typed ``FakeRecord`` for the scope of a
single test, so ``build_records()``'s pure-Python initial-value wiring can
be exercised on throwaway synthetic addresses without ever registering them
in the process-global softioc namespace described above.
"""

from __future__ import annotations

import json
import os
import socket
import sys


def _free_port() -> int:
    """Return an unused TCP port on 127.0.0.1 (same convention as
    tests/interfaces/conftest.py and friends)."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


# Loopback-only, ephemeral port pair: this worktree is shared by concurrent
# agent test runs (and by the probe/VA container), so a hardcoded port would
# race. Pick fresh ports per test session unless the environment already
# pins them. Must be set before any of this process's or the subprocesses'
# imports of epics/softioc.
os.environ.setdefault("EPICS_CA_ADDR_LIST", "127.0.0.1")
os.environ.setdefault("EPICS_CA_AUTO_ADDR_LIST", "NO")
if "EPICS_CA_SERVER_PORT" not in os.environ:
    os.environ["EPICS_CA_SERVER_PORT"] = str(_free_port())
if "EPICS_CA_REPEATER_PORT" not in os.environ:
    os.environ["EPICS_CA_REPEATER_PORT"] = str(_free_port())


def _run_ca_client_subprocess(argv: list[str]) -> None:
    """CA-client subprocess entry point: one-shot operation, JSON on stdout.

    Dispatched below *before* this module imports ``ioc.records`` /
    ``manifest``, and imports only ``epics`` itself -- see the module
    docstring for why this process must never touch ``softioc.builder``.
    """
    import epics

    op = argv[0]
    if op == "caget":
        (address,) = argv[1:]
        value = epics.caget(address, timeout=5, connection_timeout=5)
        print(json.dumps({"value": value}), flush=True)
    elif op == "caput":
        address, value = argv[1], float(argv[2])
        ok = epics.caput(address, value, wait=True, timeout=5)
        print(json.dumps({"ok": bool(ok)}), flush=True)
    else:
        raise ValueError(f"unknown CA client op {op!r}")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--ca-client":
    _run_ca_client_subprocess(sys.argv[2:])
    sys.exit(0)

# Everything below this point is only reached by pytest's own import of this
# module, or by the --run-live-ioc-subprocess entry point further down --
# never by the --ca-client path above.

import subprocess  # noqa: E402
import time  # noqa: E402

import pytest  # noqa: E402

from osprey.services.virtual_accelerator.entrypoint import (  # noqa: E402
    _channel_limits_path,
    _load_boot_values,
    _load_drive_limits,
)
from osprey.services.virtual_accelerator.ioc import records as ioc_records_module  # noqa: E402
from osprey.services.virtual_accelerator.ioc.records import (  # noqa: E402
    ManifestContractError,
    build_records,
)
from osprey.services.virtual_accelerator.manifest import (  # noqa: E402
    PARTITION_PYAT_COUPLED,
    PARTITION_SP_ECHO,
    PARTITION_STATIC_NOISY,
    RECORD_TYPE_ANALOG,
    RECORD_TYPE_BINARY,
    RECORD_TYPE_LONG_STRING,
    RECORD_TYPE_MBB,
    build_manifest,
)
from osprey.services.virtual_accelerator.manifest.loaders import (  # noqa: E402
    load_machine_json_channels,
)

_SP_ECHO_SP = "ZZTEST:MAG:ECHO:01:CURRENT:SP"
_SP_ECHO_RB = "ZZTEST:MAG:ECHO:01:CURRENT:RB"
_BRIDGE_SP = "ZZTEST:MAG:BRIDGE:01:CURRENT:SP"
_BRIDGE_RB = "ZZTEST:MAG:BRIDGE:01:CURRENT:RB"
_BRIDGE_HOOK_ECHO = "ZZTEST:MAG:BRIDGE:01:HOOK_ECHO"
# A pyat-coupled corrector (family=HCM, the ``_channel()`` default) whose
# :SP is build-time faulted stuck -- must still latch its own :SP but never
# propagate to :RB, exactly as a stuck sp-echo setpoint would.
_STUCK_SP = "ZZTEST:MAG:STUCK:03:CURRENT:SP"
_STUCK_RB = "ZZTEST:MAG:STUCK:03:CURRENT:RB"
# A second, unfaulted pyat-coupled corrector -- dedicated to the DRVH/DRVL
# clamp tests so they don't interleave with the echo/hook assertions above.
# Its clamp band is injected explicitly (see _run_live_ioc_subprocess) --
# under the injected-limits contract (task 2.3), records.py itself is
# file-blind and clamps only addresses the caller's drive_limits map names,
# never by family.
_LIMIT_SP = "ZZTEST:MAG:LIMIT:04:CURRENT:SP"
_LIMIT_RB = "ZZTEST:MAG:LIMIT:04:CURRENT:RB"
_LIMIT_DRVL, _LIMIT_DRVH = -12.0, 12.0


def _channel(
    address: str,
    *,
    partition: str,
    subfield: str,
    record_type: str = RECORD_TYPE_ANALOG,
    noise: bool = False,
    ring: str = "ZZTEST",
    system: str = "MAG",
    family: str = "HCM",
    device: str = "99",
    field: str = "CURRENT",
) -> dict:
    """Build one synthetic manifest-entry dict for a factory-level test."""
    return {
        "address": address,
        "ring": ring,
        "system": system,
        "family": family,
        "device": device,
        "field": field,
        "subfield": subfield,
        "partition": partition,
        "record_type": record_type,
        "noise": noise,
    }


def _run_live_ioc_subprocess() -> None:
    """Subprocess entry point: builds the small live-CA record set and
    serves it until killed. Imports ``ioc.records``/``softioc.builder`` --
    see the module docstring for why this must never be the same process
    that also acts as the CA client.
    """
    from softioc import asyncio_dispatcher, builder, softioc

    hook_echo = builder.aIn(_BRIDGE_HOOK_ECHO, initial_value=0.0)

    def on_pyat_setpoint(address: str, value: float) -> None:
        # Report the hook firing back over CA itself, since the real
        # assertions run in yet another process with no other channel to
        # observe this callback through.
        hook_echo.set(value)

    channels = [
        # device= must differ between the two pairs: _channel_key() (used to
        # find an sp-echo setpoint's RB partner) ignores subfield, so two
        # pairs sharing a (ring, system, family, device, field) key would
        # collide in the readback index and the echo would wire onto the
        # wrong RB.
        _channel(_SP_ECHO_RB, partition=PARTITION_SP_ECHO, subfield="RB", device="01"),
        _channel(_SP_ECHO_SP, partition=PARTITION_SP_ECHO, subfield="SP", device="01"),
        _channel(_BRIDGE_RB, partition=PARTITION_PYAT_COUPLED, subfield="RB", device="02"),
        _channel(_BRIDGE_SP, partition=PARTITION_PYAT_COUPLED, subfield="SP", device="02"),
        _channel(_STUCK_RB, partition=PARTITION_PYAT_COUPLED, subfield="RB", device="03"),
        _channel(_STUCK_SP, partition=PARTITION_PYAT_COUPLED, subfield="SP", device="03"),
        _channel(_LIMIT_RB, partition=PARTITION_PYAT_COUPLED, subfield="RB", device="04"),
        _channel(_LIMIT_SP, partition=PARTITION_PYAT_COUPLED, subfield="SP", device="04"),
    ]
    build_records(
        channels,
        on_pyat_setpoint=on_pyat_setpoint,
        stuck_setpoints=frozenset({_STUCK_SP}),
        # Injected-limits contract (task 2.3): records.py never knows this
        # address is a "corrector" -- it only clamps because the caller
        # (here, this test) put it in the map, exactly as entrypoint.py's
        # real _load_drive_limits() does from channel_limits.json.
        drive_limits={_LIMIT_SP: (_LIMIT_DRVL, _LIMIT_DRVH)},
    )

    dispatcher = asyncio_dispatcher.AsyncioDispatcher()
    builder.LoadDatabase()
    softioc.iocInit(dispatcher)

    print("live_ioc subprocess ready", flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--run-live-ioc-subprocess":
    _run_live_ioc_subprocess()


def _ca_client(*args: str, timeout: float = 10) -> dict:
    """Run one CA client operation in a fresh, ``ioc.records``-free
    subprocess (``_run_ca_client_subprocess`` above) and return its decoded
    JSON result."""
    result = subprocess.run(
        [sys.executable, __file__, "--ca-client", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"CA client subprocess failed: {result.stdout}\n{result.stderr}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def _caget(address: str) -> float | None:
    return _ca_client("caget", address)["value"]


def _caput(address: str, value: float) -> bool:
    return _ca_client("caput", address, str(value))["ok"]


@pytest.fixture(scope="module")
def manifest_channels() -> list[dict]:
    return build_manifest()["channels"]


@pytest.fixture(scope="module")
def full_manifest_records(manifest_channels):
    """Build the factory's output for the full real manifest exactly once.

    build_records() populates softioc's process-global builder registry, so
    every test that only needs to inspect the *result* shares this one
    build rather than re-registering the same ~1,228 record names.

    Wired with the same derived ``drive_limits``/``boot_values`` maps
    entrypoint.main() passes in production (see ``_load_drive_limits``/
    ``_load_boot_values``), so this shared build also doubles as an
    integration check that those derivations don't choke when combined with
    every real manifest channel at once.
    """
    return build_records(
        manifest_channels,
        drive_limits=_load_drive_limits(),
        boot_values=_load_boot_values(),
    )


class TestRecordCountsMatchManifest:
    """The factory must produce exactly one record per manifest channel --
    the served set is derived data, never hand-listed (same principle the
    manifest generator itself is built on)."""

    def test_full_manifest_produces_one_record_per_channel(
        self, manifest_channels, full_manifest_records
    ):
        assert len(full_manifest_records.all) == len(manifest_channels)

    def test_addresses_match_exactly(self, manifest_channels, full_manifest_records):
        assert set(full_manifest_records.all) == {c["address"] for c in manifest_channels}

    def test_pyat_coupled_slot_covers_every_pyat_coupled_channel(
        self, manifest_channels, full_manifest_records
    ):
        expected = {
            c["address"] for c in manifest_channels if c["partition"] == PARTITION_PYAT_COUPLED
        }
        assert set(full_manifest_records.pyat_coupled) == expected

    def test_static_noisy_slot_covers_every_static_noisy_channel(
        self, manifest_channels, full_manifest_records
    ):
        expected = {
            c["address"] for c in manifest_channels if c["partition"] == PARTITION_STATIC_NOISY
        }
        assert set(full_manifest_records.static_noisy) == expected

    def test_static_noisy_slot_excludes_other_partitions(
        self, manifest_channels, full_manifest_records
    ):
        sp_echo_addresses = {
            c["address"] for c in manifest_channels if c["partition"] == PARTITION_SP_ECHO
        }
        assert sp_echo_addresses.isdisjoint(full_manifest_records.static_noisy)


class TestDerivedDriveLimitsMatchChannelLimitsFile:
    """records.py is file-blind (task 2.3): entrypoint.py's
    ``_load_drive_limits()`` is the only place that turns
    channel_limits.json into the ``build_records(drive_limits=...)`` map
    the softioc DRVL/DRVH clamp is actually built from. This pins that
    derivation against the file's own contents directly -- read the same
    path the entrypoint reads, independently re-derive the expected map from
    the raw JSON, and require full-dict equality -- so a future
    channel_limits retune that silently changes which ``:SP`` addresses are
    writable-with-bounds fails here instead of just drifting the deployed
    clamp unnoticed."""

    def test_derived_map_matches_writable_sp_entries_in_the_file(self):
        raw = json.loads(_channel_limits_path().read_text())
        defaults = raw.get("defaults", {})
        expected: dict[str, tuple[float, float]] = {}
        for address, entry in raw.items():
            if address.startswith("_") or address == "defaults" or not address.endswith(":SP"):
                continue
            merged = {**defaults, **entry}
            if not merged.get("writable", True):
                continue
            min_value = merged.get("min_value")
            max_value = merged.get("max_value")
            if min_value is None or max_value is None:
                continue
            expected[address] = (float(min_value), float(max_value))

        derived = _load_drive_limits()
        assert derived == expected
        # Sanity count: pins today's writable-:SP-with-bounds population so
        # a channel_limits.json edit that silently drops/adds entries is
        # caught here, not only downstream in clamp behavior.
        assert len(expected) == 396


class TestBiRecordsRejectNoise:
    """A boolean channel is a status flag, not a measurement -- classify.py
    never emits noise=True for record_type 'bi', and this factory refuses to
    silently build one if that contract is ever violated upstream."""

    def test_noisy_binary_channel_raises(self):
        bad = _channel(
            "ZZTEST:MAG:BAD:01:STATUS:FAULT",
            partition=PARTITION_STATIC_NOISY,
            subfield="FAULT",
            record_type=RECORD_TYPE_BINARY,
            noise=True,
        )
        with pytest.raises(ManifestContractError, match="reject noise"):
            build_records([bad])

    def test_non_noisy_binary_channel_is_accepted(self):
        ok = _channel(
            "ZZTEST:MAG:OK:01:STATUS:FAULT",
            partition=PARTITION_STATIC_NOISY,
            subfield="FAULT",
            record_type=RECORD_TYPE_BINARY,
            noise=False,
        )
        records = build_records([ok])
        assert set(records.all) == {"ZZTEST:MAG:OK:01:STATUS:FAULT"}


class TestGatewayRecordTypes:
    """The longstringin/mbbi dispatch additions: the two gateway channel
    shapes (512-byte char waveform, multi-bit binary) the original three
    record types didn't cover. Build-time coverage only -- value round-trips
    ride the same live-CA machinery as every other type."""

    def test_long_string_sp_echo_pair_builds(self):
        channels = [
            _channel(
                "ZZTEST:GW:LSTR:01:MSG:RB",
                partition=PARTITION_SP_ECHO,
                subfield="RB",
                record_type=RECORD_TYPE_LONG_STRING,
                family="LSTR",
                field="MSG",
            ),
            _channel(
                "ZZTEST:GW:LSTR:01:MSG:SP",
                partition=PARTITION_SP_ECHO,
                subfield="SP",
                record_type=RECORD_TYPE_LONG_STRING,
                family="LSTR",
                field="MSG",
            ),
        ]
        records = build_records(channels)
        assert set(records.all) == {"ZZTEST:GW:LSTR:01:MSG:RB", "ZZTEST:GW:LSTR:01:MSG:SP"}

    def test_mbb_static_channel_builds(self):
        state = _channel(
            "ZZTEST:GW:MODE:01:STATE:RB",
            partition=PARTITION_STATIC_NOISY,
            subfield="RB",
            record_type=RECORD_TYPE_MBB,
            family="MODE",
            field="STATE",
        )
        records = build_records([state])
        assert "ZZTEST:GW:MODE:01:STATE:RB" in records.static_noisy

    def test_noisy_mbb_channel_raises(self):
        bad = _channel(
            "ZZTEST:GW:MODE:02:STATE:RB",
            partition=PARTITION_STATIC_NOISY,
            subfield="RB",
            record_type=RECORD_TYPE_MBB,
            noise=True,
        )
        with pytest.raises(ManifestContractError, match="reject noise"):
            build_records([bad])

    def test_noisy_long_string_channel_raises(self):
        bad = _channel(
            "ZZTEST:GW:LSTR:02:MSG:RB",
            partition=PARTITION_STATIC_NOISY,
            subfield="RB",
            record_type=RECORD_TYPE_LONG_STRING,
            noise=True,
        )
        with pytest.raises(ManifestContractError, match="reject noise"):
            build_records([bad])

    def test_long_string_records_declare_full_gateway_width(self):
        # NELM must be 512 explicitly: the builder's own default (256, or
        # len(initial_value)+1) would silently truncate a full-width wire
        # value. Asserted at the wrapper level so a future builder-default
        # change upstream cannot quietly shrink the record.
        rec = ioc_records_module._long_string_in("ZZTEST:GW:LSTR:03:MSG:RB")
        assert int(rec.NELM.Value()) == 512


class TestContractViolations:
    def test_unknown_record_type_raises(self):
        bad = _channel(
            "ZZTEST:MAG:UNK:01:CURRENT:RB",
            partition=PARTITION_STATIC_NOISY,
            subfield="RB",
            record_type="waveform",
        )
        with pytest.raises(ManifestContractError, match="unknown record_type"):
            build_records([bad])

    def test_sp_echo_setpoint_without_readback_raises(self):
        orphan_sp = _channel(
            "ZZTEST:MAG:ORPHAN:01:CURRENT:SP",
            partition=PARTITION_SP_ECHO,
            subfield="SP",
        )
        with pytest.raises(ManifestContractError, match="no matching RB"):
            build_records([orphan_sp])


class TestPyatCoupledCallbackSlot:
    """Partition (a) setpoints must call the injected hook. Whether a write
    actually echoes onto :RB -- and whether stuck_setpoints/drive-limit
    faults override that -- can only be observed via a live EPICS record
    processing a write (see TestLiveChannelAccessRoundTrip); build-time
    wiring here only covers hook registration and the missing-hook no-op."""

    def test_setpoint_without_hook_is_a_noop(self):
        sp = _channel(
            "ZZTEST:MAG:NOHOOK:01:CURRENT:SP",
            partition=PARTITION_PYAT_COUPLED,
            subfield="SP",
        )
        records = build_records([sp])
        # Must not raise for lack of a hook; the record simply has no wired behavior.
        assert "ZZTEST:MAG:NOHOOK:01:CURRENT:SP" in records.pyat_coupled

    def test_setpoint_with_hook_registers_without_calling_it_yet(self):
        calls = []
        sp = _channel(
            "ZZTEST:MAG:DEFERRED:01:CURRENT:SP",
            partition=PARTITION_PYAT_COUPLED,
            subfield="SP",
        )
        build_records([sp], on_pyat_setpoint=lambda addr, value: calls.append((addr, value)))
        # Building the record must not itself trigger the hook -- only a
        # live CA write does (covered by TestLiveChannelAccessRoundTrip).
        assert calls == []


class FakeRecord:
    """Duck-typed stand-in for a softioc In/Out record, swapped in for
    ``ioc.records``'s real ``_IN_BUILDERS``/``_OUT_BUILDERS`` tables via
    monkeypatching (see ``TestBootValues``). ``build_records()``'s
    boot-value wiring (``_initial_value``) is pure Python, so exercising it
    doesn't need a single real softioc record built -- and monkeypatching
    keeps these tests' throwaway addresses out of the process-global softioc
    namespace the rest of this module's non-live tests do consume (see the
    module docstring)."""

    def __init__(self, address, *, initial_value=None, on_update=None, **_ignored) -> None:
        self.address = address
        self.value = initial_value
        self.on_update = on_update

    def get(self):
        return self.value

    def set(self, value) -> None:
        self.value = value


# Real addresses (not synthetic ZZTEST ones): boot-value wiring is best
# proven against actual machine.json entries, so a real nonzero-nominal
# quad and a real zeroed corrector exercise the two ends of the range this
# factory must reproduce faithfully at boot.
_BOOT_QUAD_SP = "SR:MAG:QF:01:CURRENT:SP"
_BOOT_QUAD_RB = "SR:MAG:QF:01:CURRENT:RB"
_BOOT_CORRECTOR_SP = "SR:MAG:HCM:01:CURRENT:SP"
_BOOT_CORRECTOR_RB = "SR:MAG:HCM:01:CURRENT:RB"
# Synthetic addresses with no entry in this test's boot_values map at all --
# the fallback-to-default path.
_BOOT_FALLBACK_SP = "ZZTEST:MAG:NOBOOT:06:CURRENT:SP"
_BOOT_FALLBACK_RB = "ZZTEST:MAG:NOBOOT:06:CURRENT:RB"


class TestBootValues:
    """``build_records()``'s ``boot_values`` map seeds a ``:SP``/``:RB``
    record's initial value at construction (ioc-records-limits-injection-
    boot-init, task 2.3), mirroring entrypoint.py's real
    ``_load_boot_values()`` derivation from machine.json. An address absent
    from the map still boots at the type-appropriate default (0.0 for
    analog) -- the same behavior as when ``boot_values`` is ``None``
    entirely."""

    @pytest.fixture()
    def fake_record_builders(self, monkeypatch):
        monkeypatch.setitem(ioc_records_module._IN_BUILDERS, RECORD_TYPE_ANALOG, FakeRecord)
        monkeypatch.setitem(ioc_records_module._OUT_BUILDERS, RECORD_TYPE_ANALOG, FakeRecord)

    def test_boots_at_machine_json_value_for_nonzero_nominal_quad(self, fake_record_builders):
        nominal = load_machine_json_channels()[_BOOT_QUAD_SP]["value"]
        assert nominal != 0.0
        channels = [
            _channel(
                _BOOT_QUAD_SP,
                partition=PARTITION_PYAT_COUPLED,
                subfield="SP",
                ring="SR",
                family="QF",
            ),
            _channel(
                _BOOT_QUAD_RB,
                partition=PARTITION_PYAT_COUPLED,
                subfield="RB",
                ring="SR",
                family="QF",
            ),
        ]
        records = build_records(
            channels, boot_values={_BOOT_QUAD_SP: nominal, _BOOT_QUAD_RB: nominal}
        )
        assert records.all[_BOOT_QUAD_SP].value == pytest.approx(nominal)
        assert records.all[_BOOT_QUAD_RB].value == pytest.approx(nominal)

    def test_boots_at_zero_for_zeroed_corrector(self, fake_record_builders):
        value = load_machine_json_channels()[_BOOT_CORRECTOR_SP]["value"]
        assert value == 0.0
        channels = [
            _channel(
                _BOOT_CORRECTOR_SP, partition=PARTITION_PYAT_COUPLED, subfield="SP", ring="SR"
            ),
            _channel(
                _BOOT_CORRECTOR_RB, partition=PARTITION_PYAT_COUPLED, subfield="RB", ring="SR"
            ),
        ]
        records = build_records(
            channels, boot_values={_BOOT_CORRECTOR_SP: value, _BOOT_CORRECTOR_RB: value}
        )
        assert records.all[_BOOT_CORRECTOR_SP].value == 0.0
        assert records.all[_BOOT_CORRECTOR_RB].value == 0.0

    def test_falls_back_to_default_for_address_absent_from_map(self, fake_record_builders):
        channels = [
            _channel(_BOOT_FALLBACK_SP, partition=PARTITION_PYAT_COUPLED, subfield="SP"),
            _channel(_BOOT_FALLBACK_RB, partition=PARTITION_PYAT_COUPLED, subfield="RB"),
        ]
        # boot_values is non-empty but names only an unrelated address --
        # neither fallback address appears in it.
        records = build_records(channels, boot_values={_BOOT_QUAD_SP: 999.0})
        assert records.all[_BOOT_FALLBACK_SP].value == 0.0
        assert records.all[_BOOT_FALLBACK_RB].value == 0.0


@pytest.fixture(scope="module")
def live_ioc():
    """Launch the IOC subprocess and wait for it to start serving PVs."""
    proc = subprocess.Popen(
        [sys.executable, __file__, "--run-live-ioc-subprocess"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    deadline = time.monotonic() + 20
    connected = False
    last_error = None
    while time.monotonic() < deadline:
        try:
            value = _caget(_BRIDGE_HOOK_ECHO)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            value = None
        if value is not None:
            connected = True
            break
        time.sleep(0.5)

    if not connected:
        proc.terminate()
        output = proc.stdout.read() if proc.stdout else ""
        proc.wait(timeout=5)
        raise RuntimeError(
            f"live_ioc subprocess never came up (last_error={last_error!r}); output:\n{output}"
        )

    yield

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


class TestLiveChannelAccessRoundTrip:
    """ "echo wiring works": a real Channel Access client, in its own process
    (matching the real deployment topology -- agent and IOC are never the
    same process), must observe the SP-echo behavior and the pyat-coupled
    callback slot firing.
    """

    def test_sp_echo_write_reflects_on_readback(self, live_ioc):
        assert _caput(_SP_ECHO_SP, 3.5)
        time.sleep(0.3)
        assert _caget(_SP_ECHO_RB) == pytest.approx(3.5)

    def test_sp_echo_second_write_also_reflects(self, live_ioc):
        assert _caput(_SP_ECHO_SP, -1.25)
        time.sleep(0.3)
        assert _caget(_SP_ECHO_RB) == pytest.approx(-1.25)

    def test_pyat_coupled_write_invokes_injected_hook(self, live_ioc):
        assert _caput(_BRIDGE_SP, 2.0)
        time.sleep(0.3)
        assert _caget(_BRIDGE_HOOK_ECHO) == pytest.approx(2.0)

    def test_pyat_coupled_write_echoes_readback(self, live_ioc):
        # This is the scan-hang fix (FR10): a corrector's own CURRENT:RB
        # must track its CURRENT:SP, or the bridge's EpicsMotor.set()
        # settle-wait hangs forever. The physics bridge itself never
        # touches a magnet's own :RB (it only pushes BPM POSITION
        # readbacks), so this echo has to be the factory's doing.
        assert _caput(_BRIDGE_SP, 2.0)
        time.sleep(0.3)
        assert _caget(_BRIDGE_RB) == pytest.approx(2.0)

    def test_stuck_pyat_coupled_setpoint_freezes_readback(self, live_ioc):
        # The SP still latches the caput value itself...
        assert _caput(_STUCK_SP, 7.5)
        time.sleep(0.3)
        assert _caget(_STUCK_SP) == pytest.approx(7.5)
        # ...but the no-op on_update path wins over the echo: the readback
        # never moves, honestly, for every CA reader -- not just this one.
        assert _caget(_STUCK_RB) == pytest.approx(0.0)

    def test_setpoint_caput_beyond_injected_drive_limit_is_clamped(self, live_ioc):
        # pythonSoftIOC clamps VAL to [DRVL, DRVH] before on_update ever
        # runs, so both the SP itself and its echoed RB read the clamped
        # value, not the requested one -- a real bound below the ORM plan's
        # own pydantic schema, enforced against any writer. _LIMIT_SP is
        # clamped solely because _run_live_ioc_subprocess put it in the
        # drive_limits map it passed to build_records() -- records.py itself
        # never knows or cares that this address is a "corrector" (the
        # injected-limits contract, task 2.3).
        assert _caput(_LIMIT_SP, 50.0)
        time.sleep(0.3)
        assert _caget(_LIMIT_SP) == pytest.approx(_LIMIT_DRVH)
        assert _caget(_LIMIT_RB) == pytest.approx(_LIMIT_DRVH)

        assert _caput(_LIMIT_SP, -50.0)
        time.sleep(0.3)
        assert _caget(_LIMIT_SP) == pytest.approx(_LIMIT_DRVL)
        assert _caget(_LIMIT_RB) == pytest.approx(_LIMIT_DRVL)
