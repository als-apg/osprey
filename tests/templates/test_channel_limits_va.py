"""What a write to each channel of a served tree may do, on every tree the repo ships.

``channel_limits.json`` is the write-safety database the control assistant's
``LimitsValidator`` loads: one entry per address, and a write is refused unless
that address says it is writable and the value sits inside its band. Two kinds
of tree carry one, and this module holds both to the same contract:

* the **demo tree** (``templates/apps/control_assistant/data``), whose bands
  are committed -- the focusing-strength currents derived against its own ring
  by ``scripts/va/derive_bands.py``, the trim window committed with the tree
  for the bend the sweep does not reach, and the flat per-family bands
  everything else inherits;
* every **Middle Layer export** the repo commits a 2.0 ``va.json`` beside,
  whose bands ``osprey.services.mml.emit.va.emit_channel_limits`` derives from
  the export's own Setpoint ``Range``.

The contract both keep: every address the tree's channel database carries has
exactly one entry and no entry names an address the tree does not carry; a
channel the bindings drive is writable and banded; every readback, monitor and
undriven channel is ``writable: false``, so OSPREY's own safety layer refuses
the write rather than leaving the block to the IOC; and the whole document
loads through the validator, which fails a file closed on a single key it does
not know.

Every case here is read off the tree -- its bindings document says which
addresses are driven, of which family and to which nominal, and the export
says which band each was measured for. No family name, device count or band
value of a particular facility is written down here.

``max_step`` is set on no entry: the validator's max_step check reads the
current value through a direct ``epics.caget``, which under the mock control
system -- the preset's default -- has no server behind these addresses and
would block every write to them. A band plus the post-write confirming re-read
already satisfies "a write outside its limits is rejected" (SC9).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from osprey.services.virtual_accelerator.bindings import (
    PROVENANCE_KEY,
    BindingsDocument,
    load_bindings,
)
from osprey.services.virtual_accelerator.manifest.paths import PACKAGE_PATHS, ManifestPaths
from osprey_connectors.control_system.limits_validator import LimitsValidator

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_DATA = PACKAGE_PATHS.data_root
LIMITS_PATH = PACKAGE_PATHS.channel_limits
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "mml"
DERIVE_BANDS = REPO_ROOT / "scripts" / "va" / "derive_bands.py"

#: The keys of a limits document that are not addresses.
METADATA_KEYS = {"_comment", "_version", "_description"}
RESERVED_KEYS = METADATA_KEYS | {"defaults"}

#: Names no tree addresses: two from an obsolete bracket convention and three
#: from a tutorial that named no real channel. An entry under any of them is a
#: write band over a channel that does not exist.
STALE_ADDRESSES = (
    "MAG:HCM[H01]:CURRENT:SP",
    "MAG:QF[QF01]:CURRENT:SP",
    "ElectronGunFilamentCurrentSetPoint",
    "TerminalVoltageSetValue",
    "DIAGNOSTICS:TEMPERATURE:SP",
)

#: The FR3/3.8 demo write: ``orbit_response``'s own convention is a corrector
#: current setpoint of 10.0 A, so every corrector of the demo ring has to hold
#: it -- the orbit-response walkthrough may exercise any of them.
DEMO_WRITE_CHANNEL = "SR:MAG:HCM:01:CURRENT:SP"
DEMO_WRITE_VALUE = 10.0

#: The max-plane one-turn |trace| ``scripts/va/derive_bands.py`` stops a sweep
#: at, and the hard instability guard it keeps a margin below.
TRACE_EDGE = 1.8
HARD_TRACE_GUARD = 2.0

#: How far outside ``TRACE_EDGE`` a committed band edge may sit. The demo's
#: bands were interpolated on current against the ring as it stood when they
#: were committed; re-evaluating the trace at those edges on the ring the tree
#: serves today overshoots by up to ~1.1e-2 (measured, worst case its focusing
#: quadrupole family). This tolerance is ~2x that and still an order of
#: magnitude inside the 0.2 margin to ``HARD_TRACE_GUARD``, so it absorbs the
#: drift without accepting a band that is actually unstable. Tightening it is
#: a re-derivation (``derive_bands.py --verify``), not a test change.
TRACE_TOLERANCE = 0.02


# ===================================================================
# The trees under test
# ===================================================================


@dataclass(frozen=True)
class Tree:
    """One deployment tree's write bands, and everything that explains them.

    Attributes:
        name: The tree's name, which is the parametrised test's id.
        paths: Where the tree's files sit, so a loader reads them the way the
            served process does.
        limits: ``channel_limits.json`` as plain JSON types.
        document: The bindings the tree serves: which addresses are driven, of
            which family, to which nominal.
        addresses: Every address the tree's channel database carries.
        export: The Middle Layer export the bands were derived from, or
            ``None`` for a tree whose bands are committed rather than emitted.
    """

    name: str
    paths: ManifestPaths
    limits: dict[str, Any]
    document: BindingsDocument
    addresses: frozenset[str]
    export: dict[str, Any] | None

    @property
    def entries(self) -> dict[str, dict]:
        """Every address entry, without the metadata and ``defaults`` keys."""
        return {
            address: entry
            for address, entry in self.limits.items()
            if address not in RESERVED_KEYS and not address.startswith("_")
        }

    def entry(self, address: str) -> dict:
        """One address's entry, with the document's ``defaults`` merged under it."""
        return {**self.limits.get("defaults", {}), **self.limits.get(address, {})}

    def writable(self, address: str) -> bool:
        return bool(self.entry(address).get("writable", True))

    def band(self, address: str) -> tuple[float | None, float | None]:
        entry = self.entry(address)
        return entry.get("min_value"), entry.get("max_value")


def _demo_tree() -> Tree:
    """The tree the control assistant ships, whose bands are committed files."""
    from osprey.services.virtual_accelerator.manifest import build_manifest

    manifest = build_manifest()
    return Tree(
        name="demo",
        paths=PACKAGE_PATHS,
        limits=json.loads(LIMITS_PATH.read_text()),
        document=load_bindings(PACKAGE_PATHS.va_bindings),
        addresses=frozenset(channel["address"] for channel in manifest["channels"]),
        export=None,
    )


def emit_export_tree(name: str, directory: Path, stem: str, root: Path) -> Tree:
    """Run the virtual-accelerator emit lane over one committed 2.0 export.

    The lane is run in the order ``osprey mml emit`` runs it -- the deck first,
    because the bindings stamp its digest, then the bindings the bands are
    derived from, the starting-state seed, and the bands last -- and writes
    into ``root`` the layout :class:`ManifestPaths` resolves, so every document
    is read back by the loader the served process reads it with.

    Args:
        name: The fixture directory's name, used as the tree's name.
        directory: The fixture directory holding the export.
        stem: The export's file stem, e.g. ``quokka.sr``.
        root: An empty directory to emit the tree into.

    Returns:
        The emitted tree.
    """
    from osprey.services.mml.emit.context import build_context
    from osprey.services.mml.emit.va import (
        emit_bindings,
        emit_channel_limits,
        emit_lattice,
        emit_machine,
    )
    from osprey.services.mml.family import FamilyView
    from osprey.services.mml.loaders.mat import load_lattice
    from osprey.services.mml.normalize import normalize_family
    from osprey.services.mml.va.elements import address_elements
    from osprey.services.mml.va.verdicts import propose

    system = stem.split(".")[1].upper()
    ao = json.loads((directory / f"{stem}.ao.json").read_text())
    va = json.loads((directory / f"{stem}.va.json").read_text())
    ring = load_lattice(directory / f"{stem}.lattice.mat")
    export = {
        raw: body for raw, body in ao.items() if not raw.startswith("_") and isinstance(body, dict)
    }
    views = {raw: FamilyView(system, raw, normalize_family(body)) for raw, body in export.items()}
    verdicts = propose(va, ring, views)

    ao_path = root / "ao.json"
    ao_path.write_text(json.dumps({system: ao}))
    mapping_path = root / "mapping.yaml"
    mapping_path.write_text(f"facility:\n  token: {name}\n")
    ctx = build_context(ao_path, mapping_path, {})

    paths = ManifestPaths(data_root=root / "data")
    paths.machine_json.parent.mkdir(parents=True, exist_ok=True)
    keyed = {(system, raw): verdict for raw, verdict in verdicts.items()}
    judged_va = {(system, raw): block for raw, block in va["families"].items()}
    emit_lattice(ring, paths.lattice_json, ctx, write=False)
    paths.va_bindings.write_text(
        emit_bindings(
            keyed,
            list(views.values()),
            dict(address_elements(va, ring, verdicts).bindings),
            judged_va,
            ctx,
            system=system,
            energy_gev=va["lattice"]["energy_gev"],
        )
    )
    machine_text, _seeds = emit_machine(
        keyed, list(views.values()), judged_va, _export_mapping(name, system, views), ctx
    )
    paths.machine_json.write_text(machine_text)

    addresses = frozenset(_export_addresses(views.values()))
    limits_text, _bands = emit_channel_limits(
        None,
        load_bindings(paths.va_bindings).bindings,
        sorted(addresses),
        ctx,
        views=list(views.values()),
        system=system,
    )
    paths.channel_limits.write_text(limits_text)
    return Tree(
        name=name,
        paths=paths,
        limits=json.loads(limits_text),
        document=load_bindings(paths.va_bindings),
        addresses=addresses,
        export=export,
    )


def _export_mapping(token: str, system: str, views: dict):
    """The prose side of a mapping, for the emitters that read a channel's words.

    The bands and the seeds are read off the export, not off the mapping, so
    what a reviewer wrote about a family does not change either -- a mapping
    naming every family with no prose at all is enough to run the lane.
    """
    from osprey.services.mml.mapping.schema import (
        Facility,
        Family,
        Field,
        Mapping,
        System,
        VirtualAccelerator,
    )

    return Mapping(
        facility=Facility(token=token, title=token, description=None, provenance="human"),
        systems={system: System(raw=system, name=system, description=None, provenance="human")},
        section_order=(system,),
        families={
            raw: Family(
                raw=raw,
                rename=None,
                branch=None,
                class_="BPM",
                aliases=(),
                description=None,
                provenance="human",
                channels=1,
                fields={name: Field(description=None, provenance="human") for name in view.fields},
            )
            for raw, view in views.items()
        },
        directions={},
        judgments={},
        virtual_accelerator=VirtualAccelerator(system=system, families={}),
    )


def _export_addresses(views) -> set[str]:
    """Every address the export names, which is what its channel database carries."""
    addresses: set[str] = set()
    for view in views:
        for field in view.fields.values():
            for key in field.keys:
                for slot in field.slots(key):
                    if isinstance(slot, str) and slot.strip():
                        addresses.add(slot.strip())
    return addresses


def fixture_trees() -> tuple[str, ...]:
    """Every fixture directory a 2.0 export can land in, whether it holds one yet.

    That is a directory carrying the exporter's own paired output
    (``<machine>.<submachine>.ao.json``), because a re-export writes its
    ``va.json`` beside exactly those. Each is named rather than filtered, so a
    tree holding a 1.0 export today says so when it skips and runs the whole
    lane the day its 2.0 export lands -- without this module being edited.
    """
    return tuple(
        sorted(
            path.name
            for path in FIXTURES.iterdir()
            if path.is_dir() and any(path.glob("*.ao.json"))
        )
    )


def build_tree(name: str, root: Path) -> Tree:
    """The named tree, skipping the test where the tree states no 2.0 export."""
    if name == "demo":
        return _demo_tree()
    directory = FIXTURES / name
    exports = sorted(directory.glob("*.va.json"))
    if not exports:
        pytest.skip(f"{name} holds no 2.0 export; re-export it with mml_export 2.0 to run it here")
    return emit_export_tree(name, directory, exports[0].name[: -len(".va.json")], root)


TREE_NAMES = ("demo", *fixture_trees())


@pytest.fixture(scope="module", params=TREE_NAMES)
def tree(request, tmp_path_factory) -> Tree:
    return build_tree(request.param, tmp_path_factory.mktemp(request.param))


@pytest.fixture(scope="module")
def demo() -> Tree:
    return _demo_tree()


@pytest.fixture(scope="module")
def band_policy():
    """The band rules the demo's committed magnet bands were derived under.

    Read from ``scripts/va/derive_bands.py`` itself rather than restated here:
    which devices the edge rule sweeps, and which families a unipolar supply's
    0 A floor covers, are that script's own predicates, and this test follows
    them wherever they lead rather than pinning a second copy of the answer.
    """
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("_derive_bands", DERIVE_BANDS)
    module = importlib.util.module_from_spec(spec)
    # Its dataclasses resolve their own module while the class body runs, so
    # the module has to be registered before it is executed.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _driven(tree: Tree):
    """Every binding that writes a channel."""
    return [binding for binding in tree.document.bindings if binding.is_writable]


def _read_addresses(tree: Tree) -> list[str]:
    """Every address the tree reads and never writes."""
    addresses = [
        binding.readback_address for binding in tree.document.bindings if binding.readback_address
    ]
    addresses += [
        binding.setpoint_address for binding in tree.document.bindings if not binding.is_writable
    ]
    driven = {binding.setpoint_address for binding in _driven(tree)}
    return [address for address in dict.fromkeys(addresses) if address not in driven]


def _exported_range(tree: Tree, family: str, address: str) -> tuple[float | None, float | None]:
    """The operating band the export states for one address, read off the export.

    A flat pair is the family's whole band and reaches every device; a
    per-device table states one row each and the row is the device's own. A
    non-finite bound is no bound at all. Read from the export file rather than
    through the emitter's own accessors, so the two do not agree by sharing one
    reading of it.
    """
    body = (tree.export or {}).get(family, {})
    field = body.get("Setpoint")
    if not isinstance(field, dict):
        return None, None
    declared = field.get("Range")
    if not isinstance(declared, list) or not declared:
        return None, None
    pair = declared
    if any(isinstance(row, list) for row in declared):
        names = field.get("ChannelNames") or field.get("TangoNames") or []
        names = [name.strip() if isinstance(name, str) else name for name in names]
        if address not in names:
            return None, None
        pair = declared[names.index(address)]
    if not isinstance(pair, list) or len(pair) != 2:
        return None, None
    low, high = (_finite(bound) for bound in pair)
    if low is not None and high is not None and low > high:
        low, high = high, low
    return low, high


def _finite(value: Any) -> float | None:
    """One exported bound as a number, or ``None`` where it states no bound."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


# ===================================================================
# The contract every tree keeps
# ===================================================================


class TestTheDocumentLoadsWhereItIsRead:
    """The validator fails a whole file closed on one key it does not know, so
    a tree's bands are only real if they load through it."""

    def test_the_document_loads_through_the_write_safety_validator(self, tree):
        loaded, _raw = LimitsValidator._load_limits_database(str(tree.paths.channel_limits))
        assert loaded

    def test_no_entry_carries_a_retired_verification_key(self, tree):
        """``verification`` was replaced by ``confirm``, and the validator now
        fails the load closed on the retired key."""
        assert "verification" not in tree.limits.get("defaults", {})
        offenders = [address for address, entry in tree.entries.items() if "verification" in entry]
        assert not offenders, f"entries still carry 'verification': {sorted(offenders)[:10]}"

    def test_no_entry_sets_max_step(self, tree):
        """See the module docstring: a max_step here would block every write
        under the mock control system."""
        offenders = [address for address, entry in tree.entries.items() if "max_step" in entry]
        assert not offenders, f"entries set max_step: {sorted(offenders)[:10]}"


class TestEveryDrivenChannelIsWritableAndBanded:
    def test_every_driven_setpoint_has_an_entry(self, tree):
        missing = [
            binding.setpoint_address
            for binding in _driven(tree)
            if binding.setpoint_address not in tree.limits
        ]
        assert not missing, f"driven addresses with no limits entry: {sorted(missing)[:10]}"

    def test_every_driven_setpoint_is_writable(self, tree):
        blocked = [
            binding.setpoint_address
            for binding in _driven(tree)
            if not tree.writable(binding.setpoint_address)
        ]
        assert not blocked, f"driven addresses the file refuses writes to: {sorted(blocked)[:10]}"

    def test_every_band_is_an_ordered_pair(self, tree):
        for binding in _driven(tree):
            low, high = tree.band(binding.setpoint_address)
            if low is None or high is None:
                continue
            assert low < high, f"{binding.setpoint_address}: band [{low}, {high}] is not ordered"

    def test_every_nominal_sits_inside_its_own_band(self, tree):
        """The served model refuses to boot a channel whose starting value is
        outside its band, so a band that excludes its own nominal is a tree
        that cannot start."""
        violations = []
        for binding in _driven(tree):
            nominal = binding.nominal
            if nominal is None or not math.isfinite(nominal):
                continue
            low, high = tree.band(binding.setpoint_address)
            if (low is not None and nominal < low) or (high is not None and nominal > high):
                violations.append(
                    f"{binding.setpoint_address}: nominal {nominal} outside [{low}, {high}]"
                )
        assert not violations, "nominal outside its band:\n" + "\n".join(violations)


class TestEveryChannelNothingDrivesIsReadOnly:
    def test_every_readback_is_read_only(self, tree):
        writable = [address for address in _read_addresses(tree) if tree.writable(address)]
        assert not writable, (
            f"addresses the bindings only read are writable: {sorted(writable)[:10]}"
        )

    def test_a_readback_write_is_refused_by_the_validator(self, tree):
        """The software safety layer, not the IOC, is what refuses it."""
        from osprey.errors import ChannelLimitsViolationError

        read_addresses = _read_addresses(tree)
        assert read_addresses, f"{tree.name} reads no address it does not also write"
        limits_db, raw_db = LimitsValidator._load_limits_database(str(tree.paths.channel_limits))
        validator = LimitsValidator(limits_db, {"allow_unlisted_channels": True}, raw_db)
        with pytest.raises(ChannelLimitsViolationError) as exc:
            validator.validate(read_addresses[0], 1.0)
        assert exc.value.violation_type == "READ_ONLY_CHANNEL"

    def test_an_in_bounds_write_to_a_driven_channel_is_allowed(self, tree):
        limits_db, raw_db = LimitsValidator._load_limits_database(str(tree.paths.channel_limits))
        validator = LimitsValidator(limits_db, {"allow_unlisted_channels": True}, raw_db)
        banded = [
            binding
            for binding in _driven(tree)
            if all(bound is not None for bound in tree.band(binding.setpoint_address))
        ]
        assert banded, f"{tree.name} bands no driven channel"
        low, high = tree.band(banded[0].setpoint_address)
        validator.validate(banded[0].setpoint_address, (low + high) / 2)


class TestTheTreeStatesEveryAddressItCarries:
    def test_every_address_the_channel_database_carries_has_an_entry(self, tree):
        missing = tree.addresses - set(tree.limits)
        assert not missing, (
            f"addresses of the channel database with no entry ({len(missing)}): "
            f"{sorted(missing)[:10]}"
        )

    def test_no_entry_names_an_address_the_tree_does_not_carry(self, tree):
        known = (
            tree.addresses
            | {binding.setpoint_address for binding in tree.document.bindings}
            | {
                binding.readback_address
                for binding in tree.document.bindings
                if binding.readback_address
            }
        )
        extra = set(tree.entries) - known
        assert not extra, f"entries over addresses no tree carries: {sorted(extra)[:10]}"


# ===================================================================
# A tree whose bands the emit lane derived
# ===================================================================


class TestTheExportStatesEveryBand:
    """The bands of an emitted tree are the export's own Setpoint ``Range``,
    widened only where the device's nominal sits outside it."""

    @pytest.fixture(autouse=True)
    def _emitted_only(self, tree):
        if tree.export is None:
            pytest.skip("the demo tree commits its bands rather than deriving them from an export")

    def test_every_band_is_the_exported_range_or_the_nominal_that_widened_it(self, tree):
        for binding in _driven(tree):
            stated = _exported_range(tree, binding.family, binding.setpoint_address)
            low, high = tree.band(binding.setpoint_address)
            nominal = binding.nominal
            expected_low, expected_high = stated
            if nominal is not None and math.isfinite(nominal):
                if expected_low is not None and nominal < expected_low:
                    expected_low = nominal
                if expected_high is not None and nominal > expected_high:
                    expected_high = nominal
            assert (low, high) == (expected_low, expected_high), (
                f"{binding.setpoint_address}: banded [{low}, {high}], but its export states "
                f"{stated} and its nominal is {nominal}"
            )

    def test_a_band_the_export_does_not_state_is_left_unbounded(self, tree):
        """A bound the export states as non-finite, or does not state at all,
        is no bound -- which the write-safety database spells as an absent
        ``min_value``/``max_value``, never as a number of this lane's own."""
        for binding in _driven(tree):
            low, high = _exported_range(tree, binding.family, binding.setpoint_address)
            nominal = binding.nominal if binding.nominal is not None else None
            entry = tree.entry(binding.setpoint_address)
            if low is None:
                assert "min_value" not in entry, binding.setpoint_address
            if high is None:
                assert "max_value" not in entry, binding.setpoint_address
            assert nominal is None or low is not None or "min_value" not in entry

    def test_every_entry_carries_the_provenance_of_the_run_that_wrote_it(self, tree):
        unstamped = [
            address for address, entry in tree.entries.items() if PROVENANCE_KEY not in entry
        ]
        assert not unstamped, (
            f"entries this lane wrote with no {PROVENANCE_KEY} stamp: {sorted(unstamped)[:10]}"
        )

    def test_the_stamp_is_one_run(self, tree):
        stamps = {entry[PROVENANCE_KEY] for entry in tree.entries.values()}
        assert len(stamps) == 1, f"entries carry {len(stamps)} different stamps"


# ===================================================================
# The demo tree's committed bands
# ===================================================================


class TestTheDemoKeepsItsCommittedBands:
    """Three sources produce the demo's bands, and each says something
    different about the family it covers: a swept magnet family commits the
    stability edge derived per device, a family the sweep cannot reach commits
    a trim window around each device's own nominal, and everything else
    carries one band for the whole family."""

    @staticmethod
    def _families(demo: Tree) -> dict[str, list]:
        families: dict[str, list] = {}
        for binding in _driven(demo):
            families.setdefault(binding.family, []).append(binding)
        return families

    @staticmethod
    def _swept(demo: Tree, band_policy) -> set[str]:
        """The families the edge rule can sweep, by the rule's own predicate."""
        return {
            binding.family
            for binding in _driven(demo)
            if binding.kind == band_policy.SWEPT_KIND
            and binding.index == band_policy.FOCUSING_INDEX
        }

    @classmethod
    def _windowed(cls, demo: Tree, band_policy) -> dict[str, list]:
        """The families banded per device that the edge rule never swept.

        A band that differs device by device was set against that device, and
        one the sweep does not reach was set somewhere else: for a tree emitted
        from a Middle Layer export, from that device's own stated setpoint
        range, and for the demo tree, committed with the tree as a trim window
        about each bend's nominal.
        """
        swept = cls._swept(demo, band_policy)
        return {
            family: bindings
            for family, bindings in cls._families(demo).items()
            if family not in swept
            and len({demo.band(binding.setpoint_address) for binding in bindings}) > 1
        }

    def test_a_swept_family_bands_each_device_on_its_own(self, demo, band_policy):
        """A derived edge is measured per device, so a swept family's devices
        do not share one band -- a flat band there would mean the derivation
        never ran."""
        swept = self._swept(demo, band_policy)
        assert swept, "the demo tree binds no family the edge rule can sweep"
        for family, bindings in self._families(demo).items():
            if family not in swept:
                continue
            bands = {demo.band(binding.setpoint_address) for binding in bindings}
            assert len(bands) == len(bindings), (
                f"{family}: {len(bindings)} devices share {len(bands)} bands, so its bands "
                "were not derived device by device"
            )

    def test_a_unipolar_supply_family_never_bands_below_its_floor(self, demo, band_policy):
        """The floor is a supply-polarity decision: the ring may well be stable
        with the current reversed, and the band discards that headroom rather
        than exposing a setpoint the supply cannot reach.

        Which families it covers is read off the tree the same way the
        derivation reads it -- the supplies that never once run negative."""
        floored = band_policy.unipolar_floor_families(demo.document)
        assert floored, "the tree states a negative nominal for every family it binds"
        for family, bindings in self._families(demo).items():
            if family not in floored:
                continue
            for binding in bindings:
                low, _high = demo.band(binding.setpoint_address)
                assert low is not None and low >= band_policy.UNIPOLAR_FLOOR, (
                    f"{binding.setpoint_address}: band minimum {low} is below the "
                    f"{band_policy.UNIPOLAR_FLOOR} A unipolar-supply floor"
                )

    def test_an_unswept_per_device_family_bands_one_window_around_each_nominal(
        self, demo, band_policy
    ):
        """A band the sweep never measured still has to be a trim window: the
        same fraction either side of each device's own nominal, so the family
        is banded by one rule rather than device by device by hand."""
        windowed = self._windowed(demo, band_policy)
        assert windowed, "no family of the demo tree is banded per device without being swept"
        for family, bindings in windowed.items():
            fractions = set()
            for binding in bindings:
                low, high = demo.band(binding.setpoint_address)
                nominal = binding.nominal
                assert nominal, f"{binding.setpoint_address}: banded around no nominal"
                assert high - nominal == pytest.approx(nominal - low), binding.setpoint_address
                fractions.add(round((high - nominal) / nominal, 9))
            assert len(fractions) == 1, (
                f"{family}: its devices are windowed at {sorted(fractions)} of nominal, so the "
                "family carries no single window"
            )

    def test_every_other_family_carries_one_band_for_all_its_devices(self, demo, band_policy):
        """A family with no derivation of its own inherits the one band its
        inventory states, so every device of it is banded alike."""
        derived = self._swept(demo, band_policy) | set(self._windowed(demo, band_policy))
        flat = {
            family: bindings
            for family, bindings in self._families(demo).items()
            if family not in derived
        }
        assert flat, "every family of the demo tree is derived, so nothing states a flat band"
        for family, bindings in flat.items():
            bands = {demo.band(binding.setpoint_address) for binding in bindings}
            assert len(bands) == 1, f"{family}: devices carry {len(bands)} different bands"

    def test_every_corrector_holds_the_demo_write_magnitude(self, demo):
        """The orbit-response walkthrough writes ``DEMO_WRITE_VALUE`` to a
        corrector, and may pick any of them."""
        kicks = [binding for binding in _driven(demo) if binding.kind == "kick"]
        assert kicks, "the demo tree binds no corrector"
        for binding in kicks:
            low, high = demo.band(binding.setpoint_address)
            assert low <= -DEMO_WRITE_VALUE, binding.setpoint_address
            assert high >= DEMO_WRITE_VALUE, binding.setpoint_address

    def test_the_demo_write_channel_is_a_corrector_it_can_reach(self, demo):
        binding = next(b for b in _driven(demo) if b.setpoint_address == DEMO_WRITE_CHANNEL)
        assert binding.kind == "kick"
        low, high = demo.band(DEMO_WRITE_CHANNEL)
        assert low < DEMO_WRITE_VALUE < high


class TestEveryCommittedBandEdgeIsAStableRing:
    """SC5: a band edge of the demo tree is a current the ring it serves
    actually accepts. Each edge is written through the binding's own variable
    -- the one the serving path builds, so the current becomes a physics value
    by exactly one conversion -- and the ring's one-turn matrix is read back.

    Every edge of every driven channel is covered, not a sample: a corrector
    kick and a sextupole component move no part of the one-turn matrix and pass
    at the nominal trace, while the focusing families the edge rule was derived
    for sit at ``TRACE_EDGE``.
    """

    @pytest.fixture(scope="class")
    def traces(self, demo) -> dict[str, float | None]:
        """The max-plane one-turn |trace| at each committed band edge."""
        import warnings

        import at
        import numpy as np
        from lume_pyat.simulator import PyATSimulator, restore_element, snapshot_element

        from osprey.services.virtual_accelerator.lattice import build_ring
        from osprey.services.virtual_accelerator.model.bindings import build_action_variables

        simulator = PyATSimulator(build_ring(demo.paths))
        factories = build_action_variables(demo.document)

        def trace() -> float | None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=at.AtWarning)
                one_turn = at.find_m66(simulator.lattice)[0]
            if not np.all(np.isfinite(one_turn)):
                return None
            return max(abs(one_turn[0, 0] + one_turn[1, 1]), abs(one_turn[2, 2] + one_turn[3, 3]))

        baseline = trace()
        assert baseline is not None and baseline < TRACE_EDGE, (
            f"the tree's own nominal ring is already at |trace|={baseline}, so no edge of it "
            "can be judged"
        )

        measured: dict[str, float | None] = {}
        for binding in _driven(demo):
            variable = factories[binding.setpoint_address](
                {}, name=binding.setpoint_address, default_value=binding.nominal
            )
            elements = [simulator.element(piece.element) for piece in binding.slices]
            for edge in demo.band(binding.setpoint_address):
                if edge is None:
                    continue
                snapshots = [snapshot_element(element) for element in elements]
                variable._set(simulator, edge)
                measured[f"{binding.setpoint_address}={edge}"] = trace()
                for element, snapshot in zip(elements, snapshots, strict=True):
                    restore_element(element, snapshot)
        return measured

    def test_every_edge_leaves_a_solvable_ring(self, traces):
        assert traces, "the demo tree commits no band edge"
        unsolved = [edge for edge, value in traces.items() if value is None]
        assert not unsolved, f"edges whose one-turn matrix is not finite: {unsolved[:10]}"

    def test_every_edge_stays_inside_the_hard_instability_guard(self, traces):
        over = {
            edge: value
            for edge, value in traces.items()
            if value is not None and value >= HARD_TRACE_GUARD
        }
        assert not over, f"edges at or past the |trace| >= {HARD_TRACE_GUARD} guard: {over}"

    def test_every_edge_stays_at_the_margin_it_was_derived_at(self, traces):
        over = {
            edge: value
            for edge, value in traces.items()
            if value is not None and value > TRACE_EDGE + TRACE_TOLERANCE
        }
        assert not over, (
            f"edges past the {TRACE_EDGE} derivation margin (+/-{TRACE_TOLERANCE}): {over}"
        )


# ===================================================================
# The demo tree's shipped file, as the control assistant reads it
# ===================================================================


class TestDefaultsSemanticsPreserved:
    def test_defaults_block_present(self, demo):
        assert "defaults" in demo.limits

    def test_defaults_still_writable_and_confirmed(self, demo):
        defaults = demo.limits["defaults"]
        assert defaults["writable"] is True
        assert defaults["confirm"] is True


class TestWritableIffSetpoint:
    """The demo tree's own write-safety contract: a channel is writable if and
    only if the manifest calls it a setpoint. The bindings drive a subset of
    those; the rest are setpoints of channels the virtual accelerator does not
    model, and the IOC, not this file, is what stands behind them."""

    @staticmethod
    def _split() -> tuple[set[str], set[str]]:
        from osprey.services.virtual_accelerator.manifest import build_manifest

        channels = build_manifest()["channels"]
        return (
            {c["address"] for c in channels if c["subfield"] == "SP"},
            {c["address"] for c in channels if c["subfield"] != "SP"},
        )

    def test_every_setpoint_entry_is_writable(self, demo):
        setpoints, _ = self._split()
        readonly = [address for address in setpoints if not demo.writable(address)]
        assert not readonly, f"setpoint addresses wrongly marked read-only: {sorted(readonly)[:10]}"

    def test_every_non_setpoint_entry_is_read_only(self, demo):
        _, others = self._split()
        writable = [address for address in others if demo.writable(address)]
        assert not writable, (
            f"non-setpoint addresses not marked writable:false "
            f"({len(writable)}): {sorted(writable)[:10]}"
        )

    def test_every_setpoint_entry_has_an_ordered_band(self, demo):
        setpoints, _ = self._split()
        for address in setpoints:
            low, high = demo.band(address)
            assert low is not None and high is not None, address
            assert low < high, address


class TestStaleEntriesRemoved:
    def test_no_stale_addresses_remain(self, demo):
        for stale in STALE_ADDRESSES:
            assert stale not in demo.limits, f"stale entry {stale!r} reappeared"
