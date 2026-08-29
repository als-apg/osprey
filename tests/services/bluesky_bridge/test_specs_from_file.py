"""Unit tests for the device-file loader (``devices/_specs_from_file.py``).

Covers parsing, fail-soft degradation and de-duplication directly, plus the
``validate_device_document`` problem list the build uses to refuse a malformed
file. No module-level ``importorskip`` guard: the loader imports only
``yaml``, ``logging`` and the dependency-free ``devices/specs.py``, so it runs
in a slimmed install without ophyd-async. The two tests that carry a parsed
spec on into a *built* device guard ophyd-async themselves, in-function, so the
rest of the file stays runnable without it.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.services.bluesky_bridge.devices._specs_from_file import (
    specs_from_file,
    validate_device_document,
)


def _write(tmp_path: Path, doc: object, name: str = "devices.yml") -> Path:
    """Write ``doc`` as YAML into ``tmp_path`` and return the path."""
    path = tmp_path / name
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return path


# --- duplicate names --------------------------------------------------------


def test_duplicate_settable_name_drops_the_later_entry(tmp_path: Path) -> None:
    """Two settable entries with the same device name keep only the first."""
    path = _write(
        tmp_path,
        {
            "settables": [
                {"name": "m", "setpoint": "A:SP", "readback": "A:RB"},
                {"name": "m", "setpoint": "B:SP", "readback": "B:RB"},
            ]
        },
    )
    settables, _ = specs_from_file(path)
    assert [s.name for s in settables] == ["m"]
    assert settables[0].setpoint_pv == "A:SP"  # first wins, not the shadowing second


def test_readable_name_colliding_with_a_settable_is_dropped(tmp_path: Path) -> None:
    """A readable reusing a settable's device name is dropped (settables resolve
    first); device names become event-data column keys, so a collision must not
    stand."""
    path = _write(
        tmp_path,
        {
            "settables": [{"name": "dev", "setpoint": "A:SP", "readback": "A:RB"}],
            "readables": [{"name": "dev", "pv": "B:RB"}, {"name": "other", "pv": "C:RB"}],
        },
    )
    settables, readables = specs_from_file(path)
    assert [s.name for s in settables] == ["dev"]
    assert [s.name for s in readables] == ["other"]  # 'dev' readable dropped


def test_distinct_names_are_all_kept(tmp_path: Path) -> None:
    """No collision -> every parsed device survives (dedup is not over-eager)."""
    path = _write(
        tmp_path,
        {
            "settables": [{"name": "m1", "setpoint": "A:SP"}, {"name": "m2", "setpoint": "B:SP"}],
            "readables": [{"name": "d1", "pv": "C:RB"}, {"name": "d2", "pv": "D:RB"}],
        },
    )
    settables, readables = specs_from_file(path)
    assert [s.name for s in settables] == ["m1", "m2"]
    assert [s.name for s in readables] == ["d1", "d2"]


# --- the reason this file format exists at all ------------------------------


def test_comma_bearing_pv_survives_the_round_trip(tmp_path: Path) -> None:
    """A PV containing a comma round-trips intact -- the env format could not
    carry one (comma was its entry separator), which is why the file exists."""
    comma_pv = "BTS:QF,1:CURRENT:SP"
    path = _write(
        tmp_path,
        {
            "settables": [{"name": "qf1", "setpoint": comma_pv, "readback": "BTS:QF,1:CURRENT:RB"}],
            "readables": [{"name": "bpm", "pv": "BTS:BPM,1:X:RB"}],
        },
    )
    settables, readables = specs_from_file(path)
    assert settables[0].setpoint_pv == comma_pv
    assert settables[0].readback_pv == "BTS:QF,1:CURRENT:RB"
    assert readables[0].read_pv == "BTS:BPM,1:X:RB"


def test_colon_bearing_device_name_survives(tmp_path: Path) -> None:
    """Device names may be full colon-delimited addresses (the generator uses
    address-as-name), so nothing may split a name on ``:``."""
    path = _write(
        tmp_path,
        {"settables": [{"name": "SR:C01:QF:1", "setpoint": "SR:C01:QF:1:SP"}]},
    )
    settables, _ = specs_from_file(path)
    assert [s.name for s in settables] == ["SR:C01:QF:1"]


def test_json_document_parses(tmp_path: Path) -> None:
    """``yaml.safe_load`` covers JSON, so a ``.json`` device file needs no
    separate reader."""
    path = tmp_path / "devices.json"
    path.write_text(
        json.dumps(
            {
                "settables": [{"name": "m", "setpoint": "A:SP", "readback": "A:RB"}],
                "readables": [{"name": "d", "pv": "C:RB"}],
            }
        ),
        encoding="utf-8",
    )
    settables, readables = specs_from_file(path)
    assert [s.name for s in settables] == ["m"]
    assert settables[0].readback_pv == "A:RB"
    assert [s.name for s in readables] == ["d"]


def test_omitted_readback_becomes_none(tmp_path: Path) -> None:
    """``readback`` is optional; omitting it leaves ``readback_pv`` unset so the
    device falls back to the setpoint PV for reads."""
    path = _write(tmp_path, {"settables": [{"name": "m", "setpoint": "A:SP"}]})
    settables, _ = specs_from_file(path)
    assert settables[0].readback_pv is None


def test_path_may_be_a_string(tmp_path: Path) -> None:
    """The container passes an env-var string; ``str`` is accepted alongside
    ``Path``."""
    path = _write(tmp_path, {"readables": [{"name": "d", "pv": "C:RB"}]})
    _, readables = specs_from_file(str(path))
    assert [s.name for s in readables] == ["d"]


# --- fail-soft degradation --------------------------------------------------


def test_malformed_entry_is_skipped_and_the_rest_survive(tmp_path: Path) -> None:
    """One bad entry must not cost every other device its connection: malformed
    entries are skipped with a warning, good neighbours are kept."""
    path = _write(
        tmp_path,
        {
            "settables": [
                {"name": "good", "setpoint": "A:SP"},
                {"name": "no_setpoint"},
                {"setpoint": "B:SP"},
                {"name": "", "setpoint": "C:SP"},
                {"name": "bad_type", "setpoint": 17},
                "not-a-mapping",
                {"name": "empty_readback", "setpoint": "D:SP", "readback": ""},
                {"name": "also_good", "setpoint": "E:SP"},
            ],
            "readables": [
                {"name": "rb_good", "pv": "F:RB"},
                {"name": "rb_no_pv"},
                None,
                {"name": "rb_also_good", "pv": "G:RB"},
            ],
        },
    )
    settables, readables = specs_from_file(path)
    assert [s.name for s in settables] == ["good", "also_good"]
    assert [s.name for s in readables] == ["rb_good", "rb_also_good"]


def test_missing_file_yields_no_devices(tmp_path: Path) -> None:
    """An unreadable path degrades to an empty device set rather than raising --
    the caller warns that the worker will expose no plans."""
    assert specs_from_file(tmp_path / "does-not-exist.yml") == ([], [])


def test_directory_path_yields_no_devices(tmp_path: Path) -> None:
    """A path that is a directory is unreadable too, and degrades the same way."""
    assert specs_from_file(tmp_path) == ([], [])


def test_empty_file_yields_no_devices(tmp_path: Path) -> None:
    """An empty file parses to ``None`` -- a valid, if useless, empty device set."""
    path = tmp_path / "devices.yml"
    path.write_text("", encoding="utf-8")
    assert specs_from_file(path) == ([], [])


def test_unparseable_yaml_yields_no_devices(tmp_path: Path) -> None:
    """A YAML syntax error degrades to an empty device set, not an exception."""
    path = tmp_path / "devices.yml"
    path.write_text("settables: [unclosed\n", encoding="utf-8")
    assert specs_from_file(path) == ([], [])


def test_non_mapping_document_yields_no_devices_and_one_warning(tmp_path: Path, caplog) -> None:
    """A top-level list (a plausible mis-authoring) yields no devices and exactly
    one warning -- not one per entry."""
    path = _write(tmp_path, [{"name": "m", "setpoint": "A:SP"}])
    with caplog.at_level("WARNING"):
        assert specs_from_file(path) == ([], [])
    assert len(caplog.records) == 1


def test_unknown_top_level_key_rejects_the_document(tmp_path: Path, caplog) -> None:
    """A typo'd top-level key rejects the whole document with one warning: a
    silently-empty device set would look like a working deployment with no
    plans."""
    path = _write(
        tmp_path,
        {"settables": [{"name": "m", "setpoint": "A:SP"}], "readable": [{"name": "d", "pv": "C"}]},
    )
    with caplog.at_level("WARNING"):
        assert specs_from_file(path) == ([], [])
    assert len(caplog.records) == 1


def test_non_list_device_section_is_skipped(tmp_path: Path) -> None:
    """A section that is not a list contributes nothing, and the other section
    still loads."""
    path = _write(
        tmp_path,
        {
            "settables": {"name": "m", "setpoint": "A:SP"},
            "readables": [{"name": "d", "pv": "C:RB"}],
        },
    )
    settables, readables = specs_from_file(path)
    assert settables == []
    assert [s.name for s in readables] == ["d"]


# --- validate_device_document ----------------------------------------------


def test_validate_accepts_a_well_formed_document() -> None:
    """A document the loader keeps whole reports no problems."""
    doc = {
        "settables": [
            {"name": "m", "setpoint": "A:SP", "readback": "A:RB"},
            {"name": "m2", "setpoint": "B:SP"},
        ],
        "readables": [{"name": "d", "pv": "C:RB"}],
    }
    assert validate_device_document(doc) == []


def test_validate_accepts_an_empty_or_partial_document() -> None:
    """An empty document and a one-section document are valid (if useless): the
    build refuses malformed files, not empty ones."""
    assert validate_device_document(None) == []
    assert validate_device_document({}) == []
    assert validate_device_document({"readables": []}) == []


def test_validate_reports_a_non_mapping_document() -> None:
    """A top-level non-mapping is a single, named problem."""
    problems = validate_device_document([{"name": "m"}])
    assert len(problems) == 1
    assert "mapping" in problems[0]


def test_validate_reports_unknown_top_level_keys() -> None:
    """An unknown top-level key is named, so the typo is fixable from the
    message alone."""
    problems = validate_device_document({"settables": [], "readable": []})
    assert len(problems) == 1
    assert "readable" in problems[0]


def test_validate_names_each_malformed_entry() -> None:
    """Every malformed entry is reported, with its section and index, so a
    13k-line file can be repaired without bisecting it."""
    problems = validate_device_document(
        {
            "settables": [
                {"name": "ok", "setpoint": "A:SP"},
                {"name": "no_setpoint"},
                {"setpoint": "B:SP"},
                {"name": "bad_type", "setpoint": 17},
                "not-a-mapping",
                {"name": "typo", "setpoint": "C:SP", "setpoint_pv": "C:SP"},
            ],
            "readables": [{"name": "rb", "pv": "D:RB"}, {"name": "rb_no_pv"}],
        }
    )
    assert len(problems) == 6
    joined = "\n".join(problems)
    assert "settables[1]" in joined and "setpoint" in joined
    assert "settables[2]" in joined and "name" in joined
    assert "settables[3]" in joined
    assert "settables[4]" in joined
    assert "settables[5]" in joined and "setpoint_pv" in joined  # unknown entry key
    assert "readables[1]" in joined


def test_validate_reports_a_non_list_section() -> None:
    """A section that is not a list is named as such."""
    problems = validate_device_document({"settables": {"name": "m"}})
    assert len(problems) == 1
    assert "settables" in problems[0] and "list" in problems[0]


def test_validate_reports_duplicate_names() -> None:
    """Duplicate device names are reported: the loader drops the later entry, so
    an unrefused file would silently ship fewer devices than it lists."""
    problems = validate_device_document(
        {
            "settables": [{"name": "m", "setpoint": "A:SP"}, {"name": "m", "setpoint": "B:SP"}],
            "readables": [{"name": "m", "pv": "C:RB"}],
        }
    )
    assert len(problems) == 2
    assert all("m" in problem for problem in problems)


def test_validate_never_raises_on_arbitrary_input() -> None:
    """``validate_device_document`` reports, never raises -- the build turns its
    list into a refusal message and needs no try/except around it."""
    for doc in ("a string", 17, [], {"settables": None}, {"settables": [[]]}, object()):
        assert isinstance(validate_device_document(doc), list)


# --- from a parsed spec to a built device -----------------------------------
#
# Everything above stops at the parsed spec. The two tests below carry a spec
# on through ``devices/connector.py`` into a live device, because that is where
# the format's promise is actually kept or broken: a comma-bearing PV that
# parses correctly but is mangled on the way to the connector would still leave
# the deployment unable to address the channel.


class _FakeConnector:
    """A minimal async double for ``ControlSystemConnector``.

    Records the channel addresses the built devices actually hand to the
    control system, which is what the comma test asserts on. Reads answer
    ``0.0`` so a ``set(0.0)`` settles on its first readback poll instead of
    spinning until ``_READBACK_SETTLE_TIMEOUT_S``.
    """

    def __init__(self) -> None:
        self.reads: list[str] = []
        self.writes: list[tuple[str, Any]] = []

    async def read_channel(self, address: str) -> Any:
        self.reads.append(address)
        return _FakeChannelValue(0.0)

    async def write_channel_checked(self, address: str, value: Any, **_: Any) -> Any:
        self.writes.append((address, value))
        # The real result type, not a stand-in, so this double cannot drift from
        # the contract. ``unrequested`` (nothing confirmed) is what leaves the
        # settle poll above as the only check, which is what this fake wants.
        # Imported in-function: the module level here stays dependency-light so
        # the parsing tests run in a slimmed install.
        from osprey.connectors.control_system.base import ChannelWriteResult, WriteOutcome

        return ChannelWriteResult(
            channel_address=address, value_written=value, outcome=WriteOutcome.UNREQUESTED
        )


class _FakeChannelValue:
    """Stand-in for ``osprey.connectors.control_system.base.ChannelValue``."""

    def __init__(self, value: float) -> None:
        self.value = value
        self.timestamp = time.time()


def test_comma_bearing_pv_reaches_the_connector_of_a_built_device(tmp_path: Path) -> None:
    """A comma-bearing PV survives parse *and* device construction: the built
    device hands the connector the address verbatim.

    The round-trip test above proves the loader keeps the comma; this proves
    nothing downstream re-splits it, which is the property the env format could
    not offer at all.
    """
    pytest.importorskip("ophyd_async")
    from osprey.services.bluesky_bridge.devices.connector import build_devices

    setpoint_pv = "BTS:QF,1:CURRENT:SP"
    readback_pv = "BTS:QF,1:CURRENT:RB"
    read_pv = "BTS:BPM,1:X:RB"
    path = _write(
        tmp_path,
        {
            "settables": [{"name": "qf1", "setpoint": setpoint_pv, "readback": readback_pv}],
            "readables": [{"name": "bpm1", "pv": read_pv}],
        },
    )
    settables, readables = specs_from_file(path)

    async def build_and_exercise() -> _FakeConnector:
        connector = _FakeConnector()
        devices = await build_devices(settables, readables, connector=connector)
        await devices["qf1"].set(0.0)
        await devices["bpm1"].read()
        return connector

    connector = asyncio.run(build_and_exercise())
    assert connector.writes == [(setpoint_pv, 0.0)]
    assert readback_pv in connector.reads  # the settable's readback poll
    assert read_pv in connector.reads  # the readable's read


# --- scale: a facility-sized device file ------------------------------------


_SCALE_SETTABLES = 1_000
"""Settable entries in the synthetic scale document."""

_SCALE_READABLES = 12_000
"""Readable entries in the synthetic scale document. Together with the
settables that is ~13k entries -- the ALS-scale projection, an order of
magnitude above the 512 settable / 822 readable set the deployment starts
from, and covering the ~12.8k fields the env format left out."""


def _scale_document(settables: int, readables: int) -> dict[str, list[dict[str, str]]]:
    """Build a synthetic facility-sized device document in memory.

    Generated rather than committed as a golden file: a 13k-entry fixture is
    pure noise in review and in the repository, and nothing here depends on the
    exact addresses. One entry carries a comma so the scale case exercises the
    wide document *and* the format's reason for existing at once.
    """
    return {
        "settables": [
            {
                "name": f"SR:C{index // 40:02d}:MAG:CORR:{index:05d}",
                "setpoint": f"SR:C{index // 40:02d}:MAG:CORR:{index:05d}:CUR:SP",
                "readback": f"SR:C{index // 40:02d}:MAG:CORR:{index:05d}:CUR:RB",
            }
            for index in range(settables)
        ]
        + [{"name": "comma_bearing", "setpoint": "BTS:QF,1:CURRENT:SP"}],
        "readables": [
            {
                "name": f"SR:C{index // 400:02d}:DIAG:{index:05d}",
                "pv": f"SR:C{index // 400:02d}:DIAG:{index:05d}:VAL:RB",
            }
            for index in range(readables)
        ],
    }


@pytest.mark.slow
def test_facility_scale_document_parses_and_builds_within_the_worker_budget(
    tmp_path: Path,
) -> None:
    """A ~13k-entry device file parses and builds, and the wall-time it costs
    is reported.

    The worker awaits parse plus construction inside a single
    ``qserver_startup.CONNECT_TIMEOUT`` window, so this is a soft measurement
    of how much of that budget a facility-sized file consumes. No threshold is
    asserted: the number is machine-dependent, and a timing assertion here
    would be a flake generator rather than a guard. Run with ``-s`` to see the
    report line.
    """
    pytest.importorskip("ophyd_async")
    from osprey.services.bluesky_bridge.devices.connector import build_devices
    from osprey.services.bluesky_bridge.qserver_startup import CONNECT_TIMEOUT

    path = _write(tmp_path, _scale_document(_SCALE_SETTABLES, _SCALE_READABLES))

    parse_start = time.perf_counter()
    settables, readables = specs_from_file(path)
    parse_seconds = time.perf_counter() - parse_start

    # Built and exercised inside ONE event loop: an ophyd-async device is bound
    # to the loop it was connected on, so a `set()` issued from a second
    # `asyncio.run` would not be the same device the build produced.
    connector = _FakeConnector()

    async def build_and_probe() -> tuple[dict[str, Any], float]:
        start = time.perf_counter()
        built = await build_devices(settables, readables, connector=connector)
        elapsed = time.perf_counter() - start
        # The comma-bearing entry, exercised the way a plan would: what the
        # device hands the connector is the assertion, never its private
        # attributes -- the same public form the round-trip test above uses.
        await built["comma_bearing"].set(0.0)
        return built, elapsed

    devices, build_seconds = asyncio.run(build_and_probe())

    total_seconds = parse_seconds + build_seconds
    print(
        f"\n[scale] {len(devices)} devices "
        f"({len(settables)} settable, {len(readables)} readable): "
        f"parse {parse_seconds:.2f}s + build {build_seconds:.2f}s "
        f"= {total_seconds:.2f}s of the {CONNECT_TIMEOUT:.0f}s worker budget "
        f"({total_seconds / CONNECT_TIMEOUT:.0%})"
    )

    # The deterministic half: nothing was dropped on the way through, and the
    # comma survived the whole path at scale.
    assert len(settables) == _SCALE_SETTABLES + 1
    assert len(readables) == _SCALE_READABLES
    assert len(devices) == _SCALE_SETTABLES + 1 + _SCALE_READABLES
    assert connector.writes == [("BTS:QF,1:CURRENT:SP", 0.0)]
