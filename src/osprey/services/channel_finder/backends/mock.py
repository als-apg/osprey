"""Mock PV info backend with synthetic synchrotron-style PVs for development and testing."""

from __future__ import annotations

from fnmatch import fnmatch

from osprey.services.channel_finder.backends.base import (
    PVInfoBackend,
    PVRecord,
    SearchResult,
)

# ---------------------------------------------------------------------------
# PV generation tables — deterministic, no randomness
# ---------------------------------------------------------------------------

_DEVICE_TYPES: dict[str, dict] = {
    "BPM": {
        "subsystem": "DIAG",
        "description": "Beam Position Monitor",
        "record_types": {
            "POSITION:X": "ai",
            "POSITION:Y": "ai",
            "SIGNAL:SUM": "ai",
        },
        "signals": {
            "POSITION:X": {"desc": "Horizontal position", "units": "mm"},
            "POSITION:Y": {"desc": "Vertical position", "units": "mm"},
            "SIGNAL:SUM": {"desc": "Beam intensity sum signal", "units": "counts"},
        },
        "count": 20,
        "ioc_prefix": "IOC:DIAG:BPM",
    },
    "HCM": {
        "subsystem": "MAG",
        "description": "Horizontal Corrector Magnet",
        "record_types": {"CURRENT:SP": "ao", "CURRENT:RB": "ai", "CURRENT:MEAS": "ai"},
        "signals": {
            "CURRENT:SP": {"desc": "Current setpoint", "units": "A"},
            "CURRENT:RB": {"desc": "Current readback", "units": "A"},
            "CURRENT:MEAS": {"desc": "Measured current", "units": "A"},
        },
        "count": 20,
        "ioc_prefix": "IOC:MAG:COR",
    },
    "VCM": {
        "subsystem": "MAG",
        "description": "Vertical Corrector Magnet",
        "record_types": {"CURRENT:SP": "ao", "CURRENT:RB": "ai", "CURRENT:MEAS": "ai"},
        "signals": {
            "CURRENT:SP": {"desc": "Current setpoint", "units": "A"},
            "CURRENT:RB": {"desc": "Current readback", "units": "A"},
            "CURRENT:MEAS": {"desc": "Measured current", "units": "A"},
        },
        "count": 20,
        "ioc_prefix": "IOC:MAG:COR",
    },
    "QF": {
        "subsystem": "MAG",
        "description": "Focusing Quadrupole",
        "record_types": {"CURRENT:SP": "ao", "CURRENT:RB": "ai"},
        "signals": {
            "CURRENT:SP": {"desc": "Current setpoint", "units": "A"},
            "CURRENT:RB": {"desc": "Current readback", "units": "A"},
        },
        "count": 16,
        "ioc_prefix": "IOC:MAG:QUAD",
    },
    "QD": {
        "subsystem": "MAG",
        "description": "Defocusing Quadrupole",
        "record_types": {"CURRENT:SP": "ao", "CURRENT:RB": "ai"},
        "signals": {
            "CURRENT:SP": {"desc": "Current setpoint", "units": "A"},
            "CURRENT:RB": {"desc": "Current readback", "units": "A"},
        },
        "count": 16,
        "ioc_prefix": "IOC:MAG:QUAD",
    },
    "DCCT": {
        "subsystem": "DIAG",
        "description": "DC Current Transformer — Beam Current",
        "record_types": {"CURRENT:RB": "ai", "LIFETIME:RB": "ai"},
        "signals": {
            "CURRENT:RB": {"desc": "Stored beam current", "units": "mA"},
            "LIFETIME:RB": {"desc": "Beam lifetime", "units": "hours"},
        },
        "count": 0,  # global devices, not per-instance range
        "ioc_prefix": "IOC:DIAG:DCCT",
        "global_instances": ["01", "02"],
    },
    "RF": {
        "subsystem": "RF",
        "description": "Radio Frequency System",
        "record_types": {"VOLTAGE:RB": "ai", "POWER:FWD": "ai", "FREQUENCY:RB": "ai"},
        "signals": {
            "VOLTAGE:RB": {"desc": "Cavity voltage", "units": "MV"},
            "POWER:FWD": {"desc": "Forward power", "units": "kW"},
            "FREQUENCY:RB": {"desc": "RF frequency", "units": "MHz"},
        },
        "count": 0,
        "ioc_prefix": "IOC:RF",
        "global_instances": ["01"],
    },
}

_HOST = "synth-srv01.example.org"


def _generate_pvs() -> list[PVRecord]:
    """Generate the full set of synthetic PVs (deterministic)."""
    pvs: list[PVRecord] = []

    for dev_type, spec in _DEVICE_TYPES.items():
        subsystem = spec["subsystem"]

        if spec["count"] > 0:
            # Numbered devices
            for idx in range(1, spec["count"] + 1):
                dev_id = f"{idx:02d}"
                ioc = f"{spec['ioc_prefix']}:{dev_id}"
                for sig, sig_info in spec["signals"].items():
                    name = f"SR:{subsystem}:{dev_type}:{dev_id}:{sig}"
                    pvs.append(
                        PVRecord(
                            name=name,
                            record_type=spec["record_types"][sig],
                            description=(f"{spec['description']} #{dev_id} — {sig_info['desc']}"),
                            host=_HOST,
                            ioc=ioc,
                            units=sig_info["units"],
                            tags={
                                "subsystem": subsystem,
                                "device_type": dev_type,
                                "signal": sig,
                            },
                        )
                    )
        else:
            # Global devices
            for inst in spec.get("global_instances", []):
                ioc = f"{spec['ioc_prefix']}:{inst}"
                for sig, sig_info in spec["signals"].items():
                    name = f"SR:{subsystem}:{dev_type}:{inst}:{sig}"
                    pvs.append(
                        PVRecord(
                            name=name,
                            record_type=spec["record_types"][sig],
                            description=(f"{spec['description']} #{inst} — {sig_info['desc']}"),
                            host=_HOST,
                            ioc=ioc,
                            units=sig_info["units"],
                            tags={
                                "subsystem": subsystem,
                                "device_type": dev_type,
                                "signal": sig,
                            },
                        )
                    )

    return pvs


class MockPVInfoBackend(PVInfoBackend):
    """In-memory PV info backend with synthetic synchrotron-style PVs.

    Generates ~250 deterministic PVs with BPM, corrector, quadrupole, DCCT,
    and RF device types using unified ``SR:{system}:{family}:{device}:{field}:{subfield}``
    naming. Supports glob-pattern matching via ``fnmatch`` and filtering by
    record type, IOC, and description.
    """

    def __init__(self) -> None:
        self._pvs = _generate_pvs()
        self._by_name: dict[str, PVRecord] = {pv.name: pv for pv in self._pvs}

    @property
    def total_pv_count(self) -> int:
        return len(self._pvs)

    async def search(
        self,
        pattern: str,
        *,
        record_type: str | None = None,
        ioc: str | None = None,
        description_contains: str | None = None,
        page: int = 1,
        page_size: int = 100,
    ) -> SearchResult:
        page_size = min(page_size, 200)
        page = max(page, 1)

        matches = [pv for pv in self._pvs if fnmatch(pv.name, pattern)]

        if record_type:
            matches = [pv for pv in matches if pv.record_type == record_type]
        if ioc:
            matches = [pv for pv in matches if pv.ioc == ioc]
        if description_contains:
            lower = description_contains.lower()
            matches = [pv for pv in matches if lower in pv.description.lower()]

        total = len(matches)
        start = (page - 1) * page_size
        end = start + page_size
        page_records = matches[start:end]

        return SearchResult(
            records=page_records,
            total_count=total,
            has_more=end < total,
            page=page,
            page_size=page_size,
        )

    async def get_metadata(self, pv_names: list[str]) -> list[PVRecord]:
        return [self._by_name[name] for name in pv_names if name in self._by_name]
