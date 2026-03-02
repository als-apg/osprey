"""Mock PV info backend with synthetic ALS-style PVs for development and testing."""

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

_SECTORS = [f"{i:02d}" for i in range(1, 13)]

_DEVICE_TYPES: dict[str, dict] = {
    "BPM": {
        "description": "Beam Position Monitor",
        "record_types": {"X": "ai", "Y": "ai", "S": "ai"},
        "signals": {
            "X": {"desc": "Horizontal position", "units": "mm"},
            "Y": {"desc": "Vertical position", "units": "mm"},
            "S": {"desc": "Beam intensity sum signal", "units": "counts"},
        },
        "per_sector": 6,
        "ioc_prefix": "IOC:BPM",
    },
    "HCM": {
        "description": "Horizontal Corrector Magnet",
        "record_types": {"SP": "ao", "RB": "ai", "I": "ai"},
        "signals": {
            "SP": {"desc": "Current setpoint", "units": "A"},
            "RB": {"desc": "Current readback", "units": "A"},
            "I": {"desc": "Measured current", "units": "A"},
        },
        "per_sector": 4,
        "ioc_prefix": "IOC:COR",
    },
    "VCM": {
        "description": "Vertical Corrector Magnet",
        "record_types": {"SP": "ao", "RB": "ai", "I": "ai"},
        "signals": {
            "SP": {"desc": "Current setpoint", "units": "A"},
            "RB": {"desc": "Current readback", "units": "A"},
            "I": {"desc": "Measured current", "units": "A"},
        },
        "per_sector": 4,
        "ioc_prefix": "IOC:COR",
    },
    "QF": {
        "description": "Focusing Quadrupole",
        "record_types": {"SP": "ao", "RB": "ai"},
        "signals": {
            "SP": {"desc": "Current setpoint", "units": "A"},
            "RB": {"desc": "Current readback", "units": "A"},
        },
        "per_sector": 2,
        "ioc_prefix": "IOC:QUAD",
    },
    "QD": {
        "description": "Defocusing Quadrupole",
        "record_types": {"SP": "ao", "RB": "ai"},
        "signals": {
            "SP": {"desc": "Current setpoint", "units": "A"},
            "RB": {"desc": "Current readback", "units": "A"},
        },
        "per_sector": 2,
        "ioc_prefix": "IOC:QUAD",
    },
    "DCCT": {
        "description": "DC Current Transformer — Beam Current",
        "record_types": {"CUR": "ai", "LIFETIME": "ai"},
        "signals": {
            "CUR": {"desc": "Stored beam current", "units": "mA"},
            "LIFETIME": {"desc": "Beam lifetime", "units": "hours"},
        },
        "per_sector": 0,  # global devices, not per-sector
        "ioc_prefix": "IOC:DCCT",
        "global_instances": ["01", "02"],
    },
    "RF": {
        "description": "Radio Frequency System",
        "record_types": {"V": "ai", "P": "ai", "FREQ": "ai"},
        "signals": {
            "V": {"desc": "Cavity voltage", "units": "MV"},
            "P": {"desc": "Forward power", "units": "kW"},
            "FREQ": {"desc": "RF frequency", "units": "MHz"},
        },
        "per_sector": 0,
        "ioc_prefix": "IOC:RF",
        "global_instances": ["01"],
    },
}

_HOST = "als-srv01.lbl.gov"


def _generate_pvs() -> list[PVRecord]:
    """Generate the full set of synthetic PVs (deterministic)."""
    pvs: list[PVRecord] = []

    for dev_type, spec in _DEVICE_TYPES.items():
        if spec["per_sector"] > 0:
            # Per-sector devices
            for sector in _SECTORS:
                for idx in range(1, spec["per_sector"] + 1):
                    dev_id = f"{idx:02d}"
                    ioc = f"{spec['ioc_prefix']}:{sector}"
                    for sig, sig_info in spec["signals"].items():
                        name = f"SR:{sector}:{dev_type}:{dev_id}:{sig}"
                        pvs.append(
                            PVRecord(
                                name=name,
                                record_type=spec["record_types"][sig],
                                description=f"{spec['description']} sector {sector} #{dev_id} — {sig_info['desc']}",
                                host=_HOST,
                                ioc=ioc,
                                units=sig_info["units"],
                                tags={
                                    "sector": sector,
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
                    name = f"SR:{dev_type}:{inst}:{sig}"
                    pvs.append(
                        PVRecord(
                            name=name,
                            record_type=spec["record_types"][sig],
                            description=f"{spec['description']} #{inst} — {sig_info['desc']}",
                            host=_HOST,
                            ioc=ioc,
                            units=sig_info["units"],
                            tags={
                                "device_type": dev_type,
                                "signal": sig,
                            },
                        )
                    )

    return pvs


class MockPVInfoBackend(PVInfoBackend):
    """In-memory PV info backend with synthetic ALS-style PVs.

    Generates ~500 deterministic PVs across 12 sectors with BPM, corrector,
    quadrupole, DCCT, and RF device types. Supports glob-pattern matching
    via ``fnmatch`` and filtering by record type, IOC, and description.
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
