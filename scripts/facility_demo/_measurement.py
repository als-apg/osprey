"""The demo's ``measurement/SR.yaml``: what the deck machine's model may measure.

``kinds`` is the authored allow-list. ``groups`` names the deck machine's own
family groups, one per role; ``instruments`` names the channels the tune,
chromaticity and RF steps read or write. The step and settle keys are pyAML
0.3.1's own, carried verbatim with the tuning constants the pyAML emitter
lists.
"""

from __future__ import annotations

from typing import Any

#: The model the file belongs to; the file is ``measurement/<MODEL>.yaml``.
MODEL = "SR"

#: The measurements the model allows.
KINDS = ("orm", "dispersion", "trm", "crm", "chromaticity_monitor")

#: Role -> the deck machine's family group filling it.
GROUPS = {
    "bpm": "SR/BPM",
    "hcor": "SR/HCM",
    "vcor": "SR/VCM",
    "quad": "SR/QF",
    "sext": "SR/SF",
}

#: Role -> the channel it reads or writes.
INSTRUMENTS = {
    "tune": "SR:DIAG:TUNE:X",
    "chromaticity": "SR:DIAG:CHROM:X",
    "rf": "SR:RF:CAVITY:01:FREQUENCY:SP",
}

#: pyAML's step and settle keys: counts, the fit order, sleeps in s, and the
#: corrector (rad), quadrupole (1/m), sextupole (1/m**2) and RF (Hz) steps.
TUNING: dict[str, int | float] = {
    "n_step": 5,
    "n_avg_meas": 1,
    "fit_order": 2,
    "singular_values": 16,
    "sleep_between_step": 0.0,
    "sleep_between_meas": 0.0,
    "corrector_delta": 1.0e-5,
    "quad_delta": 1.0e-3,
    "sextu_delta": 1.0e-2,
    "frequency_delta": 100.0,
}


def path() -> str:
    """The file's path relative to ``data/facility``."""
    return f"measurement/{MODEL}.yaml"


def build_measurement() -> dict[str, Any]:
    """``measurement/SR.yaml``'s document."""
    return {
        "kinds": list(KINDS),
        "groups": dict(GROUPS),
        "instruments": dict(INSTRUMENTS),
        **TUNING,
    }
