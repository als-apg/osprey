#!/usr/bin/env python3
"""Stamp each beam-position monitor's charge-to-sum factor into the example facility.

A real monitor's sum signal is the beam charge times a per-monitor factor
``k_b`` (button gains, cable losses, electronics), so equal charge reads as
unequal sums until a reader divides the factor out. The example facility
declares that factor in the machine: every turn-by-turn sum record
(``...:TBT:SUM``) of ``models.yaml`` carries it as its wiring calibration,
``calibration.curve.linear.gain = 1/k_b``, hardware to physics. The simulator
already turns a record's physics value into hardware by its calibration, so a
monitor reads ``k_b`` times the charge it sees.

Each factor is log-normal, ``k_b = exp(SIGMA * z_b)``, with ``z_b`` the seeded,
order-free draw ``keyed_normals(channel_key_bytes(<sum address>) +
b":sum_calibration", [SEED])``. The draw happens here, once; nothing is drawn
at runtime. The gains are written rounded to ``DIGITS`` significant figures and
every other byte of ``models.yaml`` stays as it is.

Usage::

    uv run python scripts/demo_seeds/sum_calibration.py           # rewrite models.yaml
    uv run python scripts/demo_seeds/sum_calibration.py --check   # exit 1 if it differs
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np

from osprey_connectors.simulation.series import channel_key_bytes, keyed_normals

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS = REPO_ROOT / "src/osprey/templates/facilities/example/models.yaml"

#: Width of ``log k_b``.
SIGMA = 0.2
#: The counter the draws are addressed by.
SEED = 0
#: Significant figures each written gain keeps.
DIGITS = 6
#: The subkey that gives the sum factor its own stream per address.
SUBKEY = b":sum_calibration"

_ADDRESS = re.compile(r"^  - address: (?P<address>\S+:TBT:SUM)\s*$")
_GAIN = re.compile(r"^(?P<indent> +)gain: .*$")


class SumCalibrationError(Exception):
    """A sum record has no calibration gain line to stamp."""


def factor(address: str) -> float:
    """The charge-to-sum factor ``k_b`` of the sum record at ``address``."""
    draw = keyed_normals(channel_key_bytes(address) + SUBKEY, np.asarray([SEED]))[0]
    return math.exp(SIGMA * float(draw))


def gain(address: str) -> float:
    """``1/k_b`` rounded to ``DIGITS`` significant figures, as written."""
    return float(f"{1.0 / factor(address):.{DIGITS}g}")


def stamp(text: str | None = None) -> str:
    """``models.yaml`` with every sum record's calibration gain set to its ``1/k_b``.

    Only the ``gain:`` line inside each ``...:TBT:SUM`` record changes; stamping
    is idempotent.

    Args:
        text: The file's text; the committed file when None.

    Raises:
        SumCalibrationError: A sum record carries no ``gain:`` line.
    """
    if text is None:
        text = MODELS.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    current: str | None = None
    for line in lines:
        match = _ADDRESS.match(line)
        if match:
            if current is not None:
                raise SumCalibrationError(f"{current}: no calibration gain line")
            current = match["address"]
        elif current is not None and line.startswith("  - "):
            raise SumCalibrationError(f"{current}: no calibration gain line")
        elif current is not None and (gain_line := _GAIN.match(line.rstrip("\n"))):
            line = f"{gain_line['indent']}gain: {gain(current)!r}\n"
            current = None
        out.append(line)
    if current is not None:
        raise SumCalibrationError(f"{current}: no calibration gain line")
    return "".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if models.yaml differs")
    args = parser.parse_args(argv)
    text = stamp()
    target = MODELS.relative_to(REPO_ROOT)
    if args.check:
        if MODELS.read_text(encoding="utf-8") != text:
            print(f"{target} differs; run scripts/demo_seeds/sum_calibration.py")
            return 1
        print(f"{target} matches")
        return 0
    MODELS.write_text(text, encoding="utf-8")
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
