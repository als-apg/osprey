"""Draw the two probe pictures: an orbit plot with a kick near BPM 7, and an unrelated one.

Usage: ``python make_probe_plots.py OUT_DIR [DPI] [SUFFIX]``

Writes ``orbit_kick<SUFFIX>.png`` and ``tunnel_temp<SUFFIX>.png`` into OUT_DIR,
840x420 px at the default 120 dpi. The recorded llama-server vectors in this
directory were taken from the PNG bytes committed next to them, not from a
re-run of this script, so a different matplotlib may draw different bytes
without invalidating the fixtures.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main(argv: list[str]) -> None:
    if len(argv) < 2:
        raise SystemExit(__doc__)
    out = Path(argv[1])
    out.mkdir(parents=True, exist_ok=True)
    dpi = int(argv[2]) if len(argv) > 2 else 120
    suffix = argv[3] if len(argv) > 3 else ""

    rng = np.random.default_rng(3)
    bpm = np.arange(1, 25)
    x = rng.normal(0, 0.03, bpm.size)
    x[6] += 0.42  # BPM 7
    x[7] += 0.25
    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=dpi)
    ax.plot(bpm, x, "o-", color="tab:blue")
    ax.axvline(7, color="tab:red", ls="--", lw=1)
    ax.annotate(
        "SR:C07 BPM",
        (7, 0.36),
        xytext=(10, 0.30),
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": "gray"},
    )
    ax.set_xlabel("BPM index")
    ax.set_ylabel("x [mm]")
    ax.set_title("Horizontal orbit, fill 2026-09-12 14:03")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / f"orbit_kick{suffix}.png")

    t = np.linspace(0, 8, 400)
    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=dpi)
    ax.plot(
        t, 22.5 + 0.8 * np.sin(2 * np.pi * t / 8) + rng.normal(0, 0.05, t.size), color="tab:green"
    )
    ax.set_xlabel("time [h]")
    ax.set_ylabel("T [degC]")
    ax.set_title("Tunnel air temperature, sector 4")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / f"tunnel_temp{suffix}.png")
    print(f"wrote orbit_kick{suffix}.png tunnel_temp{suffix}.png at dpi {dpi} in {out}")


if __name__ == "__main__":
    main(sys.argv)
