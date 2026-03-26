"""Compute Twiss parameters and linear optics for the ALS lattice.

Demonstrates:
- Computing optics with at.get_optics() including chromaticity
- Extracting beta functions and tunes from the optics result
- Saving a matplotlib beta-function plot to the workspace
"""

import at
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Load lattice and compute optics ---
ring = at.load_m("machine_data/als.m")
refpts = range(len(ring) + 1)
ld0, rd, ld = at.get_optics(ring, refpts=refpts, get_chrom=True)

# --- Extract results ---
s_pos = ring.get_s_pos(refpts)
beta_x = ld.beta[:, 0]
beta_y = ld.beta[:, 1]
tunes = rd.tune
chroms = rd.chromaticity

# --- Print summary ---
print("ALS Linear Optics")
print("=" * 40)
print(f"  Tune X:          {tunes[0]:.6f}")
print(f"  Tune Y:          {tunes[1]:.6f}")
print(f"  Chromaticity X:  {chroms[0]:.4f}")
print(f"  Chromaticity Y:  {chroms[1]:.4f}")
print(f"  Beta X max/min:  {beta_x.max():.3f} / {beta_x.min():.3f} m")
print(f"  Beta Y max/min:  {beta_y.max():.3f} / {beta_y.min():.3f} m")

# --- Plot beta functions ---
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(s_pos, beta_x, "b-", label=r"$\beta_x$")
ax.plot(s_pos, beta_y, "r-", label=r"$\beta_y$")
ax.set_xlabel("s [m]")
ax.set_ylabel(r"$\beta$ [m]")
ax.set_title("ALS Beta Functions")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()

output_path = "_agent_data/plots/optics.png"
fig.savefig(output_path, dpi=150)
plt.close(fig)
print(f"\nPlot saved to {output_path}")
