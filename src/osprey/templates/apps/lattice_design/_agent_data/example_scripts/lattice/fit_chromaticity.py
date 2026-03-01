"""Adjust sextupole strengths to correct chromaticity.

Demonstrates:
- Reading initial chromaticity with at.get_chrom()
- Fitting chromaticity with at.fit_chrom() using named sextupole families
- Verifying the result and reporting sextupole strength changes
"""

import at

# --- Load lattice ---
ring = at.load_m("machine_data/als.m")

# --- Record initial state ---
initial_chrom = at.get_chrom(ring)
initial_sff_h = ring.get_elements("SFF")[0].H
initial_sdd_h = ring.get_elements("SDD")[0].H

print("Chromaticity Fitting")
print("=" * 50)
print(f"  Initial chrom X:  {initial_chrom[0]:.4f}")
print(f"  Initial chrom Y:  {initial_chrom[1]:.4f}")

# --- Fit to target chromaticity ---
target_chrom = [1.0, 1.0]
print(f"\n  Target chrom X:   {target_chrom[0]:.4f}")
print(f"  Target chrom Y:   {target_chrom[1]:.4f}")

at.fit_chrom(ring, "SFF", "SDD", target_chrom)

# --- Verify result ---
final_chrom = at.get_chrom(ring)
final_sff_h = ring.get_elements("SFF")[0].H
final_sdd_h = ring.get_elements("SDD")[0].H

print(f"\n  Achieved chrom X: {final_chrom[0]:.4f}")
print(f"  Achieved chrom Y: {final_chrom[1]:.4f}")

# --- Report sextupole strength changes ---
print("\nSextupole Strength Changes")
print("-" * 50)
print(f"  {'Family':<8} {'Initial H':>14} {'Final H':>14} {'Delta H':>14}")
print(f"  {'SFF':<8} {initial_sff_h:>14.4f} {final_sff_h:>14.4f} {final_sff_h - initial_sff_h:>14.4f}")
print(f"  {'SDD':<8} {initial_sdd_h:>14.4f} {final_sdd_h:>14.4f} {final_sdd_h - initial_sdd_h:>14.4f}")
print("\nDone.")
