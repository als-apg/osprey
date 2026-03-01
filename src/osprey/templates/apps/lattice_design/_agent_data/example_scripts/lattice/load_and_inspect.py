"""Load the ALS lattice and print a structural summary.

Demonstrates:
- Loading a lattice file with at.load_m()
- Inspecting element counts, circumference, and energy
- Grouping elements by family and listing quadrupole K values
"""

import at
from collections import Counter

# --- Load the lattice ---
ring = at.load_m("machine_data/als.m")

# --- Basic lattice properties ---
circumference = ring.get_s_pos(range(len(ring) + 1))[-1]
energy_gev = ring.energy / 1e9

print("=" * 60)
print("ALS Lattice Summary")
print("=" * 60)
print(f"  Total elements:  {len(ring)}")
print(f"  Circumference:   {circumference:.4f} m")
print(f"  Energy:          {energy_gev:.3f} GeV")
print()

# --- Element family breakdown ---
family_counts = Counter(elem.FamName for elem in ring)

print(f"{'Family':<16} {'Count':>6}")
print("-" * 24)
for family, count in sorted(family_counts.items(), key=lambda x: -x[1]):
    print(f"  {family:<14} {count:>6}")
print()

# --- Quadrupole families with K values ---
quad_families: dict[str, float] = {}
for elem in ring:
    if isinstance(elem, at.Quadrupole) and elem.FamName not in quad_families:
        quad_families[elem.FamName] = elem.K

print(f"{'Quadrupole':<16} {'K [1/m^2]':>12}")
print("-" * 30)
for name, k_val in sorted(quad_families.items()):
    print(f"  {name:<14} {k_val:>12.6f}")
print()
print("Done.")
