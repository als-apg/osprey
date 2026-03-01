"""Adjust quadrupole strengths to match target tunes.

Demonstrates:
- Reading initial tunes with at.get_tune()
- Fitting tunes with at.fit_tune() using named quadrupole families
- Verifying the result and reporting K-value changes
"""

import at

# --- Load lattice ---
ring = at.load_m("machine_data/als.m")

# --- Record initial state ---
initial_tunes = at.get_tune(ring)
initial_qf_k = ring.get_elements("QF")[0].K
initial_qd_k = ring.get_elements("QD")[0].K

print("Tune Fitting")
print("=" * 50)
print(f"  Initial tune X:  {initial_tunes[0]:.6f}")
print(f"  Initial tune Y:  {initial_tunes[1]:.6f}")

# --- Fit to target tunes ---
target_tunes = [0.25, 0.20]
print(f"\n  Target tune X:   {target_tunes[0]:.6f}")
print(f"  Target tune Y:   {target_tunes[1]:.6f}")

at.fit_tune(ring, "QF", "QD", target_tunes)

# --- Verify result ---
final_tunes = at.get_tune(ring)
final_qf_k = ring.get_elements("QF")[0].K
final_qd_k = ring.get_elements("QD")[0].K

print(f"\n  Achieved tune X: {final_tunes[0]:.6f}")
print(f"  Achieved tune Y: {final_tunes[1]:.6f}")

# --- Report K-value changes ---
print("\nQuadrupole Strength Changes")
print("-" * 50)
print(f"  {'Family':<8} {'Initial K':>14} {'Final K':>14} {'Delta K':>14}")
print(f"  {'QF':<8} {initial_qf_k:>14.6f} {final_qf_k:>14.6f} {final_qf_k - initial_qf_k:>14.6f}")
print(f"  {'QD':<8} {initial_qd_k:>14.6f} {final_qd_k:>14.6f} {final_qd_k - initial_qd_k:>14.6f}")
print("\nDone.")
