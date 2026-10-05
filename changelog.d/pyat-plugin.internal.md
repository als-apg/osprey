The pyat engine plug-in builds a LUME model from a model's wiring, periodic or
single-pass, and computes orbit response matrices over the wired correctors;
the response check of `osprey facility validate` measures each kept response
export on that matrix. Fixture fingerprints are re-baselined for the import
changes they now record: `tests/facility/golden/fingerprint_spear3.json` for
the energy knob wiring, the readbacks of every member of a shared supply, the
per-field signal role and the tune readback;
`tests/facility/golden/fingerprint_nsls2.json` for the device polarity carried
on its wiring slice, the per-field signal role and the wiring of the
transfer line's beam position monitors, which the import now places from the
Accelerator Objects' lattice index.
