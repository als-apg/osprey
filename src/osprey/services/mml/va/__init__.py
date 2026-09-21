"""The virtual-accelerator half of an MML export.

A 2.0 exporter writes three files beside the AO and AD of a sub-machine: the
lattice deck it saved before sampling anything, a ``va.json`` holding the deck
fingerprint and the per-family calibrations, nominals and energy tables it
sampled through MATLAB, and a ``response.json`` holding the orbit response
matrix. The modules here own that half of the import, keyed by system exactly
as the AO and AD are.
"""
