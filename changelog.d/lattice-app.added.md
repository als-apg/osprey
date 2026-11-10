A `lattice` app ships beside orbit bump, off by default (`--set
'apps.lattice={}'`): optics, chromaticity, dynamic and momentum aperture and
tune footprint of the facility's decks, each computed once per deployment and
cached. The facility chooses the analyses, their settings and deadlines, and
the engine per model (pyAT, elegant, or its own command). The agent reads the
same results with `lattice_state`, `lattice_result` and `lattice_export`.
