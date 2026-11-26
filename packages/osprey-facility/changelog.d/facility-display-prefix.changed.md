`osprey_facility.source_tree.display_path()` now names a facility path as
`facility/<relative>`, the facility tree's place at the deployment root, where
it named `data/facility/<relative>`. Messages and refusals that quote a source
path follow; a caller comparing against the old spelling must be updated.
