With `ariel.vocabulary.expand_modes` unset, semantic and hybrid search no longer
expand the query while `ariel.vocabulary.expand_entries` is off: expanding the
query alone makes dense retrieval worse than not expanding (JLAB-TN-26-024).
Keyword search still expands; turning entry expansion on brings the dense modes
back, and an explicit `expand_modes` list still names them directly.
