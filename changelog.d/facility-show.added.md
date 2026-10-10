`osprey facility show [--json] [ID]` prints the facility the tree builds, and
`--json` prints it as one document with a fixed set of keys for scripts to
read. The control-assistant preset no longer ships `data/mml/`: the MATLAB
exporter comes from `osprey facility import mml --print-exporter`, whose help
text carries what the directory's README said.
