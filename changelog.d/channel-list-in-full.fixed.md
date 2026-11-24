The hierarchical channel finder's `get_options` now offers the options under every
value a level above is selected with. Selecting all twelve storage-ring sectors used
to offer only the first sector's devices, so a request for every storage-ring BPM
came back with 6 of 72. An agent asked for a list of channel addresses now answers
with every address written out, one per line, instead of a range such as
`SR:DIAG:BPM:01:POSITION:X` through `SR:DIAG:BPM:72:POSITION:X`: the main session's
rule for a subagent's hand-back now says to relay the identifiers it lists as well as
its headline. The hierarchical channel finder is also told to select options only by
the names `get_options` lists, since a wildcard or a name built from a pattern is
refused.
