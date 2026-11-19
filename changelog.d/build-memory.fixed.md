`osprey build` of a large facility no longer runs out of memory. A facility
of 175,000 channels peaked near 6 GB resident and now stays near 2.3 GB: layer
files are read record by record, the schema check validates one record at a
time, the facility file is parsed once per build and shared with every render,
views and the facility file are copied from the first render rather than held,
the graph view is streamed to its file, and the channel search index keeps
only the triples it reads. The build's output is unchanged, byte for byte.
