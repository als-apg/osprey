The python executor now refuses writes by agent code, in either mode, to the
paths your profile names under the keys OSPREY protects: a Bluesky plan directory, a CA
bundle, a registry module, a search corpus and the other keys the python
executor architecture page lists, which also names what is not covered.
Removing, renaming or moving a directory that contains a protected location is
refused as well.
