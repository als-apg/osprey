The historical `osprey.*` import paths kept for code outside this repository now
re-export the names they forward to, so a type checker resolves a name through
the old spelling instead of reporting it as absent. Two in-repo callers that
reached through such a path for a private name now name the module that defines
it.
