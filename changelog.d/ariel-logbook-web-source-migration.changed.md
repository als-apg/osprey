A new always-on ARIEL migration, `logbook_web_source_system`, renames stored
entries whose `source_system` is exactly `"ARIEL Web"` to
`"OSPREY Logbook Web"`, so the Source System facet shows one name. Renamed
entries are re-exported to the hybrid-search mirror on its next resync.
