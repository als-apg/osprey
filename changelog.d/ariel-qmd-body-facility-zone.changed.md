The qmd mirror of an ARIEL entry now states the entry time in the facility zone with its UTC
offset, the same string the ARIEL panel and the agent show. The next export or
`osprey ariel qmd resync` rewrites every mirrored file once, so qmd re-indexes the whole
logbook one time.
