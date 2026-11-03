The `control-assistant` preset shows the CANOPY panel and opens on it at
`facility/overview`, selects the `canopy-panel-builder`, `canopy-operator` and
`canopy-screens` skills, and stages the payload in a `canopy-payload` pre-build
step; a build that names no payload leaves the panel out and says so. Screen
writes follow each login's write switch, as the agent's do: at `asked` the
panel asks before each write, at `allowed` it writes, and a readonly login's
panel is inert. Its default collection
opens the overview, orbit feedback and three tools each on a view of its own:
Orbit in Trace, RF cavity 01's temperature against its reflected power on the
correlation page, and the septum pulse on the waveform page with the last 10
shots overlaid. A repo made before this release keeps its profile: add
`canopy` to `web_panels`, the three skills and the `canopy-payload` step by
hand.
