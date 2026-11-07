Each web-terminal user gets their own CANOPY, the operator GUI with machine
screens, Trace and the correlation and waveform pages, as the builtin `canopy`
sidecar panel (`web.panels.canopy`). The terminal runs CANOPY's panel server
and its data process for that user on loopback ports it holds, so a request
made while the data process is replaced waits for the next one. The data
process reads every PV through the deployment's control session rather than a
second EPICS client, says on every socket which machine it serves (the
target's name, a fallback or unservable notice, the binding id), and keeps a
channel's kind while it is down. It is replaced at once when the control
target moves or its transport goes stale, and the panel is
retracted after five exits in a minute or a refused start. Its not-answering
statement and write state are the ones the control-target chip shows.

The panel's settings are `home_screen` (the library screen opened first;
`osprey build` and `osprey validate` refuse an id the library does not hold
and list the ones it does), `trace_default_query`,
`correlation_default_query` and `waveform_default_query` (the view a bare page
opens with), `theme_file` (relative to the project root, or absolute) and
`vole_bytes_max`. CANOPY takes its look from three stylesheets the terminal
hands it: the design tokens, the font faces and `canopy-host.css`. Each user
has a personal screen store and collection store under their agent data, and
the deployment's `data/canopy/library` and `data/canopy/collections` are
served read-only beside them. Trace's and the correlation page's autocomplete
offer the deployment's channels from one snapshot the build stages at
`build/web_terminals/channels.json` and mounts read-only into every web
container. The header command palette lists matching CANOPY screens and
collections and opens one in the panel, navigating an open viewer in place.
Only CANOPY's own application pages (`/`, `/s/*`, `/embed/*`, `/trace/*`) are
exempt from the panel sandbox; drafts stay sandboxed.
