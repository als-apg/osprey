The channel roster reads the facility file a build writes at the root of every
render (`facility.json`), whatever channel-finder mode a project configures and
whether or not it configures one. The plan-device file, the channel-suggestions
snapshot, the channel finder's membership routes and the virtual-accelerator
channel set are all taken from its channel records: a record's `role` states
the direction, a setpoint's `pair` states its readback, and the addresses a
simulator serves for its own models' status are never listed. A render no build
has written the file into reports `The facility file facility.json is not
built` and names `osprey build`. The build's virtual-accelerator fact names the
source as `this project's facility file (facility.json)`. The channel finder's
membership and enumeration routes answer 503 naming `osprey build` as the
remedy when the render holds no facility file. A facility file that declares no
channels reports `The facility file facility.json declares no channels: the
project's data/facility tree holds no channel records.`; the plan-device file
is not staged, and the channel finder's 503 names `data/facility` as the remedy.
A project that stages no channel database gets its virtual-accelerator channel
set from the facility file in every channel-finder mode, read from the render
being built. Such a manifest carries `_metadata.source_paradigms: []` — no
channel database fed it, and `source_corpus` names `facility.json` — where it
carried `["graph"]`; a manifest built from staged channel databases is
unchanged. When the facility file declares no channels, a build deploying a
virtual accelerator stops with that sentence in place of `no channel database
is staged at tier N`.
