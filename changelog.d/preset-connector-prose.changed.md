The `control-assistant` preset's comments now name every shipped connector and
archiver type, including `doocs`, `tango` and `doocs_archiver`, and `hello-world`
points at `osprey config --defaults` for the ones it does not list — which now
names them too, beside `control_system.type` and `archiver.type`. A test keeps
both in step with the types the framework actually ships.
