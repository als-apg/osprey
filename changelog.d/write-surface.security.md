A `readonly` Python run now refuses two more Channel Access routes to the
machine: `doocs4py`, the client the DOOCS connector itself writes through, and
`aioca`, which every OSPREY environment carries as ophyd-async's Channel Access
backend. Both are refused at import and at the call. Driving hardware through
ophyd-async signals or a Bluesky `RunEngine` is refused too, while importing
those libraries for analysis stays allowed. The runtime guard and the import
denylist are now generated from one table, so they cannot describe different
libraries.
