The `control-assistant` preset no longer ships an empty `data/facility/classes.yaml`, matching
`hello-world`. A tree without the file builds with no added classes, and
`osprey facility import mml` on a preset now seeds `classes.yaml` with the
classes its mapping adds.
