Naming your own image in `services.<name>.image` now runs that image as named
for the event dispatcher, the qmd sidecar, the Bluesky bridge and web panel, and
the Google Chat, Nextcloud and Teams bridges. The service renders without a
`build:` block, so a deploy no longer rebuilds OSPREY's recipe and tags it with
your image's name.
