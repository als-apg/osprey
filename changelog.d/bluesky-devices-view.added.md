`osprey build` writes each render's `data/bluesky_devices.yml` from the
facility file when the render runs a Bluesky lane: one settable per setpoint
channel, one readable per readback channel, each named by its address, under a
`schema: osprey.facility.bluesky_devices/1` first line. The Bluesky staging
copies that file unchanged into the worker's build context, and a mock control
system stages no copy.
