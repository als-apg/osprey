A plan move now settles once its readback is within the motion the facility
declares for that readback: the drift amplitude plus six times the noise of the
readback's `simulation` seed. The build writes that band into each settable's
`settle_tolerance` in `bluesky_devices.yml`, and `bluesky.settle_tolerance`
stays the floor for every device.
