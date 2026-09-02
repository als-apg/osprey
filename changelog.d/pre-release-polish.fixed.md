A batch of pre-release fixes. An EPICS put whose callback times out is now
reported as `unconfirmed` and is not re-read; pyepics answers that timeout with
`-1`, which the connector had taken as a success. The `unconfirmed` outcome now
reads "sent, but not acknowledged in time or the re-read itself failed"
everywhere it is described. A middle-layer channel database in the wrong shape
(a non-mapping root, system or family) is refused as a corrupt source instead
of loading as zero channels, so a build no longer falls back to scenario seeds
while claiming the channels came from that file; `osprey channel-finder
preview` and the spec generator raise on such a file rather than reporting no
channels. The control-target popover reads whether the deployment is running
read-only from the posture route's new `readonly_run` field instead of
inferring it, so a posture store that fails to resolve is no longer reported as
a read-only deployment, and a read-only run is named even when one target was
never armed. The posture route answers instead of crashing when the config it
falls back to does not load. In the two high-contrast themes a healthy status
dot no longer shares its grey with muted text.
