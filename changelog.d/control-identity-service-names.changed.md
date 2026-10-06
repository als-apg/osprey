**Breaking change:** shared writers reach the control system under fixed
service names. Dispatch worker *i* writes as `osprey-dispatch-<i>` instead of
`osprey`, and each Bluesky lane OSPREY renders writes as `osprey-bluesky` or
`osprey-bluesky-<lane>` instead of `root`; external-worker lanes are
unchanged. A container or lane that cannot apply its service name (a
read-only root filesystem, a non-root start) logs a warning and starts under
its old name. Update any gateway or IOC access-security (ACF) or put-log rule
keyed on `osprey` or `root`, and roll the images and the rendered project
together: `osprey health` reports a worker writing under a different name
than the render expects.
