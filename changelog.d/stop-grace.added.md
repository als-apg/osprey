A sidecar panel can now ask for time to stop. `SidecarPanel.stop_grace_s`
(default `0`) is the number of seconds its stop waits between the stop signal
and the kill, and the one number the deployment sizes the compose service's
`stop_grace_period` from: the slowest panel a terminal shows plus the
terminal's own stop, rendered whenever that exceeds compose's default. The
JupyterLab panel declares the five seconds its stop already waited.
