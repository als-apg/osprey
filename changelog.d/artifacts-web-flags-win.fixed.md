`osprey artifacts web --port 0` binds the port it is given instead of the
configured one, and with both `--host` and `--port` given the command no
longer reads the config for its address, as `osprey ariel web` and
`osprey channel-finder web` already behave.
