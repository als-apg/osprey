`osprey up` without `-d` now builds the service images in a step of its own
before it hands the terminal to compose, the way `osprey up --dev` already
did. That build's output is spooled to `var/logs/` instead of streaming;
pass `--verbose` to watch it live.
