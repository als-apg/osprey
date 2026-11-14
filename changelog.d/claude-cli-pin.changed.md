The pinned Claude Code CLI moves to 2.1.322, so the generated projects, the
dispatch worker image and the hosts that run them all use one Claude Code
version. The dispatch worker picks it up when its image is rebuilt. An existing
project follows when it is re-rendered, or when its `claude_code.cli_version` is
set to the new version.
