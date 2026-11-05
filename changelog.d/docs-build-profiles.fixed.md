The build-profiles how-to now matches what `osprey init` and `osprey build` do
today. Its troubleshooting section quotes the messages `osprey build` prints
when no deployment repo is found and when the profile's preset has moved on,
instead of two messages from a `--preset` build that no longer exists. "What the
build does" no longer mentions `--set`, a preset, a wiped `build/` or a `git
init`; the render lands in `build/.tmp` and replaces `build/` only once it
succeeds. The init tree lists `providers.yml`, `.env.lock`, `rules/`,
`watchdogs/` and `mcp_servers/`, the hook-event list adds `PostToolUseFailure`
and `StopFailure`, the build options table adds `--dev`, the views tree shows
the channel search index a graph-mode render writes, and `osprey init --force`
is described as also rewriting `watchdogs/` and resetting the shipped entries of
`providers.yml`. A new test checks the init tree, the hook events, the build
options and the quoted messages against the code.
