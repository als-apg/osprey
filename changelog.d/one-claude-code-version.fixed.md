Headless agent runs (dispatch worker, chat view, `osprey query`) now execute
the Claude Code version the project pins — `claude_code.cli_version`, else the
installed `claude` — instead of the copy bundled inside the Agent SDK, so the
terminal, the chat and the dispatcher run one build. **Behaviour change on a
developer machine:** a headless run uses only an installed `claude` that
reports the pinned version. When none does, the run is refused before anything
starts, with one message naming the version asked for, what was found, and the
two ways out: `npm install -g @anthropic-ai/claude-code@<version>`, or change
`claude_code.cli_version`. A dispatch worker in that state refuses to start
rather than accept jobs it would refuse. **`claude_code.cli_version` must now
be one exact version** such as `2.1.300`: a range, a tag (`latest`, `stable`),
an alias, a URL, a git address, a file or a path is refused, and surrounding
whitespace is no longer trimmed. The same check applies to the terminal, to
headless runs, to the health check and to the image builds. Sessions on newer
models and on a gateway's own model names also get the task list again: OSPREY
turns Claude Code's task-list tools on for every process it starts, so all
three surfaces are offered the same tools whatever model they run.
