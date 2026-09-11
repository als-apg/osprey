The shipped presets no longer render `web.docs_url`, `web.feedback.email` and
`web.feedback.github_repo` as live keys, so `osprey init` stops writing the
OSPREY project's own documentation site, mailbox and tracker into your
deployment's `profile.yml` as though you had chosen them. All three are
documented beside the keys as commented examples; the docs and tracker defaults
are unchanged, so the Documentation button and the GitHub feedback channel
still point where they did; the mail default is already blank, so an
unconfigured deployment offers no Email channel.

An existing `profile.yml` that carries any of those lines is now a difference
from its preset, so `osprey validate` reports it. Delete the line to take the
default, keep it and mark it `# DEVIATION: <why>`, or run with `--drift=warn`.
