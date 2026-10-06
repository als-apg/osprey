A deployment whose profile sets `claude_md_template:` now treats that persona as its
`CLAUDE.md` everywhere: a `CLAUDE.md` listed as user-owned under the persona's name
survives `osprey build`, the web terminal's scaffold gallery names and renders that
persona, and an unknown `claude_md_template:` fails the build by name.
