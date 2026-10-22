A deployment whose profile sets `claude_md_template:` now treats that template as its
`CLAUDE.md` everywhere: a `CLAUDE.md` listed as user-owned under the template's name
survives `osprey build`, the web terminal's scaffold gallery names and renders that
template, and an unknown `claude_md_template:` fails the build by name.
