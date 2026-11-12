`osprey app check` and `osprey build` refuse hook code in an application: a
`hooks` key in `app.yml`, a `hooks/` directory, a `.claude/` directory, or
`hooks` in a listed skill's frontmatter. The refusal (`no_hooks`) names what
to declare in `app.yml` instead.
