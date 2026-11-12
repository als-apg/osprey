`osprey app check` closes an app agent's or listed skill's frontmatter only on a
line that is exactly `---`, as Claude Code does; a line inside it starting with
`---` is refused instead of hiding the keys after it.
