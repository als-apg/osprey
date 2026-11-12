`osprey app check` closes an app agent's or listed skill's frontmatter only on a
line that is exactly `---`, as Claude Code does; a line inside it starting with
`---`, or an opening or close a looser reader would take (a byte-order mark, a
no-break space), is refused instead of hiding the keys after it. The agent tool
audit reads a file with a byte-order mark, and reports one it cannot read.
