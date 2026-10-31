`import osprey` no longer resolves `osprey.__version__`: it is resolved on first read, so a
short-lived process that imports osprey without reading the version, such as a Claude Code hook,
does not run `git describe` in a source checkout. `from osprey import __version__` and
`osprey.__version__` read as before.
