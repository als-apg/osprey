The facility graph now links places to their parent place and devices to their
place, and carries record names as `altLabel`. The Knowledge panel draws these
edges instead of a derived relation, and `/api/graph` no longer carries
`derived` or `inPlace`. The Turtle changed, so every graph store re-seeds on
the next `up`.
