A draft whose sequential algorithm may not move from the current setpoints on
its grid (for instance Nelder-Mead starting at zero) now carries a
`start_notice` saying so. The optimizer panel shows it beside Start, and the
MCP draft tools pass it on as advice; it never holds the start.
