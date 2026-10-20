A `.env` or `.env.shared` line can no longer change `PATH`, `HOME`, `SHELL` or
the Python, Node, shell start-up and dynamic-loader variables of any OSPREY
process. Such lines are ignored with a warning naming them.
