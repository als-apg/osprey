Swallowed exceptions are named in one `with contextlib.suppress(...)` instead
of a `try`/`except`/`pass` block, and ruff rule SIM105 keeps it that way.
