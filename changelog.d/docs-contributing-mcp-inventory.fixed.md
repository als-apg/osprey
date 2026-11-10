Contributor docs: the "Verify Installation" step now runs
`uv run python scripts/check.py quick`, the merge gate's quick tier, instead of a
bare `pytest` that also collects the CANOPY tests and so fails on a fresh checkout;
the pre-commit step runs through `uv run`; the CI page describes the daily
revalidation of `main` (`main-full-ci.yml`) and the `last-green` helper, and
counts ten plugin skills. The MCP server page now
lists every tool each in-tree server can register, including `control_target`,
`control_target_set`, `provenance_locator`, the pyAML tools and the `bluesky` and
`health` servers, and a docs test holds it to the code. The Python executor's
config snippets use the dotted `config:` form of `profile.yml`.
