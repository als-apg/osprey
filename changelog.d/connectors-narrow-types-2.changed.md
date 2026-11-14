A truthy `ariel:` section that is not a mapping makes simulation apply fail
with `ValueError` instead of crashing while seeding the logbook, and a
`claude_code:` block that is not a mapping reads as no pinned CLI version when
web-terminal images are rendered. A gateway port that is not a number is still
reported unreachable, now saying it is not a port number.
