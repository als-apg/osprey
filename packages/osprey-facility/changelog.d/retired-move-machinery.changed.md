`osprey_facility.retired` now offers its move machinery for any tree, not
only a facility tree. `place_moves(root, moves, hidden=False)` reports how
each `(old, new)` path under a root moves or what blocks it, and
`render_script(moves, line_edits, deletes, line_deletes=())` renders the moves,
one-line text edits, whole-line deletions (`LineDelete`) and path deletions as
one shell script per repository, and
`deck_line_edits(layout)` gives a facility layout's deck edits in the form
`render_script` takes. `DeckEdit` is renamed
`LineEdit`; `RetiredLayout.deck_edits` holds `LineEdit`s with the same fields.
An explicitly listed retired path whose name starts with `.` now counts, and a
retired directory whose successor, or a directory the move needs, is a file is
reported as a `Clash` instead of a move. What `find_retired`, `refuse_retired`
and `move_script` report for a facility tree is otherwise unchanged.
