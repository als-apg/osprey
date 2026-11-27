A `RetiredLayoutRefused` message says how its block ends. It opens with
`errors.PASTE_HEADER` and closes with `osprey init --revalidate <repo>` when
the block can be pasted whole; with a clash, or no line to run, it opens with
`errors.MOVE_HEADER` and closes with a line asking to place what is listed.
`errors.refusal_ending(script, repo, finishes=...)` gives that closing line to
any refusal built on the move machinery. `refuse_retired(machine_dir,
content_dir, repo=None)` takes the deployment repo the closing command names;
without one it names the git work tree holding the content root, or the
directory holding that root. `render_script` runs its moves in stages, each
against the tree the moves before it leave: a move gets its own `mkdir -p`
immediately before it, and `git mv` is chosen where its source lies before any
move. A source tracked by a repository nested inside the script's own moves
with `git -C <that repository> mv`, and a tracked delete falls back to `rm`
when git refuses it. `render_script(..., after=moves)` and
`move_script(layout, after=moves)` name moves an earlier script makes, so the
moves they render are traced back through them.
