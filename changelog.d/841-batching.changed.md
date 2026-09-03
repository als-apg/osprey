`osprey build` edits each rendered `config.yml` in memory across every step of
the render instead of re-reading and rewriting the file in each one, which
roughly halves the build time. The rendered file now keeps the template's
quoting and list indentation all the way through the build; previously the last
step to touch it rewrote lists at column zero and dropped the quotes.
