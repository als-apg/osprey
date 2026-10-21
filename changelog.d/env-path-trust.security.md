A `.env` or `.env.shared` line can no longer change `PATH`, `HOME`, `SHELL` or
the Python, Node, shell start-up and dynamic-loader variables of any OSPREY
process. Such lines are ignored with a warning naming them. The same holds for
the git, npm, uv, pip, compiler, Docker, Compose, Claude Code, Perl, Ruby, Java,
OpenSSL, glibc, pager, editor, browser and askpass variables, `XDG_CONFIG_*`,
`PS4`, `IFS` and exported bash functions, in any letter case: a `.env` that set
one of these (e.g. `DOCKER_HOST`) is now ignored with a warning; export it in
the shell instead.
