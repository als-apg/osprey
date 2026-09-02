Shipped templates, prompts and CLI messages no longer name commands, files,
keys and tools that do not exist. Generated READMEs point at the repo-root
`.env` and `profile.yml` instead of a `src/` package; persona headers say the
profile's `claude_md_template:` key owns `CLAUDE.md`; `config.yml` comments cite
published docs pages; the deploy's `.env` banners and `.env.auth` headers name
`osprey up`, with the older spellings still recognized on disk.
