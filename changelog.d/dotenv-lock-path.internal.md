The sibling lock file guarding a `.env` is now named in one place,
`env_lock_path`, instead of having its `.lock` suffix spelled out at each site
that derives it. Naming is unchanged; nothing on disk moves.
