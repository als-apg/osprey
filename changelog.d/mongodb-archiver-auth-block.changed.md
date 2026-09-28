**Breaking change:** the MongoDB archiver block names its login under `auth:`
(`auth.source`, `auth.username`, `auth.password_env`) and its wait as
`timeout_s`. The flat `auth`, `username`, `password_env` and `timeout` keys are
no longer read. A project that deploys its own store gets the new block from
`osprey build`.
