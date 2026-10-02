`osprey up` now refuses an archiver `auth.token_env` or `auth.password_env` with
surrounding whitespace, naming the key. Such a name used to deploy and then fail
at the first archiver read inside the web terminal.
