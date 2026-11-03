Limits checking in `optional` mode with no `database_path` no longer needs a
limits database: every channel is written as unlisted, and build and deploy
no longer refuse the configuration. A configured path must still exist.
