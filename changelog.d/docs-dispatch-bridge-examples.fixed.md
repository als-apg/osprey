Docs: the Nextcloud Talk bridge page now lists `GITLAB_URL`, `GITLAB_PROJECT`
and `GITLAB_ISSUES_TOKEN`, which the bridge reads like the other two bridges.
The Microsoft Teams page says `TEAMS_CLOUD=commercial` is the explicit form of
the default, and shows the drive-id lookup and its two `.env` lines as separate
blocks. The event-dispatch trigger examples and the three bridge profile
fragments are now files the pages include and tests load, and the pages' worker
denylist and environment variables are checked against the code.
