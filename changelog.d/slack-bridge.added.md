A Slack bridge joins the Nextcloud Talk, Google Chat and Microsoft Teams ones:
message the app directly or mention it in a channel, and the answer, plots and
files come back in the thread. `slack_bridge:` in a build profile deploys it;
it connects out over Slack's Socket Mode with `SLACK_BOT_TOKEN` and
`SLACK_APP_TOKEN`, so it opens no port.
