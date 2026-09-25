Chat bridges now tell the agent who asked each question, and every remembered
exchange records who asked it. In Google Chat, Microsoft Teams and Nextcloud
Talk the agent also sees who is in the conversation and can @mention a member
when someone asks it to pass something on; set `mentions: false` in the
bridge's build-profile block (`gchat_bridge`, `teams_bridge`,
`nextcloud_bridge`) to post mentions as plain text instead.
