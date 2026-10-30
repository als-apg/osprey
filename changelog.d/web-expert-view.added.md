`web.expert_view: false` takes the web terminal's Expert view away on every
login a render serves. Every terminal opens in Simple whatever `web.ui_mode`
says. The display menu's View row, the command palette's mode switch and
`?mode=expert` links are gone, and `/ws/terminal` refuses to start a Claude
Code session. `osprey build` refuses `web.ui_mode: expert` beside it. Absent,
nothing changes.
