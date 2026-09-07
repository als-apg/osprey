The web terminal ships no default recipient for feedback mail. A prefilled
draft can carry a session's scrollback, so the mailbox that receives it is now
named by the deployment (`web.feedback.email`, or `web.feedback.owner.email`)
rather than inherited. Until one is set the dialog offers its issue tracker
only, and `osprey feedback list` / `export` still records every submission
locally.
