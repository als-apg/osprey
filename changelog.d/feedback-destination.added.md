Feedback now goes to one destination: whoever owns the deployment. A new
`web.feedback.owner` block names them — a display name, an address and an
issue tracker on either GitHub or GitLab — so a facility redirects both halves
in one place instead of two settings that could be edited apart. Reports
arriving there carry which preset and channel-finder mode produced them, plus
a prefilled upstream issue link, so a maintainer who decides a bug belongs to
OSPREY can forward it in one click. `web.feedback.email` and
`web.feedback.github_repo` still take precedence wherever they are set, and
`osprey build` now warns when only one of the two was moved.
