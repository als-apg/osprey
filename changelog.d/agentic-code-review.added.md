A deployment can turn on code review: a second AI agent reads the Python the
OSPREY agent writes, on Python runs and on notebook cells, following
instructions the deployment writes in `data/code_review/reviewer.md`, and
either stops unsafe code or only reports it (`code_review.level`). The agent
gets the reasons; the operator gets a line in the activity log.
