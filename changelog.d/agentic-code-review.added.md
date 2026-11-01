A deployment can turn on code review: a second AI agent reads the Python the
OSPREY agent writes, on Python runs and on notebook cells, following
instructions the deployment writes in `data/code_review/reviewer.md`, and
either stops unsafe code or only reports it (`code_review.level`). On a Python
run the review comes before the approval prompt: code the reviewer blocks never
reaches the operator, and otherwise the verdict and reasons are on the prompt
and in the approval preview. The agent gets the reasons; the operator gets a
line in the activity log.
