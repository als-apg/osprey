Trajectory service: a profile can carry a `trajectory:` block, and `osprey up`
then runs a trajectory service that claims one control target for first-turn
threading through a session. Its panel door and agent door both reach it
through the web terminal and name the terminal's account as the session's
owner; every writing action is checked against that owner's write state and
narrowing and takes the target's guarded-run lock, and each refusal answers
with a code. Sessions open, take over and release.
