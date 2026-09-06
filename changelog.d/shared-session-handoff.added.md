Expert and Simple views are now two windows onto one session: switching view
hands the conversation over instead of starting a second one, with the same
transcript, write state and control target. The view you switch to waits for
the outgoing agent to finish its current turn and offers **Stop and switch
now**; deployments scaffolded before this release need the `turn-state` hook in
the profile's `hooks:` list, then a rebuild with `osprey build`.
