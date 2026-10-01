A failed dispatch run's recorded stderr no longer opens with the agent CLI's
notice that it ignored the project's allow rules. Unattended runs keep those
rules off on purpose; the trigger's allowed tools and the policy hook decide
what a run may do.
