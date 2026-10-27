The web terminal gains a supervisor for a child process kept alive behind a
listening socket its parent holds: it reads time through an injected clock, and
every exit that is not a refusal counts toward its give-up window, so a child
that keeps exiting at once is retracted instead of restarted forever. Nothing
starts it yet.
