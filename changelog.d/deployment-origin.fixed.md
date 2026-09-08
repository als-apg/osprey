Browser-facing links now follow the deployment's own origin — the dispatcher's
telemetry link, the login URL `osprey web` prints, the CI environment URL — and
are omitted rather than guessed where no origin is derivable. Assumptions about
the machine are checked instead of asserted: the container runtime, container
paths, the home directory, Docker Desktop, and the browser and OS the terminal
is open in.
