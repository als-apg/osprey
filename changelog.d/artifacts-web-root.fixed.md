`osprey artifacts web` no longer fails at startup with a `TypeError`. The
gallery serves the deployment's shared agent-data directory when no root is passed.
