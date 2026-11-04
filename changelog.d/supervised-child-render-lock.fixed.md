A supervised child's argv and environment are rendered outside the supervisor's lock, so a renderer that reads the supervisor or stops it no longer deadlocks the terminal.
