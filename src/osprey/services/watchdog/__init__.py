"""Safety watchdogs: independent judges of what an agent session is doing.

A watchdog reads a session and answers a verdict; it never acts on the machine
itself. The ``watchdogs:`` profile block and its definitions live in
:mod:`osprey.services.watchdog.config`. Submodules are imported on demand, so
validating a profile costs only the block's own parser.
"""
