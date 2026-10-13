A process forked while a guarded run was taking or dropping its lock or
opening or closing its journal could keep a copy, leaving the target busy under
the pid of a holder that was gone; forks are now ordered against those steps.
