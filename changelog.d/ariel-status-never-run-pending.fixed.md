`osprey ariel status` reported `pending: 0` for an enhancement module that had
never run, so a freshly seeded logbook looked fully enhanced until the first pass
wrote a status key. A module the store has never seen now reads every entry as
pending, the same count a module that has processed one entry reports as the
rest.
