The archive composite closes its state directory, and the mock archiver closes
the composite it replaces or drops, instead of leaving the directory to the
garbage collector.
