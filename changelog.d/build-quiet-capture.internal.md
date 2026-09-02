The quiet-build log tests capture only records from the build's own thread,
so another test's leftover background warning on a shared CI worker no longer
fails them.
