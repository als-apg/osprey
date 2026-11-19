`archiver_read` answers an archive that gave no answer with the envelope its
cause maps to (`connection_error`, `timeout_error`, or `internal_error`), its
message built from the archive's reason; the "connector was dropped" recovery
sentence is gone, because the archive reopens itself. Its coverage note for an
archive that gave no answer names the cause (`coverage probe gave no answer:
unreachable`) instead of an exception class. A read of the mock archive
for a channel outside the simulator view answers no rows, with coverage verdict
`never_recorded`, instead of the `not_found` error. The PV page's archive
section serves `description: null` for a channel the MongoDB store records, and
answers `unavailable` with the store's reason when the store cannot be read.
The `archiver_freshness` health check probes the archive of the first target
the deployment's archive serves; its error details are the archive's reason.
