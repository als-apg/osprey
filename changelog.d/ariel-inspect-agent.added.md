ARIEL: the reader's **Enhancements** tab (``e``) shows each module's state and
output for the entry, with a copyable re-run command for a failed or stale
one, and **Related** lists up to five entries nearest by embedding. Embedded,
**Ask agent** (``a``) puts a citation of the entry into the assistant prompt
without sending it, and entries the agent creates flash in the list or count
in an **n new** pill. New: ``osprey ariel enhance --module NAME --entry ID``,
``GET /api/entries/{entry_id}/enhancements`` and
``GET /api/entries/{entry_id}/related``.
