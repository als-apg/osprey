The executor's readonly import check and the new kernel-side block, which
refuses a control-system client at import and a Channel Access or PVAccess
client library at load, read one list of client packages. A test fails when a
connector imports a client the list does not name, or when a client package
ships a shared library the block's library list does not match.
