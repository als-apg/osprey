The Channel Finder web API no longer edits channel databases. Its write routes
(adding, editing and deleting channels, tree nodes, families and expansions),
the two impact-preview routes and `GET /api/tree/expansion` are gone, and a
request to one answers 404 or 405. Browsing, search, validation and the
pipeline switch answer as before.
