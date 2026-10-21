The Channel Finder no longer edits channel databases. The web API's write
routes (adding, editing and deleting channels, tree nodes, families and
expansions), its two impact-preview routes and `GET /api/tree/expansion` are
gone, and a request to one answers 404 or 405. The flat, template,
hierarchical and middle-layer databases lose their write methods with them.
Browsing, search, validation and the pipeline switch answer as before.
