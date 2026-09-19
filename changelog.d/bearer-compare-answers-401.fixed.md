A webhook trigger or a dispatch-worker call whose `Authorization` header carries
a character outside ASCII now answers 401 rather than 500. Both gates weigh the
bearer as bytes, as the dispatcher's own routes already did, so a wrong
credential is refused on every surface.
