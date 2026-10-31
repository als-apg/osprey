`osprey archive` copies the web terminals' and the dispatch worker's artifact
stores (files and index) and the plan-queue history into an append-only tree
with a sha256 manifest, beside the agent record's own daily export. It never
deletes or rewrites a copy.
