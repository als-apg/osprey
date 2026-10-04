`osprey sim apply` and scenario activation no longer refuse with "the archive
is configured but MONGO_ROOT_PASSWORD is not in .env" in a project whose
`archiver.type` is not `mongodb_archiver`: that project reads no stored
archive, so there is nothing to rewrite and no password to ask for.
