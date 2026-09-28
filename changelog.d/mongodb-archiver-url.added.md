The MongoDB archiver can name its store by `url`, a MongoDB connection string,
so it reaches TLS, replica-set and x509 stores. The url wins over `host` and
`port`, may not carry a password, and is only ever read from: the recorder and
the archive rewrite write only to the store the deployment runs.
