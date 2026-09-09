`osprey status --agents` now lists every agent the deployment can run, including
`facility-knowledge` and `facility-knowledge-graph`, which were missing from the
table. Both also gain an explicit default model tier instead of falling through
to `sonnet` unstated.
