Optimizer drafts no longer need the launch token: `POST /optimizations`,
`PUT /jobs/{id}/config` and `PUT /jobs/current` are open, so an agent can
prepare a run for an operator to review. `POST /jobs/{id}/start` now requires
`config_revision`, the revision `GET /jobs/{id}/config` returns in its
`X-Config-Revision` header, and answers `409` when the draft was written after
it was shown.
