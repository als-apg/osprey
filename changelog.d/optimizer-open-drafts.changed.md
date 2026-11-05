**Breaking change:** `POST /jobs/{id}/start` now requires `?config_revision=`.
Read the draft with `GET /jobs/{id}/config`, take its `X-Config-Revision`
header, and send that value; a draft written since answers `409`. The draft
routes (`POST /optimizations`, `PUT /jobs/{id}/config`, `PUT /jobs/current`)
no longer need the launch token.
