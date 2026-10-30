`GET /api/artifacts` on the artifact gallery takes `run=true` to list only
the records of script runs (the approval review, the executor's notebook and
result record), and `run=false` to list all other artifacts. The `total` and
the page cursor follow the filter. Each artifact that the gallery returns or
sends on its event stream has a `run_record` field with this verdict.
