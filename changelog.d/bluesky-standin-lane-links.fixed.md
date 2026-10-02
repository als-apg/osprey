A deployment that declares a `bluesky_standin` plan lane now hands the
Bluesky panel sidecar that lane's bridge URL and launch token, starts the
sidecar after that lane's bridge is healthy, and hands the dispatch worker
the lane's bridge URL, so the panel and the worker's agent can reach the
stand-in lane like any other.
