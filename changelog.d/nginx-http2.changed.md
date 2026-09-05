Deployments with TLS enabled now serve the web terminal over HTTP/2. One
multiplexed connection replaces the browser's six-per-host HTTP/1.1 limit, so
open panels and their event streams no longer compete with ordinary requests.
Plain-HTTP deployments are unchanged.
