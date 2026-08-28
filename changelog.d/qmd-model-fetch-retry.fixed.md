Building the qmd search sidecar no longer fails when a model download is cut
short. The three GGUF fetches now pass `curl --retry-all-errors`, so a stream
the CDN drops mid-transfer is retried like any other transient failure instead
of failing the image build outright.
