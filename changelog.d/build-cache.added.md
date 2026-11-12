`osprey up` can give every image build it runs a shared BuildKit layer cache:
`OSPREY_BUILD_CACHE_FROM` and `OSPREY_BUILD_CACHE_TO` take a `--cache-from` /
`--cache-to` spec, with `{key}` standing for the image's recipe. A host that
builds on fresh runners reuses another host's dependency layers instead of
reinstalling them.
