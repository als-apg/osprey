An image build no longer fails when the registry behind `OSPREY_BUILD_CACHE_FROM`
fails to serve a layer the cache lists: `osprey up` logs a warning and builds that
image again once without the cache source, keeping `OSPREY_BUILD_CACHE_TO`. Any
other build failure is reported as before.
