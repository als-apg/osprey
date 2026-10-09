Image builds install the framework with uv. The deps layer of the project,
persona and service images resolves and downloads with uv instead of pip,
which is much faster on a slow package index; uv is installed from the same
index the site configures, honours the same constraints and index settings,
and is removed again before the layer is committed.
