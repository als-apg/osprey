`osprey.apps.sdk.testing`: an application's own test suite builds a synthetic
deployment with a real `osprey build` — its facility tree, its applications and
one login per write ceiling — and serves the backend in process on one login's
render. Installing the framework registers the pytest fixtures
`synthetic_deployment` and `rendered_tree`.
