The framework release workflow no longer grants write scopes to every job:
only the PyPI upload mints an OIDC token and only the GitHub-release job
writes contents, as in the connectors release workflow. Both release
workflows refuse a tag that `main` cannot reach, the framework release
refuses a stable tag whose `osprey-connectors` dependency resolved to a
pre-release, and the framework jobs are guarded on a tag ref rather than a
ref name a branch could carry.
