# Deploy-scaffolding fixtures

This directory holds the inputs `tests/deployment/test_deploy_scaffold_goldens.py`
renders the CI and verify templates against. It is no longer where the byte
specification lives.

`exemplar-profile/` is a build profile for a facility running exactly three
services: the virtual accelerator, OpenObserve, and one facility-owned container
(`services/facility-mcp/`, which carries a Dockerfile and therefore earns its own
image-build job in the pipeline). It is the reference deployment the feature's
own exemplar is not — it names a registry and builds an image, so it reaches the
images stage and the registry credential that an `image_source: local` profile
never does. Its `deploy:` block is the only input the CI template reads that a
plain profile does not already provide.

`ci-extra.yml` is not a golden of anything the scaffolder re-emits: it is written
once, by `osprey init`, and belongs to the deployment from then on. It lives here
because the pipeline `include:`s it, and a preview run needs something to find.

## Where the byte specification is

It used to be `gitlab-ci.yml` and `verify.sh` beside this file, hand-built for
the retired `profile/` layout. Both are gone, and
`test_deploy_scaffold_goldens.py` asserts they stay gone. The three-zone
exemplar in `tests/fixtures/lifecycle_repo.py` is the specification now, and
`tests/cli/test_emitted_artifacts_clean.py` holds the templates to it byte for
byte. What this directory's tests assert instead is behaviour: the branches a
registry turns on, and the security properties that hold whatever the profile
says.

## Regenerating the profile fixture

`exemplar-profile/` is a trimmed materialization of the bundled
`control-assistant` preset — `data/` is empty on purpose (the fixture is
validated and parsed, never built). The exact commands and edits that produced
it are recorded outside the test tree, with the walkthrough this exemplar
became.
