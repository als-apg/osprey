# Deploy-scaffolding goldens

`gitlab-ci.yml` and `verify.sh` are the hand-built reference deployment for a
facility repo. They were written first, by hand, and validated mechanically
(`npx --yes gitlab-ci-local --file gitlab-ci.yml --preview`) before any template
existed. **They are the specification**: the templates under
`src/osprey/templates/deploy/` must reproduce them byte for byte.

`exemplar-profile/` is the profile those two files were emitted for — a facility
running exactly three services: the virtual accelerator, OpenObserve, and one
facility-owned container (`services/facility-mcp/`, which carries a Dockerfile
and therefore earns its own image-build job in the pipeline). Its `deploy:`
block is the only input the CI template reads that a plain profile does not
already provide.

## Comparing against a render

One value in each golden is not reproducible: the `osprey-version:` provenance
line, which is derived from the git tag and moves with every release. It is
frozen here as the literal `OSPREY_VERSION`, so normalize the rendered file's
version token to that string before diffing. Everything else is exact —
whitespace, comment wording, and key order included.

`ci-extra.yml` is not a golden of anything the scaffolder re-emits: it is
written once, by `osprey profile new`, and belongs to the facility from then on.
It lives here because the pipeline `include:`s it, and a preview run needs
something to find.

## Regenerating the profile fixture

`exemplar-profile/` is a trimmed materialization of the bundled
`control-assistant` preset — `data/` is empty on purpose (the fixture is
validated and parsed, never built). The exact commands and edits that produced
it are recorded outside the test tree, with the walkthrough this exemplar
became.
