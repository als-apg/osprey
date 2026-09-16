`osprey up` without `-d` no longer leaves a copy of the site CA bundle under
`build/services/<name>/` once the images that needed it are built. Both start
shapes now clear it, as the project, persona and auth-sidecar builds already
do.
