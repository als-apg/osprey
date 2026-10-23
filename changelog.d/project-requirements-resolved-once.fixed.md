A build run with `--skip-deps` now renders the same dependency list into the
generated Dockerfile as a full build: `environment.packages` and a declared
base interpreter's frozen pins join `dependencies` in the image's install line.
`osprey facility validate` renders that list too.
