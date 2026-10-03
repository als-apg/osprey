`osprey build` builds the facility file once from the profile's
`data/facility/`, before the project environment and every render, and writes
it as `facility.json` at the root of every render: the deployment's own, each
persona's and each container image's. A facility error stops the build with its
one line before anything is installed or rendered.
