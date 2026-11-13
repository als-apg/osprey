`apps.<name>.services.<key>.mounts` binds host directories or files read-only
into an app's `runtime: osprey` container, for data and binaries the project
image does not carry (a lattice repository on a network share, a physics
code's install).
