A deployment whose `.env` pins `VA_LATTICE` at a lattice the built tree does
not carry no longer starts the image builds and then fails to boot: `osprey up`
refuses up front, naming the file it looked for, `osprey build` repoints the
retired `builtin` spelling at the lattice the project's own tree serves, and
`osprey reset` discards the build-derived pointers along with the `build/` tree
they address.
