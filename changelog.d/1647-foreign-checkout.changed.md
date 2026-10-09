`osprey up` and `osprey restart` refuse to start while containers of the same
project name, created from another copy of the repo, are on the host. Nothing
is stopped. The refusal names that copy's path, the `osprey down --repo`
command that stops it, and the `profiles/<variant>.yml` overlay that gives this
copy a project name of its own. Volumes no longer carry the
`com.osprey.repo-id` label; they belong to the project by name. The first
`osprey up` after upgrading keeps every existing volume, and `osprey reset`
now removes all of the project's volumes once no other copy's container holds
the name.
