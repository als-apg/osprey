Re-reading the directory listing a file watcher diffs has one definition,
shared by the workspace watcher and the artifacts store-index watcher, so the
rule that a directory found already gone keeps no listing is stated once. The
stamp the store-index debounce compares is the same shape a directory listing
holds, rather than a second one describing the same file.
