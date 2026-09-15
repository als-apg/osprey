A directory removed from the workspace or from the artifacts store now drops
the file listings of everything that was under it rather than only its own, so
a long-running terminal's memory stays bounded by the tree it is watching
however finely the platform reports the removal.
