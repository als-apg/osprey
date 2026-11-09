The repository-boundary check now follows directory links the way the copy does, so a second link that leaves the repository is refused, and so is a link that leads back into its own tree.
