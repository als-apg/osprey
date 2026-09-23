`osprey.utils.identity` is now the same module object as
`osprey_connectors.identity`, as every other path that moved into the
connectors package already is, so patching a name through either path reaches
the identity ladder.
