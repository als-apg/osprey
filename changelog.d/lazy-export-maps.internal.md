The packages that resolve their public names on first attribute access now all
declare that resolution as one mapping from name to the module that answers for
it, so a name and its module cannot drift apart unnoticed.
