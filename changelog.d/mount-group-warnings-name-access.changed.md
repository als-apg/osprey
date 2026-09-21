A container that cannot join a bind mount's group now warns that the mount
will be unreachable rather than unwritable. The same step joins the read-only
control-state tree, which nothing in the container writes, so the old wording
sent operators looking for a writer that does not exist.
