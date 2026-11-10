On a deployment whose write tier is `never`, a chat tool call that is not a
write (Read, Glob, a read-only MCP tool) now raises the approval card instead
of being refused with "This deployment does not arm writes".
