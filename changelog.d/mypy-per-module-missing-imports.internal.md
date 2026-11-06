The type check no longer treats every unresolvable import as `Any`. Each package
that ships no types, or is installed only with an optional platform, is named in
its own override; a new one is an error until it is listed. The deployed hook
scripts and the Teams relay resolve their sibling imports, so a call from one
hook into another is checked, and `psutil` is checked against `types-psutil`.
