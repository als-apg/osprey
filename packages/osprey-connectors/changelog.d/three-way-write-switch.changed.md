**Breaking change:** the login tier is three-valued (`never | asked | allowed`)
and caps the switch. The switch record stores a position per machine, `asked`
or `allowed` (absent is off); a stored `on` reads as `asked`, and an unknown
position refuses. `set_switch` takes `off`, `asked` or `allowed`. An older
package reads an `asked` tier as unreadable and refuses, so release this
package before the framework.
