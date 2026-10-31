The login tier is three-valued (`never | asked | allowed`) and caps the
switch. The switch record stores a position per machine, `asked` or `allowed`
(absent is off); a stored `on` reads as `asked`, and an unknown position
refuses. `set_switch` also takes `asked` and `allowed`, and still takes `on`,
as `asked`. An older package reads a login-tier map holding any `asked` entry
as unreadable and refuses every login, so release this package before the
framework.
