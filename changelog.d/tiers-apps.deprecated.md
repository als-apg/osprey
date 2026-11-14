The roster's old spellings still load until the next stable OSPREY release, each
with a warning that names its successor; `osprey init --revalidate` rewrites a repo to the new
spellings in place:

- `users[].persona` and `authorization.roles.<role>.persona`, read as `tier` <!-- retired-spelling -->
- `modules.web_terminals.default_persona`, read as `default_tier` <!-- retired-spelling -->
- a `personas/` folder beside `profile.yml`, read as `tiers/` <!-- retired-spelling -->
- the `modules.web_terminals.personas` catalog, ignored: each login is its own file in `tiers/` <!-- retired-spelling -->
