`osprey knowledge build-ttl` now orders the corpus by the section order the
channel database's own tree lists, instead of by a built-in `SR, BR, BTS` ring
order that only the demo machine has. `--section-order` names the order
explicitly for a database whose key order carries no meaning; a section neither
source names sorts after the ones they do, alphabetically.

The facility token every IRI and identifier embeds now defaults to the
project's own `facility.prefix` rather than to `demo`, so a project that
already names its facility does not name it twice. Each run reports the token
it minted with and the ontology table it emitted against, and warns when a
corpus built from a database that is not the packaged demo one still carries
`demo`. **Behaviour change:** a project with `facility.prefix` set that
regenerated its corpus on the old default mints different IRIs on the next
`build-ttl` — pass `--facility demo` to keep the old ones.
