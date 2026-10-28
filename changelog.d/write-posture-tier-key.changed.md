**Breaking change:** whether a login may write is now one key,
`control_system.writes: allowed | never`, set in a tier file, the deployment
root or an app (apps always `never`); a tier overrides the root and silence is
`never`. The old `writes_enabled` keys are rewritten at build with one warning
each (a literal `true` reads as `allowed`, anything else as `never`); a file
spelling both is refused. `osprey init --revalidate` rewrites the files. A
render built before this release reads as `never` until rebuilt.
