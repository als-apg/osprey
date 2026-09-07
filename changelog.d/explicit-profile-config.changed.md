`profile.yml` is the complete declarative statement of a deployment: every
preset carries the full `config:` block it deploys. The provider catalog moves
to a `providers.yml` beside the profile, which `osprey build` renders into
`api.providers`.

`osprey profile expand` fills in a profile written against the old layout: it
writes every key its preset documents and the profile lacks, stamps
`provenance:`, and reports the differences that `osprey validate` refuses from
then on. Drift checking now splits by kind — a structural difference is
refused unless a `# DEVIATION:` comment claims it, a value difference is only
reported. Six keys with no safe unstated answer, among them
`control_system.type` and `approval.default_policy`, are refused by
`osprey build` when the rendered config leaves them silent.
