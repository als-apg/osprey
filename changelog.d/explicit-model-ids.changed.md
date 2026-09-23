Configuration names models by the id the gateway serves. The haiku / sonnet /
opus tier words are gone from profiles, presets and the provider catalog; each
catalog entry lists the models its gateway serves and names its default.
Claude Code's own three alias names are filled at build from that list (or
from a gateway's optional `claude_code_aliases`). `claude_code.models` is now
`claude_code.aliases`; `logbook.composition.default_tier` is now
`logbook.composition.model`. Model names shown to people drop the vendor
prefix: Sonnet 5, Haiku 4.5, Fable 5.1.
A deployment on anthropic that names no model now runs Sonnet 5 where it ran
Haiku 4.5. A deployment on als-apg that names no model now runs Sonnet 5 too,
and its health check still probes Haiku 4.5.
