Application SDK: `osprey.apps.sdk.bridges_for_target(target)` names the
Bluesky bridges a deployment runs for a control target, `audit_surface()` the
word an application's audit records file under (its name), and
`canonical_json` / `input_digest` are the digest recipe an approval binds to.
`serve_tools` takes an `on_shutdown` hook that runs once when the tool server
stops.
An application's `bind.state_fields` key may be a dotted path into nested
state (`step.digest`).
