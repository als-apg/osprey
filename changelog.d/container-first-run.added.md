Containers now open their first terminal session on the control-room prompt
instead of the agent CLI's own setup: the entrypoint seeds the missing
first-run state (onboarding, workspace trust, and — under a raw-key
provider — API-key approval) into the session volume at start, merge-only,
so a returning operator's choices are never rewritten. Seeding trust also
means the rendered permission allow-list applies from the very first
session rather than waiting on a dialog. A deployment whose `web.theme`
pins a mode (a concrete theme id such as `desy-light`) now renders the
matching terminal theme too, so one config key governs both surfaces.
