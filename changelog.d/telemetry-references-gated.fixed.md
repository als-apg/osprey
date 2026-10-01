A web-terminal deploy now refuses when any `${VAR}` without a default of its
own anywhere in the `claude_code.telemetry` block is unset in the project's
`.env`, not only one in the OpenObserve password, and names the key that asks
for it. None of those values is copied into `.env.users`.
