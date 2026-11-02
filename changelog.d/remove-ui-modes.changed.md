**Breaking change:** the web terminal's Expert and Simple views are gone. The
Agent pane is the operator chat for everyone, and the terminal is an opt-in per
tab from the Agent bar's ⋯ menu. `web.ui_mode` is no longer read: a build warns
and drops it. `web.expert_view` becomes `web.terminal`, and a build refuses the
old key.
