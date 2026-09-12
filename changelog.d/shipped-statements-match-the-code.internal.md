Shipped comments and docstrings that described something other than what the
code does now match it: a lane target is documented as one of the control-target
constant's members rather than as two of its three, and the compose templates
state what each service actually does when `CONFIG_FILE` is unset. Two hand-kept
reference tables — the host-knob page's build args and the profile `dispatch:`
key list — are pinned against their producers in the source.
