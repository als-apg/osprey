An approval prompt's answer is now in the audit trail: a `hook_approval` entry
records `approved` when the call ran and `denied` when the turn ended without
it, on the same `tool_use_id` as the prompt, and a control-target switch
records the target it left and the one it asked for.
