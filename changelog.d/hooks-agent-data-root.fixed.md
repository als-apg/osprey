Hooks now follow a relocated `agent_data.base_dir`: the prompt's gallery focus block is read from and
cleaned under the configured agent-data root instead of the default one.
Channel-finder captures land under the configured agent-data root, and the review app reads its
pending reviews from the same place.
When no agent-data root is stamped into the session, the write-posture reader looks for the
control-target record under the configured root too.
