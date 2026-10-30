A channel bridge reads a conversation's earlier turns back from the agent
record instead of keeping its own copy, so how far back a follow-up can refer
is `agent_record.retention_days` rather than a fixed 180 days. When the record
cannot answer that read in full, the bridge sends no history at all and the
payload's `conversation_record` key says why (`unavailable` or `incomplete`).
