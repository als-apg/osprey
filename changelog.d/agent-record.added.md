The agent record keeps what the agent was asked, what it answered and what it
did in one place: prompts, answers, thinking, tool calls with their arguments
and results, transcripts, safety decisions, dispatch runs, bridge turns and
feedback, each as one entry. Every deployment runs it as the `agent-record`
service, and `osprey chat`, `osprey web` and `osprey query` on the host run a
local one. It is configured in the top-level `agent_record:` block: entries
are kept for `agent_record.retention_days` (default 14), and each closed day is
exported as Parquet into `var/archive/<day>/` before that day is deleted, so a
day leaves the record only once the archive holds it. `agent_record.backend`
chooses where entries are kept; the default, `embedded`, needs no extra
service. Claude Code keeps its transcripts on the deployment's volumes for the
larger of `agent_record.retention_days` and 30 days. `osprey health` reports
when each writer was last seen and how far the export lags.
