Writes can be paused for one session: while a session has an unanswered hold,
every write it executes is refused with the reason `session_held`, naming what
raised the hold, and `execution_verdict` orders that refusal into the write
verdict. The connection and gateway stay as they were across hold and release.
