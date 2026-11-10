---
name: scaffold-app
description: >
  Read and set the scaffold-app channel with the scaffold-app tools:
  read_channel shows what the panel shows, set_channel writes behind the
  operator's approval. Use when asked what the scaffold-app channel reads or
  to set it to a value.
---

# Scaffold app

The panel and these tools share one backend: what `read_channel` answers is
what the panel shows.

| Tool | Panel equivalent | Writes to the machine |
|---|---|---|
| `read_channel()` | the channel, its readback and the machine | no |
| `set_channel(value)` | — (the scaffold's panel only shows) | yes, behind approval |

Read first, tell the operator the current value, then call `set_channel`
with the value they asked for; leave the `approved_*` arguments unset. The
approval card shows the channel, the new value and the readback it was
approved over. A refusal says why: writes off for the machine, a login that
may not write, or a readback that moved since the approval — read again and
ask for a new approval. Do not look for another way to write.

Keep this table complete as the app grows: every value the panel shows is in
a read tool's answer, and every control it offers is a tool or an argument.
