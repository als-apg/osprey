In the Simple view, each action of an agent turn — a thought, a tool call, a
subagent — is one row of the same shape: an icon in a left column, a short
phrase, the detail dimmed, and the result at the right edge. A hairline joins
the rows, so a turn reads as a timeline. The icon shows the action's state: a
colour of its own kind while it runs (pink for a thought, blue for a tool call,
gold for a subagent), with a soft pulse on the current action only, grey once it
settles, and the error colour when it failed. The reduced-motion setting stops
the pulse. A settled thought says how long it took, also in a replayed
conversation. The osprey no longer sits on each reasoning row; it
stays on the turn's status line and in the header. Consecutive tool calls of
one kind fold into one row that counts them and opens to the calls, with a
failure inside shown on the folded row. A call that needs the operator's
approval, and a control-system write, each stay on a row of their own, so the
decision stays in view. A refused write's reason shows under its row, or under
the folded row or subagent row that holds it, without opening anything. The
line that closes a turn now states its duration, then how many actions and
approvals the turn had, and leaves out what did not happen.
