The Simple view no longer stays in the working state after its live feed
reconnects. A tab that follows a turn from another tab now ends that turn when
the server says that no turn runs or that no agent runs. The tab then reads
the conversation again, so the turn shows complete. When the feed misses part
of a running turn, the tab reads the conversation again and continues the
turn. A new turn from another tab now shows as a new turn with its prompt.

A reload during a turn no longer shows that turn as finished. The turn stays
open, and the live stream continues it in the same reply. A turn that waits
for its background agents also stays open until they report.

A reloaded turn that failed with a model API error now shows the error and a
failed result. Before, it showed as a success.

When a prompt does not reach the agent, the chat now removes its message and
puts the text back in the message box. This applies to a refused prompt, a
server error and a network error. Before, the text was lost.
