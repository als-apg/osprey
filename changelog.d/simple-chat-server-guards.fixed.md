The Simple view chat now refuses a slash command that it does not list, such
as `/login`, `/model` or `/config`. The server refuses the prompt before the
turn starts, and the chat shows that the command is not available. The chat
lists the project's skills and commands, `/clear` and `/compact`. A prompt
that starts with a path, such as `/home/x is full`, still goes to the agent.

The operator's skill row no longer shows the list of attached files as part
of the skill's arguments.

When the agent process stops between turns, the chat now shows that the agent
process ended. The next prompt starts a new agent process that continues the
same conversation. Before, every prompt failed until the operator started a
new conversation.
