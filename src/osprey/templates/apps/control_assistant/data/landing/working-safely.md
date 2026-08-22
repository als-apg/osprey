# Before you start: working safely with the agent

OSPREY connects an AI agent to **live control systems**. Every action it proposes
can affect real hardware. You are the operator and the agent is a tool: you are
responsible for every action it takes.

This file is yours to edit. It is listed under `modules.web_terminals.landing.notices`
in `config.yml`, and every file listed there becomes one section on the landing
page. Add your own for local procedures, contacts, or anything else operators
should read before they start.

## Be mindful of what you approve

OSPREY asks for your approval before any hardware write. That gate is where you
decide, and it is worth the moment it takes.

- Read the proposed command before approving it: the target channels, the values
  and the units.
- If something looks unfamiliar or unexpected, find out why it was proposed
  before you approve it.
- Go through a batch item by item. Approving a batch as one thing is how an
  unread command slips through.

## Follow what the agent is doing

The agent works on its own between approval gates. Following along is how you
learn the workflow, the tool, and where its information came from, which is what
lets you judge the next thing it proposes.

- Read its reasoning, not only the action at the end of it.
- Notice which channels and sources it drew on.
- Stop the session if it starts looping or heads somewhere you did not intend.

## Agents are built to find you an answer

These agents are biased toward producing a solution for whatever you ask them
for. Ask for something this deployment cannot actually do and you are more likely
to get an answer-shaped attempt than a refusal.

- Know which subsystems and channels this deployment is configured for.
- Narrow open-ended asks ("fix everything", "optimize the beam") yourself before
  handing them over.
- Treat a confident answer about something outside that scope as a reason to
  check, not as a result.

When in doubt, stop the session and talk to your control system team.
