The channel-finder benchmark's `sdk` backend records each tool call's result,
waits for the channel-finder server to connect before the first turn, and
routes OpenAI-protocol providers through the translation proxy. Scores are
still computed from the agent's answer text alone.
