Simulation: `osprey_connectors.simulation.sessions` parses a scenario's `sessions`
block (past agent sessions, their tool calls, answers and artifacts) into frozen
records, refusing a malformed block, a session less than a day back, one that runs
past midnight, or a `controls` tool step naming a channel its turn does not list;
it places every turn at an aware instant against an anchor and round-trips the
result through JSON.
