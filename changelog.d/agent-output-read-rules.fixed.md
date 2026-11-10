The agent can read its own artifacts and execution data without an approval
card again: the Read rules over the agent-data root now name its `artifacts/`
and `data/` directories by absolute path. The rest of the root still asks.
The Edit rules over its `artifacts/` and `notebooks/` trees are named the same
way, so its notebook edits there no longer ask either.
