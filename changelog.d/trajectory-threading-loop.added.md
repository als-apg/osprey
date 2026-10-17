First-turn threading runs: the trajectory service takes a shot (arm the BPMs,
fire one injection through the facility's sequence module, read the first
turn), computes the corrector step, and applies it by its digest or puts
recorded values back, listing every setting it moved with its value before.
A Stop is answered at once and ends the action within its stop budget; the
panel door adds setup, continuous shots and references. The agent drives the
loop with the `trajectory_shot`, `trajectory_step` and `trajectory_apply`
tools, and every shot and apply asks for approval bound to the service's
target and the exact step shown.
