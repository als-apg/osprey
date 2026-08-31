The web terminal's control-target chip and popover now name machines by what
they are — Real machine, Rehearsal, Simulator, Demo, or per-target names set
with `control_system.target_display_names` — and speak one write-state
vocabulary (writes on / off / locked) across the chip, the popover, the
confirmation dialogs, and the agent's refusal messages. The popover shows the
machine the agent is on as a card with one write control, other machines as
rows with Turn writes on/off and Switch to, and reports reachability only when
a machine is not answering.
