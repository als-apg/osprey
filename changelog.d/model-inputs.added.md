A simulation model may declare `inputs`, records another model of the facility
wires, and its engine receives their values through `take_inputs` before each
advance; the build refuses a cycle, an address no model wires, the model's own
record, a duplicate, and an engine without `take_inputs` or whose
`describe_input` refuses the input. The simulator view is now written as
`osprey.facility.simulator/4`; a `/3` view still loads, with no inputs.
