A virtual-accelerator write now posts only the channels it changed, plus the
written setpoints, instead of all of them, so the server's posting step costs
about a third of what it did and monitor clients see one event per real
change. Every channel still refreshes on the periodic pass.
