The virtual accelerator serves the simulator view through the composite and the model runner, on Channel Access and PVAccess.
It reads the view from `/data/simulator/` (`VA_DATA_DIR` names a different data root) and requires `VA_INSTANCE` (`virtual_accelerator` or `live_standin`); a missing or unknown value refuses the boot, and the model RPC's `status` reply reports the value.
The image runs `osprey.services.virtual_accelerator.entrypoint` by name; `VA_ENTRYPOINT_MODULE` is not read.
