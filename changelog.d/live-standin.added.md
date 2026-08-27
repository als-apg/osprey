`virtual_accelerator.live_standin: <port>` in a build profile deploys a second
copy of the simulator as the deployment's `live` target, so `control_target_set
live` rehearses the whole go-live path — probe, acknowledgment, strict limits,
approval prompts — with no config edits. The banner and posture badge read
`LIVE MACHINE (stand-in)` throughout, and the `control-assistant` preset ships
it on at port 5074.
