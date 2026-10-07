The control-target switch refuses a `va` or `standin` target whose selected
gateway has the same address and port as one of the `live` target's gateways.
The roster reports it as ineligible with reason `reaches_live_machine` ("points
at live machine"). A deployment that pointed its virtual accelerator or stand-in
at a live gateway can no longer switch to it.
