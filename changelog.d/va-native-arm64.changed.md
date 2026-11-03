The virtual accelerator image builds natively for linux/amd64 and linux/arm64:
on arm64 it compiles the Channel Access server its serving stack needs, so
Apple Silicon no longer runs the simulator under emulation. The
`OSPREY_VA_PLATFORM` override and the `platform:` pin are gone.
