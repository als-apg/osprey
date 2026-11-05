The virtual accelerator image now installs the framework in the same layer as
the other service images and adds its `virtual-accelerator` extra on top, so a
deploy that builds the VA and the bluesky bridge installs the framework once
instead of twice: one fewer multi-minute install, and about 2.7 GB less image
data. On arm64 the image no longer compiles its own PyAT wheel, which the
shared layer already installs.
