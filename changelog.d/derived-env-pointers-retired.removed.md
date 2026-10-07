`osprey up` no longer checks `.env` for the virtual accelerator's build-derived
channels-file and lattice pointers, and `osprey reset` no longer strips or
counts a build-derived `.env` section: reset strips only the minted blocks
OSPREY wrote, and a line under no OSPREY banner is left alone.
