`osprey up` refuses a deployment whose `.env` has lost a key the build wrote.
The build writes `VA_CHANNELS_FILE` and `VA_LATTICE` together whenever it
generates a channel manifest, so a manifest on disk beside a chain missing
either is a lost key, and the virtual accelerator would exit on the empty
value. The preflight names the key and how the build writes it back. The
archiver seed no longer falls back to the framework's bundled channel set
when nothing names a manifest.
