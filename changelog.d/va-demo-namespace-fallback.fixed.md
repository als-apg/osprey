A virtual accelerator deployed for a project no longer falls back to the
framework's bundled demo channel namespace when the project's own data tree
stages no channel database. The build now refuses outright, naming what is
missing from the data tree, so an operator never drives a simulator quietly
serving the framework's tutorial channels instead of the facility's own. A
channel database that is present but cannot be read is reported by name at
build time and contributes no channels; the build refuses only when no
readable database remains.
