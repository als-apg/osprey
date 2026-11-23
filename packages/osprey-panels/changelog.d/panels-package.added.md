First version of `osprey-panels`: the panel engine that generates control-room
screens from a facility.json, moved out of the framework under the import root
`osprey_panels`. It installs and tests without the framework, depends on
`jsonschema` only, and is run as the `osprey-panels` command.
