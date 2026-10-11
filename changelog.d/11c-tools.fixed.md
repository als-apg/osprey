A measurement file in which two group roles name the same group stops the build
as `value-invalid`, naming both roles, unless the two roles are `hcor` and
`vcor`: those may share a group, which is split by plane. The pyAML name mapping
refuses the same shape.
