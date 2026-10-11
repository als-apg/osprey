A measurement file in which two group roles name the same group stops the build
as `value-invalid`, naming both roles, unless the two roles are `hcor` and
`vcor`: those may share a group, which is split by plane. The pyAML name mapping
refuses the same shape.

A profile whose `approval.enabled` is stated as anything but `true` or `false` (an empty
line, a quoted string, a number) stops the build as `profile-invalid` on
`approval.enabled`: read by truthiness, such a value turned every approval prompt off
without a word. An unstated switch still reads as asking.
