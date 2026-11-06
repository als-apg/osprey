The "Describe your facility" page's examples now work on the control-assistant
preset it describes. The fixes example and the seed examples named channels
the preset does not have and stopped the build; they now use preset channels.
The channel-list CSV now gives its device a position, so the import writes the
device the channels name instead of leaving a `reference-missing` build. The
page no longer says an MML import never rewrites an existing `classes.yaml`
(it appends the classes it lacks), and it lists every top-level mapping block
and family key, including `branches`. The limits and measurement examples are
the preset's own files, included rather than copied. The MML import page marks
its two YAML blocks as fragments of `mapping.yaml` and documents `branches` and
`models.<name>.tune`. A docs test pastes each example into a copy of the preset
and runs the facility validator on it.
