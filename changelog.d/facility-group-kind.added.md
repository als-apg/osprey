A facility group can state its `kind`: `family` (the default) or `section`.
Only a family gives its members their purpose text in the graph and files as
a channel-finder Family; a section, such as the example's vacuum section
`SR/VS03`, keeps its description and members and is neither. The graph writes
`groupKind` on every group node, and the relations members table takes an
optional `group_kind` column.
