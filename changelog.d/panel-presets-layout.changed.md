A panel preset (`panel_presets` in a profile, `web.presets` in config) is the
dock layout it opens: a tile `{tabs: [<panel id>, ...], showing?, size?}` or a
split `{split: row|column, items: [...]}`, never naming the Agent pane. The
"Layouts" click and `arrange_workspace(preset=...)` apply it exactly. The
`label: [ids]` list form fails the build with the layout to write instead.
