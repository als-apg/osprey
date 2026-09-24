A custom panel whose id has a dot in it, such as `beam.viewer` written as a
key under `web.panels:`, can now be selected in `web_panels`, and the build
switches it on or off in its own block. It used to write the switch under
`beam` instead, so an unselected dotted panel stayed on. A `panels.<id>` key
written inside a `web:` mapping is now refused, because it renders as a
literal key under `web` and never as a panel.
