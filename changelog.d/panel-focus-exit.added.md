An embedded panel can hand keyboard focus back to the web terminal by posting
`{type: 'osprey-focus-exit', direction: 'forward' | 'backward'}` to its parent;
the terminal moves focus to the element before or after the panel's tile.
