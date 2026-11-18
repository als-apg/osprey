**Knowledge bundles hold images.** A concept embeds a PNG, JPEG, GIF, WebP or
SVG stored in the bundle with `![alt](path)`, a reference-style image or
`<img>`, and the KNOWLEDGE panel shows it. `read_concept` lists the images a
concept embeds, and the new `view_asset` tool lets the agent look at one (SVG
excepted). `osprey knowledge validate` warns about an image that is missing,
of a disallowed type, over 10 MB or without alt text.
