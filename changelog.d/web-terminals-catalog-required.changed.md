**Breaking change:** web terminals need a `modules.web_terminals.renders`
catalog with a `default_tier` in registry mode too, as they already did with
`image_source: local`; `osprey up`, the render and
`osprey scaffold web-terminals lint` (code `web_terminals.requires_catalog`)
refuse a config without one. A terminal now always runs in `/app/<project>`,
where `<project>` is its tier's or app's project, and one whose image is built
elsewhere must state that `project` (code `web_terminals.render_missing_project`).
A registry deployment that has no catalog adds one whose default tier names
the project its image was built from — for example `default_tier: main` with
`renders: {main: {project: <that project>}}`. That keeps each user's saved
Claude Code sessions, which are stored under the terminal's working directory.
