**Breaking change:** web terminals need a `modules.web_terminals.personas`
catalog with a `default_persona` in registry mode too, as they already did with
`image_source: local`; `osprey up`, the render and
`osprey scaffold web-terminals lint` (code `web_terminals.requires_catalog`)
refuse a config without one. A terminal now always runs in `/app/<project>`,
where `<project>` is its persona's project, and a persona whose image is built
elsewhere must state that `project` (code `web_terminals.persona_missing_project`).
A registry deployment that has no catalog adds one whose default persona names
the project its image was built from — for example `default_persona: main` with
`personas: {main: {project: <that project>}}`. That keeps each user's saved
Claude Code sessions, which are stored under the terminal's working directory.
