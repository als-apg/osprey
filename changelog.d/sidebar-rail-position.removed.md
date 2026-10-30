**Breaking change:** A labelled sidebar replaces the web terminal's panel rail
and header in both views: **Workspace** (Agent, Artifacts, Script runs,
Events), **Tools** and a footer with Docs, Feedback and **☰ More**, whose
labels ⌘\ (Ctrl+\ elsewhere) hides. The rail's top position goes with it:

- `web.rail_position` is no longer read, and the retro theme uses the sidebar.
- `GET /api/panels` no longer returns `rail_position`,
  `rail_position_configured` or `family_rail_defaults`.
- The onboarding tour walks the sidebar, and invites every user once more.
