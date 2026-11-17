The web terminal loads faster and in one step. Its scripts, styles and fonts
are served gzipped under a content fingerprint, so a browser keeps them until
a deploy changes them and a reload fetches almost nothing; the page names its
whole module graph up front instead of discovering it one import level per
round trip. While it loads, the sidebar and a "Loading workspace" veil stand in
for the workspace, which then appears arranged as it will stay. The veil's bar
fills as the page's files arrive; the veil says when the connection is slow
and offers a reload when a file cannot be fetched.
