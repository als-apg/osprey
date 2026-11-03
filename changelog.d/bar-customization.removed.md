**Breaking change:** The web terminal's customizable header and status bars are
removed, together with Customize, the `GET`/`PUT`/`DELETE /api/bar-items`
route, the per-operator saved bar layouts under `bar_items/` in the agent-data
directory, and the `web.bar_items` configuration key. The sidebar takes over
the bars' controls; the Bluesky queue chip and the stopwatch are dropped, and a
saved bar layout is not migrated. The command palette no longer offers
`Customize bars` or `Move panel rail to top/left`.
