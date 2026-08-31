Per-user web terminals on one origin no longer share browser-stored UI
state — mode, theme, dock layout, rail position, drawer width, terminal
session, palette history, panel-attention acknowledgements, the one-time rail
hint, and the settings-acknowledgement flag. Each user's choices are now scoped
to their own terminal. Previously stored values are not carried over: on a
multi-user deployment each user starts from the server defaults once and their
choices persist from there.
