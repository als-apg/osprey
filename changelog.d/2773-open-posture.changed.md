**Breaking change:** `osprey up` now refuses an open deployment
(`modules.web_terminals.auth.method: none`) whose render shows the notebook
panel, and `osprey scaffold web-terminals lint` reports it as
`web_terminals.open_mode_code_panel`. A kernel runs what is typed into a cell
with no guard in front of it, so behind a proxy that vouches for every request
it could open any login's terminal. Put a login wall in front of the terminals
(`token`, `password` or `oidc`), or stop selecting `jupyter` in the renders
the message names and rebuild.
