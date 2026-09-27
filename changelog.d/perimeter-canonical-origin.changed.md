`deploy.fqdn` and `modules.web_terminals.external_origin` must name a DNS host
name or IPv4 address, and a value nginx could not use as a server name is
refused at render.

nginx now serves a multi-user deployment only on its external origin's host and
answers any other name (another DNS name, the bare IP, `localhost`) with a
`301` to the same path on the origin. Before, such a page loaded and every
action on it was refused.
