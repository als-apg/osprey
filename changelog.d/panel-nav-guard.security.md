The web terminal refuses to navigate a panel anywhere but within that panel: a
path under its own mount, or an `http(s)://` URL on its own server, which opens
through the mount. An agent's `open` URL and a runtime panel registration URL
are read the way a browser's URL parser reads them (encoded `..`, `//` and `\`
included), and another path on this origin, a foreign host, or a `javascript:`,
`data:`, `vbscript:`, `blob:` or scheme-relative `//host` URL is answered 422;
the agent's `open_panel` returns the refusal as an error, and the browser
refuses the same URL again before an iframe moves. A registered panel's URL is stored
normalised; an empty or bare relative URL is now refused.
