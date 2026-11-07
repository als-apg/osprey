The web terminal serves every panel document an agent could have written into
an opaque origin: proxied panels and discovered `/panel-static/` bundles answer
with their own `Content-Security-Policy: sandbox allow-scripts allow-downloads`
on every response that is not JSON, a raster image, CSS, JavaScript or a font,
and an upstream policy cannot loosen it. Only a panel's declared entry pages
are exempt: the new `SidecarPanel.entry_pages` (JupyterLab's application
pages), each companion server's root page, and a config-declared panel's own
`path`. The knowledge (OKF) panel renders its markdown through DOMPurify. A
discovered panel that loads its own assets from the server, or a config-declared
panel whose pages lie outside its `path`, no longer loads them on a gated
deployment.
