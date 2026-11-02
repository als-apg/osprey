The artifact gallery serves every content page (artifact files, rendered
notebooks and markdown) with `Content-Security-Policy: sandbox allow-scripts
allow-downloads`, so agent-authored HTML runs in an opaque origin without the
operator's session; only the gallery shell is exempt. The pages it builds are
self-contained (offline, the vendor libraries are inlined), rendered markdown
goes through DOMPurify, preview iframes lose `allow-same-origin`, HTML and
notebook cards show a summary instead of a live page, and printing such an
artifact opens it in a new tab for the browser's own print.
