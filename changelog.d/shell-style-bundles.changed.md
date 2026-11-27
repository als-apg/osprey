**The Web Terminal's shell page fetches its own style sheets in two requests
instead of twenty-three.** The sheets the shell owns are served concatenated
at fingerprinted, immutable addresses (`/static/_v/<fingerprint>/_bundle/base.css`
ahead of the vendored sheets, `…/shell.css` behind them), in the order the
page used to link them, compressed and kept by the browser until a file
changes. Each sheet is still served at its own address for the pages that
link it singly. A sheet is added to the shell in `SHELL_STYLE_BUNDLES`
(`web_terminal/app.py`), not as a `<link>` in `index.html`.
