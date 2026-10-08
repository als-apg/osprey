A picture backfill run from the command line no longer ends with
`RuntimeError: Event loop is closed` on Python 3.11 and 3.12. The picture render
worker is now closed and reaped before its event loop shuts down, including when
a render is cancelled.
