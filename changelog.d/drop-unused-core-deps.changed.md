The framework no longer installs `anthropic`, `ollama`, `nltk` or `unique-namer`,
which nothing in OSPREY imports; code of your own that uses one of them now has to
install it. `nbclient`, `tenacity`, `websocket-client` and `python-dateutil` are no
longer declared either but still arrive through other dependencies. `starlette`,
`anyio` and `packaging`, which OSPREY imports directly, are now declared.
