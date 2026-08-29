`deployment.port_base` moves a whole deployment onto another block of a
thousand host ports in one step — set it and rebuild — so two deployments
coexist on one host by taking two bases. `dispatch.worker_port_stride`
(default `1`) spaces host-network dispatch workers inside their band. The
docs gain a ports reference page rendered from the layout table itself, and
the test suite a lint that refuses a retired framework port literal anywhere
in `src/`, `packages/` or `docs/`.
