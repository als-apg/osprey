The MCP readiness budget is now `OSPREY_MCP_READY_TIMEOUT`, and it is
documented. It gates every run — `osprey query` exits 1 when a declared server
has not registered in time — not only the end-to-end tests its old
`OSPREY_E2E_MCP_READY_TIMEOUT` name implied. The old name is still read and
will be dropped after one release.
