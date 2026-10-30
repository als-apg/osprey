Code that calls ARIEL's `hybrid_search` directly passes the search service's
search-call pool as `search_calls=`, as `semantic_search` already requires. The
ARIEL service passes it itself, so searches through the service, the MCP tools
and the web interface are unchanged.
