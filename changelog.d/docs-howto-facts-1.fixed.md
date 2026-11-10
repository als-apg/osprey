Several how-to pages now state what the code does. The code-review page puts
the `code_review` section under `config:` (a top-level `code_review:` is refused
as an unknown profile key); the virtual accelerator page names the archive
store's host port as 10801 at the default port base, not the in-container
27017; the add-MCP-server note says an `mcp_servers:` entry takes `hooks:` and
`writes:`, and the read-only query page says a server's `writes:` list is
blocked too; the Bluesky queue route table lists the move and export routes;
the channel-finder page counts four pipelines, spells ReAct, and names preset
channel addresses. A new test pins these and other quoted facts (the retired
`mock` refusal, the mya client floor, the Phoebus `curl` hint and the
`osprey query --json` keys) to the source that defines them.
