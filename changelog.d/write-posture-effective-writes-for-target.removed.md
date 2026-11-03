**Breaking change:** `osprey.mcp_server.control_system.target_eligibility` no
longer has `effective_writes_for_target` or `narrowing_refusal`. Use
`target_write_verdict(section, target).allowed`.
