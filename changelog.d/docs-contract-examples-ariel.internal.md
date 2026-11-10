The ARIEL contract page's JSON examples are files under
`reference/contracts/examples/`, and `tests/docs/test_contract_examples_ariel.py`
reads each one with the code that writes it: the entry-field error envelopes
are the route's own bodies, the health, search and status responses round-trip
through their pydantic models, the capabilities document is what
`get_capabilities()` returns, and the attachment summary field list is
`SUMMARY_KEYS`.
