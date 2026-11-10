A registered kind's `facts_writer` output lands in the facts document under
`kinds.<kind>`, which an application reads with `osprey.apps.sdk.read_facts`; the
key is absent when no kind writes facts.
