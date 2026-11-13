A registered kind's `facts_writer` output lands in the facts document as `facts`
in each model's kind block, `measurement_models.<model>.kinds.<kind>`, which an
application reads with `osprey.apps.sdk.read_facts`; two models may give one kind
different blocks.
