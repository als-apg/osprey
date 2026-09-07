`ariel.ingestion.adapter` is now required. An `ingestion:` block that names no
adapter is refused when the configuration loads, with the registered adapter
names in the message, instead of defaulting to `generic` — a name no adapter
was ever registered under, so the block failed at the first ingest anyway.
