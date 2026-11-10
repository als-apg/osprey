The ARIEL contract page's `/api/capabilities` example is the whole response
the service returns: it now shows each mode's own parameters (the keyword
mode's `include_highlights`, the semantic mode's `similarity_threshold`), every
shared parameter with its labels, and the `default_mode`, `vocabulary` and
`attachments` blocks. The search response example carries every field the
API sends, including the required `created_at` and `updated_at`, and names one
search mode, as a search reports. The entry-field error envelopes show the
detail text the service writes.
