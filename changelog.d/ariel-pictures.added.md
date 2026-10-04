ARIEL can now search logbook entries by their pictures and show a stored picture
to the agent. With `ariel.attachments.copy_on_ingest: images`, ingest copies each
entry's pictures into the database; the `image_caption` enhancement module asks a
vision model to describe each copied picture and list its visible text, and folds
that caption into the entry's searchable text, so keyword, semantic and hybrid
search find an entry by what its plots show. The `image_embedding` module embeds
each copied picture with a multimodal model served by a site-run `llama-server`
(the new `llama-cpp` provider), and `hybrid` search also ranks pictures against
the query text. The agent's `attachment_view` tool returns one stored picture.
The `control-assistant` and `ariel-standalone` presets turn all three on; each
picture module runs when its server and model answer and is otherwise skipped,
with `osprey ariel status` naming the skipped module and why.
A simulation scenario's logbook entries can carry pictures (`attachments`), and
three of the control-assistant demo entries now do.

Upgrade notes: an existing `profile.yml` does not gain the new keys, so the
picture modules stay off until they are added (`osprey validate` lists them as
drift; `osprey profile expand` writes them). For generic file sources, sidecar
metadata is now fetched only from relative paths under the entry's
`source_url` directory, and an ingest without an adapter fetches no sidecar.
The upgrade builds new full-text and trigram indexes and folds captions already
present upstream into entries' searchable text, which marks those entries' text
embeddings as owed again; see the picture-search guide for counting and draining
that backlog.
