ARIEL logbook ingestion honors the `ariel.ingestion` transport settings on every
adapter. The generic-JSON, JLab and ORNL fetches previously ignored them, so on
a proxied deployment their traffic went direct and a site CA named in the
config was not used. An `ariel.ingestion.verify_ssl:` key left without a value
no longer turns verification off; the default stays on unless the key says
`false`.
