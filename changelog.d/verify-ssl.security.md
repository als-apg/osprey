ARIEL logbook ingestion now verifies TLS certificates by default. A deployment
that ingests from a logbook with a certificate the image cannot verify must
either name its site CA with the new `ariel.ingestion.ca_bundle` key or opt out
explicitly with `ariel.ingestion.verify_ssl: false`; the previous default
silently accepted any certificate.
