"""Ingest routes of the agent record: transcripts, OTLP and the spool.

Deliberately empty of re-exports, like the package above it: each route is
imported by the module that needs it, so importing one never pulls in the
others or a backend.
"""
