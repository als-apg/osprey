"""The agent record — one record of every session fact, one writer, one reader.

Deliberately empty of re-exports. Consumers import the module they need
(``osprey.agent_record.contract`` for the entry shape and the backend
protocol, ``osprey.agent_record.identities`` for the source identities and
the credential table) so that importing the package costs nothing: the
contract is a stdlib-only leaf that writers, ingest, backends and readers all
depend on, and a package ``__init__`` that pulled a backend in would hand
every one of them the backend's imports as well.
"""
