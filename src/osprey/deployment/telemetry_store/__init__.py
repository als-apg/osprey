"""The telemetry store: its schema, its accounts, and how ``osprey up`` applies them.

:mod:`.ddl` renders every table and reconciles a running store against the
render; :mod:`.accounts` renders the store's own accounts, profiles, quotas and
workload; :mod:`.render` builds the context the store's compose and server
config templates render from; :mod:`.stage` runs all of it against the store
container during ``osprey up``.
"""
