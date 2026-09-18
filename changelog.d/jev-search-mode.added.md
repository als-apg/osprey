New opt-in ARIEL search mode, ``jev``: PostgreSQL full-text search fetches a
candidate pool and `Jev <https://docs.typesafe.ai/>`__, TypeSafe's System One
decision model, reviews the ranking in one batched request — a relevance
question per candidate plus two about the query itself. It is the search bar's
alternative for queries whose wording differs from the logbook's, and the only
mode that ships with search-as-you-type: the panel runs it on every keystroke,
debounced, with slower responses superseded by newer ones.

The mode is registered but **disabled**, and stays unreachable until a
deployment writes ``search_modules.jev.enabled: true`` and exports
``TYPESAFE_API_KEY``. It is the one mode that sends entry text to a third-party
endpoint, so enabling it is a facility's decision to make explicitly. Every way
the endpoint can fail — no key, a timeout, an error status — returns the
keyword ranking with a warning rather than a failed search.

``scripts/demos/jev_instant_search.py`` runs the whole path over a small
in-memory corpus, with no database and no deployment.
