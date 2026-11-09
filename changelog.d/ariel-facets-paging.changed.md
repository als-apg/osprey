ARIEL: the web interface filters by facet counts in the rail, a day histogram
and ``author:``/``date:`` search operators, and pages with **Load more**; entry
fields declared ``facet: true`` become filter dimensions. ``/api/entries``,
``/api/search`` and the new ``/api/facets`` take a ``filters`` object, page by
cursor or offset and report ``total``/``total_exact``/``has_more``. Run
``osprey ariel migrate`` to backfill entry subjects. The MCP tools now return a
null ``entry_url`` for every ARIEL-only entry, including local ARIEL Web saves.
**Search modules:** descriptors gain ``accepts_entry_filter`` and
``accepts_offset``, ``ModuleOutput`` gains totals, and a module declaring
neither is called with a wider ``max_results``.
