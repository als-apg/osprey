"""Cypher the graph paradigm's channel census is counted with.

The census is asked by two callers that share nothing else: the benchmark
runner, which opens its own driver against a project directory it was handed,
and the web explorer, which reads the running app's store context. Both must
count the same population or the number an operator reads in the browser and
the number a benchmark reports would quietly disagree.

This module is deliberately dependency-free — a constant and its docstring, no
imports at all. The explorer reaches it from
``osprey.interfaces.channel_finder.database_api``, which is imported whenever
the web app starts; pulling the benchmark runner in for one string dragged the
whole agent SDK into that import closure.
"""

from __future__ import annotations

#: The graph paradigm's channel census.
#:
#: ``ChannelBinding`` is the graph's channel: one node per address the facility
#: exposes, so counting them answers the same question the file-backed
#: paradigms answer by counting rows in their database.
GRAPH_CHANNEL_COUNT_CYPHER = "MATCH (b:ChannelBinding) RETURN count(b) AS n"
