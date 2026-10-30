-- What the agent recorded, from the record's archive: every closed day the
-- record has exported, counted by who wrote each entry and what kind it is,
-- the most frequent first. Run it from the repo root.
SELECT
    identity,
    kind,
    count(DISTINCT key) AS entries,
    min(ts) AS first_seen,
    max(ts) AS last_seen
FROM read_parquet('var/archive/*/agent_record/entries.parquet')
GROUP BY identity, kind
ORDER BY entries DESC, identity, kind
