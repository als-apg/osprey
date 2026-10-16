-- The preflight band: queueserver preflight checks run on a user's behalf in
-- [from, to). Never a bare OnBehalfOf match, which would also take the
-- sidecar's sign-in rows. The page matches each row to the nearest preceding
-- queue-submit tool call of the viewer within 60 s; in a person's own view an
-- unmatched row is never shown.
SELECT
    User,
    OnBehalfOf,
    Surface,
    Timestamp,
    LogAttributes['osprey.audit.decision'] AS decision,
    cityHash64(Body) AS body_hash,
    substring(Body, 1, 4096) AS body,
    length(Body) AS body_length
FROM telemetry.audit_ledger
WHERE User IN {queueservers:Array(String)}
    AND Surface = 'preflight'
    AND OnBehalfOf = {user:String}
    AND Timestamp >= {from:DateTime64(9)}
    AND Timestamp < {to:DateTime64(9)}
ORDER BY Timestamp, body_hash
LIMIT 1 BY User, Timestamp, body_hash
LIMIT 500
