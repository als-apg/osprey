A Channel Access target whose block spells `gateways: []` declares that it has
nothing to probe: a return to it as the deployment baseline skips the
reachability check and says "target declares no endpoints". A missing
`gateways` table is still refused as `gateways_missing`, so a forgotten table
is never read as a direct connection, and a table with gateways is probed as
before.
