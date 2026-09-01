**Breaking change:** derived bluesky plan-device files now come from the
facility's own channel roster — the knowledge graph, or a channel-finder
database — instead of the channel-limits database, and derivation now runs
for every plan lane, including the live one. A facility whose limits
database named fewer channels than its graph or channel-finder store now
stages the full roster as plan devices. A build refuses outright when a lane
would arm writes over roster-derived devices while limits checking is not
enabled for that lane's target.
