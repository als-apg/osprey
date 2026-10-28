**Breaking change:** a write with no owner is refused, since no switch covers
it. Scheduled and triggered dispatch jobs run for no login, so their writes are
refused (`owner_unscheduled`), with no route back in this release; Bluesky
items with no owner no longer write either.
