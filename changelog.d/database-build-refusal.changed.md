`osprey channel-finder build-database` now fails, naming the family, when a
device family cannot be turned into a template. Such a family was previously
degraded to plain rows and the run still reported success, so the database
silently lost the family's device navigation.
