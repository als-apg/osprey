The `control-assistant` preset runs the graph channel finder
(`channel_finder_mode: graph`) for every persona, and its standalone cards are
shared (`access: any`): any roster login opens them with their own password
instead of a per-card login link. The ARIEL card is renamed `logbook` (persona
`logbook`, preset `control-assistant-logbook`); the ARIEL product name is
unchanged everywhere else. `osprey init` refuses an `extends:` override before
resolving the layered profile, and the profile card marks shared cards.
