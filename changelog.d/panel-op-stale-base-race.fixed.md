An agent's second panel step in a row is no longer refused as `stale_base`
when it reaches the page before the page has taken in the first step's
answer. The page now runs agent panel operations one at a time, each after
the previous one's answer is acknowledged (waiting at most the server's own
5-second answer timeout), so the revision an agent was handed is always the
one the page checks against. A step with nothing ahead of it still runs at
once.
