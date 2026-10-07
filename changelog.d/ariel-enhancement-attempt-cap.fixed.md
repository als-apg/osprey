An entry whose enhancement fails three times is left out of later enhancement
passes instead of being retried every cycle; the count is kept in the entry's
enhancement status, and a success clears it. Each module now runs only on the
entries it has not finished.
