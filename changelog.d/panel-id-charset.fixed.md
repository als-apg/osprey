A `web.panels` id outside `[A-Za-z0-9._-]`, or one that does not start with a
letter or digit, no longer answers 500 on every request for that panel. The
terminal refuses to start and names the block, `osprey validate` refuses the
same id in a profile, and a runtime registration under such an id answers 422.
