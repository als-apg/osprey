A release tag is now refused unless a full CI run, including the lanes that
drive a real agent, passed on the tagged tree; the release workflow checks this
before it builds anything, and the `/osprey:release` skill says how to satisfy
it. The Teams bridge end-to-end module skips itself when the `teams` extra is
absent instead of ending collection for the whole run.
