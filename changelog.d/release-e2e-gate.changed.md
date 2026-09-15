The `/osprey:release` skill now requires the full end-to-end suite, including
the lanes that drive a real agent, to pass on the exact tree being tagged, and
spells out how to check each of those lanes individually. The Teams bridge
end-to-end module skips itself when the `teams` extra is absent instead of
ending collection for the whole run.
