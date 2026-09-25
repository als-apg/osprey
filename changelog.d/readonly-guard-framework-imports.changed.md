A ``readonly`` Python execution no longer imports Bluesky and ophyd-async
before the script runs. Their write entry points are refused the moment the
script imports one of them, so a run that reads and computes starts sooner.
Control-system client libraries are unchanged: their writes are still refused
before any line of the script runs.
