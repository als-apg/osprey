A container runtime that is slow to answer is no longer reported as not
running. The deploy verbs wait for `docker ps` / `podman ps` and the compose
version probe to answer, say they are still waiting after 5 seconds, and stop
only after 2 minutes, saying the runtime did not answer. A refusal names only
the runtimes it probed, each with the probe that failed.
