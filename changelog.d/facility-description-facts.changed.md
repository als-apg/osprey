The `facility_description` tool returns the build's generated facts page
(`data/facility_facts.md`) under `generated` alongside the hand-written
description, and `source` names both paths. The not-found envelope is raised
only when both files are absent, so an agent sees the generated facts where no
hand-written page exists.
