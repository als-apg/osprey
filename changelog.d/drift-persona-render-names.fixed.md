`osprey validate` no longer reports preset drift for a persona's `project`
and `project_path`. Those two name the render this deployment builds and
mounts, so a repo checked out under any directory name validates the same
way.
