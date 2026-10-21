Deployment: `osprey build` provisions a shared `var/guarded_run/<target>/`
directory for each configured control target before compose runs, setgid and
group-writable, and every web terminal and dispatch worker mounts
`var/guarded_run` read-write under its project directory. A session run on the
host creates the directory under the repo root when it is missing.
`osprey.runtime.guarded_run.guarded_run_dir(target)` resolves it, keyed by the
target name.
