The control target and write posture are now one record per deployment
instance. A switch or a posture change made in any window applies to every
session, notebook kernel and hook. Notebook cells read the deployment's
current target and posture on each run, so a popped-out notebook window or a
kernel restart keeps following the deployment. The Simple view's roster shows
the current target and its switch works.

**Breaking change:** the record schema changed together with the hooks. Run
`osprey build` to regenerate the deployment before starting it.
