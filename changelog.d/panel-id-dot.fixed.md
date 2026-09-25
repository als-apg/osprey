`osprey build` now refuses a `web.panels` id with a dot in it when the web
terminal would refuse the same id at startup, such as `a.b c` written as a
key under `web.panels:`. The build used to check only the part before the
first dot, so the deployment built and then failed to start.
