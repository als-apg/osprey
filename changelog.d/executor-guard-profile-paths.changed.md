**Rebuild required:** after upgrading, run `osprey build`. The python executor
refuses agent code on a build output without the protected-path list it now
writes, and `osprey up` and `osprey restart` refuse to start such a build
unless you pass `--as-built`. `osprey build` warns when your profile states a
path-shaped value under a configuration key OSPREY does not know as a path;
that path is not protected.
