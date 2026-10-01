An OSPREY source checkout on a heavily loaded host reports the commit it runs
rather than the version stamped at its last rebuild: the `git describe` probe
behind the running version waits up to 30 seconds, a bound only a hung git
reaches.
