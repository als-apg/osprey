The config-key guard now asserts the rendered union's size instead of noting a
drift against it, so a stale count in the manifest fails the check with the
number to record.
