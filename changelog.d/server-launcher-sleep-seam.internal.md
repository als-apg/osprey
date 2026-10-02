The companion-server launcher waits through a module-level seam, so its tests
count only the launcher's own waits and no longer fail when another thread in
the same test process is sleeping.
