The workspace and store-index watchers take the watchdog observer they run on
as a constructor argument, defaulting to the platform's native one. Tests that
ask which filesystem change becomes which broadcast now run on a polling
observer, which reports what is on disk rather than what a notification stream
happened to carry; one test per watcher stays on the native backend.
