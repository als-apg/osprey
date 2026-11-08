Internal: the Channels panel resolves the `pv_directory` section at start, holds a
directory service for it, mounts `/api/directory` and adds the `directory`
descriptor to `/api/info`, all only when the section is configured.
