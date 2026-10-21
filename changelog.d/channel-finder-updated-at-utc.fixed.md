The middle-layer channel index stores `channels.updated_at` as a timestamp with
time zone, so it reads back as the import's instant instead of local wall time
labelled as UTC.
