The largest timeseries file the artifact gallery will chart or tabulate is now
a config key, `artifact_server.max_timeseries_file_mb` (default 200, the
previous fixed bound). Over the cap those two views still refuse with a 413 and
the file stays downloadable; raising it costs resident memory on the machine
serving the gallery, since the handler loads the whole file to build the view.
