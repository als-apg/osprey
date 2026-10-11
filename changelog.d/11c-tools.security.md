Files an agent's Python run leaves for the artifact gallery are read back as
untrusted input. A saved artifact is collected only when its manifest entry
names a plain file sitting directly in the run's `artifacts/` folder, and a
figure only when it is a plain file of the run's folder or its `figures/`
folder: a name that climbs out of the folder, an absolute path, a symbolic
link, a hard link to another file, a file over 512 MiB, and anything past
1000 files or 2 GiB in one run are skipped with a warning, and the run still
succeeds with its other artifacts. The bytes are read from the same open file
the checks were made on, so a file swapped after the check is not read.
