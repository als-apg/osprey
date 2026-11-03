**Breaking change:** history read on a control target comes only from that
machine's own archive. The facility's archive (the EPICS appliance, MYA,
DOOCS), when the `archiver:` block names one, serves the live machine and the
stand-in that rehearses it; the deployment's own store serves the virtual
accelerator, and the recorder samples only while the `va` target lands on
this deployment's simulator. The stand-in no longer owns a store, the live
target is no longer refused on the archive (`archive_belongs_to_standin` is no
longer read), and `archiver_read` answers `no_archive` on a target the archive
does not serve, naming the machine whose past it holds.
