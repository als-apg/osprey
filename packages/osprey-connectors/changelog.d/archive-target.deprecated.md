`standin.archive_belongs_to_standin`, `standin.ARCHIVER_RECORDER_SERVICE` and
`standin.DEPLOYED_SERVICES_KEY` are kept only for the framework releases that
import them and go in a later release: the deployment's store is the virtual
accelerator's and no store belongs to the stand-in. Ask `archive_targets()`
which targets an archive serves.
