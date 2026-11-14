`ArchiverMetadata.is_archived` and `check_availability` answer `None` when the
archiver cannot say; the EPICS Archiver Appliance and DOOCS connectors no
longer claim every channel archived. New `ArchiverConnector.store_extent()`,
implemented by the MongoDB archiver.
