The Control Assistant template no longer ships the ALS Archiver Appliance
URL in its `epics_archiver` block. The block ships as a commented
placeholder, and authoring it with your facility's values travels with the
flip to `archiver.type: epics_archiver`. Until then the connector refuses
to start without a `url`, which is the truthful state of a fresh
deployment.
