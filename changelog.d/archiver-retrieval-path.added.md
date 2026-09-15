The EPICS Archiver Appliance connector takes an optional
`archiver.epics_archiver.retrieval_path`, the prefix under which the
appliance's retrieval servlet is reached. A bare appliance serves it at
`/retrieval`, which stays the default; a facility whose appliance sits behind
a reverse proxy that publishes the servlet under another name can now point
the connector at the proxy instead of needing a route to the appliance itself.
