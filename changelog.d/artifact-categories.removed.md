Artifact categories are retired: the agent's save tools no longer take a
category or data type, and `osprey build` refuses `artifact_server.categories`.
Migrate each category to `artifact_server.extension_types.<key>: {label,
base_form}`, optionally with a `viewer` module; the sidebar now lists artifacts
under their agent session, grouped by kind, in place of the Artifacts and
Script runs rows.
