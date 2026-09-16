A deployment's service build contexts no longer keep a copy of the site CA
bundle under `build/services/<name>/` after the images that needed it are
built, matching what the project, persona and auth-sidecar builds already do.
