The Nextcloud Talk bridge no longer tells a room that "nothing was changed"
when a run fails. A run that was approved a write and then timed out reaches
the same notice, so the claim was a safety assurance the bridge could not
check.
