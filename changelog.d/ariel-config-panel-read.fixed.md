The ARIEL panel reads `web.config_panel.enabled` through the loaded
configuration, as the web terminal already did. A deployment that writes the
switch as an environment-variable reference — `enabled: "${PANEL:-false}"` —
now closes the Config panel on both surfaces; before, it closed only on the
terminal and ARIEL left the panel reachable.
