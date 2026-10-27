Web terminal: an application panel's declared `web.panels.<id>.agent_routes`
are admitted on the panel credential, exact match only. On a login that does
not arm the application (`arm_writes` false or missing), the panel proxy
refuses write methods and websocket upgrades to it, naming the login and the
application.
