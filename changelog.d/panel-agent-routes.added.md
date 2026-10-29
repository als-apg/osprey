Web terminal: an application panel's declared `web.panels.<id>.agent_routes`
are admitted on the panel credential, exact match only. On a login whose
`control_system.writes` tier is not `allowed`, the panel proxy refuses write
methods and websocket upgrades to the application, naming the login and the
application; a panel block spelling `arm_writes` is refused at load.
