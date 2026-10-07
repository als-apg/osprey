The EPICS Archiver Appliance connector can reach an appliance behind a login
or a site CA: `archiver.settings.auth` names the environment variable that
holds a bearer token (`token_env`) or a user and the variable that holds the
password (`username`, `password_env`), and `archiver.settings.tls.ca_bundle`
names the CA file the appliance's certificate is checked against. The login is
sent only to the configured host.
