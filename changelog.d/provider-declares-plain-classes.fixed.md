providers: a registered provider class that does not subclass BaseProvider is read as declaring no capability flags, so it needs neither a key nor an endpoint, instead of failing every flag check.
