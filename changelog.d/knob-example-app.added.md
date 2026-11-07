The control-assistant preset ships `knob-example`, a small application with a panel, a read tool and an approved write tool.
The application SDK adds `owner_target`, the machine a request's owner is on, for a backend that answers from it.
`serve_tools` takes a `register(server)` function: the SDK builds the application's server and hands it over.
