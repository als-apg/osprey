The web-terminal landing title, the sign-in page's facility name, the event
dispatcher's default name and the agent prompts take the facility's name from
the build's facility identity (`data/facility/identity.yaml`), else the project
name; the config's `facility.name` and top-level `facility_name` no longer name
anything. A facility file that is present but unreadable, not JSON, or missing
its identity code is refused with an error that names the file.
