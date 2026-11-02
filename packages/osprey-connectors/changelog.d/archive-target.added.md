`archive_targets()` says which control targets the deployment's `archiver:`
block serves: the deployment's own MongoDB store serves the virtual
accelerator on every deployment that builds one; every other archiver but the
mock (the EPICS appliance, MYA, DOOCS) is the facility's and serves the live
machine and the stand-in that rehearses it, when its block names an address.
