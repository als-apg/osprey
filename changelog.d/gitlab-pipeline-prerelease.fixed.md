A deployment whose `requires_osprey_version` names a pre-release now gets a
GitLab pipeline that installs it with `--pre`. OSPREY and its connectors ship
as a pair from one tag, and pip admits a pre-release only for the requirement
that names one, so without the flag the validation job could pair a
pre-release framework with the older stable connectors. A stable floor renders
exactly as before.
