A deployment whose `requires_osprey_version` names a pre-release now gets a
GitLab pipeline that installs it with `--pre`. pip admits a pre-release only
for the requirement that names one, and OSPREY's floor on `osprey-connectors`
names none, so without the flag the validation job could pair a pre-release
framework with an older stable connectors than the pre-release it was tested
with. A stable floor renders exactly as before.
