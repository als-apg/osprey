Image builds keep their pip and apt downloads between builds. Every pip and
apt install in the project, persona and service images fetches through a
BuildKit cache mount, so a layer that has to rebuild (a new framework pin, a
changed dependency manifest) downloads only what changed, and images building
in the same deploy share one copy.
