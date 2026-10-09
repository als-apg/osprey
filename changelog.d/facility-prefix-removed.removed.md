The `facility.prefix` config key is removed, with the lint that reported it
empty (`web_terminals.empty_facility_prefix`). Container names and persona
projects come from the project name, and the knowledge graph's identifiers
from the facility file's identity code; a profile that still sets the key
loads unchanged.
