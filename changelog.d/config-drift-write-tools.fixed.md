The session-start config drift check no longer goes quiet on a deployment that
lists extra `control_system.write_tools`. Those tools are approved per call and
never enter the agent config's deny list, so the check read their absence as a
hand-edited file and said nothing when config.yml had writes on but the agent
config still blocked them.
