A `data/textbooks` directory beside a deployment no longer grants the agent a
`Read` permission by its name alone. The convention was undeclared and could
not be revoked from a profile; grant extra read roots through
`claude_code.permissions.allow`, which is auditable in the profile.
