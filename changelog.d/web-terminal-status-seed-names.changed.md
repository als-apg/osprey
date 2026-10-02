`osprey status`, container seeding, `osprey reset` and `osprey feedback`
address web-terminal containers by the project name (`project_name`, else the
project directory's name): the status table looks up `<project>-web-<user>`,
and seeding writes each user's context into that container.
