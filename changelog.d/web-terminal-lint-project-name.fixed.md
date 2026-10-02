`osprey` web-terminal lint resolves each roster entry under the project name
(`project_name`, else the project directory's name), the same name provisioning
uses, so its persona checks no longer report against a `facility.prefix` project
that is never built.
