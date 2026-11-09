`osprey build` serialises the facility file once per build instead of three times, and reads hook
frontmatter and the facility schema with the libyaml loader; the build output is unchanged.
