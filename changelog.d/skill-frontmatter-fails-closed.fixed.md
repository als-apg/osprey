`osprey app check` reads a listed skill's `SKILL.md` past a leading byte-order
mark, so `hooks` in its frontmatter is refused (`no_hooks`); a frontmatter that
does not parse as a YAML mapping is refused (`skill_frontmatter`) instead of
being read as declaring no hooks.
