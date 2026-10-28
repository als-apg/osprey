**Breaking change:** on a render whose tier is `allowed`, a
`claude_code.permissions.remove_deny` that lifts `Bash` or `Monitor` is now a
build refusal; on a `never` render it stays a warning.
