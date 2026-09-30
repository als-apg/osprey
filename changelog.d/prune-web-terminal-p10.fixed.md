The scaffold gallery refuses writes from an app that never resolved
`web.scaffold_gallery.write_enabled`, refuses to delete an owned artifact as an
orphan (release it with `delete_file` instead), deletes the file an ownership
record actually names when releasing with `delete_file`, and says so when a
release cannot restore the framework copy.
