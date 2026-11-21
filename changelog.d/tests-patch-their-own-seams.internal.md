`osprey reset`'s container probe looks up `subprocess.run` on every call.
Every module imports `subprocess` and `shutil` at its top and looks their functions up where it calls them.
