Editing a framework file from the settings drawer's scaffold gallery now
always lands in the editor. Taking ownership loads the file twice — once for
the preview it reopens on, once for the editor — and when those two loads came
back out of order the panel showed the rendered preview under an active Edit
tab, with nothing to type into.
